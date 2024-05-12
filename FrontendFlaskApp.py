#FrontendFlaskapp.py
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
import json
from tika_server import TikaServer, close_tika_server
from Embed_Backend import get_embedding, classify_document_with_title, extract_parties_from_document, save_embedding, search_embeddings, rerank_results, generate_modified_query, send_to_claude_and_get_chunks

logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3', 'rtf'}

# Start the Tika server
tika_server = TikaServer()

#the following code is used to render the Frontend.html file
@app.route('/')
def home():
    return render_template('Frontend.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

'''UPLOAD FUNCTIONALITY'''
@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    #get file
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    #ensure it has a working name and an allowed filetype
    filename = secure_filename(file.filename)
    if filename.split('.')[-1].lower() not in app.config['ALLOWED_EXTENSIONS']:
        return jsonify(error=f"Unsupported file type: {filename}"), 400

    #save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    #tika parses the content from the file & stores it in the text variable. Metadata also stored in metadata.
    parsed = parser.from_file(file_path)
    text = parsed["content"] if parsed["content"] else ""
    metadata = parsed["metadata"]


    #text is sent for classification via Haiku model and we receive both a document_type id and the document_type_name
    document_type_id, document_type_name, document_title = classify_document_with_title(text)  # Receive both ID and name
    if document_type_id is None:
        return jsonify(error="Failed to classify document."), 400

    #text is sent for extraction of parties names via Haiku model
    document_parties = extract_parties_from_document(text)  # Receive parties names
    if document_parties is None:
        return jsonify(error="Failed to classify document."), 400


    #Below cleans the doc of whitespace and then adds numbers. Ensures numbering is sequential.
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    numbered_sentences = {i + 1: line.strip() for i, line in enumerate(cleaned_lines)}

    #TESTING SECTION
    #Below skips over whitespace when numbering. Results in non-sequential numbers sent to llm.
    #lines = text.split('\n')
    #numbered_sentences = {i + 1: line.strip() for i, line in enumerate(lines) if line.strip()}

    #logging.info(f"numbered sentences {numbered_sentences}")
    chunks = send_to_claude_and_get_chunks(numbered_sentences)

    #Code to create chunks for embedding and save chunk json files
    for chunk_id, sentence_nums in chunks.items():
        chunk_text = " ".join(numbered_sentences[num] for num in sentence_nums)
        #document type/classification written in this format for purposes of embedding
        document_type_and_title_descriptor = f"[Type: {document_type_name}] [Parent Document Title: {document_title}] [Parent Document Title: {document_parties}]"
        #content + document type/classification
        chunk_text_with_type_title = document_type_and_title_descriptor + " " + chunk_text
        #filename
        chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{chunk_id}.json"
        #file path
        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], chunk_filename)
        # Log the text being sent for embedding
        logging.info(f"Chunk {chunk_id} being sent for embedding: {chunk_text_with_type_title}")

        #get the embeddings for the chunk
        embedding = get_embedding(chunk_text_with_type_title, input_type='document')
        if embedding is None:
            logging.error(f"Failed to generate embedding for chunk {chunk_id} of {filename}")
            continue  # Skip this chunk if embedding generation fails

        # Save the embedding along with the original document filename, chunk filename, and document type
        save_embedding(filename, chunk_filename, document_title, document_parties, embedding, document_type_id, document_type_name)

        # Save JSON for the chunk with only the original chunk text
        chunk_data = {'text': chunk_text, 'metadata': metadata, 'document_type_id': document_type_id, 'document_type_name': document_type_name, 'document_title': document_title, 'document_title': document_parties}
        with open(chunk_path, 'w', encoding='utf-8') as json_file:
            json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)

    return jsonify(success=True, message="Document processed into chunks and saved", file_type=document_type_id, file_name=filename)


'''SEARCH FUNCTIONALITY'''
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    doc_type = int(request.form['doc_type'])  # Expecting 0, 1, 2, or 3

    # Generate a spell checked query based on the original query for embedding
    modified_query = generate_modified_query(query)
    if not modified_query:
        return jsonify(error="Error generating modified query"), 400
    logging.info(f"modified query: {modified_query}")

    # Obtain the embedding for the spell checked/modified query
    query_embedding = get_embedding(modified_query, input_type='query')
    if query_embedding is None:
        return jsonify(error="Error generating query embedding"), 400

    # Search for the most similar embeddings
    results = search_embeddings(query_embedding, doc_type, top_n=15)
    if not results:
        return jsonify(error="No matching documents found"), 404

    summaries = []
    for original_filename, chunk_filename, similarity in results:
        json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], chunk_filename)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                preview_text = ' '.join(data['text'].split()[:2000])  # Extracted from the actual document. (if you want to keep some formatting) preview_text = data['text'][:10000]
                preview_text_with_type = f"{preview_text}\n<Document Type: {data['document_type_name']}> </Document Type>"
        except IOError:
            preview_text = "Preview not available"

        match_score = similarity * 100
        summaries.append({
            'file_name': original_filename,
            'preview_text': preview_text_with_type,
            'match_score': match_score
        })
    # Re-rank based on the relevance of the preview text to the query
    reranked_summaries = rerank_results(summaries, modified_query)
    return jsonify(results=reranked_summaries)


'''DOWNLOAD FILE FUNCTIONALITY'''
@app.route('/files/<filename>')
def uploaded_file(filename):
    secure_name = secure_filename(filename)
    file_path = safe_join(app.config['UPLOAD_FOLDER'], secure_name)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify(error="File not found"), 404

#code to close tika server
close_tika_server


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
