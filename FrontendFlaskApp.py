#FrontendFlaskapp.py
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
import json
from Embed_Backend import get_embedding,classify_document, TikaServer, save_embedding, search_embeddings, rerank_results, generate_better_query, send_to_claude_and_get_chunks
import atexit
import signal


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

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    #get file
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    #ensure it has a working name
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
    document_type, document_type_name = classify_document(text)  # Receive both ID and name
    if document_type is None:
        return jsonify(error="Failed to classify document."), 400

    #TESTING SECTION
    #Below cleans the doc of whitespace and then adds numbers. Ensures numbering is sequential.
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    numbered_sentences = {i + 1: line.strip() for i, line in enumerate(cleaned_lines)}

    #Below skips over whitespace when numbering. Results in non-sequential numbers sent to llm.
    #lines = text.split('\n')
    #numbered_sentences = {i + 1: line.strip() for i, line in enumerate(lines) if line.strip()}

    #logging.info(f"numbered sentences {numbered_sentences}")
    chunks = send_to_claude_and_get_chunks(numbered_sentences)

    for chunk_id, sentence_nums in chunks.items():
        chunk_text = " ".join(numbered_sentences[num] for num in sentence_nums)
        # Append the document type descriptor for embedding
        document_type_descriptor = f"\n<Document Type: {document_type_name}> </Document Type>"
        chunk_text_with_type = chunk_text + document_type_descriptor

        chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{chunk_id}.json"
        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], chunk_filename)
        # Log the text being sent for embedding
        #logging.info(f"Chunk {chunk_id} being sent for embedding: {chunk_text_with_type}")


        #logging.info(f"Generating embedding for chunk {chunk_id} of {filename}")
        embedding = get_embedding(chunk_text_with_type, input_type='document')  # Use modified text for embedding
        if embedding is None:
            logging.error(f"Failed to generate embedding for chunk {chunk_id} of {filename}")
            continue  # Skip this chunk if embedding generation fails

        # Save the embedding along with the original document filename, chunk filename, and document type
        save_embedding(filename, chunk_filename, embedding, document_type, document_type_name)


        # Create and save JSON for the chunk with only the original chunk text
        chunk_data = {'text': chunk_text, 'metadata': metadata, 'document_type_id': document_type, 'document_type_name': document_type_name}
        with open(chunk_path, 'w', encoding='utf-8') as json_file:
            json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)

    return jsonify(success=True, message="Document processed into chunks and saved", file_type=document_type, file_name=filename)



@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    doc_type = int(request.form['doc_type'])  # Expecting 0, 1, 2, or 3

    # Generate a better query based on the original query for embedding
    better_query = generate_better_query(query)
    if not better_query:
        return jsonify(error="Error generating better query"), 400
    logging.info(f"better query: {better_query}")

    # Obtain the embedding for the better query
    query_embedding = get_embedding(better_query, input_type='query')
    if query_embedding is None:
        return jsonify(error="Error generating query embedding"), 400

    # Search for the most similar embeddings
    results = search_embeddings(query_embedding, doc_type, top_n=5)
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
    reranked_summaries = rerank_results(summaries, query)
    return jsonify(results=reranked_summaries)


#the following code is used to download the file
@app.route('/files/<filename>')
def uploaded_file(filename):
    secure_name = secure_filename(filename)
    file_path = safe_join(app.config['UPLOAD_FOLDER'], secure_name)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify(error="File not found"), 404

def setup_signal_handlers(app, tika_server):
    def stop_tika_server():
        if tika_server:
            tika_server.stop()
            print("Tika server stopped")

    def signal_handler(sig, frame):
        stop_tika_server()
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(stop_tika_server)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
