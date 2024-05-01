#FrontendFlaskapp.py
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
import json
from Embed_Backend import get_embedding,classify_document, save_embedding, search_embeddings, rerank_results, generate_better_query, decode_formatting, encode_formatting, send_to_claude_and_get_chunks
import nltk
from nltk.tokenize import sent_tokenize
if not os.path.exists(nltk.data.find('tokenizers/punkt')):
    nltk.download('punkt')


logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'  # Ensure this directory exists within your project structure
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3', 'rtf'}

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

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    filename = secure_filename(file.filename)
    if filename.split('.')[-1].lower() not in app.config['ALLOWED_EXTENSIONS']:
        return jsonify(error=f"Unsupported file type: {filename}"), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    parsed = parser.from_file(file_path)
    text = parsed["content"] if parsed["content"] else ""
    metadata = parsed["metadata"]

    # Classification step here
    document_type = classify_document(text)  # Call the classification function
    if document_type is None:
        return jsonify(error="Failed to classify document."), 400

    # Encode formatting in the entire document text
    encoded_text = encode_formatting(text)

    sentences = sent_tokenize(encoded_text)
    numbered_sentences = {i+1: sentence for i, sentence in enumerate(sentences)}

    chunks = send_to_claude_and_get_chunks(numbered_sentences)

    for chunk_id, sentence_nums in chunks.items():
        chunk_text = " ".join(decode_formatting(numbered_sentences[num]) for num in sentence_nums)
        chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{chunk_id}.json"
        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], chunk_filename)

        logging.info(f"Generating embedding for chunk {chunk_id} of {filename}")
        embedding = get_embedding(chunk_text)
        if embedding is None:
            logging.error(f"Failed to generate embedding for chunk {chunk_id} of {filename}")
            continue  # Skip this chunk if embedding generation fails

        # Save the embedding along with the original document filename, chunk filename, and document type
        save_embedding(filename, chunk_filename, embedding, document_type)  # Updated save function to include document type

        # Create and save JSON for the chunk
        chunk_data = {'text': chunk_text, 'metadata': metadata, 'document_type': document_type}
        with open(chunk_path, 'w', encoding='utf-8') as json_file:
            json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)

    return jsonify(success=True, message="Document processed into chunks and saved", file_type=document_type, file_name=filename)


#the following code is used to search the file
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Generate a better query based on the query for embedding
    better_query = generate_better_query(query)
    if not better_query:
        return jsonify(error="Error generating better query"), 400
    # Obtain the embedding for the better query
    logging.info(f"better query: {better_query}")
    query_embedding = get_embedding(better_query)
    if query_embedding is None:
        return jsonify(error="Error generating query embedding"), 400

    results = search_embeddings(query_embedding, top_n=5)
    if not results:
        return jsonify(error="No matching documents found"), 404

    summaries = []
    for original_filename, chunk_filename, similarity in results:
        json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], chunk_filename)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                preview_text = ' '.join(data['text'].split()[:384])  # Extracted from the actual document
        except IOError:
            preview_text = "Preview not available"

        match_score = similarity * 100
        summaries.append({
            'file_name': original_filename,
            'preview_text': preview_text,
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

if __name__ == '__main__':
    app.run(debug=True)
