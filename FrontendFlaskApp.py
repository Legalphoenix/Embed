#Flaskapp
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
import json
from Embed_Backend import get_embedding, save_embedding, search_embeddings, rerank_results, generate_hypothetical_document, validate_json



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
    logging.info(f"Received files: {request.files}")
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify(error="No file part"), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify(error="No selected file"), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        logging.error(f"Invalid file type: {filename}")
        allowed_exts = ", ".join(app.config['ALLOWED_EXTENSIONS'])
        return jsonify(error=f"The file type you uploaded is not supported. The following file types are supported: {allowed_exts}"), 400


    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info(f"File {filename} saved to {file_path}")

    parsed = parser.from_file(file_path)
    text = parsed["content"] if parsed["content"] else ""
    metadata = parsed["metadata"]

    # New chunking logic
    CHUNK_SIZE = 2000  # Maximum characters per chunk
    text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    for i, chunk in enumerate(text_chunks):
        chunk_filename = f"{os.path.splitext(filename)[0]}_part_{i+1}.json"
        chunk_path = os.path.join(app.config['UPLOAD_FOLDER'], chunk_filename)

        embedding = get_embedding(chunk)
        if embedding is None:
            logging.error(f"Failed to generate embedding for chunk {i+1} of {filename}")
            continue  # Skip this chunk

        # Save the embedding along with the original document filename and chunk filename
        save_embedding(filename, chunk_filename, embedding)

        # Create and save JSON for the chunk
        chunk_data = {'text': chunk.strip(), 'metadata': metadata}
        with open(chunk_path, 'w', encoding='utf-8') as json_file:
            json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)

    return jsonify(success=True, message="Document processed and embeddings generated", file_name=filename)


#the following code is used to search the file
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']

    # Generate a hypothetical document based on the query for embedding
    hypothetical_doc = generate_hypothetical_document(query)
    if not hypothetical_doc:
        return jsonify(error="Error generating hypothetical document"), 400
    # Obtain the embedding for the hypothetical document
    logging.info(f"hyp doc: {hypothetical_doc}")
    query_embedding = get_embedding(hypothetical_doc)
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
