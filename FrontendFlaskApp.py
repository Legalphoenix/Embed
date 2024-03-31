from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
import json
from Embed_Backend import get_embedding, save_embedding, search_embeddings, validate_json



logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'  # Ensure this directory exists within your project structure
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3'}

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
        return jsonify(error="Invalid file type"), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info(f"File {filename} saved to {file_path}")

    # Use Apache Tika for content and metadata extraction
    parsed = parser.from_file(file_path)
    text = parsed["content"]  # The extracted text content
    metadata = parsed["metadata"]  # The extracted metadata

    # Example of processing and saving metadata (adjust according to your schema)
    # For simplicity, converting all metadata values to strings
    metadata_processed = {key: str(value) for key, value in metadata.items()}

    if not text:
        logging.error(f"No text extracted from {filename}")
        return jsonify(error="Failed to extract text"), 500

    data = {'text': text.strip()}  # Ensure text is not None or only whitespace
    if not validate_json(data):
        logging.error("Data structure from extracted text does not meet expectations")
        return jsonify(error="Invalid data structure"), 400

    embedding = get_embedding(data['text'])
    if embedding is None:
        logging.error("An error occurred during embedding")
        return jsonify(error="An error occurred during embedding"), 500

    save_embedding(filename, embedding, file_path)
    json_data = {'text': text.strip(), 'metadata': metadata_processed}  # Include metadata if needed
    json_filename = os.path.splitext(filename)[0] + '.json'
    with open(os.path.join(app.config['UPLOAD_FOLDER'], json_filename), 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file)
    # Save metadata alongside embeddings or in a separate process
    # Consider how to integrate metadata into your search and cataloging system

    return jsonify(success=True, metadata=metadata_processed)

#the following code is used to search the file
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return jsonify(error="Error generating query embedding"), 400

    result_filename, _ = search_embeddings(query_embedding)
    if not result_filename:
        return jsonify(error="No matching documents found"), 404
    #the following code is used to read the json file
    json_filename = os.path.splitext(result_filename)[0] + '.json'
    json_file_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
    try:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            original_text = data['text']
    except IOError:
        return jsonify(error="Failed to read the result file"), 500

    return jsonify(original_text=original_text, file_name=result_filename)

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
