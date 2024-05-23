from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tika import parser
import os
import logging
import uuid
import hashlib
from shared_config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, TIKA_SERVER_PORT
from embed_backend import get_embedding, classify_extract_and_chunk, save_embedding, process_chunks_in_batches, save_embeddings_in_batches


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO, filename='upload_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def home():
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    filename = secure_filename(file.filename)
    if filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type: {filename}"), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    parsed = parser.from_file(file_path)
    text = parsed["content"].strip() if parsed["content"] else ""
    metadata = '1'

    parent_document_filesize = os.path.getsize(file_path)
    document_type_id, document_type_name, document_title, document_parties, numbered_sentences, chunks = classify_extract_and_chunk(text)
    if document_type_id is None or document_parties is None:
        return jsonify(error="Failed to classify document or extract parties."), 400

    parent_document_family_id = str(uuid.uuid4())
    parent_document_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    parent_document_type_id = document_type_id + 100

    parent_embedding = get_embedding(text, input_type='document')
    if parent_embedding is None:
        logging.error(f"Failed to generate embedding for parent document {filename}")
        return jsonify(error="Failed to generate embedding for parent document."), 500

    save_embedding(
        filename,
        document_title,
        document_parties,
        parent_embedding,
        parent_document_type_id,
        document_type_name,
        text,
        metadata,
        parent_document_family_id,
        parent_document_hash,
        parent_document_filesize,
    )

    embeddings, chunk_texts = process_chunks_in_batches(chunks, numbered_sentences, document_type_name, document_title, document_parties)
    save_embeddings_in_batches(
        embeddings,
        chunk_texts,
        filename,
        document_title,
        document_parties,
        document_type_id,
        document_type_name,
        metadata,
        parent_document_family_id,
        parent_document_hash,
        parent_document_filesize
    )

    return jsonify(success=True, message="Document processed into chunks and saved", file_type=document_type_id, file_name=filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
