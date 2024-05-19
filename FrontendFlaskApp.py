#Frontendflaskapp.py
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
import json
from tika_server import TikaServer, close_tika_server
from Embed_Backend import get_embedding, search_embeddings, flatten_metadata, classify_document_with_title, extract_parties_from_document, save_embedding, send_to_claude_and_get_chunks
import time
import uuid
import hashlib

logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3', 'rtf'}

# Start the Tika server
#tika_server = TikaServer()

@app.route('/')
def home():
    return render_template('Frontend.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Start the Tika server
tika_server = TikaServer()

@app.route('/upload', methods=['POST'])
def upload_file():

    '''GET FILE AND PARSE TEXT + metadata'''
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
    #metadata = parsed["metadata"]
    #metadata = flatten_metadata(metadata)
    metadata = '1'

    '''GENERATE META DATA'''
    parent_document_filesize = os.path.getsize(file_path)

    document_type_id, document_type_name, document_title = classify_document_with_title(text)
    if document_type_id is None:
        return jsonify(error="Failed to classify document."), 400

    document_parties = extract_parties_from_document(text)
    if document_parties is None:
        return jsonify(error="Failed to classify document."), 400

    parent_document_family_id = str(uuid.uuid4())  # Generate a unique document family ID
    parent_document_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()  # Generate a hash for the whole document

    parent_document_type_id = document_type_id + 100  # Generate parent document type ID

    '''EMBED AND SAVE PARENT DOCUMENT'''
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
        text,  # Use the entire document text
        metadata,
        parent_document_family_id,
        parent_document_hash,
        parent_document_filesize,
        is_parent=True  # Indicate that this is the parent document
    )

    '''CHUNK AND SAVE'''
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    numbered_sentences = {i + 1: line.strip() for i, line in enumerate(cleaned_lines)}

    chunks = send_to_claude_and_get_chunks(numbered_sentences)

    for chunk_id, sentence_nums in chunks.items():
        chunk_text = " ".join(numbered_sentences[num] for num in sentence_nums)
        document_type_and_title_descriptor = f"[Type: {document_type_name}] [Parent Document Title: {document_title}] [Parent Document Parties: {document_parties}]"
        chunk_text_with_type_title = document_type_and_title_descriptor + " " + chunk_text
        logging.info(f"Chunk {chunk_id} being sent for embedding: {chunk_text_with_type_title}")

        embedding = get_embedding(chunk_text_with_type_title, input_type='document')
        if embedding is None:
            logging.error(f"Failed to generate embedding for chunk {chunk_id} of {filename}")
            continue

        save_embedding(
            filename,
            document_title,
            document_parties,
            embedding,
            document_type_id,
            document_type_name,
            chunk_text,
            metadata,
            parent_document_family_id,
            parent_document_hash,
            parent_document_filesize,
            is_parent=False  # Indicate that this is a chunk
        )

    return jsonify(success=True, message="Document processed into chunks and saved", file_type=document_type_id, file_name=filename)


@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form['query']
        doc_type = int(request.form['doc_type'])

        logging.info(f"Search request received: query={query}, doc_type={doc_type}")

        query_embedding = get_embedding(query, input_type='query')
        if query_embedding is None:
            logging.error("Error generating query embedding")
            return jsonify(error="Error generating query embedding"), 400

        results = search_embeddings(query_embedding, doc_type, top_n=10)
        if not results:
            logging.info("No matching documents found")
            return jsonify(error="No matching documents found"), 404

        summaries = []
        for result in results:
            metadata_list = result["metadata"]
            similarity = result.get("distance", [])  # Get the list of distances

            if isinstance(metadata_list, list) and metadata_list:  # Ensure it's a non-empty list
                for index, meta in enumerate(metadata_list):
                    if index >= len(similarity):
                        break  # If there are more metadata items than similarity scores, break

                    original_filename = meta.get("original_file_name")
                    preview_text = meta.get("chunk_text", "Preview not available")  # Get preview text from metadata
                    document_type_name = meta.get("document_type_name")
                    match_score = 1 - similarity[index]  # Get the corresponding similarity score & make it more user friendly (higher is better instead of lower)

                    summaries.append({
                        'file_name': original_filename,
                        'preview_text': f"{preview_text}\n<Document Type: {document_type_name}>",
                        'match_score': match_score
                    })
                logging.info(f"Summaries: {summaries}")
        if not summaries:
            logging.info("No matching documents found in summaries")
            return jsonify(error="No matching documents found"), 404
        return jsonify(results=summaries)
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return jsonify(error="An error occurred during the search"), 500

#The below adds 1.5 seconds
    #modified_query = generate_modified_query(query)
    #if not modified_query:
        #return jsonify(error="Error generating modified query"), 400
    #logging.info(f"Modified query: {modified_query}")

#old code getting preview text took 0.2

    #reranked_summaries = rerank_results(summaries, modified_query)
    #return jsonify(results=reranked_summaries)



@app.route('/files/<filename>')
def uploaded_file(filename):
    secure_name = secure_filename(filename)
    file_path = safe_join(app.config['UPLOAD_FOLDER'], secure_name)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify(error="File not found"), 404

close_tika_server

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
