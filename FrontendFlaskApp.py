from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename, safe_join
from tika import parser
import os
import logging
from tika_server import TikaServer, close_tika_server
from Embed_Backend import get_embedding, classify_extract_and_chunk, search_embeddings, save_embeddings_in_batches, process_chunks_in_batches, classify_document_with_title, extract_parties_from_document, save_embedding, send_to_claude_and_get_chunks
import time
import uuid
import hashlib
import anthropic

def load_api_key(env_path='./Claude.env'):
    with open(env_path, 'r') as file:
        return file.read().strip()

api_key = load_api_key()
client = anthropic.Anthropic(api_key=api_key)

logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3', 'rtf'}

# Start the Tika server
tika_server = TikaServer()

@app.route('/')
def home():
    return render_template('Frontend.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    '''GET FILE AND PARSE TEXT + metadata'''
    start_time = time.time()

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
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Grab file elapsed time: {elapsed_time:.2f} seconds")

    start_time = time.time()
    parsed = parser.from_file(file_path)
    text = parsed["content"].strip() if parsed["content"] else ""
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Parse file {elapsed_time:.2f} seconds")
    metadata = '1'

    '''GENERATE META DATA'''
    parent_document_filesize = os.path.getsize(file_path)

    start_time = time.time()
    document_type_id, document_type_name, document_title, document_parties, numbered_sentences, chunks = classify_extract_and_chunk(text)
    if document_type_id is None or document_parties is None:
        return jsonify(error="Failed to classify document or extract parties."), 400

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"classify doc {elapsed_time:.2f} seconds")

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
    )

    '''CHUNK AND SAVE'''
    start_time = time.time()
    embeddings, chunk_texts = process_chunks_in_batches(chunks, numbered_sentences, document_type_name, document_title, document_parties)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"process and embed chunks {elapsed_time:.2f} seconds")

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



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    context = data.get('context')

    if not message:
        return jsonify(error="Message is required"), 400

    full_context = f"Legal Text Context:\n{context}\n\nUser Message: {message}"
    try:
        messages = [{"role": "user", "content": full_context}]
        #logging.info(f"Sending to Claude: {messages}")

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            system = "You are a legal assistant. Provide research and analysis based on the provided legal text. Please be concise. Please respond using html tag where applicable for formatting.",
            temperature=0,
            messages=messages
        )
        logging.info(f"Sending to Claude: {response}")
        # Extract the text content from the TextBlock
        reply = ''.join(block.text for block in response.content)
        return jsonify(reply=reply)
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return jsonify(error="An error occurred during the chat"), 500




""" @app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    context = data.get('context')

    if not message:
        return jsonify(error="Message is required"), 400

    full_context = f"Legal Text Context:\n{context}\n\nUser Message: {message}"
    try:
        response = openai.ChatCompletion.create(
            model="GPT-4o",
            messages=[
                {"role": "system", "content": "You are a legal assistant. Provide research and analysis based on the provided legal text. Please be concise. Please respond using html tag where applicable for formatting."},
                {"role": "user", "content": full_context}
            ]
        )
        reply = response.choices[0].message['content']
        return jsonify(reply=reply)
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return jsonify(error="An error occurred during the chat"), 500 """

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
