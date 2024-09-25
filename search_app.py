# search.py

from flask import Flask, request, jsonify, render_template
import logging
from embed_backend import get_embedding, search_embeddings, rerank_results, doc_type_id_to_collection
from werkzeug.utils import secure_filename, safe_join
import os
import threading

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
logging.basicConfig(level=logging.INFO, filename='search_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form['query']
        doc_type = request.form.get('doc_type', '0')  # Default to "All Categories"

        logging.info(f"Search request received: query={query}, doc_type={doc_type}")

        query_embedding = get_embedding(query, input_type='query')
        if query_embedding is None:
            logging.error("Error generating query embedding")
            return jsonify(error="Error generating query embedding"), 400
        logging.info(f"embedding {query_embedding}")

        # Convert doc_type to list of integers
        doc_type_list = list(map(int, doc_type.split(',')))

        results = search_embeddings(query_embedding, doc_type_list, top_n=10)
        if not results:
            logging.info("No matching documents found")
            return jsonify(error="No matching documents found"), 404

        logging.info(f"all results{results}")
        summaries = []
        for result in results:
            metadata_list = result["metadata"]
            similarity = result.get("distance", [])

            if isinstance(metadata_list, list) and metadata_list:
                for index, meta in enumerate(metadata_list):
                    if index >= len(similarity):
                        break

                    original_filename = meta.get("original_file_name")
                    preview_text = meta.get("full_preview_text", "No preview found")
                    document_type_name = meta.get("document_type_name")
                    match_score = 1 - similarity[index]

                    logging.info(f"File: {original_filename}, Full Preview Text: {preview_text}")

                    summaries.append({
                        'file_name': original_filename,
                        'preview_text': f"{preview_text}\n<Document Type: {document_type_name}>",
                        'match_score': match_score
                    })
                logging.info(f"Summaries: {summaries}")

        if not summaries:
            logging.info("No matching documents found in summaries")
            return jsonify(error="No matching documents found"), 404
        summaries = sorted(summaries, key=lambda x: x['match_score'], reverse=True)
        # ranked_summaries = rerank_results(summaries, query)
        monitor_memory_usage()
        return jsonify(results=summaries)
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return jsonify(error="An error occurred during the search"), 500


def preload_collections():
    """
    Preloads all collections by querying a small batch of embeddings or documents.
    This forces ChromaDB to load the collections into memory.
    """
    logging.info("Starting to preload all collections...")
    try:
        # Access the global collection mapping (doc_type_id_to_collection) and preload each collection
        for doc_type_id, collection in doc_type_id_to_collection.items():
            try:
                # Query a small batch of embeddings to load the collection into memory
                result = collection.get(limit=10, include=["embeddings"])
                logging.info(f"Preloaded collection '{collection.name}' with {len(result['embeddings'])} embeddings.")
            except Exception as e:
                logging.error(f"Error preloading collection '{collection.name}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error during collection preloading: {e}")
    logging.info("Completed preloading all collections.")



""" @app.route('/files/<filename>')
def uploaded_file(filename):
    secure_name = secure_filename(filename)
    file_path = safe_join(app.config['UPLOAD_FOLDER'], secure_name)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify(error="File not found"), 404
 """
if __name__ == '__main__':
    # Preload all collections before starting the Flask app
    '''preload_thread = threading.Thread(target=preload_collections)
    preload_thread.start()
    preload_thread.join()'''
    app.run(host='0.0.0.0')

'''if __name__ == '__main__':
   app.run(debug=True, use_reloader=True, host='0.0.0.0', port=3005)'''
