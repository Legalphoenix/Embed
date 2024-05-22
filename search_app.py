from flask import Flask, request, jsonify, render_template
import logging
from embed_backend import get_embedding, search_embeddings

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, filename='search_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def home():
    return render_template('search.html')

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
            similarity = result.get("distance", [])

            if isinstance(metadata_list, list) and metadata_list:
                for index, meta in enumerate(metadata_list):
                    if index >= len(similarity):
                        break

                    original_filename = meta.get("original_file_name")
                    preview_text = meta.get("chunk_text", "Preview not available")
                    document_type_name = meta.get("document_type_name")
                    match_score = 1 - similarity[index]

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

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
