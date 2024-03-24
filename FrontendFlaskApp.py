# FrontendFlaskApp.py adjustments
from flask import render_template
from flask import send_from_directory
from flask import request, render_template
from flask import Flask, request, jsonify
from flask import send_from_directory, send_file
from werkzeug.utils import secure_filename, safe_join
import os
from Embed_Backend import get_embedding, json_to_embedding, save_embedding, search_embeddings

import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'  # Ensure this directory exists within your project structure
app.config['ALLOWED_EXTENSIONS'] = {'json'}

@app.route('/')
def home():
    return render_template('Frontend.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        embedding = json_to_embedding(file_path)
        if embedding:
            # Save embedding to CSV
            save_embedding(filename, embedding, file_path)
            return jsonify(embedding=embedding)
        else:
            return jsonify(error="An error occurred during embedding"), 500
    else:
        return jsonify(error="Invalid file type"), 400

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return jsonify(error="Error generating query embedding"), 400
    result_filename, result_file_path = search_embeddings(query_embedding)
    if result_filename:
        # Assuming the file is a plain text file. Modify if your files are in a different format.
        result_file_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        try:
            with open(result_file_path, 'r') as file:
                original_text = file.read()
            # Return the original text and the filename in the response
            return jsonify(original_text=original_text, file_name=result_filename, file_path = result_file_path)
        except IOError as e:
            return jsonify(error="Failed to read the result file"), 500
    else:
        return jsonify(error="No matching documents found"), 404

@app.route('/files/<filename>')
def uploaded_file(filename):
    filename = secure_filename(filename)  # Correctly reassign the sanitized filename
    file_path = safe_join(app.config['UPLOAD_FOLDER'], filename)  # Use correct case for UPLOAD_FOLDER
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)
    else: 
        return jsonify(error="File not found"), 404  

if __name__ == '__main__':
    app.run(debug=True)
