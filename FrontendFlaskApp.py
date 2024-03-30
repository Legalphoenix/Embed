#flaskap
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename, safe_join
import os
import logging
import textract
import io
import pandas as pd
from Embed_Backend import get_embedding, save_embedding, search_embeddings, validate_json
import json
from bs4 import BeautifulSoup
import lxml.html as lh
from lxml import etree

logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'  # Ensure this directory exists within your project structure
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3'}

def html_to_structured_text(html_content):
    """Convert HTML content to structured text."""
    soup = BeautifulSoup(html_content, 'lxml')
    structured_texts = []

    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol']):
        text = f"{tag.name}: {tag.get_text(separator=' ', strip=True)}"
        structured_texts.append(text)

    return '\n'.join(structured_texts)

def xml_to_structured_text(xml_bytes):
    try:
        root = etree.fromstring(xml_bytes)
        # Temporarily return a simple JSON for testing
        return json.dumps({"tag": root.tag})
    except etree.XMLSyntaxError as e:
        logging.error(f"XML parsing error: {e}")
        return None

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
    file_extension = filename.rsplit('.', 1)[1].lower()
    if not allowed_file(filename):
        logging.error(f"Invalid file type: {filename}")
        return jsonify(error="Invalid file type"), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info(f"File {filename} saved to {file_path}")

    if file_extension in ['xlsx', 'xls']:
        # Read Excel file into DataFrame
        try:
            df = pd.read_excel(file_path)
            text = df.to_json(orient='records')
        except Exception as e:
            logging.error(f"Error processing Excel file: {e}")
            return jsonify(error="Failed to process Excel file"), 500
    elif file_extension == 'html':
        try:
            html_content = file.read().decode('utf-8')
            text = html_to_structured_text(html_content)
        except Exception as e:
            logging.error(f"Failed to process HTML file: {e}")
            return jsonify(error="Failed to process HTML file"), 500
    elif file_extension == 'xml':
        try:
            file.seek(0)  # Ensure we're reading from the beginning of the file
            xml_bytes = file.read()  # Read the file content as bytes
            text = xml_to_structured_text(xml_bytes)  # Process the XML bytes
            if text is None:  # Check if the conversion was successful
                raise ValueError("Failed to convert XML to JSON")
        except Exception as e:
            logging.error(f"Failed to process XML file: {e}")
            return jsonify(error="Failed to process XML file"), 500
    else:
            # Use textract for other file types
        try:
            text = textract.process(file_path).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to extract text: {e}")
            return jsonify(error="Failed to extract text"), 500

    data = {'text': text}
    if not validate_json(data):
        logging.error("Data structure from extracted text does not meet expectations")
        return jsonify(error="Invalid data structure"), 400

    embedding = get_embedding(data['text'])
    if embedding is None:
        logging.error("An error occurred during embedding")
        return jsonify(error="An error occurred during embedding"), 500

    save_embedding(filename, embedding, file_path)
    json_filename = os.path.splitext(filename)[0] + '.json'
    with open(os.path.join(app.config['UPLOAD_FOLDER'], json_filename), 'w', encoding='utf-8') as f:
        json.dump(data, f)

    return jsonify(success=True)

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
