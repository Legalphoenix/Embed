import os

UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx', 'xls', 'pptx', 'odt', 'json', 'html', 'xml', 'wav', 'mp3', 'rtf'}

TIKA_SERVER_JAR = 'tika/tika-server-standard-2.9.2.jar'
TIKA_SERVER_PORT = 9998  # You can choose any available port

CHROMADB_PATH = './chromadb'
