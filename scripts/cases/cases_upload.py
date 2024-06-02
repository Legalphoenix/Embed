import os
import uuid
import hashlib
import logging
from chromadb import HttpClient, Settings
import voyageai
from tika import parser

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Function to get embedding
def get_embedding(text, input_type=None):
    vo = voyageai.Client(api_key='pa-HDqADASg6DbYlfiZhbtf2n5HCek1CpiouLod9AGALzA')
    documents_embeddings = vo.embed(text, model="voyage-law-2", input_type=input_type).embeddings
    if documents_embeddings and isinstance(documents_embeddings[0], list):
        return [item for sublist in documents_embeddings for item in sublist]
    return documents_embeddings

# Function to save embeddings
def save_embedding(collection, document, embedding, metadata):
    unique_id = str(uuid.uuid4())
    collection.add(documents=[document], embeddings=[embedding], metadatas=[metadata], ids=[unique_id])

# Initialize ChromaDB client
chroma_client = HttpClient(
    host="localhost",
    port="8000",
    ssl=False,
    headers={},
    settings=Settings()
)

# Create collection for court cases
parent_collection_court_cases = chroma_client.get_or_create_collection(name="parent_collection_court_cases", metadata={"hnsw:space": "ip"})

# Directory containing court cases
directory_path = '/Users/volumental/Desktop/Embed-search/scripts/dpcuria_cases'

# Constants
document_title = 'PRIVACY COURT CASE'
document_parties = 'European Union'
document_type_name = 'parent_collection_court_cases'
parent_document_type_id = 103

# Process each document in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        logging.info(f"Processing file: {file_path}")

        # Read the document
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Generate unique identifiers
        parent_document_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        parent_document_family_id = str(uuid.uuid4())
        parent_document_filesize = os.path.getsize(file_path)

        # Embed the parent document
        parent_embedding = get_embedding(text, input_type='document')
        parent_metadata = {
            'original_file_name': filename,
            'document_title': document_title,
            'document_parties': document_parties,
            'document_type_id': parent_document_type_id,
            'document_type_name': document_type_name,
            'chunk_text': text,
            'full_preview_text': text,
            'metadata': '1',
            'document_family_id': parent_document_family_id,
            'parent_hash': parent_document_hash,
            'parent_document_filesize': parent_document_filesize
        }

        # Save the parent document embedding
        save_embedding(parent_collection_court_cases, text, parent_embedding, parent_metadata)
        logging.info(f"Saved embedding for file: {filename}")

print("Processed and saved all documents into ChromaDB.")
