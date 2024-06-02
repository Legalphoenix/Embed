#embedrecitals.py
import docx
import os
import uuid
import hashlib
from chromadb import HttpClient, Settings
import voyageai
import logging
import re

# Initialize logging
logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Load the document
doc = docx.Document('/Users/volumental/Desktop/Embed-search/Scripts/fullrecitals.docx')

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

# Create collections
collection_recitals = chroma_client.get_or_create_collection(name="recitals", metadata={"hnsw:space": "ip"})
parent_collection_recitals = chroma_client.get_or_create_collection(name="parent_recitals", metadata={"hnsw:space": "ip"})

# Constants
filename = 'input_document.docx'
document_title = 'Regulation (EU) 2016/679 (General Data Protection Regulation)'
document_parties = 'European Union'
document_type_name = 'GDPR Recitals'
parent_document_type_id = 4

# Initialize variables
recitals = []
current_recital = ""
pattern = re.compile(r"^Recital (\d+)")

# Read document and split into recitals
for paragraph in doc.paragraphs:
    if pattern.match(paragraph.text):
        if current_recital:
            recitals.append(current_recital.strip())
        current_recital = paragraph.text
    else:
        current_recital += "\n" + paragraph.text

if current_recital:
    recitals.append(current_recital.strip())

# Full document text
text = "\n".join(recitals)
parent_document_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
parent_document_family_id = str(uuid.uuid4())
parent_document_filesize = os.path.getsize('/Users/volumental/Desktop/Embed-search/Scripts/fullrecitals.docx')

# Embed and save the parent document
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
save_embedding(parent_collection_recitals, text, parent_embedding, parent_metadata)

# Embed and save each recital
for i, recital in enumerate(recitals):
    chunk_embedding = get_embedding(recital, input_type='document')
    chunk_metadata = {
        'original_file_name': filename,
        'document_title': document_title,
        'document_parties': document_parties,
        'document_type_id': parent_document_type_id,
        'document_type_name': document_type_name,
        'chunk_text': recital,
        'full_preview_text': recital,
        'metadata': '1',
        'document_family_id': parent_document_family_id,
        'parent_hash': parent_document_hash,
        'parent_document_filesize': parent_document_filesize
    }
    save_embedding(collection_recitals, recital, chunk_embedding, chunk_metadata)

print(f"Processed and saved the document and its chunks into ChromaDB.")
