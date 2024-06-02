import chromadb
from docx import Document

# Define the path to your ChromaDB database
db_path = "/Users/volumental/Desktop/Embed-search/chromadb"

# Create a persistent client to connect to the database
client = chromadb.PersistentClient(path=db_path)

# Define the collection name
collection_name = "recitals"

# Get the collection
collection = client.get_collection(collection_name)

# Retrieve all items from the collection
items = collection.get()

# Create a new Document
doc = Document()

# Iterate through the items and add the text from 'preview_full_text' metadata to the document
if 'metadatas' in items:
    for item in items['metadatas']:
        if 'full_preview_text' in item:
            doc.add_paragraph(item['full_preview_text'])

# Save the document
output_path = '/Users/volumental/Desktop/recitals_full_text.docx'
doc.save(output_path)

print(f"Document created successfully at {output_path}.")
