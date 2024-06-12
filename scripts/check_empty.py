import chromadb
import json

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "parent_court_cases"
collection = chroma_client.get_collection(name=collection_name)

# Search for the document with "court" in the chunk_text
result = collection.get(where_document={"$contains": "court"}, include=["documents", "metadatas"])

# Check if any documents were found
if result["documents"]:
    # Assuming we want the first matching document
    chunk_text = result["documents"][0]
    full_preview_text = result["metadatas"][0].get("full_preview_text", "")

    print("Chunk Text:")
    print(chunk_text)
    print("\nFull Preview Text:")
    print(full_preview_text)
else:
    print("No document found with 'C-604/22' in chunk_text.")
