import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "parent_court_cases"
collection = chroma_client.get_collection(name=collection_name)

# IDs to delete
ids_to_delete = [
    "5afa2d73-e9b6-45c4-8d51-bb78e2b13bda",
]

# Delete the specified IDs
collection.delete(ids=ids_to_delete)
print(f"Deleted the following IDs: {', '.join(ids_to_delete)}")