from chromadb import HttpClient, Settings

# Initialize ChromaDB client
chroma_client = HttpClient(
    host="localhost",
    port="8000",
    ssl=False,
    headers={},
    settings=Settings()
)

# Collection names
old_collection_name = "parent_collection_court_cases"
new_collection_name = "parent_court_cases"

# Delete old collection if it exists
try:
    chroma_client.delete_collection(name=new_collection_name)
    print(f"Existing collection '{new_collection_name}' deleted.")
except Exception as e:
    print(f"No existing collection named '{new_collection_name}' found. Proceeding with renaming.")

# Fetch the old collection
try:
    old_collection = chroma_client.get_collection(name=old_collection_name)
except Exception as e:
    print(f"Collection named '{old_collection_name}' not found.")
    exit()

# Rename the collection
try:
    old_collection.modify(name=new_collection_name)
    print(f"Collection renamed from '{old_collection_name}' to '{new_collection_name}' successfully.")
except Exception as e:
    print(f"Failed to rename collection: {e}")
