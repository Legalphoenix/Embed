import chromadb

# Define the path to your ChromaDB database
db_path = "/Users/volumental/Desktop/Embed-search/chromadb"

# Create a persistent client to connect to the database
client = chromadb.PersistentClient(path=db_path)

# Define the collection name you want to delete
collection_name = "recitals"

# Delete the specified collection
try:
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' has been successfully deleted.")
except ValueError as e:
    print(f"Error: {e}")
