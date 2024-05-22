import chromadb
from chromadb.config import Settings

def count_files_in_collections():
    # Initialize the persistent client with the path to your database
    settings = Settings()
    client = chromadb.PersistentClient(path="./chromadb", settings=settings)

    # Define the collections
    collections = {
        "legislation": client.get_or_create_collection(name="legislation"),
        "guidelines": client.get_or_create_collection(name="guidelines"),
        "court_cases": client.get_or_create_collection(name="court_cases"),
        "contracts": client.get_or_create_collection(name="contracts"),
        "parent_legislation": client.get_or_create_collection(name="parent_legislation"),
        "parent_guidelines": client.get_or_create_collection(name="parent_guidelines"),
        "parent_court_cases": client.get_or_create_collection(name="parent_court_cases"),
        "parent_contracts": client.get_or_create_collection(name="parent_contracts")
    }

    for collection_name, collection in collections.items():
        # Get all the IDs in the collection
        ids = collection.peek(n=10000)["ids"]
        
        # Output the number of files and file names
        print(f"Collection '{collection_name}' contains {len(ids)} files:")
        for file_id in ids:
            print(f" - {file_id}")

if __name__ == "__main__":
    count_files_in_collections()
