import chromadb
from chromadb.config import Settings

def reset_chromadb():
    # Initialize the settings with allow_reset set to True
    settings = Settings(allow_reset=True)

    # Initialize the persistent client with the settings and path to your database
    client = chromadb.PersistentClient(path="./chromadb", settings=settings)

    # Perform a heartbeat check to ensure the client is connected
    heartbeat = client.heartbeat()
    print(f"Heartbeat: {heartbeat}")

    # Ask the user for confirmation before resetting the database
    user_input = input("Do you want to proceed with resetting ChromaDB? (y/n): ")

    if user_input.lower() == 'y':
        # Reset the ChromaDB database
        client.reset()
        print("ChromaDB has been reset.")
    else:
        print("Reset operation cancelled.")

if __name__ == "__main__":
    reset_chromadb()
