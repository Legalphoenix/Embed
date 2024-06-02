import chromadb
import json

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "parent_court_cases"
collection = chroma_client.get_collection(name=collection_name)

# Retrieve all items in the collection
items = collection.get(include=["documents", "metadatas"])

# Prepare data for Batch API
batch_requests = []
for doc, metadata in zip(items["documents"], items["metadatas"]):
    custom_id = metadata["id"]
    full_preview_text = doc  # Assuming 'documents' contain 'full_preview_text'
    instructions = "Your specific instructions here"

    # Prepare the request for the batch file
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "chat-gpt4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{instructions}\n\n{full_preview_text}"}
            ],
            "max_tokens": 4096
        }
    }
    batch_requests.append(request)

# Save batch requests to a .jsonl file
batch_file_path = "batch_input.jsonl"
with open(batch_file_path, 'w') as batch_file:
    for request in batch_requests:
        batch_file.write(json.dumps(request) + "\n")

print("Batch requests prepared and saved to batch_input.jsonl.")
