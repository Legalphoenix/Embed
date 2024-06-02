import chromadb
import json
import glob

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "parent_court_cases"
collection = chroma_client.get_collection(name=collection_name)

# Retrieve all documents and print the list of IDs
all_items = collection.get()
all_ids = all_items["ids"]
print(f"List of IDs in the collection: {all_ids}")

# Retrieve all batch output files
batch_output_files = glob.glob("batch_*_output.jsonl")

# Parse the output and update the collection
for batch_output_file in batch_output_files:
    with open(batch_output_file, 'r') as batch_output:
        batch_output_lines = batch_output.readlines()

    for line in batch_output_lines:
        response = json.loads(line)
        custom_id = response["custom_id"]
        new_full_preview_text = response["response"]["body"]["choices"][0]["message"]["content"]

        # Check if the custom_id exists in the collection's IDs
        if custom_id not in all_ids:
            print(f"custom_id {custom_id} not found in the collection's IDs")
            continue

        # Fetch the current metadata using the custom_id (document ID)
        result = collection.get(ids=[custom_id], include=["metadatas"])
        
        if not result["metadatas"]:
            print(f"No metadata found for custom_id: {custom_id}")
            print(f"Result from collection.get: {result}")
            continue
        
        current_metadata = result["metadatas"][0]
        
        # Update the full_preview_text in the metadata
        current_metadata["full_preview_text"] = new_full_preview_text
        
        # Update the collection with the new metadata
        collection.update(
            ids=[custom_id],  # The ID of the document to update
            metadatas=[current_metadata]  # The updated metadata
        )

    print(f"Collection updated with new full_preview_text from {batch_output_file}.")
