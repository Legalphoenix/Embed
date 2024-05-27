from openai import OpenAI
import json
import time
import logging
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'gpt_api.env'))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Load the specific .env file


# Initialize OpenAI client with the API key from the environment variable


def send_to_claude_and_get_chunks(document_text):
    # Create batch request JSONL file
    jsonl_file_path = create_batch_request_jsonl(document_text)

    # Upload JSONL file and create batch
    batch_id = create_batch_from_jsonl(jsonl_file_path)

    # Periodically check batch status until completed
    while True:
        batch_status = check_batch_status(batch_id)
        if batch_status["status"] == "completed":
            break
        elif batch_status["status"] == "failed":
            logging.error("Batch processing failed.")
            return {}
        time.sleep(60)  # Wait for 1 minute before checking again

    # Retrieve the batch results
    output_jsonl_file = retrieve_batch_results(batch_id)

    # Process the batch output to extract chunks
    if output_jsonl_file:
        chunks = process_batch_output(output_jsonl_file)
        return chunks
    else:
        logging.error("Batch processing not completed or failed.")
        return {}

# Function to create batch request JSONL file
def create_batch_request_jsonl(document_text, custom_id="large-document"):
    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": '''You are an expert legal document chunker. Please segment the document below into legally relevant chunks.
                    Output each chunk with its corresponding line numbers in the format: Chunk 1: 1,2,3,4'''
                },
                {
                    "role": "user",
                    "content": document_text
                }
            ],
            "max_tokens": 4000
        }
    }

    # Write the request to a JSONL file
    with open("batch_input.jsonl", "w") as file:
        file.write(json.dumps(batch_request) + "\n")

    return "batch_input.jsonl"

# Function to upload the JSONL file to OpenAI and create a batch
def create_batch_from_jsonl(jsonl_file_path):
    # Upload the file
    with open(jsonl_file_path, "rb") as file:
        batch_input_file = client.files.create(
            file=file,
            purpose="batch"
        )

    # Create the batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "large document chunking job"
        }
    )

    return batch.id

# Function to check the status of the batch
def check_batch_status(batch_id):
    batch_status = client.batches.retrieve(batch_id)
    return batch_status

# Function to retrieve the results of the batch
def retrieve_batch_results(batch_id):
    batch_status = client.batches.retrieve(batch_id)
    if batch_status["status"] == "completed":
        output_file_id = batch_status["output_file_id"]
        content = client.files.download(output_file_id)
        with open("batch_output.jsonl", "wb") as output_file:
            output_file.write(content)
        return "batch_output.jsonl"
    else:
        return None

# Function to process the batch output and extract chunks
def process_batch_output(jsonl_file_path):
    chunks = {}
    with open(jsonl_file_path, "r") as file:
        for line in file:
            response = json.loads(line)
            message_content = response["response"]["body"]["choices"][0]["message"]["content"].strip()

            # Parse the chunk data
            for line in message_content.split('\n'):
                if line.startswith('Chunk'):
                    chunk_number, sentence_numbers = line.split(': ')
                    chunk_sentences = [int(num) for num in sentence_numbers.split(',')]
                    chunks[int(chunk_number.split(' ')[1])] = chunk_sentences

    return chunks
