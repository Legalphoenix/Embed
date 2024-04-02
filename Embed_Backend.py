#embded_backend.py
import csv
import numpy as np
import openai
import json
import logging
import os
from dotenv import load_dotenv
from schema import Schema, And, Use, SchemaError
openai.api_key_path = './API.env'


# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define a schema for JSON validation based on expected structure and content
def get_json_schema():
    return Schema({
        'text': Use(str)  # Remove the length condition for now
    })

def validate_json(data):
    schema = get_json_schema()
    try:
        schema.validate(data)
        return True
    except SchemaError as e:
        logging.error(f'JSON validation error: {e}')
        return False

# Implement the get_embedding function
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")  # Ensure no newlines interfere with the API call
    try:
        response = openai.Embedding.create(
            model=model,
            input=text,
        )
        return response['data'][0]['embedding']
    except openai.error.InvalidRequestError as e:
        if "tokens" in str(e):
            logging.error(f'Text exceeds the maximum token limit: {e}')
        else:
            logging.error(f'An unexpected error occurred with OpenAI API: {e}')
        return None
    except Exception as e:
        logging.error(f'An unexpected error occurred: {e}')
        return None


def json_to_embedding(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            if not validate_json(data):
                return None  # Logging is handled in the validate_json function

            # Using the get_embedding function to get the embedding
            return get_embedding(data['text'], "text-embedding-3-small")

    except FileNotFoundError:
        logging.error('File not found.')
        return None
    except json.JSONDecodeError:
        logging.error('Invalid JSON content.')
        return None
    except Exception as e:
        logging.error(f'An unexpected error occurred: {e}')
        return None

def save_embedding(original_file_name, chunk_file_name, embedding):
    with open('embeddings.csv', 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([original_file_name, chunk_file_name] + list(embedding))

def cosine_similarity(vec_a, vec_b):
    """Calculate the cosine similarity between two vectors."""
    cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return cos_sim


# Function to search for the most similar embedding
def search_embeddings(query_embedding, top_n=5):
    matches = []
    with open('embeddings.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            original_filename, chunk_filename, *embedding_values = row
            embedding = np.array(embedding_values, dtype=float)
            similarity = cosine_similarity(query_embedding, embedding)
            matches.append((original_filename, chunk_filename, similarity))

    # Sort matches by their similarity, descending
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

    # Return the top_n matches, now including the chunk filename
    top_matches = sorted_matches[:top_n]
    return top_matches  # Each item will have (original_filename, chunk_filename, similarity)

    # Sort matches by their similarity, descending
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

    # Return the top_n matches, excluding the similarity score if not needed
    top_matches = sorted_matches[:top_n]
    return [(match[0], match[2]) for match in top_matches]  # Return filename and similarity score
