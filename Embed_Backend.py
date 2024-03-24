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
        'text': And(Use(str), lambda s: 0 < len(s) <= 1000)  # Validate 'text' is a string and within the character limit
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

# Function to save embedding with associated file name
def save_embedding(file_name, embedding, file_path):
    with open('embeddings.csv', 'a', newline='') as f:
        csv_writer = csv.writer(f)
        # Convert the numpy array to a list for csv writer
        csv_writer.writerow([file_name, file_path] + list(embedding))

# Function to search for the most similar embedding
def search_embeddings(query_embedding):
    best_match = None
    best_match_path = None
    smallest_distance = float('inf')
    with open('embeddings.csv', 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            file_name, file_path, *embedding_values = row
            embedding = np.array(embedding_values, dtype=float)
            distance = np.linalg.norm(query_embedding - embedding)
            if distance < smallest_distance:
                smallest_distance = distance
                best_match = file_name
                best_match_path = file_path
    return best_match, best_match_path
