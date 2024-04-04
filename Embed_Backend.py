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

def generate_hypothetical_document(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",  # Ensure this model identifier is correct
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"A user is attempting to search an embedding space of legal court cases with this query: \"{query}\". "
                               "Your role is to boost its semantic meaning to increase the likelihood of a match in an embedding space. "
                               "For example, if the query is: 'is it lawful to detain a person who has applied for refugee status who otherwise cannot be removed from the country?' "
                               "You should respond with: 'Can a person who has applied for refugee status and cannot be removed from the country be lawfully detained by the government pending their asylum decision?' "
                               "Remember, please respond without any additional explanations or text."
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            stop=None,
        )
        generated_document = response.choices[0]['message']['content'].strip()
        logging.info(f"GPT-3.5 generated document: {generated_document}")

        return generated_document
    except Exception as e:
        logging.error(f"Error generating hypothetical document: {e}")
        return None



def rerank_results(summaries, query):
    """
    Re-rank search results using GPT-3.5 Turbo Instruct based on the relevance of the preview text.
    """
    prompt = ("I have listed below 5 summaries derived from a search based on a specific query. "
              "Each summary is identified by a number (1 to 5). Your task is to carefully review each summary "
              "and then rank them purely based on their relevance to the original search query, from the most "
              "relevant to the least relevant.\n\n"
              "Please respond with only the numbers in the new order of relevance. Your response should be a "
              "simple, comma-separated list of numbers, indicating this new order from most to least relevant. "
              "For instance, if you find the third summary to be the most relevant, followed by the first, "
              "second, fifth, and finally the fourth, you should respond with: '3, 1, 2, 5, 4'.\n\n"
              "Remember, I need just the list of numbers in the correct order, without any additional explanations or text.\n")

    for index, summary in enumerate(summaries, start=1):
        prompt += f"{index}. {summary['preview_text']}\n"

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            stop=None,
            n=1
        )
        logging.info(f"GPT-3.5 Turbo response: {response.choices[0].text}")
        # Example of how to parse a simple ordered list response: "1, 3, 2, 4, 5"
        new_order = response.choices[0].text.strip().split(', ')
        # Apply the new order to summaries
        ordered_summaries = [summaries[int(idx) - 1] for idx in new_order]

        return ordered_summaries
    except Exception as e:
        logging.error(f"Error re-ranking results with GPT-3.5 Turbo Instruct: {e}")
        return summaries  # Return original summaries in case of an error
