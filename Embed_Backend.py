#embded_backend.py
import csv
import numpy as np
import openai
import subprocess
import logging
import os
from os.path import dirname, abspath, join
from dotenv import load_dotenv
from schema import Schema, And, Use, SchemaError
import anthropic
import voyageai
import socket

#API Key handling
openai.api_key_path = './API.env'
voyageai.api_key_path = './Voyage.env'
vo = voyageai.Client()
def load_api_key(env_path='./Claude.env'):
    with open(env_path, 'r') as file:
        return file.read().strip()

api_key = load_api_key()
client = anthropic.Anthropic(api_key=api_key)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Singleton pattern to ensure we only start one Tika server
class TikaServer:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(TikaServer, cls).__new__(cls)
            cls.instance.process = cls.start_tika_server()
        return cls.instance

    @staticmethod
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    @staticmethod
    def start_tika_server():
        tika_jar = join(dirname(abspath(__file__)), 'tika', 'tika-server-standard-2.9.2.jar')
        if not os.path.isfile(tika_jar):
            raise FileNotFoundError(f"Tika server JAR not found: {tika_jar}")

        port = TikaServer.find_free_port()
        os.environ['TIKA_SERVER_ENDPOINT'] = f'http://localhost:{port}'
        os.environ['TIKA_SERVER_JAR'] = f"file:///{tika_jar}"
        command = ['java', '-jar', tika_jar, f'-p{port}']
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self):
        if self.instance and self.instance.process:
            self.instance.process.terminate()
            self.instance.process.wait()

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

#Fetches the embedding for the given text, specifying if it's a query or a document.
def get_embedding(text, input_type=None):
    text = text.replace("\n", " ")  # Ensure no newlines interfere with the API call
    # Specify input_type if provided, otherwise default to None
    documents_embeddings = vo.embed(text, model="voyage-law-2", input_type=input_type).embeddings
    # Flatten the embeddings list if it's a list of lists
    if documents_embeddings and isinstance(documents_embeddings[0], list):
        return [item for sublist in documents_embeddings for item in sublist]
    return documents_embeddings  # Return as is if it's already flat



    #try:
        #response = openai.Embedding.create(
         #   model=model,
          #  input=text,
        #)
        #return response['data'][0]['embedding']
    #except openai.error.InvalidRequestError as e:
        #if "tokens" in str(e):
         #   logging.error(f'Text exceeds the maximum token limit: {e}')
        #else:
         #   logging.error(f'An unexpected error occurred with OpenAI API: {e}')
        #return None
    #except Exception as e:
        #logging.error(f'An unexpected error occurred: {e}')
        #return None

def save_embedding(original_file_name, chunk_file_name, embedding, document_type_id, document_type_name):
    with open('embeddings.csv', 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([original_file_name, chunk_file_name, document_type_id, document_type_name] + list(embedding))



def cosine_similarity(vec_a, vec_b):
    """Calculate the cosine similarity between two vectors."""
    cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return cos_sim


# Function to search for the most similar embedding, filtered by document type
def search_embeddings(query_embedding, doc_type, top_n=5):
    matches = []
    with open('embeddings.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            # Adjusted to account for the additional document_type_name column
            original_filename, chunk_filename, document_type_id, document_type_name, *embedding_values = row
            # Filter results based on the document type
            if doc_type == 0 or int(document_type_id) == doc_type:
                embedding = np.array(embedding_values, dtype=float)
                similarity = cosine_similarity(query_embedding, embedding)
                matches.append((original_filename, chunk_filename, similarity))
    # Sort matches by their similarity, descending
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
    # Return the top_n matches, now including the chunk filename
    top_matches = sorted_matches[:top_n]
    return top_matches  # Each item will have (original_filename, chunk_filename, similarity)


#refine the users query
def generate_better_query(query):
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
                    "content": f"A user is attempting to search an embedding space with this query: \"{query}\". Fix any spelling errors but otherwise replicate the exact query in your response "
                               #"Your role is to boost its semantic meaning to increase the likelihood of a match in an embedding space. "
                               #"For example, if the query is: 'is it lawful to detain a person who has applied for refugee status who otherwise cannot be removed from the country?' "
                               #"You should respond with: 'Can a person who has applied for refugee status and cannot be removed from the country be lawfully detained by the government pending their asylum decision?' "
                               #"Remember, please respond without any additional explanations or text."
                }
            ],
            temperature=0,
            max_tokens=1024,
            stop=None,
        )
        generated_document = response.choices[0]['message']['content'].strip()
        logging.info(f"GPT-3.5 generated document: {generated_document}")

        return generated_document
    except Exception as e:
        logging.error(f"Error generating better query: {e}")
        return None


def rerank_results(summaries, query):
    """
    Re-rank search results using Voyage AI reranker.
    """
    #for i, summary in enumerate(summaries):
     #   logging.info(f"Summary {i + 1} for reranking: {summary}")
    # Extract preview texts for reranking
    documents = [f"{summary['file_name']}: {summary['preview_text']}" for summary in summaries]
    #logging.info(f"Sending the following previews for reranking: {documents}")
    #logging.info(f"Starting rerank with query: {query} and {len(documents)} documents.")

    # Call the rerank method from the Voyage AI library
    try:
        reranking = vo.rerank(query, documents, model="rerank-lite-1")
        # Collect reranked results according to the new relevance scores
        ordered_summaries = [summaries[r.index] for r in sorted(reranking.results, key=lambda x: -x.relevance_score)]
        logging.info(f"Reranking successful. Reordered indices: {[r.index for r in reranking.results]}")
        relevance_scores = [round(r.relevance_score * 100, 2) for r in reranking.results]
        logging.info(f"Relevance scores (as percentages): {relevance_scores}")
        return ordered_summaries
    except Exception as e:
        logging.error(f"Error re-ranking results with Voyage AI: {e}")
        return summaries  # Return original summaries in case of an error




#Chunking function
def send_to_claude_and_get_chunks(numbered_sentences):

    # Prepare the content by encoding formatting and joining sentences with a unique identifier
    # test with opening and closing tags - works well - use the below again
    sentences_content = ''.join([f'<line id="{num}">{sentence}</line>' for num, sentence in numbered_sentences.items()])
    messages = [{"role": "user", "content": '<documents> ' + sentences_content + '''
                 </documents> <instructions> Check the final <line id="num">sentence</line> number first to ensure you
                do not go past that number when generating your chunks.  </instructions>  '''}]

    logging.info(f"Sending to Claude: {messages}")

    # Create a message to Claude.
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0,
        system='''<role>You are an expert legal embedding chunking program. I am embedding a court case, legilsation, contract, or guidelines.
        Please read the document below carefully and do nothing except follow the exact instructions.
        </role> <instruction 1> You should segment the document that has been provided into legally relevant chunks.
        Imagine how a lawyer might group the passages together for an embedding database.
        For legislation and contracts, keep whole sections together, including headings and section numbers.
        For court case or guidelines, keep lines of reasoning, arguments, guidance, entire orders sections, ratio decidendi (reasons for a decision)  or similar ideas together.</instruction>
        <instruction 2> In order to minimize the amount of text in your output, we will do the following:
        1. You will be provided with the document formatted such that each sentence or group of sentences
        from the case are numbered top to bottom idicated by '<line id=num>sentence</line>' where num is the line number.
        2. When you chunk the document, you will output your segmentated chunks strictly in the following example format:
        </instruction>
        <example> Please only follow this exact format - Chunk 1: 1,2,3,4 Chunk 2: 5,6,7 Chunk 3: 8,9,10,11,12 Chunk 4: ...
        No matter what, follow only this exact format. </example>''',
        messages=messages
    )

    logging.info(f"Claude's raw response: {message}")
    logging.info(f"Claude's content: {message.content}")


    # Attempt to extract text again with additional checks
    try:
        # Example of accessing 'text' if 'TextBlock' is a custom object with attributes
        content_texts = [item.text for item in message.content]
        combined_text = "\n".join(content_texts)
    except Exception as e:
        logging.error(f"Error processing Claude's response: {e}")
        combined_text = ""

    logging.info(f"Combined text for chunk processing: {combined_text}")

    # Assuming combined_text is now correctly populated, continue to process for chunks
    chunks = {}
    if combined_text:
        for line in combined_text.split('\n'):
            if line.startswith('Chunk'):
                chunk_number, sentence_numbers = line.split(': ')
                chunk_sentences = [int(num) for num in sentence_numbers.split(',')]
                chunks[int(chunk_number.split(' ')[1])] = chunk_sentences
        logging.info(f"Extracted chunks: {chunks}")
    else:
        logging.error("Combined text is empty, cannot extract chunks.")

    # Return the dictionary containing chunks information
    return chunks

#Use Claud to classify the document using integers, and map them to the category so that other functions can use them.
def classify_document(text):
    encoded_text = (text[:2000])
    document_type_map = {
        1: "Legislation",
        2: "Legal Guidelines",
        3: "Court Case",
        4: "Contracts"
    }

    messages = [
        {
            "role": "user",
            "content": f"<documents> {encoded_text} </documents> <instructions> Classify this document as 1 (Legislation), 2 (Guidelines), 3 (Court Cases), or 4 (Contracts). </instructions>"
        }
    ]

    logging.info(f"Sending to Claude for classification: {messages}")

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=5,
            temperature=0,
            system="<role> You are an expert legal document classifier. Read the document below carefully and classify it according to the given categories. Respond only with a number and nothing else. </role>"
        )
        logging.info(f"API Response: {message}")

        # Extract the document type ID from the response
        content_texts = [item.text for item in message.content if hasattr(item, 'text')]
        if content_texts:
            response_text = content_texts[0].strip()
            document_type_id = int(response_text)
            document_type_name = document_type_map.get(document_type_id, "Unknown")
            logging.info(f"Document classified as type: {document_type_id}, {document_type_name}")
            return document_type_id, document_type_name
        else:
            logging.error("Unexpected response structure or missing 'text' attribute in message content")
            return None, None
    except Exception as e:
        logging.error(f"Error in classification: {e}")
        return None, None




