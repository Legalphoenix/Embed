#embed_backend.py
import numpy as np
from openai import OpenAI
import logging
from dotenv import load_dotenv
from schema import Schema, Use, SchemaError
import anthropic
import voyageai
from chromadb import HttpClient, Settings
import uuid
from tenacity import retry, stop_after_attempt, wait_random_exponential
import concurrent.futures
import os
load_dotenv(os.path.join(os.path.dirname(__file__), 'gpt_api.env'))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO, filename='embedding_log.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
#from WIP_gpt_batches import send_to_claude_and_get_chunks

# API Key handling
voyageai.api_key_path = './Voyage.env'
vo = voyageai.Client()
def load_api_key(env_path='./Claude.env'):
    with open(env_path, 'r') as file:
        return file.read().strip()

api_key = load_api_key()
anthropic_client = anthropic.Anthropic(api_key=api_key)

# Initialize persistent ChromaDB client
#java -jar tika/tika-server-standard-2.9.2.jar -p 9998
#from chromadb.config import Settings --might need
#chroma run --path ./chromadb
chroma_client = HttpClient(
    host="localhost",
    port="8000",
    ssl=False,
    headers={},
    settings=Settings(anonymized_telemetry=False)
)
# Define all collections with Item Count > 0
collections_info = [
    {"name": "collection_NSWFTT", "doc_type_id": 128, "doc_type_name": "NSWFTT"},
    {"name": "collection_NSWIRCOMM", "doc_type_id": 136, "doc_type_name": "NSWIRCOMM"},
    {"name": "collection_NSWCA", "doc_type_id": 107, "doc_type_name": "NSWCA"},
    {"name": "collection_NSWCATGD", "doc_type_id": 123, "doc_type_name": "NSWCATGD"},
    {"name": "collection_NSWMT", "doc_type_id": 130, "doc_type_name": "NSWMT"},
    {"name": "collection_NSWDC", "doc_type_id": 109, "doc_type_name": "NSWDC"},
    {"name": "legislation", "doc_type_id": 1, "doc_type_name": "Legislation"},
    {"name": "parent_legislation", "doc_type_id": 101, "doc_type_name": "Legislation"},
    {"name": "collection_ADT", "doc_type_id": 132, "doc_type_name": "ADT"},
    {"name": "collection_NSWCATEN", "doc_type_id": 125, "doc_type_name": "NSWCATEN"},
    {"name": "collection_NSWDRGC", "doc_type_id": 110, "doc_type_name": "NSWDRGC"},
    {"name": "collection_NSWCATOD", "doc_type_id": 124, "doc_type_name": "NSWCATOD"},
    {"name": "parent_recitals", "doc_type_id": 104, "doc_type_name": "GDPR Recitals"},
    {"name": "collection_NSWDDT", "doc_type_id": 126, "doc_type_name": "NSWDDT"},
    {"name": "collection_NSWCC", "doc_type_id": 106, "doc_type_name": "NSWCC"},
    {"name": "collection_NSWIC", "doc_type_id": 115, "doc_type_name": "NSWIC"},
    {"name": "collection_NSWCATAP", "doc_type_id": 120, "doc_type_name": "NSWCATAP"},
    {"name": "collection_NSWLEC", "doc_type_id": 112, "doc_type_name": "NSWLEC"},
    {"name": "collection_NSWADTAP", "doc_type_id": 118, "doc_type_name": "NSWADTAP"},
    {"name": "collection_NSWCHC", "doc_type_id": 135, "doc_type_name": "NSWCHC"},
    {"name": "collection_NSWSC", "doc_type_id": 114, "doc_type_name": "NSWSC"},
    {"name": "collection_NSWCCA", "doc_type_id": 108, "doc_type_name": "NSWCCA"},
    {"name": "collection_NSWEOT", "doc_type_id": 127, "doc_type_name": "NSWEOT"},
    {"name": "collection_NSWCATCD", "doc_type_id": 122, "doc_type_name": "NSWCATCD"},
    {"name": "collection_NSWLC", "doc_type_id": 113, "doc_type_name": "NSWLC"},
    {"name": "collection_NSWTAB", "doc_type_id": 131, "doc_type_name": "NSWTAB"},
    {"name": "collection_NSWEOD", "doc_type_id": 116, "doc_type_name": "NSWEOD"},
    {"name": "collection_NSWCATAD", "doc_type_id": 121, "doc_type_name": "NSWCATAD"},
    {"name": "recitals", "doc_type_id": 4, "doc_type_name": "GDPR Recitals"},
    {"name": "collection_NSWLST", "doc_type_id": 129, "doc_type_name": "NSWLST"},
    {"name": "parent_collection_guidelines", "doc_type_id": 102, "doc_type_name": "Australian High Court Cases"},
    {"name": "parent_court_cases", "doc_type_id": 103, "doc_type_name": "ECJU Privacy Court Cases"},

]

# Create collections and map Document Type IDs to collections
doc_type_id_to_collection = {}

for collection in collections_info:
    col = chroma_client.get_or_create_collection(
        name=collection["name"],
        metadata={"hnsw:space": "ip"}
    )
    doc_type_id_to_collection[collection["doc_type_id"]] = col

# Load environment variables
load_dotenv()

def get_json_schema():
    return Schema({'text': Use(str)})

def validate_json(data):
    schema = get_json_schema()
    try:
        schema.validate(data)
        return True
    except SchemaError as e:
        logging.error(f'JSON validation error: {e}')
        return False


'''CHUNK AND SAVE'''
def save_embeddings_in_batches(embeddings, chunk_texts, filename, document_title, document_parties, document_type_id, document_type_name, metadata, document_family_id, parent_hash, parent_document_filesize):
    for i, embedding in enumerate(embeddings):
        chunk_text = chunk_texts[i]  # Use the plain chunk text here
        unique_id = str(uuid.uuid4())
        chunk_metadata = {
            'original_file_name': filename,
            'document_title': document_title,
            'document_parties': document_parties,
            'document_type_id': document_type_id,
            'document_type_name': document_type_name,
            'chunk_text': chunk_text,
            'full_preview_text': chunk_text,
            'metadata': metadata,
            'document_family_id': document_family_id,
            'parent_hash': parent_hash,
            'parent_document_filesize': parent_document_filesize
        }

        # Adjust the collection based on the original document type ID
        if document_type_id == 1:
            collection_legislation.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])
        elif document_type_id == 2:
            collection_guidelines.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])
        elif document_type_id == 3:
            collection_court_cases.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])
        elif document_type_id == 4:
            collection_recitals.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])



def process_chunks_in_batches(chunks, numbered_sentences, document_type_name, document_title, document_parties):
    chunk_texts = []
    chunk_texts_with_descriptor = []
    for sentence_nums in chunks.values():
        chunk_text = " ".join(numbered_sentences[num] for num in sentence_nums)
        document_type_and_title_descriptor = f"[Type: {document_type_name}] [Parent Document Title: {document_title}] [Parent Document Parties: {document_parties}]"
        logging.info(f"title embeded: {document_title}")
        chunk_text_with_type_title = document_type_and_title_descriptor + " " + chunk_text
        chunk_texts.append(chunk_text)
        chunk_texts_with_descriptor.append(chunk_text_with_type_title)

    batch_size = 128
    all_embeddings = []

    for i in range(0, len(chunk_texts_with_descriptor), batch_size):
        batch = chunk_texts_with_descriptor[i:i + batch_size]
        embeddings = embed_with_backoff(documents=batch)
        all_embeddings.extend(embeddings)

    return all_embeddings, chunk_texts




@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def embed_with_backoff(documents, model="voyage-law-2", input_type="document"):
    return vo.embed(documents, model=model, input_type=input_type).embeddings


'''SAVE PARENT DOCUMENT'''
def get_embedding(text, input_type=None):
    text = text.strip()
    logging.info(f"query stripped text: {text}")
    documents_embeddings = vo.embed(text, model="voyage-law-2", input_type=input_type).embeddings
    if documents_embeddings and isinstance(documents_embeddings[0], list):
        return [item for sublist in documents_embeddings for item in sublist]
    return documents_embeddings


def save_embedding(original_file_name, document_title, document_parties, embedding, document_type_id, document_type_name, chunk_text, metadata, document_family_id, parent_hash, parent_document_filesize):
    unique_id = str(uuid.uuid4())
    chunk_metadata = {
        'original_file_name': original_file_name,
        'document_title': document_title,
        'document_parties': document_parties,
        'document_type_id': document_type_id,
        'document_type_name': document_type_name,
        'chunk_text': chunk_text,
        'full_preview_text': chunk_text,
        'metadata': metadata,
        'document_family_id': document_family_id,  # Include the document family ID in the metadata
        'parent_hash': parent_hash,  # Include the parent hash in the metadata
        'parent_document_filesize': parent_document_filesize  # Include the parent document filesize in the metadata
    }

    # Adjust the collection based on the parent document type ID
    if document_type_id == 101:
        parent_collection_legislation.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])
    elif document_type_id == 102:
        parent_collection_guidelines.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])
    elif document_type_id == 103:
        parent_collection_court_cases.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])
    elif document_type_id == 104:
        parent_collection_recitals.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])

# embed_backend.py

def search_embeddings(query_embedding, doc_types, top_n=10):
    """
    Search embeddings across the collections specified by the doc_types list.

    Parameters:
    - query_embedding: The embedding vector of the search query.
    - doc_types: A list of Document Type IDs to search across.
    - top_n: The number of top results to return.

    Returns:
    - A sorted list of search results with metadata and distances.
    """
    collections = []

    # If "All Categories" (doc_type == 0) is selected, include all collections
    if 0 in doc_types:
        collections = list(doc_type_id_to_collection.values())
    else:
        # Otherwise, add collections corresponding to the selected doc_type IDs
        for doc_type_id in doc_types:
            collection = doc_type_id_to_collection.get(doc_type_id)
            if collection:
                collections.append(collection)
            else:
                logging.warning(f"Unsupported Document Type ID: {doc_type_id}")

    if not collections:
        logging.error("No valid collections found for the provided Document Type IDs.")
        return []

    # Search results will be stored in this list
    results = []

    # Iterate through all selected collections and query each one
    for collection in collections:
        try:
            # Perform the query on the current collection
            query_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n,
                include=["metadatas", "distances"]
            )

            # Append results with metadata and distances
            for i in range(len(query_results["ids"])):
                results.append({
                    "metadata": query_results["metadatas"][i],
                    "distance": query_results["distances"][i]
                })
        except Exception as e:
            logging.error(f"Error querying collection {collection.name}: {e}")

    # Sort results by distance (assuming lower distance means a better match)
    results = sorted(results, key=lambda x: x["distance"])[:top_n]
    return results




def generate_modified_query(query):
    try:
        response = openai_client.chat.completions.create(model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f'''A user is attempting to search an embedding space with this query: \"{query}\"
                                    Output a new query using the exact format below.
                                    Query Subject: which could be seen as an encapsulation of the overall subject or theme of the query.
                                    Query Description: which provides a more detailed description of what the query is about.
                                    Query Tags: which include specific tags that categorize the legal themes or keywords associated with the question.
                                    Query content: (exact query here with fixed spelling)


                                Examples
                                Original User Query: "What are the implications of filing for bankruptcy under Chapter 7?"
                                Modified Query

                                    Query Subject: Bankruptcy
                                    Query Description: Implications of filing
                                    Query Tags: Chapter 7; Personal Bankruptcy
                                    Query content: What are the implications of filing for bankruptcy under Chapter 7?

                                Original User Query: "Can I modify a custody agreement without going to court?"
                                Modified Query

                                    Query Subject: Child Custody
                                    Query Description: Modifying agreement without court
                                    Query Tags: Custody Modification; Legal Procedure
                                    Query content: Can I modify a custody agreement without going to court?




                                    )'''
            }
        ],
        temperature=0,
        max_tokens=1024,
        stop=None)
        generated_document = response.choices[0].message.content.strip()
        logging.info(f"GPT-3.5 generated document: {generated_document}")
        return generated_document
    except Exception as e:
        logging.error(f"Error generating modified query: {e}")
        return None

def rerank_results(summaries, modified_query):
    documents = [f"{summary['preview_text']}" for summary in summaries]
    try:
        reranking = vo.rerank(modified_query, documents, model="rerank-1")
        ordered_summaries = [summaries[r.index] for r in sorted(reranking.results, key=lambda x: -x.relevance_score)]
        logging.info(f"Reranking successful. Reordered indices: {[r.index for r in reranking.results]}")
        return ordered_summaries
    except Exception as e:
        logging.error(f"Error re-ranking results with Voyage AI: {e}")
        return summaries

'''GET CHUNKS FOR EMBEDDINGS AND CHUNKS'''


def send_to_claude_and_get_chunks(numbered_sentences):
    sentences_content = ''.join([f'<line id="{num}">{sentence}</line>' for num, sentence in numbered_sentences.items()])
    logging.info(f"Sending to ChatGPT: {sentences_content}")
    messages = [
        {
            "role": "system",
            "content": '''
                Please read and parse the legal document below very carefully and do nothing except follow the exact instructions.
                <instruction 1> Segment the document into one chunk per "Recital". Each Recital is contained in a <line id="X">Recital 173</line> followed by its child text.  </instruction>
                <instruction 2> In order to minimize the amount of text in your output, we will do the following:
                1. You will be provided with the document formatted such that each sentence or group of sentences.
                from the document are numbered top to bottom indicated by '<line id=num>sentence</line>' where num represents the line number.
                2. When you chunk each Recital, you should combine the paragraphs that follow the Recital heading until you hit the next recital. Output following this example format:
                </instruction>
                <example> Please only follow this exact format - Chunk 1: 1 Chunk 2: 2 Chunk 3: 4 Chunk 4: 5,6 ...
                No matter what, follow only this exact format. Check the last line ID number before you begin and ensure you include all line id's into the chunks. All sections in each Recital should be grouped in one chunk. </example>
            '''
        },
        {
            "role": "user",
            "content": '<documents> ' + sentences_content + '''
                         </documents> <instructions> Check the final <line id="num">sentence</line> number first to ensure you
                        do not go past that number when generating your chunks.  </instructions>  '''
        }
    ]

    logging.info(f"SxxxxxxxxT: {messages}")

    try:
        response = openai_client.chat.completions.create(model="gpt-4o",
        messages=messages,
        max_tokens=4096,
        temperature=0)
        logging.info(f"ChatGPT's raw response: '{response}'")
        message_content = response.choices[0].message.content
        logging.info(f"ChatGPT's content: '{message_content}'")

        combined_text = message_content.strip()
    except Exception as e:
        logging.error(f"Error processing ChatGPT's response: {e}")
        combined_text = ""

    logging.info(f"Combined text for chunk processing: {combined_text}")

    chunks = {}
    if combined_text:
        for line in combined_text.split('\n'):
            if line.startswith('Chunk'):
                chunk_number, sentence_numbers = line.split(': ')
                chunk_sentences = [int(num) for num in sentence_numbers.split(',')]
                chunks[int(chunk_number.split(' ')[1])] = chunk_sentences
    else:
        logging.error("Combined text is empty, cannot extract chunks.")

    return chunks

""" def send_to_claude_and_get_chunks(numbered_sentences):
    sentences_content = ''.join([f'<line id="{num}">{sentence}</line>' for num, sentence in numbered_sentences.items()])
    logging.info(f"Sending to ChatGPT: {sentences_content}")
    messages = [
        {
            "role": "system",
            "content": '''
                You are an expert legal embedding chunking program.
                Please read the GDPR document below very carefully and do nothing except follow the exact instructions.
                <instruction 1> Segment the GDPR Articles document into one chunk per entire and complete Article of GDPR. When you see "Article X" alone between two lines like this: <line id=num>Article X</line> then you know you should create a new chunk and keep all clauses following inside that chunk. The title of the document should be chunked by itself. </instruction>
                <instruction 2> In order to minimize the amount of text in your output, we will do the following:
                1. You will be provided with the document formatted such that each sentence or group of sentences
                from the document are numbered top to bottom indicated by '<line id=num>sentence</line>' where num is the line number.
                2. When you chunk the document, you will output your segmented chunks strictly in the following example format:
                </instruction>
                <example> Please only follow this exact format - Chunk 1: 1,2,3,4 Chunk 2: 5,6,7 Chunk 3: 8,9,10,11,12 Chunk 4: ...
                No matter what, follow only this exact format. All sections in each Article should be grouped in one chunk.


                </example>
            '''
        },
        {
            "role": "user",
            "content": '<documents> ' + sentences_content + '''
                         </documents> <instructions> Check the final <line id="num">sentence</line> number first to ensure you
                        do not go past that number when generating your chunks.  </instructions>  '''
        }
    ]

    logging.info(f"SxxxxxxxxT: {messages}")

    try:
        response = openai_client.chat.completions.create(model="gpt-4o",
        messages=messages,
        max_tokens=4096,
        temperature=0)
        logging.info(f"ChatGPT's raw response: '{response}'")
        message_content = response.choices[0].message.content
        logging.info(f"ChatGPT's content: '{message_content}'")

        combined_text = message_content.strip()
    except Exception as e:
        logging.error(f"Error processing ChatGPT's response: {e}")
        combined_text = ""

    logging.info(f"Combined text for chunk processing: {combined_text}")

    chunks = {}
    if combined_text:
        for line in combined_text.split('\n'):
            if line.startswith('Chunk'):
                chunk_number, sentence_numbers = line.split(': ')
                chunk_sentences = [int(num) for num in sentence_numbers.split(',')]
                chunks[int(chunk_number.split(' ')[1])] = chunk_sentences
    else:
        logging.error("Combined text is empty, cannot extract chunks.")

    return chunks """

""" def send_to_claude_and_get_chunks(numbered_sentences):
    sentences_content = ''.join([f'<line id="{num}">{sentence}</line>' for num, sentence in numbered_sentences.items()])
    messages = [{"role": "user", "content": '<documents> ' + sentences_content + '''
                 </documents> <instructions> Check the final <line id="num">sentence</line> number first to ensure you
                do not go past that number when generating your chunks.  </instructions>  '''}]

    logging.info(f"Sending to Claude: {messages}")

    message = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0,
        system='''<role> You are an expert legal chunking program.
                Please read and parse the document below very carefully and do nothing except follow the exact instructions.
                <instruction 1> Segment the document into one chunk per "Recital". The name Recital is a heading for a block of text. Each Recital is numbered 1 - 173, each containing either one or two paragraphs of text within it. </instruction>
                <instruction 2> In order to minimize the amount of text in your output, we will do the following:
                1. You will be provided with the document formatted such that each sentence or group of sentences.
                from the document are numbered top to bottom indicated by '<line id=num>sentence</line>' where num represents the line number.
                2. When you chunk each recital, you should combine the sentence id's that follow the Recital heading until you hit the next recital. Output following this example format:
                </instruction>
                <example> Please only follow this exact format - Chunk 1: 1 Chunk 2: 2 Chunk 3: 4 Chunk 4: 5,6 ...
                No matter what, follow only this exact format. Check the last line ID number before you begin and ensure you include all line id's into the chunks. All sections in each Recital should be grouped in one chunk. </example>
            ''',
        messages=messages
    )

    logging.info(f"Claude's raw response: {message}")
    logging.info(f"Claude's content: {message.content}")

    try:
        content_texts = [item.text for item in message.content]
        combined_text = "\n".join(content_texts)
    except Exception as e:
        logging.error(f"Error processing Claude's response: {e}")
        combined_text = ""

    logging.info(f"Combined text for chunk processing: {combined_text}")

    chunks = {}
    if combined_text:
        for line in combined_text.split('\n'):
            if line.startswith('Chunk'):
                chunk_number, sentence_numbers = line.split(': ')
                chunk_sentences = [int(num) for num in sentence_numbers.split(',')]
                chunks[int(chunk_number.split(' ')[1])] = chunk_sentences
    else:
        logging.error("Combined text is empty, cannot extract chunks.")

    return chunks """

'''CLASSIFY AND EXTRACT INFORMATION'''

def classify_extract_and_chunk(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        classify_future = executor.submit(classify_document_with_title, text)
        extract_future = executor.submit(extract_parties_from_document, text)
        clean_and_chunk_future = executor.submit(clean_and_chunk, text)

        # Wait for all futures to complete
        document_type_id, document_type_name, document_title = classify_future.result()
        document_parties = extract_future.result()
        numbered_sentences, chunks = clean_and_chunk_future.result()

    return document_type_id, document_type_name, document_title, document_parties, numbered_sentences, chunks

def clean_and_chunk(text):
    cleaned_lines = [line.strip() for line in text.split('\n') if line.strip()]
    numbered_sentences = {i + 1: line.strip() for i, line in enumerate(cleaned_lines)}
    chunks = send_to_claude_and_get_chunks(numbered_sentences)

    return numbered_sentences, chunks



def classify_document_with_title(text):
    encoded_text = (text[:2000])
    document_type_map = {
        1: "Legislation",
        2: "Legal Guidelines",
        3: "Court Case",
        4: "GDPR Recitals"
    }

    messages = [
        {
            "role": "user",
            "content": f"<documents> {encoded_text} </documents> <instructions> Classify this document and extract or guess the full title. Classify as 1 (Legislation), 2 (Guidelines), 3 (Court Cases), or 4 (GDPR Recitals). </instructions>"
        }
    ]

    logging.info(f"Sending to Claude for classification and title extraction: {messages}")

    try:
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=100,
            temperature=0,
            system="<role> You are an expert legal document classifier. Read the document below carefully and classify it according to the given categories. Respond with a number that maps to the classification followed by a comma and the full document title. </role>"
        )
        logging.info(f"API Response: {message}")

        content_texts = [item.text for item in message.content if hasattr(item, 'text')]
        if content_texts:
            response_parts = content_texts[0].strip().split(',', 1)
            document_type_id = int(response_parts[0])
            document_title = response_parts[1].strip() if len(response_parts) > 1 else "Unknown Title"
            document_type_name = document_type_map.get(document_type_id, "Unknown")
            logging.info(f"Document classified as type: {document_type_id}, {document_type_name} with title: '{document_title}'")
            return document_type_id, document_type_name, document_title
        else:
            logging.error("Unexpected response structure or missing 'text' attribute in message content")
            return None, None, None
    except Exception as e:
        logging.error(f"Error in classification: {e}")
        return None, None, None

def extract_parties_from_document(text):
    encoded_text = (text[:2000])

    messages = [
        {
            "role": "user",
            "content": f"<documents> {encoded_text} </documents> <instructions> Extract the names of any companies or parties involved in this document. If it is the GDPR, then output 'European Union.' List the parties only and do not output anything else. Remove all formatting. </instructions>"
        }
    ]

    logging.info(f"Sending to Claude for party extraction: {messages}")

    try:
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=100,
            temperature=0,
            system="<role> You are an expert in identifying relevant parties from legal documents. Read the document below carefully and list the names of all identified parties, separated by commas. If it is the GDPR, then output 'European Union'. </role>"
        )
        logging.info(f"API Response: {message}")

        content_texts = [item.text for item in message.content if hasattr(item, 'text')]
        if content_texts:
            parties = content_texts[0].strip()
            logging.info(f"Parties extracted: '{parties}'")
            return parties
        else:
            logging.error("Unexpected response structure or missing 'text' attribute in message content")
            return "Parties not found"
    except Exception as e:
        logging.error(f"Error in party extraction: {e}")
        return "Error during extraction"
