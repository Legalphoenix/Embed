#embed_backend.py
import numpy as np
import openai
import logging
from dotenv import load_dotenv
from schema import Schema, Use, SchemaError
import anthropic
import voyageai
import chromadb
import uuid

# API Key handling
openai.api_key_path = './API.env'
voyageai.api_key_path = './Voyage.env'
vo = voyageai.Client()
def load_api_key(env_path='./Claude.env'):
    with open(env_path, 'r') as file:
        return file.read().strip()

api_key = load_api_key()
client = anthropic.Anthropic(api_key=api_key)

# Initialize persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chromadb")

# Create collections for each document type
collection_legislation = chroma_client.get_or_create_collection(name="legislation")
collection_guidelines = chroma_client.get_or_create_collection(name="guidelines")
collection_court_cases = chroma_client.get_or_create_collection(name="court_cases")
collection_contracts = chroma_client.get_or_create_collection(name="contracts")

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

def get_embedding(text, input_type=None):
    text = text.replace("\n", " ")
    documents_embeddings = vo.embed(text, model="voyage-law-2", input_type=input_type).embeddings
    if documents_embeddings and isinstance(documents_embeddings[0], list):
        return [item for sublist in documents_embeddings for item in sublist]
    return documents_embeddings

def save_embedding(original_file_name, chunk_file_name, document_title, document_parties, embedding, document_type_id, document_type_name):
    unique_id = str(uuid.uuid4())
    metadata = {
        'original_file_name': original_file_name,
        'chunk_file_name': chunk_file_name,
        'document_title': document_title,
        'document_parties': document_parties,
        'document_type_id': document_type_id,
        'document_type_name': document_type_name
    }
    if document_type_id == 1:
        collection_legislation.add(embeddings=[embedding], metadatas=[metadata], ids=[unique_id])
    elif document_type_id == 2:
        collection_guidelines.add(embeddings=[embedding], metadatas=[metadata], ids=[unique_id])
    elif document_type_id == 3:
        collection_court_cases.add(embeddings=[embedding], metadatas=[metadata], ids=[unique_id])
    elif document_type_id == 4:
        collection_contracts.add(embeddings=[embedding], metadatas=[metadata], ids=[unique_id])

def search_embeddings(query_embedding, doc_type, top_n=15):
    collections = []
    if doc_type == 1:
        collections.append(collection_legislation)
    elif doc_type == 2:
        collections.append(collection_guidelines)
    elif doc_type == 3:
        collections.append(collection_court_cases)
    elif doc_type == 4:
        collections.append(collection_contracts)
    else:
        collections = [collection_legislation, collection_guidelines, collection_court_cases, collection_contracts]

    results = []
    for collection in collections:
        query_results = collection.query(query_embeddings=[query_embedding], n_results=top_n, include=["metadatas", "distances"])
        for i in range(len(query_results["ids"])):
            results.append({
                "metadata": query_results["metadatas"][i],
                "distance": query_results["distances"][i]
            })

    results = sorted(results, key=lambda x: x["distance"])[:top_n]
    return results







def cosine_similarity(vec_a, vec_b):
    cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return cos_sim

def generate_modified_query(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
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
            stop=None,
        )
        generated_document = response.choices[0]['message']['content'].strip()
        logging.info(f"GPT-3.5 generated document: {generated_document}")
        return generated_document
    except Exception as e:
        logging.error(f"Error generating modified query: {e}")
        return None

def rerank_results(summaries, modified_query):
    documents = [f"{summary['preview_text']}" for summary in summaries]
    try:
        reranking = vo.rerank(modified_query, documents, model="rerank-lite-1")
        ordered_summaries = [summaries[r.index] for r in sorted(reranking.results, key=lambda x: -x.relevance_score)]
        logging.info(f"Reranking successful. Reordered indices: {[r.index for r in reranking.results]}")
        return ordered_summaries
    except Exception as e:
        logging.error(f"Error re-ranking results with Voyage AI: {e}")
        return summaries

def send_to_claude_and_get_chunks(numbered_sentences):
    sentences_content = ''.join([f'<line id="{num}">{sentence}</line>' for num, sentence in numbered_sentences.items()])
    messages = [{"role": "user", "content": '<documents> ' + sentences_content + '''
                 </documents> <instructions> Check the final <line id="num">sentence</line> number first to ensure you
                do not go past that number when generating your chunks.  </instructions>  '''}]

    logging.info(f"Sending to Claude: {messages}")

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

    return chunks

def classify_document_with_title(text):
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
            "content": f"<documents> {encoded_text} </documents> <instructions> Classify this document and extract or guess the full title. Classify as 1 (Legislation), 2 (Guidelines), 3 (Court Cases), or 4 (Contracts). </instructions>"
        }
    ]

    logging.info(f"Sending to Claude for classification and title extraction: {messages}")

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=10,
            temperature=0,
            system="<role> You are an expert legal document classifier. Read the document below carefully and classify it according to the given categories. Respond with a number followed by a comma and the full document title. </role>"
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
            "content": f"<documents> {encoded_text} </documents> <instructions> Extract the names of any companies or parties involved in this document. List the parties only and do not output anything else. Remove all formatting. </instructions>"
        }
    ]

    logging.info(f"Sending to Claude for party extraction: {messages}")

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=25,
            temperature=0,
            system="<role> You are an expert in identifying relevant parties from legal documents. Read the document below carefully and list the names of all identified parties, separated by commas. </role>"
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
