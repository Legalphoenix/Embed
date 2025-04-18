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
from tenacity import retry, stop_after_attempt, wait_random_exponential
import concurrent.futures


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

# Create collections with inner product similarity (dot product)
collection_legislation = chroma_client.get_or_create_collection(
    name="legislation",
    metadata={"hnsw:space": "ip"}
)

collection_guidelines = chroma_client.get_or_create_collection(
    name="guidelines",
    metadata={"hnsw:space": "ip"}
)

collection_court_cases = chroma_client.get_or_create_collection(
    name="court_cases",
    metadata={"hnsw:space": "ip"}
)

collection_contracts = chroma_client.get_or_create_collection(
    name="contracts",
    metadata={"hnsw:space": "ip"}
)
parent_collection_legislation = chroma_client.get_or_create_collection(
    name="parent_legislation",
    metadata={"hnsw:space": "ip"}
)

parent_collection_guidelines = chroma_client.get_or_create_collection(
    name="parent_guidelines",
    metadata={"hnsw:space": "ip"}
)

parent_collection_court_cases = chroma_client.get_or_create_collection(
    name="parent_court_cases",
    metadata={"hnsw:space": "ip"}
)

parent_collection_contracts = chroma_client.get_or_create_collection(
    name="parent_contracts",
    metadata={"hnsw:space": "ip"}
)


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
            collection_contracts.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])



def process_chunks_in_batches(chunks, numbered_sentences, document_type_name, document_title, document_parties):
    chunk_texts = []
    chunk_texts_with_descriptor = []
    for sentence_nums in chunks.values():
        chunk_text = " ".join(numbered_sentences[num] for num in sentence_nums)
        document_type_and_title_descriptor = f"[Type: {document_type_name}] [Parent Document Title: {document_title}] [Parent Document Parties: {document_parties}]"
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
    logging.info(f"stripped text: {text}")
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
        parent_collection_contracts.add(documents=[chunk_text], embeddings=[embedding], metadatas=[chunk_metadata], ids=[unique_id])

'''SEARCH EMBEDDINGS'''
def search_embeddings(query_embedding, doc_type, top_n=15):
    collections = []
    if doc_type in [1, 101]:  # Legislation or Parent Legislation
        if doc_type == 1:
            collections.append(collection_legislation)
        elif doc_type == 101:
            collections.append(parent_collection_legislation)
    elif doc_type in [2, 102]:  # Guidelines or Parent Guidelines
        if doc_type == 2:
            collections.append(collection_guidelines)
        elif doc_type == 102:
            collections.append(parent_collection_guidelines)
    elif doc_type in [3, 103]:  # Court Cases or Parent Court Cases
        if doc_type == 3:
            collections.append(collection_court_cases)
        elif doc_type == 103:
            collections.append(parent_collection_court_cases)
    elif doc_type in [4, 104]:  # Contracts or Parent Contracts
        if doc_type == 4:
            collections.append(collection_contracts)
        elif doc_type == 104:
            collections.append(parent_collection_contracts)
    else:  # All categories
        collections = [
            collection_legislation, collection_guidelines, collection_court_cases, collection_contracts,
            parent_collection_legislation, parent_collection_guidelines, parent_collection_court_cases, parent_collection_contracts
        ]

    results = []
    for collection in collections:
        query_results = collection.query(query_embeddings=[query_embedding], n_results=top_n, include=["metadatas", "distances"])
        for i in range(len(query_results["ids"])):
            results.append({
                "metadata": query_results["metadatas"][i],
                "distance": query_results["distances"][i]
            })

    results = sorted(results, key=lambda x: x["distance"])[:top_n]
    logging.info(f"results: {results}")
    return results


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

'''GET CHUNKS FOR EMBEDDINGS AND CHUNKS'''

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
