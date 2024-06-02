import chromadb
import json

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "parent_court_cases"
collection = chroma_client.get_collection(name=collection_name)

# Retrieve all items in the collection
items = collection.get(include=["documents", "metadatas"])

# Prepare the instructions and format for ChatGPT
instructions = """
Please act as a legal summarization tool that creates a very concise summary that would be useful for a lawyer who is doing legal research and needs to see if a case is relevant to her task quickly. The goal is for her to be able to see if the cases is relevant within 30 seconds.

Do a summary in the format of the following example. The case to summarize is provided below.

Important Factors for Quick Relevance Assessment:

    Legal Issues: Key legal principles and interpretations involved.
    Facts: Core facts that underpin the legal issues.
    Judgment: Court's final decision and its implications.
    Applicable Law: Specific provisions of the law applied.
    Reasoning: Court's rationale in arriving at its decision.
    Impact: Potential implications for future cases or practices.

EXAMPLE - Concise Summary:
Case Overview:

    Case Name: IAB Europe v. Gegevensbeschermingsautoriteit
    Court: Fourth Chamber of the Court of Justice of the European Union
    Date: 7 March 2024
    Reference: C-604/22

Key Legal Issues:

    Interpretation of "personal data" (Article 4(1) GDPR)
    Definition of "data controller" (Article 4(7) GDPR)
    Concept of "joint controllers" (Article 26(1) GDPR)

Core Facts:

    IAB Europe, a non-profit in the digital advertising sector, created the Transparency & Consent Framework (TCF) to help its members comply with GDPR.
    TCF includes guidelines for obtaining user consent for data processing, which is recorded in a Transparency and Consent String (TC String).
    The Belgian Data Protection Authority (DPA) found IAB Europe as a controller of the TC String data and ordered compliance with GDPR, along with fines.

Judgment:

    TC String as Personal Data:
        The TC String, containing user consent preferences, is considered personal data if it can be associated with an identifier like an IP address.
        This applies even if IAB Europe cannot directly access or combine the TC String with other identifiers.

    IAB Europe as Joint Controller:
        IAB Europe is a joint controller because it influences the processing of personal data through its TCF, setting binding technical rules.
        This status holds despite IAB Europe not having direct access to the personal data processed by its members.

    Scope of Joint Controllership:
        IAB Europe’s joint controllership does not automatically extend to subsequent processing by third parties (e.g., for targeted advertising) unless it has a direct role in determining the purposes and means of such processing.

Applicable Law:

    GDPR Articles: 4(1), 4(7), 6(1)(f), 24(1), 26(1)
    EU Charter: Articles 7 and 8
    Recitals of GDPR: 1, 10, 26, 30

Reasoning:

    Personal Data Definition: Emphasizes the broad scope intended by the GDPR.
    Controller Definition: Focuses on the influence over purposes and means of processing.
    Joint Controllership: Considers both direct and indirect influence over data processing operations.

Impact:

    Clarifies the scope of what constitutes personal data under GDPR.
    Defines the responsibilities and scope of control for organizations setting data processing frameworks.
    Establishes limits on the extension of joint controllership to subsequent data processing activities.
"""

# Prepare data for Batch API
batch_size = 10
batch_requests = []
batch_index = 0

for i in range(len(items["documents"])):
    custom_id = items["ids"][i]  # Ensure custom_id is taken from the actual document IDs
    full_preview_text = items["metadatas"][i]['full_preview_text']  # Accessing 'full_preview_text' from metadata

    # Log the custom_id and full_preview_text for verification
    print(f"Preparing batch request for custom_id: {custom_id}")

    # Prepare the request for the batch file
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": full_preview_text}
            ],
            "max_tokens": 4096
        }
    }
    batch_requests.append(request)

    # Write batch file every batch_size requests
    if len(batch_requests) == batch_size:
        batch_file_path = f"batch_input_{batch_index}.jsonl"
        with open(batch_file_path, 'w') as batch_file:
            for request in batch_requests:
                batch_file.write(json.dumps(request) + "\n")
        print(f"Batch requests prepared and saved to {batch_file_path}.")
        batch_requests = []
        batch_index += 1

# Save any remaining requests in the final batch file
if batch_requests:
    batch_file_path = f"batch_input_{batch_index}.jsonl"
    with open(batch_file_path, 'w') as batch_file:
        for request in batch_requests:
            batch_file.write(json.dumps(request) + "\n")
    print(f"Batch requests prepared and saved to {batch_file_path}.")
