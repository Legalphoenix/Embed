import chromadb
import voyageai

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "legislation"
collection = chroma_client.get_collection(name=collection_name)

# Initialize VoyageAI client
voyage_client = voyageai.Client(api_key='pa-HDqADASg6DbYlfiZhbtf2n5HCek1CpiouLod9AGALzA')

# Function to get embedding
def get_embedding(text, model="voyage-law-2"):
    response = voyage_client.embed(text, model=model)
    return response.embeddings[0] if response.embeddings else None

# Function to update embedding
def update_embedding(collection, doc_id, embedding):
    collection.update(
        ids=[doc_id],
        embeddings=[embedding]
    )

# Texts to embed and their respective IDs with proper indentation
texts_to_embed = {
    "dd890f03-2a7a-4d39-9e0d-313fedbc31ab": """[Type: Legislation] [Parent Document Title: Regulation (EU) 2016/679 (General Data Protection Regulation) GDPR] [Parent Document Parties: European Union]
Article 58
Powers
1. Each supervisory authority shall have all of the following investigative powers:
    1. to order the controller and the processor, and, where applicable, the controller's or the processor's representative to provide any information it requires for the performance of its tasks;
    2. to carry out investigations in the form of data protection audits;
    3. to carry out a review on certifications issued pursuant to Article 42(7);
    4. to notify the controller or the processor of an alleged infringement of this Regulation;
    5. to obtain, from the controller and the processor, access to all personal data and to all information necessary for the performance of its tasks;
    6. to obtain access to any premises of the controller and the processor, including to any data processing equipment and means, in accordance with Union or Member State procedural law.
2. Each supervisory authority shall have all of the following corrective powers:
    1. to issue warnings to a controller or processor that intended processing operations are likely to infringe provisions of this Regulation;
    2. to issue reprimands to a controller or a processor where processing operations have infringed provisions of this Regulation;
    3. to order the controller or the processor to comply with the data subject's requests to exercise his or her rights pursuant to this Regulation;
    4. to order the controller or processor to bring processing operations into compliance with the provisions of this Regulation, where appropriate, in a specified manner and within a specified period;
    5. to order the controller to communicate a personal data breach to the data subject;
    6. to impose a temporary or definitive limitation including a ban on processing;
    7. to order the rectification or erasure of personal data or restriction of processing pursuant to Articles 16, 17 and 18 and the notification of such actions to recipients to whom the personal data have been disclosed pursuant to Article 17(2) and Article 19;
    8. to withdraw a certification or to order the certification body to withdraw a certification issued pursuant to Articles 42 and 43, or to order the certification body not to issue certification if the requirements for the certification are not or are no longer met;
    9. to impose an administrative fine pursuant to Article 83, in addition to, or instead of measures referred to in this paragraph, depending on the circumstances of each individual case;
    10. to order the suspension of data flows to a recipient in a third country or to an international organisation.
3. Each supervisory authority shall have all of the following authorisation and advisory powers:
    1. to advise the controller in accordance with the prior consultation procedure referred to in Article 36;
    2. to issue, on its own initiative or on request, opinions to the national parliament, the Member State government or, in accordance with Member State law, to other institutions and bodies as well as to the public on any issue related to the protection of personal data;
    3. to authorise processing referred to in Article 36(5), if the law of the Member State requires such prior authorisation;
    4. to issue an opinion and approve draft codes of conduct pursuant to Article 40(5);
    5. to accredit certification bodies pursuant to Article 43;
    6. to issue certifications and approve criteria of certification in accordance with Article 42(5);
    7. to adopt standard data protection clauses referred to in Article 28(8) and in point (d) of Article 46(2);
    8. to authorise contractual clauses referred to in point (a) of Article 46(3);
    9. to authorise administrative arrangements referred to in point (b) of Article 46(3);
    10. to approve binding corporate rules pursuant to Article 47.
4. The exercise of the powers conferred on the supervisory authority pursuant to this Article shall be subject to appropriate safeguards, including effective judicial remedy and due process, set out in Union and Member State law in accordance with the Charter.
5. Each Member State shall provide by law that its supervisory authority shall have the power to bring infringements of this Regulation to the attention of the judicial authorities and where appropriate, to commence or engage otherwise in legal proceedings, in order to enforce the provisions of this Regulation.
6. Each Member State may provide by law that its supervisory authority shall have additional powers to those referred to in paragraphs 1, 2 and 3. The exercise of those powers shall not impair the effective operation of Chapter VII.""",
    "d3fd03f5-01a2-486f-ba14-7ef8bda45132": """[Type: Legislation] [Parent Document Title: Regulation (EU) 2016/679 (General Data Protection Regulation) GDPR] [Parent Document Parties: European Union]
Article 37
Designation of the data protection officer
1. The controller and the processor shall designate a data protection officer in any case where:
    1. the processing is carried out by a public authority or body, except for courts acting in their judicial capacity;
    2. the core activities of the controller or the processor consist of processing operations which, by virtue of their nature, their scope and/or their purposes, require regular and systematic monitoring of data subjects on a large scale; or
    3. the core activities of the controller or the processor consist of processing on a large scale of special categories of data pursuant to Article 9 and personal data relating to criminal convictions and offences referred to in Article 10.
2. A group of undertakings may appoint a single data protection officer provided that a data protection officer is easily accessible from each establishment.
3. Where the controller or the processor is a public authority or body, a single data protection officer may be designated for several such authorities or bodies, taking account of their organisational structure and size.
4. In cases other than those referred to in paragraph 1, the controller or processor or associations and other bodies representing categories of controllers or processors may or, where required by Union or Member State law shall, designate a data protection officer. The data protection officer may act for such associations and other bodies representing controllers or processors.
5. The data protection officer shall be designated on the basis of professional qualities and, in particular, expert knowledge of data protection law and practices and the ability to fulfil the tasks referred to in Article 39.
6. The data protection officer may be a staff member of the controller or processor, or fulfil the tasks on the basis of a service contract.
7. The controller or the processor shall publish the contact details of the data protection officer and communicate them to the supervisory authority."""
}

# Process each text and update the corresponding document in the collection
for doc_id, text in texts_to_embed.items():
    print(f"Embedding text for document ID: {doc_id}")
    embedding = get_embedding(text)
    if embedding:
        update_embedding(collection, doc_id, embedding)
        print(f"Updated embedding for document ID: {doc_id}")
    else:
        print(f"Failed to get embedding for document ID: {doc_id}")