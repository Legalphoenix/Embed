import chromadb
import voyageai

# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# Fetch the collection
collection_name = "parent_court_cases"
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
    "5babfa2f-eec1-4ce2-8272-8303ee65dafd": """JUDGMENT OF THE COURT (Third Chamber)

21 December 2023 ( *1 )

"Preliminary ruling – Protection of individuals with regard to the processing of personal data – Regulation (EU) 2016/679 – Article 6, paragraph 1 – Conditions for lawful processing – Article 9, paragraphs 1 to 3 – Processing of special categories of data – Data concerning health – Assessment of an employee's work capacity – Medical service dealing with the health insurance of its own employees – Admissibility and conditions of such processing – Article 82, paragraph 1 – Right to compensation and liability – Compensation for moral damage – Compensatory function – Impact of the fault committed by the data controller"

In case C-667/21,

concerning a request for a preliminary ruling under Article 267 TFEU, made by the Bundesarbeitsgericht (Federal Labour Court, Germany) by decision of 26 August 2021, received by the Court on 8 November 2021, in the proceedings

ZQ

against

Medizinischer Dienst der Krankenversicherung Nordrhein, Körperschaft des öffentlichen Rechts,

THE COURT (Third Chamber),

composed of Ms K. Jürimäe, President of the Chamber, Mr. N. Piçarra, Mr. Safjan, Mr. Jääskinen (Rapporteur), and Mr. Gavalec, Judges,

Advocate General: Mr. M. Campos Sánchez-Bordona,

Registrar: Mr. A. Calot Escobar,

having regard to the written procedure,

considering the observations submitted:

– for ZQ, by Mr. E. Daun, Rechtsanwalt,

– for the Medizinischer Dienst der Krankenversicherung Nordrhein, Körperschaft des öffentlichen Rechts, by Mr. M. Wehner, Rechtsanwalt,

– for Ireland, by Ms. M. Browne, Chief State Solicitor, Mr. A. Joyce, and Ms. M. Lane, acting as agents, assisted by Mr. D. Fennelly, BL,

– for the Italian government, by Ms. G. Palmieri, acting as agent, assisted by Ms. M. Russo, avvocato dello Stato,

– for the European Commission, by Mr. A. Bouchagiar, Ms. M. Heller, and Mr. H. Kranenborg, acting as agents,

having heard the Advocate General in his Opinion at the hearing on 25 May 2023,

delivers the following

Judgment

    The request for a preliminary ruling concerns the interpretation of Article 9, paragraph 1, paragraph 2, subparagraph h), and paragraph 3, of Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016, on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation) (OJ 2016, L 119, p. 1, hereinafter 'the GDPR'), read in conjunction with Article 6, paragraph 1, of that regulation, as well as the interpretation of Article 82, paragraph 1, of the same regulation.

    This request has been submitted in the context of a dispute between ZQ and his employer, the Medizinischer Dienst der Krankenversicherung Nordrhein (Medical Service of the Health Insurance of North Rhine, Germany) (hereinafter 'MDK Nordrhein'), regarding the compensation for damage that the former claims to have suffered due to a processing of data concerning his health that was allegedly carried out unlawfully by the latter.

Legal framework

Union law

    Recitals 4 to 8, 10, 35, 51 to 53, 75, and 146 of the GDPR are worded as follows:

"(4) The processing of personal data should be designed to serve humanity. The right to the protection of personal data is not an absolute right; it must be considered in relation to its function in society and be balanced against other fundamental rights, in accordance with the principle of proportionality. This Regulation respects all fundamental rights and observes the freedoms and principles recognized in the Charter of Fundamental Rights of the European Union, enshrined in the Treaties, in particular respect for private and family life, [...] protection of personal data, [...]

(5) The economic and social integration resulting from the functioning of the internal market has led to a substantial increase in cross-border flows of personal data. The exchange of personal data between public and private actors, including individuals, associations, and undertakings, has increased in the Union. Union law requires the authorities of the Member States to cooperate and exchange personal data to be able to discharge their duties or perform tasks on behalf of an authority of another Member State.

(6) The rapid technological developments and globalization have brought new challenges for the protection of personal data. The scale of the collection and sharing of personal data has increased significantly. Technology allows both private companies and public authorities to use personal data on an unprecedented scale in order to pursue their activities. Individuals increasingly make information about themselves publicly and globally available. Technology has transformed both the economy and social life, and should further facilitate the free flow of personal data within the Union and the transfer to third countries and international organizations, while ensuring a high level of protection of personal data.

(7) These developments require a strong and more coherent data protection framework in the Union, backed by strong enforcement, given the importance of creating the trust that will allow the digital economy to develop across the internal market. Individuals should have control of their own personal data. Legal and practical certainty for individuals, economic operators, and public authorities should be enhanced.

(8) Where this Regulation provides for specifications or restrictions of its rules by Member State law, Member States may incorporate elements of this Regulation into their law as necessary to ensure coherence and to make the national provisions comprehensible to the persons to whom they apply.

[...]

(10) To ensure a consistent and high level of protection of natural persons and to remove the obstacles to flows of personal data within the Union, the level of protection of the rights and freedoms of natural persons with regard to the processing of such data should be equivalent in all Member States. [...] This Regulation also allows Member States to provide for specifications or restrictions of its rules, including with regard to the processing of special categories of personal data (hereinafter referred to as 'sensitive data'). [...]

[...]

(35) Personal data concerning health should include all data pertaining to the health status of a data subject that reveal information about the past, present, or future physical or mental health status of the data subject. [...]

[...]

(51) Personal data that are, by their nature, particularly sensitive in relation to fundamental rights and freedoms merit specific protection as the context of their processing could create significant risks to the fundamental rights and freedoms. [...] In addition to the specific requirements for such processing, the general principles and other rules of this Regulation should apply, in particular as regards the conditions for lawful processing. Derogations from the general prohibition on processing such special categories of personal data should be explicitly provided for, among other things, where the data subject gives explicit consent or for specific needs [...]

(52) Derogations from the prohibition on processing special categories of personal data should also be allowed where Union or Member State law provides for and subject to suitable safeguards, to protect personal data and other fundamental rights, where the public interest so justifies, in particular processing personal data in the field of employment and social security and social protection law, including pensions, and for purposes of health security, monitoring, and alert, prevention or control of communicable diseases and other serious threats to health. These derogations should be possible for health purposes, including public health and the management of health-care services, in particular to ensure the quality and efficiency of the procedures for settling claims for benefits and services in the health insurance system, or for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes. [...]

(53) Special categories of personal data that merit higher protection should be processed only for health-related purposes where necessary to achieve those purposes for the benefit of natural persons and society as a whole, in the context of the management of health-care services and systems or social protection, including processing by the management and central national health authorities of such data with a view to monitoring the quality, providing management information, and general oversight at national and local level of the health or social protection system, and to ensure continuity of health care or social protection. [...] This Regulation should therefore provide for harmonized conditions for the processing of special categories of personal data concerning health, to meet specific needs, in particular where the processing of such data is carried out for certain health-related purposes by persons subject to a legal obligation of professional secrecy. Union or Member State law should provide for specific and suitable measures to protect the fundamental rights and personal data of natural persons. Member States should be allowed to maintain or introduce further conditions, including limitations, with regard to the processing of genetic data, biometric data, or data concerning health. [...]

[...]

(75) The risk to the rights and freedoms of natural persons, which may vary in likelihood and severity, may result from the processing of personal data which could cause physical, material, or non-material damage, in particular: where the processing may give rise to discrimination, identity theft or fraud, financial loss, damage to reputation, loss of confidentiality of data protected by professional secrecy, unauthorized reversal of pseudonymization, or any other significant economic or social disadvantage; where data subjects might be deprived of their rights and freedoms or prevented from exercising control over their personal data; where personal data are processed which reveal [...] health data [...] where personal aspects are evaluated, in particular analyzing or predicting aspects concerning work performance, economic situation, health, [...] for creating or using personal profiles; [...]

[...]

(146) The controller or processor should compensate any damage which a person may suffer as a result of processing that infringes this Regulation. The controller or processor should be exempt from liability if it proves that it is not in any way responsible for the damage. The concept of damage should be broadly interpreted in light of the case law of the Court of Justice, in a manner that fully reflects the objectives of this Regulation. This is without prejudice to any claims for damage deriving from a breach of other rules in Union or Member State law. Processing that violates this Regulation includes processing that violates delegated and implementing acts adopted in accordance with this Regulation and Member State law specifying rules of this Regulation. Data subjects should receive full and effective compensation for the damage they have suffered. [...]"

    Chapter I of this Regulation, concerning 'General provisions', includes Article 2, entitled 'Material scope', which states, in paragraph 1:

"This Regulation applies to the processing of personal data wholly or partly by automated means and to the processing other than by automated means of personal data which form part of a filing system or are intended to form part of a filing system."

    Article 4 of this Regulation, entitled 'Definitions', states:

"For the purposes of this Regulation:

    'personal data' means any information relating to an identified or identifiable natural person ('data subject'); [...]

    'processing' means any operation or set of operations which is performed on personal data or on sets of personal data, whether or not by automated means [...]

[...]

    'controller' means the natural or legal person, public authority, agency or other body which, alone or jointly with others, determines the purposes and means of the processing of personal data; [...]

[...]

    'data concerning health' means personal data related to the physical or mental health of a natural person, including the provision of health care services, which reveal information about the health status of that person;

[...]"

    Chapter II of the GDPR, concerning the 'Principles' set by the Regulation, includes Articles 5 to 11.

    Article 5 of this Regulation, entitled 'Principles relating to processing of personal data', states:

"1. Personal data shall be:

a) processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness, transparency');

[...]

f) processed in a manner that ensures appropriate security of the personal data, including protection against unauthorized or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organizational measures ('integrity and confidentiality');

    The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1 ('accountability')."

    Article 6 of this Regulation, entitled 'Lawfulness of processing', states, in paragraph 1:

"Processing shall be lawful only if and to the extent that at least one of the following applies:

a) the data subject has given consent to the processing of his or her personal data for one or more specific purposes;

b) processing is necessary for the performance of a contract to which the data subject is party or in order to take steps at the request of the data subject prior to entering into a contract;

c) processing is necessary for compliance with a legal obligation to which the controller is subject;

d) processing is necessary in order to protect the vital interests of the data subject or of another natural person;

e) processing is necessary for the performance of a task carried out in the public interest or in the exercise of official authority vested in the controller;

f) processing is necessary for the purposes of the legitimate interests pursued by the controller or by a third party, except where such interests are overridden by the interests or fundamental rights and freedoms of the data subject which require protection of personal data, in particular where the data subject is a child.

Point (f) of the first subparagraph shall not apply to processing carried out by public authorities in the performance of their tasks."

    Article 9 of this Regulation, entitled 'Processing of special categories of personal data', states:

"1. Processing of personal data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, or trade union membership, and the processing of genetic data, biometric data for the purpose of uniquely identifying a natural person, data concerning health or data concerning a natural person's sex life or sexual orientation shall be prohibited.

    Paragraph 1 shall not apply if one of the following applies:

[...]

b) processing is necessary for the purposes of carrying out the obligations and exercising specific rights of the controller or of the data subject in the field of employment and social security and social protection law in so far as it is authorized by Union or Member State law or a collective agreement pursuant to Member State law providing for appropriate safeguards for the fundamental rights and the interests of the data subject;

[...]

h) processing is necessary for the purposes of preventive or occupational medicine, for the assessment of the working capacity of the employee, medical diagnosis, the provision of health or social care or treatment or the management of health or social care systems and services on the basis of Union or Member State law or pursuant to contract with a health professional and subject to the conditions and safeguards referred to in paragraph 3;

[...]

    Personal data referred to in paragraph 1 may be processed for the purposes referred to in paragraph 2, point (h), if those data are processed by or under the responsibility of a professional subject to the obligation of professional secrecy under Union or Member State law or rules established by national competent bodies or by another person subject to an obligation of secrecy under Union or Member State law or rules established by national competent bodies.

    Member States may maintain or introduce further conditions, including limitations, with regard to the processing of genetic data, biometric data or data concerning health."

    Chapter IV of the GDPR, entitled 'Controller and processor', includes Articles 24 to 43.

    Under section 1 of this chapter, entitled 'General obligations', Article 24 of this Regulation, entitled 'Responsibility of the controller', states, in paragraph 1:

"Taking into account the nature, scope, context and purposes of processing as well as the risks of varying likelihood and severity for the rights and freedoms of natural persons, the controller shall implement appropriate technical and organizational measures to ensure and to be able to demonstrate that processing is performed in accordance with this Regulation. Those measures shall be reviewed and updated where necessary."

    Under section 2 of this chapter, entitled 'Security of personal data', Article 32 of this Regulation, entitled 'Security of processing', states, in paragraph 1:

"Taking into account the state of the art, the costs of implementation and the nature, scope, context and purposes of processing as well as the risk of varying likelihood and severity for the rights and freedoms of natural persons, the controller and the processor shall implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk, including inter alia as appropriate:

a) the pseudonymization and encryption of personal data;

b) the ability to ensure the ongoing confidentiality, integrity, availability and resilience of processing systems and services;

[...]"

    Chapter VIII of the GDPR, entitled 'Remedies, liability and penalties', includes Articles 77 to 84.

    Under Article 82 of this Regulation, entitled 'Right to compensation and liability':

"1. Any person who has suffered material or non-material damage as a result of an infringement of this Regulation shall have the right to receive compensation from the controller or processor for the damage suffered.

    Any controller involved in processing shall be liable for the damage caused by processing which infringes this Regulation. [...]

    A controller or processor shall be exempt from liability under paragraph 2 if it proves that it is not in any way responsible for the event giving rise to the damage.

[...]"

    Article 83 of the GDPR, entitled 'General conditions for imposing administrative fines', states:

"1. Each supervisory authority shall ensure that the imposition of administrative fines pursuant to this Article in respect of infringements of this Regulation referred to in paragraphs 4, 5 and 6 shall in each individual case be effective, proportionate and dissuasive.

    [...] When deciding whether to impose an administrative fine and deciding on the amount of the administrative fine in each individual case due regard shall be given to the following:

a) the nature, gravity and duration of the infringement taking into account the nature, scope or purpose of the processing concerned as well as the number of data subjects affected and the level of damage suffered by them;

b) the intentional or negligent character of the infringement;

[...]

d) the degree of responsibility of the controller or processor taking into account technical and organizational measures implemented by them pursuant to Articles 25 and 32;

[...]

k) any other aggravating or mitigating factor applicable to the circumstances of the case, such as financial benefits gained or losses avoided, directly or indirectly, from the infringement.

    If a controller or processor intentionally or negligently infringes several provisions of this Regulation, in connection with the same processing operations or linked processing operations, the total amount of the administrative fine shall not exceed the amount specified for the gravest infringement.

[...]"

    Article 84 of this Regulation, entitled 'Penalties', states, in paragraph 1:

"Member States shall lay down the rules on other penalties applicable to infringements of this Regulation in particular for infringements which are not subject to administrative fines pursuant to Article 83, and shall take all measures necessary to ensure that they are implemented. Such penalties shall be effective, proportionate and dissuasive."

German law

    Under Article 275, paragraph 1, of the Sozialgesetzbuch, Fünftes Buch (Social Code, Book V), in the version applicable to the dispute in the main proceedings, the compulsory health insurance funds are required to request a Medizinischer Dienst (Medical Service), which assists them, in particular, to conduct an expert opinion to dispel doubts about the incapacity to work of an insured person, in cases determined by law or when the illness of the latter necessitates it.

    Article 278, paragraph 1, of this code provides that such a medical service is established in each of the Länder as a public-law entity.

The main proceedings and the questions referred for a preliminary ruling

    The MDK Nordrhein is a public-law entity which, as a medical service of health insurance funds, has the legal task, among others, of conducting medical assessments aimed at dispelling doubts about the incapacity to work of persons insured with the compulsory health insurance funds within its jurisdiction, including when these assessments concern its own employees.

    In such a case, only members of a special unit, called the 'special cases unit', are authorized to process the 'social' data of this employee, using a locked domain of the IT system of this organization, and to access digital archives after the closure of the expert opinion file. An internal service note regarding these cases stipulates, among other things, that a limited number of authorized agents, including some IT staff, have access to said data.

    The applicant in the main proceedings worked in the IT department of the MDK Nordrhein before being placed on incapacity to work for medical reasons. At the end of the semester during which this organization, as an employer, continued to pay him, the compulsory health insurance fund to which he was affiliated began to pay him sickness benefits.

    This fund then requested the MDK Nordrhein to conduct an assessment of the incapacity to work of the applicant in the main proceedings. A doctor working in the 'special cases unit' of the MDK Nordrhein conducted the assessment, in particular, by obtaining information from the attending physician of the applicant in the main proceedings. When the latter was informed by his attending physician, he contacted one of his colleagues in the IT department and asked her to take and send him photographs of the assessment report that was in the digital archives of the MDK Nordrhein.

    Believing that data concerning his health had thus been processed unlawfully by his employer, the applicant in the main proceedings requested that the latter pay him compensation of EUR 20,000, which the MDK Nordrhein refused.

    Subsequently, the applicant in the main proceedings brought an action before the Arbeitsgericht Düsseldorf (Labor Court, Düsseldorf, Germany) seeking, based on Article 82, paragraph 1, of the GDPR and provisions of German law, to have the MDK Nordrhein ordered to compensate for the damage he claims to have suffered due to the processing of personal data carried out in this manner. He essentially argued, first, that the assessment in question should have been conducted by another medical service to prevent his colleagues from having access to data concerning his health, and second, that the security measures surrounding the archiving of the assessment report were insufficient. He also argued that this processing constituted a violation of the legal rules protecting such data, which caused him both moral and material damage.

    In defense, the MDK Nordrhein primarily argued that the collection and storage of data concerning the health of the applicant in the main proceedings were carried out in compliance with the provisions concerning the protection of such data.

    Having been dismissed at first instance, the applicant in the main proceedings appealed to the Landesarbeitsgericht Düsseldorf (Higher Labor Court, Düsseldorf, Germany), which also rejected his appeal. He then lodged an appeal before the Bundesarbeitsgericht (Federal Labor Court, Germany), which is the referring court in this case.

    This court starts from the premises that, in the main proceedings, the assessment conducted by the MDK Nordrhein, as a medical service, constitutes a 'processing' of 'personal data' and, more specifically, 'data concerning health', within the meaning of Article 4, points 1, 2, and 15, of the GDPR, so that this operation falls within the material scope of this regulation, as defined in Article 2, paragraph 1, thereof. Moreover, it considers that the MDK Nordrhein is the 'controller' concerned, within the meaning of Article 4, point 7, of that regulation.

    Its questions concern, first, the interpretation of several provisions of Article 9 of the GDPR, which relates to the processing of special categories of personal data, particularly given that the processing in question in the main proceedings was carried out by an organization that is also the employer of the data subject, as defined in Article 4, point 1, of this regulation.

    Initially, the referring court doubts that the processing of data concerning health in the main proceedings can fall under one of the exceptions provided in paragraph 2 of Article 9 of the GDPR. According to this court, only the exceptions listed in points b) and h) of this paragraph 2 are relevant in this case. However, it immediately excludes applying the derogation provided in point b) in the present case, on the grounds that the processing in question in the main proceedings was not necessary for the rights and obligations of the controller, taken in its capacity as the employer of the data subject. Indeed, this processing would have been initiated by another organization, which requested the MDK Nordrhein to conduct a check in its capacity as a medical service. Conversely, although it is inclined not to apply the derogation provided in point h) either, as it seems that only processing carried out by a 'neutral third party' should be able to fall under this and that an organization cannot rely on its 'dual role' as employer and medical service to circumvent the prohibition on such processing, the referring court is not categorical in this regard.

    Then, assuming that the processing of data concerning health is authorized in such circumstances under Article 9, paragraph 2, point h), of the GDPR, the referring court questions the data protection rules that must be respected in this context. In its opinion, this regulation implies that it is not sufficient for the controller to meet the requirements set out in Article 9, paragraph 3, thereof. The controller should, in addition, ensure that none of the colleagues of the data subject can have any access to data concerning the health of the latter.

    Finally, still in the same hypothesis, this court wishes to know whether at least one of the conditions set out in Article 6, paragraph 1, of the GDPR must also be met for such processing to be lawful. According to it, this should be the case and, in the context of the main proceedings, only points c) and e) of this Article 6, paragraph 1, first subparagraph, could a priori be relevant. However, these two points c) and e) should not apply, on the grounds that the processing concerned is not 'necessary', within the meaning of these provisions, as it could just as well be carried out by another medical service than the MDK Nordrhein.

    Secondly, assuming that a violation of the GDPR is established in this case, the referring court questions the compensation possibly due to the applicant in the main proceedings under Article 82 of this regulation.

    On the one hand, it wishes to know whether the rule provided in Article 82, paragraph 1, of the GDPR has a deterrent or punitive character, in addition to its compensatory function, and, if so, whether this character should be taken into account when determining the amount of damages awarded for moral damage, particularly in light of the principles of effectiveness, proportionality, and equivalence established in other areas of Union law.

    On the other hand, this court tends to consider that the liability of the controller can be established on the basis of this Article 82, paragraph 1, without the need to prove that it committed a fault. However, having doubts, mainly in view of German law rules, it wonders whether it is necessary to verify that the violation of the GDPR in question is attributable to the controller due to an intentional act or negligence on its part and whether the level of gravity of the fault possibly committed by it should influence the damages awarded for moral damage.

    In these circumstances, the Bundesarbeitsgericht (Federal Labour Court) decided to stay the proceedings and to refer the following questions to the Court for a preliminary ruling:

"1) Must Article 9, paragraph 2, subparagraph h), of the [GDPR] be interpreted as prohibiting the medical service of a health insurance fund from processing health data of one of its employees, whose work capacity is being assessed?

    If the Court answers the first question in the negative, so that a derogation from the prohibition on processing health data set out in Article 9, paragraph 1, of the GDPR is envisaged, pursuant to Article 9, paragraph 2, subparagraph h), of the GDPR, in a case such as the present one, in addition to the conditions set out in Article 9, paragraph 3, of the GDPR, must other data protection requirements be complied with, and if so, which ones?

    If the Court answers the first question in the negative, so that a derogation from the prohibition on processing health data set out in Article 9, paragraph 1, of the GDPR is envisaged, pursuant to Article 9, paragraph 2, subparagraph h), of the GDPR, in a case such as the present one, does the lawfulness of processing health data also depend on compliance with at least one of the conditions set out in Article 6, paragraph 1, of the GDPR?

    Does Article 82, paragraph 1, of the GDPR have a special or general deterrent character and should this be taken into account in the assessment of the moral damage compensable that the controller or processor is required to repair on the basis of this provision?

    Does the degree of severity of the fault of the controller or processor influence the assessment of the moral damage compensable on the basis of Article 82, paragraph 1, of the GDPR? In particular, can the absence of fault or slight fault on the part of the controller or processor be considered in its defense?"

On the questions referred for a preliminary ruling

On the first question

    By its first question, the referring court asks, in essence, whether, given the prohibition on processing health data provided in Article 9, paragraph 1, of the GDPR, paragraph 2, subparagraph h), of that article should be interpreted as meaning that the exception it provides is applicable to situations where a medical control body processes health data of one of its employees in the capacity not of an employer but of a medical service, to assess the work capacity of that employee.

    According to consistent case law, the interpretation of a provision of Union law requires taking into account not only its wording but also the context in which it occurs and the objectives and purpose pursued by the act of which it forms part. The origin of a provision of Union law can also reveal relevant elements for its interpretation (judgment of 16 March 2023, Towercast, C-449/21, EU:C:2023:207, point 31 and cited case law).

    First, it should be noted that Article 9 of the GDPR addresses, as indicated in its title, the 'Processing of special categories of personal data', also referred to as 'sensitive data' in recitals 10 and 51 of this regulation.

    Recital 51 of the GDPR states that personal data that are, by nature, particularly sensitive in relation to fundamental rights and freedoms merit specific protection as the context of their processing could create significant risks to those freedoms and rights.

    Thus, Article 9, paragraph 1, of the GDPR lays down the principle of the prohibition on processing the special categories of personal data it lists. Among these, 'data concerning health', as defined in Article 4, point 15, of this regulation, read in light of recital 35 thereof, are referred to in the present case.

    The Court has clarified that the purpose of Article 9, paragraph 1, of that regulation is to ensure enhanced protection against processing which, due to the particular sensitivity of the data in question, is likely to constitute particularly serious interference with the fundamental rights to respect for private life and to the protection of personal data, guaranteed by Articles 7 and 8 of the Charter of Fundamental Rights [see, to that effect, judgment of 5 June 2023, Commission/Poland (Independence and privacy of judges), C-204/21, EU:C:2023:442, point 345 and cited case law].

    However, Article 9, paragraph 2, subparagraphs a) to j), of the GDPR provides an exhaustive list of exceptions to the principle of the prohibition on processing such sensitive data.

    In particular, Article 9, paragraph 2, subparagraph h), of the GDPR authorizes such processing if it is 'necessary for [...] the assessment of the working capacity of the employee [...] based on Union or Member State law or pursuant to a contract with a health professional'. This provision specifies that any processing based on it is also 'subject to the conditions and safeguards referred to in paragraph 3' of that Article 9.

    It follows from Article 9, paragraph 2, subparagraph h), of the GDPR, read in conjunction with paragraph 3 of that article, that the possibility of processing sensitive data, such as health data, is strictly framed by a series of cumulative conditions. These relate, first, to the purposes listed in that point h) – including the assessment of an employee's work capacity –, secondly, to the legal basis for such processing – whether Union law, Member State law, or a contract with a health professional, as per that point h) – and finally, thirdly, to the duty of confidentiality to which the persons authorized to carry out such processing are subject, under Article 9, paragraph 3, all these persons being subject to an obligation of secrecy under this latter provision.

    As the Advocate General observed in substance in points 32 and 33 of his Opinion, neither the wording of Article 9, paragraph 2, subparagraph h), of the GDPR nor the origin of this provision provide any elements to consider that the application of the derogation provided in that provision would be reserved, as the referring court suggests, for situations where the processing is carried out by a 'neutral third party' and not by the employer of the data subject, as defined in Article 4, point 1, of this regulation.

    In view of the referring court's opinion that an organization should not be able to rely on its 'dual role' as the employer of the data subject and as a medical service to escape the principle of the prohibition on processing health data, set out in Article 9, paragraph 1, of the GDPR, it is important to consider the capacity in which such data processing is carried out.

    Indeed, if Article 9, paragraph 1, prohibits, in principle, the processing of health data, paragraph 2 of that article provides, in subparagraphs a) to j), ten derogations that are independent of each other and must therefore be assessed autonomously. It follows that the fact that the conditions for applying one of the derogations provided in that paragraph 2 are not met does not prevent a controller from relying on another derogation mentioned in that provision.

    It follows from the above that Article 9, paragraph 2, subparagraph h), of the GDPR, read in conjunction with paragraph 3 of that article, does not exclude the applicability of the exception provided in that point h) to situations where a medical control body processes health data of one of its employees in the capacity of a medical service, and not as an employer, to assess the work capacity of that employee.

    Second, such an interpretation is supported by the consideration of the system in which Article 9, paragraph 2, subparagraph h), of the GDPR is situated as well as the objectives pursued by this regulation and this provision.

    Firstly, while it is true that Article 9, paragraph 2, of the GDPR must be interpreted restrictively, since it provides an exception to the principle of the prohibition on processing special categories of personal data [judgment of 4 July 2023, Meta Platforms et al. (Terms of use of a social network), C-252/21, EU:C:2023:537, point 76], respect for the principle of prohibition set out in Article 9, paragraph 1, of the GDPR cannot lead to a reduction of the scope of another provision of this regulation in a way that would go against the clear wording of the latter. However, the interpretation suggested, according to which the scope of the exception provided in that Article 9, paragraph 2, subparagraph h), should be limited to situations where a 'neutral third party' processes health data for the purpose of assessing the working capacity of a worker, would add a requirement that does not appear in the clear wording of this latter provision.

    In this respect, it is irrelevant that, in the present case, if the MDK Nordrhein were prohibited from fulfilling its mission as a medical service when it concerns one of its own employees, another medical control body could be able to take over. It is important to emphasize that this alternative, mentioned by the referring court, is not necessarily present or feasible in all Member States and in all situations that could be covered by Article 9, paragraph 2, subparagraph h), of the GDPR. Therefore, the interpretation of this provision should not be guided by considerations drawn from the health system of a single Member State or circumstances specific to the main proceedings.

    Secondly, the interpretation set out in point 48 of this judgment is consistent with the objectives of the GDPR and those of Article 9 of this regulation.

    Thus, recital 4 of the GDPR states that the right to the protection of personal data is not an absolute right, as it must be considered in relation to its function in society and be balanced against other fundamental rights, in accordance with the principle of proportionality (see, to that effect, judgment of 22 June 2023, Pankki S, C-579/21, EU:C:2023:501, point 78). Furthermore, the Court has already emphasized that the mechanisms to achieve a fair balance between the different rights and interests at stake are embedded in the GDPR itself (see, to that effect, judgment of 17 June 2021, M.I.C.M., C-597/19, EU:C:2021:492, point 112).

    These considerations apply even when the data concerned belong to the special categories referred to in Article 9 of this regulation [see, to that effect, judgment of 24 September 2019, GC et al. (De-referencing of sensitive data), C-136/17, EU:C:2019:773, points 57 and 66 to 68], such as health data.

    More specifically, recital 52 of the GDPR indicates that 'derogations from the prohibition on processing such special categories of data' should be authorized 'where the public interest so justifies, in particular [...] in the field of employment and social security and social protection law,' as well as 'for health purposes, [...] in particular to ensure the quality and efficiency of the procedures for settling claims for benefits and services in the health insurance system'. Recital 53 of this regulation also states that processing carried out 'for health-related purposes' should be possible 'where necessary to achieve those purposes for the benefit of individuals and society as a whole, particularly in the context of the management of health-care services or social protection systems'.

    It is in this overall perspective and in view of the various legitimate interests at stake that the Union legislature has provided, in Article 9, paragraph 2, subparagraph h), of the GDPR, a possibility to derogate from the principle of the prohibition on processing health data set out in paragraph 1 of that article, provided that the processing concerned meets the conditions and safeguards expressly imposed by that point h) and by the other relevant provisions of this regulation, in particular paragraph 3 of that Article 9, which do not include the requirement that a medical service processing such data under that point h) be an entity distinct from the employer of the data subject.

    In light of the foregoing, and without prejudice to the answers to be provided to the second and third questions, it must be concluded that Article 9, paragraph 2, subparagraph h), of the GDPR should be interpreted as meaning that the exception provided in that provision is applicable to situations where a medical control body processes health data of one of its employees in the capacity of a medical service, and not as an employer, to assess the work capacity of that employee, provided that the processing concerned meets the conditions and safeguards expressly imposed by that point h) and by paragraph 3 of that Article 9.

On the second question

    According to the referring court, it follows from recitals 35, 51, 53, and 75 of the GDPR that it is not sufficient to meet the requirements of Article 9, paragraph 3, thereof, in a situation such as the main proceedings, where the controller is also the employer of the person whose work capacity is being assessed. This regulation would also require excluding from the processing of health data all employees of the controller who are likely to have any professional contact with that person. According to this court, any controller with several establishments, such as the MDK Nordrhein, should ensure that the entity responsible for processing health data of the controller's employees always belongs to a different establishment than the one where the data subject works. Moreover, the professional secrecy obligation imposed on employees authorized to process such data would not, in practice, prevent a colleague of the data subject from accessing data concerning them, which would pose risks of harm, such as damage to their reputation.

    In these circumstances, by its second question, the referring court essentially asks whether the provisions of the GDPR should be interpreted as meaning that the controller of health data processing, based on Article 9, paragraph 2, subparagraph h), of this regulation, is required to ensure that no colleague of the data subject can access data relating to their health status.

    It should be recalled that, pursuant to Article 9, paragraph 3, of the GDPR, processing of the data and for the purposes listed in paragraph 1 and paragraph 2, subparagraph h), of that Article 9, in this case, data concerning the health of a worker for the assessment of their work capacity, may only be carried out if such data are processed by or under the responsibility of a professional subject to the obligation of professional secrecy under Union or Member State law or rules established by national competent bodies or by another person also subject to an obligation of secrecy under Union or Member State law or rules established by national competent bodies.

    By adopting Article 9, paragraph 3, of this regulation, which refers precisely to paragraph 2, subparagraph h), of that Article 9, the Union legislature defined the specific protection measures it intended to impose on controllers of such processing, which consist of ensuring that such processing is reserved for persons subject to an obligation of secrecy, in accordance with the conditions detailed in that paragraph 3. There is therefore no need to add to the wording of this latter provision requirements that it does not mention.

    It follows, as the Advocate General observed in substance in point 43 of his Opinion, that Article 9, paragraph 3, of the GDPR cannot serve as the legal basis for a measure ensuring that no colleague of the data subject can access data relating to their health status.

    However, it is necessary to assess whether the requirement to ensure that no colleague of the data subject has access to data relating to their health status can be imposed on the controller of health data processing, based on Article 9, paragraph 2, subparagraph h), of the GDPR, on the basis of another provision of this regulation.

    In this respect, it is important to specify that the only possibility for Member States to add such a requirement, in addition to those set out in paragraphs 2 and 3 of Article 9 of this regulation, lies in the explicit possibility granted to them by paragraph 4 of that article to 'maintain or introduce further conditions, including limitations, with regard to the processing of genetic data, biometric data or data concerning health'.

    However, these possible additional conditions do not result from the provisions of the GDPR themselves but, where applicable, from national rules governing such processing, with respect to which this regulation expressly leaves a margin of appreciation to Member States (see, to that effect, judgment of 30 March 2023, Hauptpersonalrat der Lehrerinnen und Lehrer, C-34/21, EU:C:2023:270, points 51 and 78).

    Moreover, it should be emphasized that a Member State intending to make use of the possibility provided by Article 9, paragraph 4, of that regulation must, in accordance with the principle of proportionality, ensure that the practical consequences, particularly of an organizational, economic, and medical nature, resulting from the additional requirements it intends to impose on such processing, are not excessive for the controllers of such processing, who do not necessarily have sufficient dimensions or technical and human resources to meet these requirements. Indeed, these should not undermine the effectiveness of the authorization of processing expressly provided for in Article 9, paragraph 2, subparagraph h), of that regulation and framed in paragraph 3 of that Article 9.

    Finally, it is important to emphasize that, under Article 32, paragraph 1, subparagraphs a) and b), of the GDPR, which implements the principles of integrity and confidentiality set out in Article 5, paragraph 1, subparagraph f), of that regulation, every controller of personal data processing is required to implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk, particularly the pseudonymization and encryption of such data, as well as means to ensure, in particular, the confidentiality and integrity of processing systems and services. To determine the practical modalities of this obligation, the controller must, in accordance with that Article 32, paragraph 1, take into account the state of the art, the costs of implementation, and the nature, scope, context, and purposes of the processing as well as the risks, whose likelihood and severity vary, to the rights and freedoms of natural persons.

    However, it will be for the referring court to assess whether all the technical and organizational measures implemented, in this case, by the MDK Nordrhein, are in accordance with the requirements of Article 32, paragraph 1, subparagraphs a) and b), of the GDPR.

    Therefore, it must be concluded that Article 9, paragraph 3, of the GDPR should be interpreted as meaning that the controller of health data processing, based on Article 9, paragraph 2, subparagraph h), of this regulation, is not required, under these provisions, to ensure that no colleague of the data subject can access data relating to their health status. However, such an obligation may be imposed on the controller of such processing either under a regulation adopted by a Member State based on Article 9, paragraph 4, of that regulation or under the principles of integrity and confidentiality set out in Article 5, paragraph 1, subparagraph f), of that regulation and implemented in Article 32, paragraph 1, subparagraphs a) and b), thereof.

On the third question

    By its third question, the referring court asks, in essence, whether Article 9, paragraph 2, subparagraph h), and Article 6, paragraph 1, of the GDPR should be interpreted as meaning that a processing of health data based on the first provision must, to be lawful, not only comply with the requirements arising from it but also fulfill at least one of the conditions of lawfulness set out in that Article 6, paragraph 1.

    In this regard, it should be noted that Articles 5, 6, and 9 of the GDPR are part of Chapter II of this regulation, entitled 'Principles', and relate, respectively, to the principles relating to the processing of personal data, the conditions of lawfulness of processing, and the processing of special categories of personal data.

    Moreover, it should be noted that recital 51 of the GDPR explicitly states that, 'in addition to the specific requirements' applicable to the processing of 'particularly sensitive' data, which are set out in Article 9, paragraphs 2 and 3, of this regulation, without prejudice to any measures adopted by a Member State based on paragraph 4 of that article, 'the general principles and other rules [of that] regulation should [also] apply [to such processing], in particular as regards the conditions for lawful processing', as they result from Article 6 of that regulation.

    Consequently, under Article 6, paragraph 1, first subparagraph, of the GDPR, processing of 'particularly sensitive' data, such as health data, is lawful only if at least one of the conditions listed in that paragraph 1, first subparagraph, subparagraphs a) to f), is met.

    Article 6, paragraph 1, first subparagraph, of that regulation provides an exhaustive and restrictive list of cases in which personal data processing can be considered lawful. Thus, for it to be considered legitimate, processing must fall under one of the cases provided in that provision [judgment of 4 July 2023, Meta Platforms et al. (Terms of use of a social network), C-252/21, EU:C:2023:537, point 90 and cited case law].

    Therefore, the Court has repeatedly ruled that any processing of personal data must comply with the principles relating to processing set out in Article 5, paragraph 1, of the GDPR and meet the conditions of lawfulness of processing listed in Article 6 of this regulation [judgment of 4 May 2023, Bundesrepublik Deutschland (Electronic court mailbox), C-60/22, EU:C:2023:373, point 57 and cited case law].

    Furthermore, it has been ruled that, as Articles 7 to 11 of the GDPR, which appear, like Articles 5 and 6 of it, in Chapter II of this regulation, aim to define the scope of the obligations incumbent on the controller under Article 5, paragraph 1, subparagraph a), and Article 6, paragraph 1, of that regulation, the processing of personal data, to be lawful, must also comply with these other provisions of that chapter, which concern, in substance, consent, the processing of special categories of sensitive personal data, and the processing of personal data relating to criminal convictions and offenses [judgment of 4 May 2023, Bundesrepublik Deutschland (Electronic court mailbox), C-60/22, EU:C:2023:373, point 58 and cited case law].

    It follows, in particular, that, to the extent that Article 9, paragraph 2, subparagraph h), of the GDPR aims to define the scope of the obligations incumbent on the controller under Article 5, paragraph 1, subparagraph a), and Article 6, paragraph 1, of that regulation, a processing of health data based on that first provision must, to be lawful, comply with both the requirements arising from it and the obligations resulting from these latter two provisions and, in particular, fulfill at least one of the conditions of lawfulness listed in that Article 6, paragraph 1.

    In light of the foregoing, it must be concluded that Article 9, paragraph 2, subparagraph h), and Article 6, paragraph 1, of the GDPR should be interpreted as meaning that a processing of health data based on that first provision must, to be lawful, not only comply with the requirements arising from it but also fulfill at least one of the conditions of lawfulness set out in that Article 6, paragraph 1.

On the fourth question

    By its fourth question, the referring court asks, in essence, whether Article 82, paragraph 1, of the GDPR should be interpreted as meaning that the right to compensation provided in that provision fulfills not only a compensatory function but also a deterrent or punitive function and, if so, whether this latter should be taken into account when determining the amount of damages awarded for moral damage based on this provision.

    It should be recalled that Article 82, paragraph 1, of the GDPR states that 'any person who has suffered material or non-material damage as a result of an infringement of this Regulation shall have the right to receive compensation from the controller or processor for the damage suffered'.

    The Court has interpreted this provision as meaning that the mere infringement of the GDPR is not sufficient to confer a right to compensation, after highlighting, in particular, that the existence of 'damage' or 'harm' that has been 'suffered' constitutes one of the conditions of the right to compensation provided in that Article 82, paragraph 1, along with the existence of an infringement of that regulation and a causal link between that damage and that infringement, these three conditions being cumulative [see, to that effect, judgment of 4 May 2023, Österreichische Post (Non-material damage related to data processing), C-300/21, EU:C:2023:370, points 32 and 42].

    Moreover, the Court has ruled that, since the GDPR does not contain a provision defining the rules relating to the assessment of damages due under the right to compensation provided in Article 82 of that regulation, national courts must apply, in this respect, the internal rules of each Member State relating to the scope of pecuniary compensation, provided that the principles of equivalence and effectiveness of Union law are respected, as defined by the Court's consistent case law [see, to that effect, judgment of 4 May 2023, Österreichische Post (Non-material damage related to data processing), C-300/21, EU:C:2023:370, points 53, 54, and 59].

    In this context and in light of recital 146, sixth sentence, of the GDPR, which states that this instrument aims to ensure 'full and effective compensation for the damage suffered', the Court noted that, given the compensatory function of the right to compensation provided in Article 82 of that regulation, pecuniary compensation based on this article should be considered 'full and effective' if it allows full compensation for the harm actually suffered as a result of the infringement of that regulation, without it being necessary, for such full compensation, to impose the payment of punitive damages [see, to that effect, judgment of 4 May 2023, Österreichische Post (Non-material damage related to data processing), C-300/21, EU:C:2023:370, points 57 and 58].

    In this regard, it is important to emphasize that Article 82 of the GDPR fulfills a compensatory, not a punitive, function, unlike other provisions of this regulation also appearing in Chapter VIII thereof, namely its Articles 83 and 84, which have, for their part, an essentially punitive purpose, since they respectively allow the imposition of administrative fines as well as other penalties. The articulation between the rules set out in that Article 82 and those set out in those Articles 83 and 84 demonstrates that there is a difference between these two categories of provisions, but also a complementarity, in terms of encouraging compliance with the GDPR, noting that the right of any person to seek compensation for harm strengthens the operational nature of the protection rules provided by this regulation and is likely to deter the repetition of unlawful behavior [see, to that effect, judgment of 4 May 2023, Österreichische Post (Non-material damage related to data processing), C-300/21, EU:C:2023:370, points 38 and 40].

    Therefore, since the right to compensation provided in Article 82, paragraph 1, of the GDPR does not fulfill a deterrent or punitive function, as envisaged by the referring court, the severity of the infringement of that regulation that caused the damage concerned should not influence the amount of damages awarded under that provision, even when it concerns non-material damage. It follows that this amount should not be set at a level exceeding full compensation for that harm.

    Consequently, it must be concluded that Article 82, paragraph 1, of the GDPR should be interpreted as meaning that the right to compensation provided in that provision fulfills a compensatory function, in that pecuniary compensation based on that provision must allow full compensation for the harm actually suffered as a result of the infringement of that regulation, and not a deterrent or punitive function.

On the fifth question

    It follows from the elements provided by the referring court, in response to a request for clarification addressed to it under Article 101 of the Court's Rules of Procedure, that the fifth question seeks to determine, on the one hand, whether the existence and/or proof of fault are required conditions for the liability of the controller or processor to be engaged, and on the other hand, what impact the degree of severity of the controller's or processor's fault is likely to have on the concrete assessment of damages to be awarded for non-material harm suffered.

    In light of this response from the referring court, the fifth question should be understood as aiming, in essence, to know, on the one hand, whether Article 82 of the GDPR should be interpreted as meaning that the liability of the controller is conditional on the existence of a fault committed by it, and, on the other hand, whether the degree of severity of that fault should be taken into account when determining the amount of damages awarded for non-material harm based on that provision.

    Regarding the first part of this question, it should be noted that, as recalled in point 82 of this judgment, Article 82, paragraph 1, of the GDPR makes the right to compensation conditional on the existence of three elements, namely the existence of an infringement of that regulation, the existence of harm suffered, and the existence of a causal link between that harm and that infringement.

    Article 82, paragraph 2, of the GDPR states that any controller involved in processing shall be liable for the harm caused by processing that infringes this regulation. However, the wording of this provision in certain linguistic versions, including the German version, which is the language of the proceedings in this case, does not make it possible to determine with certainty whether the infringement in question must be attributable to the controller for its liability to be engaged.

    In this respect, it follows from an analysis of the different linguistic versions of the first sentence of Article 82, paragraph 2, of the GDPR that the controller is presumed to have participated in the processing constituting the infringement of this regulation that is referred to. Indeed, while the versions in German, French, or Finnish are formulated in an open manner, a number of other linguistic versions are more precise and use a demonstrative determiner for the third occurrence of the term 'processing', or for the third reference to this term, making it clear that this third occurrence or reference refers to the same operation as the second occurrence of this term. This is the case for the versions in Spanish, Estonian, Greek, Italian, or Romanian.

    Article 82, paragraph 3, of the GDPR further specifies, in this perspective, that a controller is exempt from liability under paragraph 2 of that Article 82 if it proves that the event that caused the harm is not in any way attributable to it.

    It follows from a combined analysis of these different provisions of Article 82 of the GDPR that this article provides a fault-based liability regime in which the burden of proof lies not on the person who has suffered harm but on the controller.

    Such an interpretation is corroborated by the context in which this Article 82 occurs and the objectives pursued by the Union legislature through the GDPR.

    In this respect, first, it follows from the wording of Articles 24 and 32 of the GDPR that these provisions merely require the controller to adopt technical and organizational measures aimed at preventing, to the greatest extent possible, any data breaches. The appropriateness of such measures must be assessed concretely by examining whether these measures have been implemented by the controller, taking into account the various criteria set out in those articles and the specific data protection needs inherent in the processing concerned, as well as the risks induced by it (see, to that effect, judgment of 14 December 2023, Natsionalna agentsia za prihodite, C-340/21, EU:C:2023:986, point 30).

    However, such an obligation would be called into question if the controller were subsequently required to compensate for any damage caused by processing carried out in violation of the GDPR.

    Second, regarding the objectives of the GDPR, it follows from recitals 4 to 8 of this regulation that it aims to achieve a balance between the interests of controllers of personal data and the rights of individuals whose personal data are processed. The intended purpose is to enable the development of the digital economy while ensuring a high level of protection for individuals. Thus, a balancing of the interests of the controller and those of the individuals whose personal data are processed is sought. However, a fault-based liability regime with a reversal of the burden of proof, as provided by Article 82 of the GDPR, precisely ensures such a balance.

    On the one hand, as the Advocate General observed in substance in point 93 of his Opinion, it would not be consistent with the objective of such high protection to opt for an interpretation whereby individuals who have suffered harm due to a violation of the GDPR would, in the context of an action for compensation based on Article 82 of that regulation, bear the burden of proving not only the existence of that violation and the resulting harm but also the existence of a fault committed by the controller either intentionally or negligently, or the degree of severity of that fault, when that Article 82 does not formulate such requirements (see, by analogy, judgment of 14 December 2023, Natsionalna agentsia za prihodite, C-340/21, EU:C:2023:986, point 56).

    On the other hand, a strict liability regime would not ensure the achievement of the objective of legal certainty pursued by the legislature, as indicated in recital 7 of the GDPR.

    Regarding the second part of the fifth question, relating to the determination of the amount of damages that may be due under Article 82 of the GDPR, it should be recalled that, as emphasized in point 83 of this judgment, for the assessment of these damages, national courts must apply the internal rules of each Member State relating to the scope of pecuniary compensation, provided that the principles of equivalence and effectiveness of Union law are respected, as defined by the consistent case law of the Court.

    It is important to specify that, given its compensatory function, Article 82 of the GDPR does not require the severity of the violation of that regulation, which the controller is presumed to have committed, to be taken into account when determining the amount of damages awarded for non-material harm based on that provision, but requires that this amount be set in a manner that fully compensates the harm actually suffered as a result of the violation of that regulation, as follows from points 84 and 87 of this judgment.

    Therefore, it must be concluded that Article 82 of the GDPR should be interpreted as meaning that, on the one hand, the liability of the controller is conditional on the existence of a fault committed by it, which is presumed unless it proves that the event causing the harm is not in any way attributable to it, and, on the other hand, this Article 82 does not require that the severity of that fault be taken into account when determining the amount of damages awarded for non-material harm based on that provision.

On costs

    Since these proceedings are, for the parties to the main proceedings, a step in the action pending before the referring court, it is for that court to decide on the costs. The costs incurred for submitting observations to the Court, other than those of the said parties, are not recoverable.

For these reasons, the Court (Third Chamber) rules:

    Article 9, paragraph 2, subparagraph h), of Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation),

must be interpreted as meaning that:

the exception provided in that provision is applicable to situations where a medical control body processes health data of one of its employees in the capacity of a medical service, and not as an employer, to assess the work capacity of that employee, provided that the processing concerned meets the conditions and safeguards expressly imposed by that point h) and by paragraph 3 of that Article 9.

    Article 9, paragraph 3, of Regulation 2016/679

must be interpreted as meaning that:

the controller of health data processing, based on Article 9, paragraph 2, subparagraph h), of this regulation, is not required, under these provisions, to ensure that no colleague of the data subject can access data relating to their health status. However, such an obligation may be imposed on the controller of such processing either under a regulation adopted by a Member State based on Article 9, paragraph 4, of that regulation or under the principles of integrity and confidentiality set out in Article 5, paragraph 1, subparagraph f), of that regulation and implemented in Article 32, paragraph 1, subparagraphs a) and b), thereof.

    Article 9, paragraph 2, subparagraph h), and Article 6, paragraph 1, of Regulation 2016/679

must be interpreted as meaning that:

a processing of health data based on this first provision must, to be lawful, not only comply with the requirements arising from it but also fulfill at least one of the conditions of lawfulness set out in that Article 6, paragraph 1.

    Article 82, paragraph 1, of Regulation 2016/679

must be interpreted as meaning that:

the right to compensation provided in that provision fulfills a compensatory function, in that pecuniary compensation based on that provision must allow full compensation for the harm actually suffered as a result of the infringement of that regulation, and not a deterrent or punitive function.

    Article 82 of Regulation 2016/679

must be interpreted as meaning that:

on the one hand, the liability of the controller is conditional on the existence of a fault committed by it, which is presumed unless it proves that the event causing the harm is not in any way attributable to it, and, on the other hand, this Article 82 does not require that the severity of that fault be taken into account when determining the amount of damages awarded for non-material harm based on that provision."""
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