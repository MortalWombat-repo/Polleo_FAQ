import pysqlite3  # noqa: F401
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pickle
import time
import requests
import zipfile
from google import genai
from google.genai import types
from IPython.display import Markdown
from IPython.display import display
from dotenv import load_dotenv
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
import re
from langcodes import Language


def get_language_name(lang_code):
    language = Language.make(language=lang_code).language_name()
    print(language)
    return language


def import_google_api():
    #importing Google api key
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    for m in client.models.list():
        if "embedContent" in m.supported_actions:
            print(m.name)

    return client


def embedding_function(client):
    class GeminiEmbeddingFunction(EmbeddingFunction):
        document_mode = True

        def __init__(self, client):
            self.client = client
            self._retry = retry.Retry(predicate=lambda e: isinstance(e, genai.errors.APIError) and e.code in {429, 503})

        def __call__(self, input: Documents) -> Embeddings:
            embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
            response = self._retry(self.client.models.embed_content)(
                model="models/text-embedding-004",
                contents=input,
                config=types.EmbedContentConfig(task_type=embedding_task),
            )
            return [e.values for e in response.embeddings]

    return GeminiEmbeddingFunction(client)


def create_collection(chroma_client, gemini_embedding_function, documents_list):
    DB_NAME = "polleo_faq"
    embed_fn = gemini_embedding_function
    embed_fn.document_mode = True

    db = chroma_client.get_or_create_collection(
        name=DB_NAME,
        metadata={"model": "text-embedding-004", "dimension": 768},
        embedding_function=embed_fn
    )

    # Check if the collection is empty before adding documents
    if db.count() == 0: # <--- Re-added this critical check
        documents = documents_list
        print(f"Adding {len(documents)} documents to ChromaDB collection: {DB_NAME}")
        for i, doc in enumerate(documents):
            try:
                # Using more robust IDs to prevent potential clashes
                db.add(documents=[doc], ids=[f"{DB_NAME}_doc_{i}"])
                #time.sleep(0.1) # Consider re-enabling small sleep for API rate limits if many docs
                print(f"Added document with ID: {DB_NAME}_doc_{i}, Content (first 100 chars): {str(doc[:100])}")
            except Exception as e:
                print(f"Error adding document {DB_NAME}_doc_{i}: {e}")
    else:
        print(f"Collection '{DB_NAME}' already contains {db.count()} documents. Skipping document addition.")


def persistent_client(embed_fn):

    # Initialize PersistentClient with desired path
    persist_dir = "./output"  # Use one directory for persistence
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    DB_NAME = "polleo_faq"  # Use a more specific name for clarity, e.g., 'polleo_sport_faqs_hr'
    embed_fn = embed_fn
    collection = chroma_client.get_collection(DB_NAME, embedding_function=embed_fn)

    # List collection names to verify the database
    #collection_names = chroma_client.list_collections()
    #print("Collection names:", collection_names)

    # Access a specific collection by its name
    #collection_name = collection_names[0]  # You can select the first collection or any other collection by name

    # Peek into the collection (view first item)
    #print(collection.peek(1))
    #print(collection.count())
    print(collection.metadata)  # Check metadata for clues about the embedding model
    #print(collection.count())  # Verify the collection has data

    # Peek at a sample document
    #print(f"Sample document: {collection.peek(1)}")

    return embed_fn, collection


def get_article(user_query, embed_fn, collection, client, user_language):
    print(user_language.upper())
    # Switch to query mode when generating embeddings
    embed_fn.document_mode = False

    result = collection.query(query_texts=[user_query], n_results=7)
    [all_passages] = result["documents"]

    query_oneline = user_query.replace("\n", " ")

    print(query_oneline)

    faq_prompt = f"""
    You are a friendly and clear customer support assistant who replies in the **same language as the user's question**, which is {get_language_name(user_language.upper())}, using information from the FAQ excerpt provided below.

    Your tone and style:
    - Be friendly, informative, and professional.
    - Respond clearly without using technical jargon.
    - If the question is ambiguous, use context to infer the most likely meaning.
    - If there is no direct answer in the text, reply based on general knowledge and give helpful advice.

    Formatting rules:
    - If contacting support is mentioned, provide the contact information (if available).
    - Use bullet points where possible to improve readability.
    - If the text mentions conditions (e.g., minimum order amount, delivery times, return policy), highlight them clearly.
    - **If the text contains a URL (marked as "URL is [some_link]"), place it at the beginning of the response and separate it with a blank line.**

    Instructions for structuring your response:
    - Write in full sentences.
    - Start with a direct answer to the question.
    - Then, if needed, explain any relevant background or related policies (e.g., delivery, availability, difference between Web shop and physical stores).
    - **Do not repeat the user's question.**
    - Use examples where helpful to increase clarity.

    QUESTION (in {get_language_name(user_language.upper())}): {query_oneline}
    """

    for passage in all_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"PASSAGE: {passage_oneline}\n"

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)

    #return Markdown(answer.text)
    return answer.text


def get_article_hr(user_query, embed_fn, collection, client, user_language):
    print(user_language.upper())
    embed_fn.document_mode = False

    # Retrieve more results for better context
    result = collection.query(query_texts=[user_query], n_results=7)
    # Check if documents were found
    if not result["documents"]:
        return "Ispričavamo se, ali ne mogu pronaći relevantan odgovor u našoj bazi FAQ. Molimo kontaktirajte našu podršku za više informacija." # Fallback if no docs found

    # Access documents; result["documents"] is a list of lists, so [0] gives the list of docs
    all_passages = result["documents"][0] # Correctly access the list of documents

    query_oneline = user_query.replace("\n", " ")
    print(f"Query: {query_oneline}")
    print(f"Retrieved passages: {len(all_passages)}")


    # Start building the prompt with strong instructions
    prompt = f"""
    Ti si ljubazan i jasan asistent korisničke podrške koji odgovara na **istom jeziku kao korisnički upit**, koji je {get_language_name(user_language.upper())}.

    **Koristi ISKLJUČIVO informacije iz sljedećih FAQ odlomaka za odgovor na pitanje. Ako odgovor nije pronađen u dostavljenim odlomcima, jasno navedi da ne možeš pronaći odgovor u FAQ i uputi korisnika na kontaktiranje podrške.**

    Tvoj ton i stil:
    - Budi prijateljski, informativan i profesionalan.
    - Odgovaraj jasno i bez stručnog žargona.

    Pravila formatiranja:
    - Ako se spominje mogućnost kontaktiranja podrške, ponudi kontakt informacije (ako su dostupne).
    - Koristi popise s grafičkim oznakama gdje je moguće za veću preglednost.
    - Ako su u tekstu spomenuti uvjeti (npr. minimalni iznos narudžbe, rokovi dostave, povrat), istakni ih jasno.
    - **Ako tekst sadrži URL (označen kao "URL je [some_link]"), dodaj ga na početak odgovora i odvoji praznim redom.**

    Upute za strukturiranje odgovora:
    - Odgovaraj u potpunim rečenicama.
    - Prvo odgovori na pitanje izravno.
    - Zatim objasni širi kontekst ako je potrebno (e.g., pravila dostave, dostupnost proizvoda, razlike Web shopa i poslovnica).
    - **Ne ponavljaj korisničko pitanje.**
    - Koristi primjere gdje je moguće kako bi korisniku bilo jasnije.

    **--- DOSTUPNI FAQ ODLOMCI ---**
    """

    for i, passage in enumerate(all_passages):
        passage_oneline = passage.replace("\n", " ")
        prompt += f"ODLOMAK {i+1}: {passage_oneline}\n\n" # Clearly label passages

    prompt += f"**--- KRAJ ODLOMAKA ---**\n\n"
    prompt += f"PITANJE (na {get_language_name(user_language.upper())}): {query_oneline}\n"
    prompt += "ODGOVOR:" # Signal to the LLM to start answering

    # print(prompt) # Uncomment this to inspect the full prompt being sent to Gemini! This is very helpful for debugging.

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)

    return answer.text


def parse_faq_file(file_path):
    """
    Reads a file with Q: and A: pairs and combines them into single documents.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Use a regular expression to find all Q: and A: pairs.
    # The pattern finds "Q: " followed by text, then "A: " followed by text,
    # until the next "Q:" or the end of the file.
    faq_pairs = re.findall(r'Q: (.*?)\n(A: .*?)(?=\nQ: |\Z)', content, re.DOTALL)
    
    # Concatenate each question and answer into a single, cohesive document.
    faq_documents = [f"Pitanje: {q.strip()}\nOdgovor: {a.strip()}" for q, a in faq_pairs]
    
    return faq_documents


# Call the new function to process your FAQ file
faq = parse_faq_file("faq_croatian.txt")

# You can add a print statement to confirm it worked correctly
print(f"Processed {len(faq)} combined Q&A documents from the file.")
print(f"Sample document content: \n---\n{faq[0]}\n---")


client = import_google_api()
gemini_embedding_function = embedding_function(client)
chroma_persistent_client = chromadb.PersistentClient(path="./output") # Choose a suitable path for your DB
create_collection(chroma_persistent_client, gemini_embedding_function, faq)

user_query = "Imate li program vjernosti i nagrađivanja i kako se zove?"

embed_fn, collection = persistent_client(gemini_embedding_function) # Make sure this also connects to the correct DB_NAME and path

# And for your get_article calls:
user_lang = "HR" # Assuming you'll determine this dynamically
print(f"User query language: {get_language_name(user_lang)}")


print("\n--- Testing get_article_hr ---")
try:
    response_text_hr = get_article_hr(user_query, embed_fn, collection, client, user_lang)
    display(Markdown(response_text_hr))
except NameError as e:
    print(f"Error: {e}. Ensure all functions are defined correctly and 'prompt' variable is used properly inside them.")
    print("Specifically, check that 'prompt' is initialized within get_article and get_article_hr.")



