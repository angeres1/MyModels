# test_context_retrieval_debug.py
import os
import re
import logging
import time
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the embedding model and vector database
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("incident_knowledge")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def retrieve_relevant_context(query):
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        logger.info(f"Full query results: {results}")  # Log full results for debugging
        if results.get('documents') and results['documents'][0]:
            return " ".join(results['documents'][0])
        else:
            logger.warning("No relevant Incident context found.")
            return "No relevant Incident context found."
    except Exception as e:
        logger.error(f"Error retrieving context for query '{query}': {e}")
        return "Error retrieving context."

def test_context_retrieval():
    # Insert known test content into the collection
    known_text = "This is a test incident report about a network outage in building A."
    known_id = "test-incident-1"
    
    # Create embedding for the known text
    embedding = embedding_model.encode([known_text], convert_to_tensor=True)
    embedding_list = [embedding[0].tolist()]
    
    # Insert the known text into the collection
    collection.add(ids=[known_id], documents=[known_text], embeddings=embedding_list)
    logger.info("Inserted known test text for context retrieval.")
    
    # Optional: Wait a moment to ensure the DB is updated
    time.sleep(1)
    
    # Test query and variations for debugging
    test_queries = [
        "What is Cherwell?",
        "Where can I log a ticket?",
        "How to escalate a P1 ticket?"
    ]
    
    for query in test_queries:
        # Calculate and log cosine similarity between known text and query
        known_embedding = embedding_model.encode([known_text], convert_to_tensor=False)[0]
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
        similarity = cosine_similarity(np.array(known_embedding), np.array(query_embedding))
        logger.info(f"Cosine similarity for query '{query}': {similarity}")
        
        # Retrieve context
        retrieved_context = retrieve_relevant_context(query)
        print("\nTest Query:")
        print(query)
        print("\nRetrieved Context:")
        print(retrieved_context)
        
        if known_text in retrieved_context:
            print("Test Passed: The known test text was successfully retrieved.")
        else:
            print("Test Failed: The known test text was not retrieved as expected.")

if __name__ == "__main__":
    test_context_retrieval()

test_query = "What is Cherwell?"
known_text = "Cherwell is the ITSM tool in McClatchy"

# Compute cosine similarity
known_embedding = embedding_model.encode([known_text], convert_to_tensor=False)[0]
query_embedding = embedding_model.encode([test_query], convert_to_tensor=False)[0]
similarity = cosine_similarity(np.array(known_embedding), np.array(query_embedding))
print(f"Cosine similarity for query '{test_query}': {similarity}")

# Test text relevant to escalation
relevant_text = "To escalate a P1 ticket, you should contact the incident management team immediately and follow the escalation protocol as outlined in our incident management guidelines."
relevant_id = "test-escalation-1"

# Create embedding and add to collection
relevant_embedding = embedding_model.encode([relevant_text], convert_to_tensor=True)
collection.add(ids=[relevant_id], documents=[relevant_text], embeddings=[relevant_embedding[0].tolist()])

# Now test the retrieval
test_query = "How to escalate a P1 ticket?"
retrieved_context = retrieve_relevant_context(test_query)
print("Retrieved Context:")
print(retrieved_context)

# Optionally, check cosine similarity
known_embedding = embedding_model.encode([relevant_text], convert_to_tensor=False)[0]
query_embedding = embedding_model.encode([test_query], convert_to_tensor=False)[0]
similarity = cosine_similarity(np.array(known_embedding), np.array(query_embedding))
print(f"Cosine similarity for query '{test_query}': {similarity}")


import os
import re
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import ollama
import chromadb
from fastapi import FastAPI, Query
from pypdf import PdfReader
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from sentence_transformers import SentenceTransformer

# --------------------------
# Configuration and Logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_FOLDER = "/Users/aotalora/Downloads/MyModels/PDFs"
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
EXPECTED_EMBEDDING_DIM = 768  # Model output dimension
COLLECTION_NAME = "incident_knowledge"

# --------------------------
# Initialize FastAPI and Services
# --------------------------
app = FastAPI()
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_embedding_dim(sample_embedding) -> int:
    return len(sample_embedding) if isinstance(sample_embedding, list) else None

# Generate a sample embedding to determine dimension
sample_embedding = embedding_model.encode("sample text").tolist()
current_embedding_dim = get_embedding_dim(sample_embedding)
logger.info(f"Current embedding dimension from model: {current_embedding_dim}")

# --------------------------
# Collection Setup with Graceful Handling
# --------------------------
try:
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    metadata = collection.get_metadata() if hasattr(collection, "get_metadata") else {}
    if metadata and "embedding_dim" in metadata:
        if metadata["embedding_dim"] != EXPECTED_EMBEDDING_DIM:
            logger.warning(
                f"Existing collection embedding dimension ({metadata['embedding_dim']}) "
                f"does not match expected ({EXPECTED_EMBEDDING_DIM}). Recreating collection..."
            )
            chroma_client.delete_collection(COLLECTION_NAME)
            try:
                collection = chroma_client.create_collection(
                    COLLECTION_NAME, metadata={"embedding_dim": EXPECTED_EMBEDDING_DIM}
                )
            except chromadb.errors.UniqueConstraintError:
                logger.info("Collection already exists after deletion, retrieving existing collection.")
                collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    else:
        try:
            collection = chroma_client.create_collection(
                COLLECTION_NAME, metadata={"embedding_dim": EXPECTED_EMBEDDING_DIM}
            )
        except chromadb.errors.UniqueConstraintError:
            logger.info("Collection already exists, retrieving existing collection.")
            collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
except Exception as e:
    logger.error(f"Error setting up collection: {e}")
    raise

# --------------------------
# Thread Pool for Offloading Tasks
# --------------------------
executor = ThreadPoolExecutor(max_workers=4)

# --------------------------
# PDF Processing Functions
# --------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF file.
    """
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def process_pdf(pdf_path: str) -> None:
    """
    Processes a PDF file: extracts text, splits it into chunks, generates embeddings,
    and stores them in the vector DB.
    """
    logger.info(f"Processing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.warning(f"No text extracted from {pdf_path}. Skipping.")
        return

    # Split text into coherent chunks (paragraphs)
    text_chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    if not text_chunks:
        logger.warning(f"No valid text chunks found in {pdf_path}")
        return

    try:
        embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True)
        ids = [f"{os.path.basename(pdf_path)}-{i}" for i in range(len(text_chunks))]
        embedding_list = [emb.tolist() for emb in embeddings]
        collection.add(ids=ids, documents=text_chunks, embeddings=embedding_list)
        logger.info(f"Stored {len(text_chunks)} chunks from {pdf_path} in vector DB.")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")

# --------------------------
# Watchdog PDF Folder Monitoring
# --------------------------
class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            logger.info(f"Detected new PDF: {event.src_path}")
            executor.submit(process_pdf, event.src_path)

def watch_pdf_folder(folder_path: str) -> Observer: # type: ignore
    """
    Monitors a folder for new PDFs and processes them.
    """
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    logger.info(f"Watching folder: {folder_path} for new PDFs...")
    return observer

# --------------------------
# Query and Chat Functions
# --------------------------
def retrieve_relevant_context(query: str) -> str:
    """
    Retrieves relevant context from the vector DB based on the query.
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        if results.get("documents"):
            return " ".join(results["documents"][0])
        else:
            return "No relevant incident context found."
    except Exception as e:
        logger.error(f"Error retrieving context for query '{query}': {e}")
        return "Error retrieving context."

def chat_with_ollama(user_query: str) -> str:
    """
    Combines the retrieved context with the user query and sends it to ollama for a response.
    """
    context = retrieve_relevant_context(user_query)
    full_query = f"Context: {context}\n\nUser Query: {user_query}"
    try:
        response = ollama.chat(
            model="phi4",
            messages=[{"role": "user", "content": full_query}]
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error during chat_with_ollama: {e}")
        return "Error generating response."

@app.get("/chat")
async def chat(query: str = Query(..., description="User's incident-related question")):
    """
    FastAPI endpoint for processing chat queries asynchronously.
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, chat_with_ollama, query)
    return {"response": response}

# --------------------------
# Main Application Runner
# --------------------------
if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    observer = watch_pdf_folder(PDF_FOLDER)
    
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()