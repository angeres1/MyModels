import os
import re
import logging
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import ollama
import chromadb
from fastapi import FastAPI, Query, HTTPException, status
from pypdf import PdfReader
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from sentence_transformers import SentenceTransformer
import uvicorn
import threading

# --------------------------
# Configuration and Logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_FOLDER = os.getenv("PDF_FOLDER", "/Users/aotalora/Documents/Projects/MyModels/PDFs")  # Use environment variable, fallback to default
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "multi-qa-mpnet-base-dot-v1")
EXPECTED_EMBEDDING_DIM = 768  # Model output dimension
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "incident_knowledge")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b-base")  # Corrected model name
MAX_RETRIES = 3  # For connection retries
RETRY_DELAY = 2  # Seconds
MAX_TEXT_LENGTH = 2048  # Added a max text length to avoid issues with the model.

# --------------------------
# Initialize FastAPI and Services
# --------------------------
app = FastAPI()

# Initialize embedding model with retry logic
for attempt in range(MAX_RETRIES):
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        break  # Success, exit the retry loop
    except Exception as e:
        logger.error(f"Attempt {attempt + 1} failed to load embedding model: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
        else:
            logger.error("Max retries reached.  Exiting.")
            raise

def get_embedding_dim(model: SentenceTransformer) -> int:
    """Gets the embedding dimension from the model."""
    sample_embedding = model.encode("sample text")
    return sample_embedding.shape[0]  # More robust dimension check

current_embedding_dim = get_embedding_dim(embedding_model)
logger.info(f"Current embedding dimension from model: {current_embedding_dim}")
if current_embedding_dim != EXPECTED_EMBEDDING_DIM:
    logger.warning(f"Model embedding dimension ({current_embedding_dim}) does not match expected ({EXPECTED_EMBEDDING_DIM}).")

# Initialize ChromaDB client with retry logic
for attempt in range(MAX_RETRIES):
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        break
    except Exception as e:
        logger.error(f"Attempt {attempt+1} failed to connect to ChromaDB: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
        else:
            logger.error("Max retries reached. Exiting.")
            raise

# --------------------------
# Collection Setup with Comprehensive Handling
# --------------------------
def setup_collection(client: chromadb.PersistentClient, collection_name: str, expected_dim: int) -> chromadb.Collection:
    """Sets up the ChromaDB collection, handling existing collections and dimension mismatches."""
    try:
        collection = client.get_or_create_collection(collection_name)
        metadata = collection.get()  # Get *all* collection data

        # Check for metadata and nested 'metadata' key safely
        if metadata and metadata.get('metadata') and "embedding_dim" in metadata['metadata']:
            if metadata['metadata']["embedding_dim"] != expected_dim:
                logger.warning(
                    f"Existing collection embedding dimension ({metadata['metadata']['embedding_dim']}) "
                    f"does not match expected ({expected_dim}). Recreating collection..."
                )
                client.delete_collection(collection_name)
                collection = client.create_collection(
                    collection_name, metadata={"embedding_dim": expected_dim}
                )
            else:
                logger.info(f"Collection '{collection_name}' exists and has correct dimensions.")
        else:  # No metadata, no 'metadata' key, or no 'embedding_dim'
            logger.info(f"Collection '{collection_name}' metadata needs updating or does not exist. Creating/Recreating...")
            client.delete_collection(collection_name)  # Simplest way to update.
            collection = client.create_collection(collection_name, metadata={"embedding_dim": expected_dim})

    except Exception as e:
        logger.exception(f"Error setting up collection: {e}")  # Use exception for full traceback
        raise
    return collection

collection = setup_collection(chroma_client, COLLECTION_NAME, EXPECTED_EMBEDDING_DIM)



# --------------------------
# Thread Pool for Offloading Tasks
# --------------------------
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers as needed

# --------------------------
# PDF Processing Functions
# --------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF file, handling common errors.
    """
    try:
        with open(pdf_path, "rb") as file:  # Open in binary mode
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: # Check if page_text is not None or empty
                  text += page_text + "\n"
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def truncate_text(text: str, max_length: int) -> str:
    """Truncates text to a maximum length, ensuring whole words."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def process_pdf(pdf_path: str) -> None:
    """
    Processes a PDF: extracts text, chunks it, generates embeddings, and stores in ChromaDB.
    """
    logger.info(f"Processing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.warning(f"No text extracted from {pdf_path}. Skipping.")
        return

    # Split text into paragraphs, handling various newline/whitespace scenarios
    text_chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]
    if not text_chunks:
        logger.warning(f"No valid text chunks (paragraphs) found in {pdf_path}")
        return

    # Truncate very long chunks *before* embedding
    truncated_chunks = [truncate_text(chunk, MAX_TEXT_LENGTH) for chunk in text_chunks]

    try:
        embeddings = embedding_model.encode(truncated_chunks, convert_to_tensor=False) # No need for PyTorch tensors here
        embedding_list = embeddings.tolist()

        # Generate unique IDs *per chunk*, not per run.  This is important for updates.
        ids_to_manage = [f"{os.path.basename(pdf_path)}-{i}-{uuid.uuid4()}" for i in range(len(truncated_chunks))]

        # Add to ChromaDB.  Consider using upsert if your ChromaDB version supports it.
        collection.add(ids=ids_to_manage, documents=truncated_chunks, embeddings=embedding_list)
        logger.info(f"Stored {len(truncated_chunks)} chunks from {pdf_path} in vector DB.")
    except Exception as e:
        logger.error(f"Error processing or storing embeddings for {pdf_path}: {e}")


# --------------------------
# Watchdog PDF Folder Monitoring
# --------------------------
processed_files = set()  # Tracks processed files to prevent reprocessing

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event: FileCreatedEvent):
        if not event.is_directory and event.src_path.endswith(".pdf"):
            if event.src_path in processed_files:
                logger.info(f"PDF already processed in this session: {event.src_path}")
                return
            processed_files.add(event.src_path)
            logger.info(f"Detected new PDF: {event.src_path}")
            executor.submit(process_pdf, event.src_path)

def watch_pdf_folder(folder_path: str) -> Observer:
    """
    Starts the Watchdog observer to monitor the specified folder.
    """
    os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    logger.info(f"Watching folder: {folder_path} for new PDFs...")
    return observer

# --------------------------
# Query and Chat Functions
# --------------------------
def retrieve_relevant_context(query: str, n_results: int = 5) -> str:
    """
    Retrieves relevant context from ChromaDB using the embedding model.
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

        if results and results.get("documents"):
            #  Join the documents, handling potential nested list structure.
            return " ".join([doc for sublist in results["documents"] for doc in sublist])
        else:
            return "No relevant incident context found."
    except Exception as e:
        logger.error(f"Error retrieving context for query '{query}': {e}")
        return "Error retrieving context."

def chat_with_ollama(user_query: str, context: str) -> str:
    """
    Interacts with the Ollama model, providing context and the user's query.
    """
    full_query = f"Context: {context}\n\nUser Query: {user_query}"
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": full_query}],
            stream=False,  # Disable streaming for simplicity
            options={'temperature': 0.2, 'top_p': 0.9}  # Example options
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Error during chat_with_ollama: {e}")
        return "Error generating response."

@app.get("/chat")
async def chat(query: str = Query(..., description="User's incident-related question")):
    """
    FastAPI endpoint for handling chat queries, now with proper error handling.
    """
    try:
        context = retrieve_relevant_context(query)
        if "Error" in context:  # Check for error messages from retrieval
            raise HTTPException(status_code=500, detail=context)

        loop = asyncio.get_running_loop()  # Use get_running_loop
        response = await loop.run_in_executor(executor, chat_with_ollama, query, context)
        return {"response": response}

    except HTTPException as e:  # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --------------------------
# Debug and Test Endpoints
# --------------------------

@app.get("/test")
def test():
    return {"message": "Test endpoint working"}

@app.get("/debug/collection")
def debug_collection():
    try:
        data = collection.get()
        return {"data": data}
    except Exception as e:
        logger.error(f"Error fetching collection data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/pdf_folder")
def debug_pdf_folder():
    return {"pdf_folder": PDF_FOLDER}

@app.get("/debug/processed_files")
def debug_processed_files():
     return {"processed_files": list(processed_files)}

@app.get("/debug_context")
def debug_context(query: str = Query(..., description="Test query for vector DB retrieval")):
    context = retrieve_relevant_context(query)
    return {"query": query, "retrieved_context": context}

# --------------------------
# Main Application Runner
# --------------------------
def run_api():
    """Runs the FastAPI server using uvicorn."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    observer = watch_pdf_folder(PDF_FOLDER)

    # Run FastAPI in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("FastAPI server is running in the background...")

    try:
        while True:
            time.sleep(1)  # Keep main thread alive for Watchdog
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        observer.stop()
        observer.join()