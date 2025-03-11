from flask import Flask, request, jsonify
import faiss
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

app = Flask(__name__)

# Load local embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load local LLM model (LLaMA or Mistral)
LLM_MODEL_PATH = "mistral-7b.gguf"  # Change to "llama-2-7b.gguf" if using LLaMA 2
llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=4096, n_threads=8)

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Embed text using local model
def embed_text(chunks):
    return embedding_model.encode(chunks, convert_to_numpy=True)

# Store embeddings in FAISS
def store_in_faiss(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

# Search FAISS for relevant text
def search_faiss(query, index, text_chunks, k=3):
    query_vector = embedding_model.encode([query])
    D, I = index.search(query_vector, k)
    return [text_chunks[i] for i in I[0]]

# Generate response using local LLM
def generate_response(context, user_query):
    prompt = f"""
    Based on the following document content, answer the user's question.

    Context:
    {context}

    Question:
    {user_query}

    Answer:
    """
    response = llm(prompt, max_tokens=250, temperature=0.7)
    return response["choices"][0]["text"]

# Load PDF, create vector database
pdf_path = "McC.pdf"  # Replace with the actual PDF file
pdf_text = extract_text_from_pdf(pdf_path)
text_chunks = split_text(pdf_text)
vector_store = embed_text(text_chunks)
faiss_index = store_in_faiss(vector_store)

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")
    relevant_chunks = search_faiss(user_query, faiss_index, text_chunks)
    
    # Combine top results into context
    context = "\n\n".join(relevant_chunks)
    
    # Get LLM-generated response
    llm_response = generate_response(context, user_query)
    
    return jsonify({"response": llm_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)