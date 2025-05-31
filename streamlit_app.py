import streamlit as st 
import json
import faiss
import numpy as np
import time
import torch
import requests
import re
import os
import subprocess
from transformers import AutoTokenizer, AutoModel
import openai
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import httpx

st.set_page_config(page_title="Agentic RAG", layout="centered")

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ares_api_key = os.getenv("ARES_API_KEY")

# Qdrant setup for local file-based DB
qdrant_data_path = r"/Data/"

if 'qdrant_initialized' not in st.session_state:
    if not os.path.exists(qdrant_data_path):
        try:
            with st.spinner("Downloading Qdrant data..."):
                subprocess.run(
                    ["git", "clone", "https://github.com/hamzafarooq/multi-agent-course.git", qdrant_data_path],
                    check=True
                )
                st.success("\u2705 Qdrant data downloaded.")
        except Exception as e:
            st.error(f"Git clone failed: {e}")

    try:
        qdrant_collection_path = os.path.join(qdrant_data_path)
        qdrant = QdrantClient(path=qdrant_collection_path)

        st.success(f"\u2705 Qdrant collections loaded.")

        st.session_state.qdrant = qdrant
        st.session_state.qdrant_initialized = True

    except Exception as e:
        st.error(f"❌ Error initializing Qdrant client: {e}")

else:
    qdrant = st.session_state.qdrant

try:
    collections = qdrant.get_collections()
    st.write("✅ Available Qdrant Collections:", collections)
except Exception as e:
    st.error(f"❌ Failed to get Qdrant collections: {e}")

from qdrant_client.models import VectorParams, PointStruct

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Then define this:
from qdrant_client.models import VectorParams, PointStruct

def create_dummy_collection(name):
    st.warning(f"Creating dummy collection '{name}' for testing...")

    dim = embed_model.get_sentence_embedding_dimension()
    qdrant.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance="Cosine")
    )

    # Define sample text based on collection name
    if name == "10k_data":
        sample_texts = [
            "Apple competes with Google in mobile operating systems.",
            "The iPhone has a dominant market share in the U.S.",
            "Apple's services segment is growing rapidly."
        ]
    elif name == "openai_data":
        sample_texts = [
            "OpenAI released GPT-4, a powerful language model.",
            "ChatGPT is widely used for natural language tasks.",
            "OpenAI emphasizes safe and responsible AI use."
        ]
    else:
        sample_texts = ["Placeholder text."]

    points = []
    for i, text in enumerate(sample_texts):
        embedding = embed_model.encode(text).tolist()
        points.append(PointStruct(id=i, vector=embedding, payload={"text": text}))

    qdrant.upsert(collection_name=name, points=points)
    st.success(f"✅ Dummy collection '{name}' created with sample data.")

collections = qdrant.get_collections().collections
existing = [col.name for col in collections]

# Ensure both collections exist
for collection_name in ["10k_data", "openai_data"]:
    if collection_name not in existing:
        create_dummy_collection(collection_name)





# Map actions to actual Qdrant collection names
action_to_collection = {
    "OPENAI_QUERY": "openai_data",
    "10K_DOCUMENT_QUERY": "10k_data"
}

# Embedding model
#embed_model = SentenceTransformer("all-MiniLM-L6-v2")

import json
import os
import numpy as np

cache_file_path = "cache.json"
semantic_cache = []

# Load cache from file if it exists
if os.path.exists(cache_file_path):
    try:
        with open(cache_file_path, "r") as f:
            raw_cache = json.load(f)

        # Validate and process
        if isinstance(raw_cache, list):
            for item in raw_cache:
                if isinstance(item, dict) and "embedding" in item:
                    item["embedding"] = np.array(item["embedding"])  # Convert list back to array
                    semantic_cache.append(item)

        st.info("🧠 Cache loaded from cache.json")

    except Exception as e:
        st.error(f"Failed to load cache: {e}")
else:
    semantic_cache = []




# In-memory semantic cache
#semantic_cache = []  # List of dicts: {"query": ..., "embedding": ..., "answer": ...}

# Semantic cache check
def check_cache(query, threshold=0.9):
    query_emb = embed_model.encode([query])
    for item in semantic_cache:
        sim = cosine_similarity(query_emb, np.array(item["embedding"]).reshape(1, -1))[0][0]
        if sim > threshold:
            return item["answer"]
    st.write("🔎 Checking cache...")
    return None

# Save to cache
def update_cache(query, embedding, answer):
    cache_entry = {
        "query": query,
        "embedding": embedding.tolist(),  # convert ndarray to list
        "answer": answer
    }
    semantic_cache.append(cache_entry)

    try:
        # Convert entire cache for JSON (make sure embeddings are lists)
        json_ready_cache = []
        for item in semantic_cache:
            json_ready_cache.append({
                "query": item["query"],
                "embedding": item["embedding"].tolist() if isinstance(item["embedding"], np.ndarray) else item["embedding"],
                "answer": item["answer"]
            })

        with open(cache_file_path, "w") as f:
            json.dump(json_ready_cache, f)

        st.success("✅ Cache saved to cache.json")

    except Exception as e:
        st.error(f"❌ Failed to save cache: {e}")




# Retrieve documents from Qdrant
def retrieve_from_qdrant(query, action="10K_DOCUMENT_QUERY", top_k=5):
    try:
        vector = embed_model.encode(query).tolist()
        return qdrant.search(
            collection_name=action_to_collection.get(action),
            query_vector=vector,
            limit=top_k
        )
    except Exception as e:
        st.error(f"Qdrant search failed: {e}")
        return []


# Generate answer using OpenAI
def generate_answer_with_context(query, docs):
    context = "\n\n".join([doc.payload.get("text", "") for doc in docs])
    
    prompt = f"You are a financial assistant. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    from openai import OpenAI

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.2,
    #     max_tokens=500
    # )
    
    return response.choices[0].message.content

# Web search with ARES
def get_web_answer(query):
    if not ares_api_key:
        return "ARES API key missing."
    try:
        res = httpx.post(
            "https://api-ares.traversaal.ai/live/predict",
            headers={"x-api-key": ares_api_key, "content-type": "application/json"},
            json={"query": [query]},
            timeout=10.0
        )
        res.raise_for_status()
        data = res.json().get('data', {})
        return data.get('response_text', "No valid response received.")
    except Exception as e:
        return f"Web search error: {e}"

# Route query based on toggle or intent
def route_query(query, use_web):
    cached = check_cache(query)
    if cached:
        return cached + "\n(From cache)"

    if use_web:
        answer = get_web_answer(query)
    else:
        docs = retrieve_from_qdrant(query, action="10K_DOCUMENT_QUERY")
        answer = generate_answer_with_context(query, docs)
    embedding = embed_model.encode(query)
    update_cache(query, embedding, answer)
    #update_cache(query, embed_model.encode([query]), answer)
    return answer

# Streamlit UI
st.title("🔍 Agentic RAG Search with Web + Local + Cache")

query = st.text_input("Ask your question:", "What does Apple say about competition?")
use_web = st.toggle("Use Web Search (ARES)?")


if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Thinking..."):
            response = route_query(query, use_web)
        st.markdown("### 🧠 Answer")
        st.write(response)
        st.markdown("---")
        st.markdown("✅ **Search Mode:** " + ("Web Search" if use_web else "Local RAG"))
        st.markdown("📌 **Cached:** " + ("Yes" if "(From cache)" in response else "No"))
