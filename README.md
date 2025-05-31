Here’s what I implemented:

🔍 User Interface:

Streamlit-based query box with an optional “Allow Web Search” toggle

Displays answers, supporting docs, and cache/log info

🧠 RAG Pipeline:

Indexed 10-K document embeddings using FAISS/Qdrant

Used LLMs to generate answers from top retrieved chunks

🤖 Agentic Routing:

Smartly routes queries to local docs or web search based on intent

Simulated web search with tools like DuckDuckGo/ARES

Logs every routing decision for transparency

⚡ Semantic Caching:

Stores question embeddings + responses

Returns cached answers for similar queries to speed up response
