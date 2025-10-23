Groq-Powered Retrieval Augmented Generation (RAG) System
Overview
This project implements a Retrieval Augmented Generation system that retrieves relevant context from PDF and TXT document collections and answers user queries using Groq's OpenAI-compatible language models. It leverages sentence embeddings for semantic search and Groq's high-performance inference API for generating natural language responses.

Features
Extracts and chunks text from PDF and TXT files.

Uses sentence-transformers to generate embeddings for efficient semantic search.

Retrieves the most relevant chunks based on cosine similarity and keyword matching.

Calls Groq's LLM API for answer generation with the retrieved context.

Supports flexible parameters such as top-k retrieval, minimum similarity score for chunks.

Highlights queried terms in retrieved text for better readability.

Prerequisites
Python 3.8 or higher

Groq API key for access to Groq's LLM inference API

Installation
Clone the repo:

bash
git clone https://github.com/yourusername/groq-rag.git
cd groq-rag
Create and activate a virtual environment:

bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows
Install dependencies:

bash
pip install -r requirements.txt
Export your Groq API key:

bash
export GROQ_API_KEY="your_actual_groq_api_key"
Usage
Run the main script with your documents folder and query:

bash
python rag_demo.py --docs /path/to/docs --ask "your question here" --topk 3 --minscore 0.3
--docs: Folder containing .txt and .pdf files.

--ask: User query to answer.

--topk: Number of top relevant chunks to retrieve (default: 3).

--minscore: Minimum cosine similarity threshold to filter chunks (default: 0.3).

How It Works
Document Loading: Loads all text and PDF files from specified folder. Extracts text and chunks it with a sliding window approach.

Embedding Generation: Uses the all-MiniLM-L6-v2 SentenceTransformer model to create vector embeddings for all chunks.

Semantic Search: For a given query, generates query embedding and calculates cosine similarity against all chunk embeddings.

Keyword Filtering: Filters chunks based on the presence of relevant keywords.

Answer Generation: Aggregates top relevant chunks as context, sends prompt to Groq's chat completion API for answer synthesis.

Result Display: Prints out highlighted chunks and Groq-generated answer.

Environment Variables
GROQ_API_KEY: Your API key for Groq’s OpenAI-compatible API.

Dependencies
See requirements.txt for versions. Includes:

openai

sentence-transformers

numpy

pymupdf

scikit-learn

