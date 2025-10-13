# Lightweight Retrieval-Augmented Generation (RAG) Demo

This is a simple RAG-style project implemented in Python to retrieve relevant text chunks from `.txt` documents based on user queries.

## Features

- Loads local text documents from a folder
- Splits documents into paragraph chunks
- Builds a vocabulary and vectorizes chunks using bag-of-words
- Computes cosine similarity to find most relevant chunks to query
- Outputs top-k retrieved chunks with source info and similarity scores
- No heavy dependencies; uses only `numpy` and built-in libraries

## Setup

1. Clone the repo or download the code.

2. (Optional but recommended) Create and activate a virtual environment:

