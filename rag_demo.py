import os
import glob
import argparse
import numpy as np
import re
from collections import Counter

def tokenize(text):
    # Simple tokenizer: lower, split by non-alphabetic
    return re.findall(r'\b[a-z]+\b', text.lower())

def build_vocab(docs):
    vocab = set()
    for doc in docs:
        vocab.update(tokenize(doc))
    return sorted(vocab)

def vectorize(doc, vocab):
    words = tokenize(doc)
    counter = Counter(words)
    vec = np.zeros(len(vocab), dtype=float)
    for i, word in enumerate(vocab):
        vec[i] = counter[word]
    # normalize vector
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_docs(doc_dir):
    files = sorted(glob.glob(os.path.join(doc_dir, "*.txt")))
    docs = []
    filenames = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
            docs.extend(chunks)
            filenames.extend([os.path.basename(fpath)] * len(chunks))
    return docs, filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="./data")
    parser.add_argument("--ask", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    docs, filenames = load_docs(args.docs)
    if not docs:
        print("No documents found!")
        exit(1)

    vocab = build_vocab(docs)
    doc_vecs = np.array([vectorize(doc, vocab) for doc in docs])

    query_vec = vectorize(args.ask, vocab)
    sims = np.array([cosine_similarity(query_vec, dvec) for dvec in doc_vecs])
    top_k_idx = sims.argsort()[-args.topk:][::-1]

    print("\nTop relevant chunks:\n")
    for i in top_k_idx:
        print(f"[{i+1}] Source: {filenames[i]}, Score: {sims[i]:.4f}")
        print(docs[i])
        print()
