import glob
import re
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text()
    return full_text

def chunk_text_sliding_window(text, window_size=3, step_size=1):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i + window_size]).strip()
        if chunk:
            chunks.append(chunk)
        i += step_size
    return chunks

def load_docs(path):
    txt_files = glob.glob(f"{path}/*.txt")
    pdf_files = glob.glob(f"{path}/*.pdf")

    docs = []
    filenames = []
    raw_texts = []

    for fpath in txt_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            raw_texts.append(text)
            chunks = chunk_text_sliding_window(text)
            docs.extend(chunks)
            filenames.extend([fpath] * len(chunks))

    for fpath in pdf_files:
        try:
            text = extract_text_from_pdf(fpath).strip()
            raw_texts.append(text)
            chunks = chunk_text_sliding_window(text)
            docs.extend(chunks)
            filenames.extend([fpath] * len(chunks))
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    return docs, filenames, raw_texts

def get_relevant_chunks(docs, embeddings, query, model, keywords, minscore=0.3, topk=3):
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    indices = np.argsort(sims)[-topk*10:][::-1]
    results = []
    for i in indices:
        if sims[i] < minscore:
            continue
        chunk_lower = docs[i].lower()
        if any(re.search(r'\b' + re.escape(k) + r'\b', chunk_lower) for k in keywords):
            results.append((docs[i], sims[i]))
        if len(results) >= topk:
            break
    return results

def highlight_terms(text, terms):
    for term in terms:
        text = re.sub(r'(?i)(' + re.escape(term) + r')', r'\033[1;32m\1\033[0m', text)
    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--ask", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--minscore", type=float, default=0.3)
    args = parser.parse_args()

    docs, filenames, raw_texts = load_docs(args.docs)
    if not docs:
        print("No documents found!")
        exit(1)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)

    keywords = ["list", "tuple", "operator", "operators", "arithmetic", "comparison", "logical"]

    relevant_chunks = get_relevant_chunks(docs, embeddings, args.ask, model, keywords, args.minscore, args.topk)

    if not relevant_chunks:
        print("No relevant chunks found. Try lowering minscore or refining your query.")
    else:
        query_terms = args.ask.lower().split()
        for chunk, score in relevant_chunks:
            print(f"\nScore: {score:.3f}")
            print(highlight_terms(chunk, query_terms))


