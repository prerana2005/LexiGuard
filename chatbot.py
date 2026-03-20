import os
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import json
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_embedder = None
_faiss_index = None
_metadata = None
_bm25 = None

def load_resources():
    global _embedder, _faiss_index, _metadata, _bm25
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _faiss_index = faiss.read_index("faiss_index/indian_laws.index")
        with open("faiss_index/indian_laws_metadata.json", "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        # Build BM25 index once
        corpus = [chunk.get("text", "") for chunk in _metadata]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        _bm25 = BM25Okapi(tokenized_corpus)


def is_retrieval_relevant(clause_text, retrieved_laws):
    """CRAG check — verify retrieved laws are actually relevant"""
    if not retrieved_laws:
        return False
    combined_law_text = " ".join([l['text'] for l in retrieved_laws]).lower()
    query_words = set(clause_text.lower().split())
    law_words = set(combined_law_text.split())
    overlap = len(query_words & law_words)
    return overlap > 3


def hybrid_search(query, top_k=3):
    """BM25 + FAISS with RRF fusion"""
    load_resources()

    # Dense search — FAISS
    query_embedding = _embedder.encode(
        [query], normalize_embeddings=True).astype(np.float32)
    scores, indices = _faiss_index.search(query_embedding, top_k * 10)

    dense_results = {}
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:
            dense_results[int(idx)] = 1 / (rank + 1)

    # Sparse search — BM25
    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    top_bm25 = bm25_scores.argsort()[-top_k * 10:][::-1]

    sparse_results = {}
    for rank, idx in enumerate(top_bm25):
        sparse_results[int(idx)] = 1 / (rank + 1)

    # RRF Fusion
    all_indices = set(dense_results.keys()) | set(sparse_results.keys())
    fused = {}
    for idx in all_indices:
        fused[idx] = dense_results.get(idx, 0) + sparse_results.get(idx, 0)

    top_indices = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)

    results = []
    for idx in top_indices:
        chunk = _metadata[idx]
        if len(chunk.get("text", "").split()) < 15:
            continue
        results.append({
            "law": chunk.get("law_name", ""),
            "section": chunk.get("section_number", ""),
            "text": chunk.get("text", "")[:300]
        })
        if len(results) == top_k:
            break

    return results


def retrieve_relevant_laws(query, top_k=3):
    """Hybrid search with CRAG fallback"""
    results = hybrid_search(query, top_k)

    # CRAG check — if not relevant try broader query
    if not is_retrieval_relevant(query, results):
        broader_query = query + " Indian contract law rights obligations"
        results = hybrid_search(broader_query, top_k)

    return results


def answer_query(user_query, contract_text):
    load_resources()

    relevant_laws = retrieve_relevant_laws(user_query)

    law_context = ""
    for i, law in enumerate(relevant_laws, 1):
        law_context += f"{i}. {law['law']} — {law['section']}\n{law['text']}\n\n"

    contract_excerpt = contract_text[:3000]

    prompt = f"""You are LexiGuard, an AI legal assistant helping an Indian user understand their contract.

CONTRACT TEXT (excerpt):
{contract_excerpt}

RELEVANT INDIAN LAWS:
{law_context}

USER QUESTION: {user_query}

Answer the question based on the contract text and Indian laws provided.
- Be specific — quote relevant parts of the contract if needed
- Mention which Indian law applies if relevant
- Keep answer concise and in plain English
- If the contract doesn't address the question, say so clearly
- Do not give generic answers — base everything on the actual contract"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()