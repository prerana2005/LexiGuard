from groq import Groq
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_embedder = None
_faiss_index = None
_metadata = None
_bm25 = None


def load_resources():
    global _embedder, _faiss_index, _metadata, _bm25
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _faiss_index = faiss.read_index("faiss_index/indian_laws_v3.index")
        with open("embeddings/indian_laws_metadata_v3.json", "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        corpus = [chunk.get("text", "") for chunk in _metadata]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        _bm25 = BM25Okapi(tokenized_corpus)


def is_retrieval_relevant(query, retrieved_laws):
    if not retrieved_laws:
        return False
    combined = " ".join([l['text'] for l in retrieved_laws]).lower()
    query_words = set(query.lower().split())
    law_words = set(combined.split())
    overlap = len(query_words & law_words)
    return overlap > 3


def hybrid_search(query, top_k=2):
    load_resources()

    # Dense — FAISS
    query_embedding = _embedder.encode(
        [query], normalize_embeddings=True).astype(np.float32)
    scores, indices = _faiss_index.search(query_embedding, top_k * 10)

    dense_results = {}
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:
            dense_results[int(idx)] = 1 / (rank + 1)

    # Sparse — BM25
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
            "text": chunk.get("text", "")[:200]
        })
        if len(results) == top_k:
            break

    return results


def retrieve_laws(query, top_k=2):
    results = hybrid_search(query, top_k)
    if not is_retrieval_relevant(query, results):
        broader_query = query + " Indian contract law risk penalty obligation"
        results = hybrid_search(broader_query, top_k)
    return results

RISK_PROMPT = """You are a senior Indian contract lawyer.

Relevant Indian Laws:
{law_context}

Analyze this contract clause based on the Indian laws above and return ONLY this JSON:
{{
    "risk_level": "High" or "Medium" or "Low",
    "risk_type": "short name e.g. Termination Without Notice",
    "explanation": "1-2 sentences why this is risky under Indian law",
    "impact": "real-world impact on the person signing",
    "applicable_law": "which Indian law applies here if any"
}}

Rules:
- High Risk: Only for genuinely one-sided or dangerous clauses
- Medium Risk: Clauses with some concern but not immediately dangerous
- Low Risk: Standard fair clauses

Always High Risk:
- Termination without notice or with less than 14 days notice
- Salary or payment reduction at employer's sole discretion
- Non-compete clauses post-employment (void under Section 27 Indian Contract Act)
- IP assignment including work done outside working hours
- Payment conditional on subjective evaluation with no defined criteria
- Perpetual confidentiality obligations (forever)
- Liquidated damages disproportionate to actual loss
- Asymmetric notice periods heavily favouring one party

Do NOT flag as High or Medium Risk:
- Introduction or preamble clauses
- Entire agreement or integration clauses
- Governing law and jurisdiction clauses
- Standard IP assignment where full payment triggers transfer
- Standard arbitration under Arbitration Act 1996
- Mutual termination with 30 days or more notice

Return ONLY JSON. No text outside JSON.

Clause: {clause_text}"""
def analyze_risk(clause_text):
    relevant_laws = retrieve_laws(clause_text)
    law_context = ""
    for i, law in enumerate(relevant_laws, 1):
        law_context += f"{i}. {law['law']} — {law['section']}\n{law['text']}\n\n"

    if not law_context:
        law_context = "No specific Indian law retrieved — use general Indian contract law principles."

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": RISK_PROMPT.format(
                clause_text=clause_text,
                law_context=law_context
            )
        }],
        temperature=0.1,
        max_tokens=300
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return {
            "risk_level": "Medium",
            "risk_type": "Unknown",
            "explanation": raw,
            "impact": "Unable to parse",
            "applicable_law": "Unknown"
        }


if __name__ == "__main__":
    test = "The employee shall be paid Rs. 10,000 if work done is satisfactory as determined by management."
    result = analyze_risk(test)
    print(json.dumps(result, indent=2))