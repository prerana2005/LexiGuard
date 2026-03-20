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
        _faiss_index = faiss.read_index("faiss_index/indian_laws.index")
        with open("faiss_index/indian_laws_metadata.json", "r", encoding="utf-8") as f:
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
        broader_query = query + " Indian contract law clarity definite terms"
        results = hybrid_search(broader_query, top_k)
    return results


VAGUE_WORDS = [
    "satisfactory", "reasonable", "adequate",
    "at discretion", "as needed", "as soon as possible",
    "reasonable time", "good condition", "tenable condition",
    "minor repairs", "normal wear and tear", "well performed",
    "if work done is well", "subject to approval"
]

AMBIGUITY_PROMPT = """You are a legal analyst specializing in Indian contract law.

Relevant Indian Laws that require clear contract terms:
{law_context}

Analyze this clause for vague undefined terms that could realistically be EXPLOITED by one party against the other.

Only flag as ambiguous if there is a genuine exploitable vague term — not standard legal language.
Do NOT flag: party names, standard definitions, boilerplate introductions.
DO flag: subjective performance standards, undefined timeframes, discretionary decisions.

Return ONLY this JSON:
{{
    "is_ambiguous": true or false,
    "vague_phrases": ["only", "genuinely", "exploitable", "terms"],
    "how_exploited": "specific realistic way this could be misused",
    "suggested_fix": "exact replacement with specific measurable terms",
    "indian_law_requirement": "what Indian law says about clarity in contracts"
}}

Return ONLY JSON. No text outside JSON.

Clause: {clause_text}"""

def analyze_ambiguity(clause_text):
    found_vague = [w for w in VAGUE_WORDS if w.lower() in clause_text.lower()]

    relevant_laws = retrieve_laws(clause_text + " vague terms contract clarity")
    law_context = ""
    for i, law in enumerate(relevant_laws, 1):
        law_context += f"{i}. {law['law']} — {law['section']}\n{law['text']}\n\n"

    if not law_context:
        law_context = "Indian Contract Act 1872 requires contracts to have clear and definite terms."

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": AMBIGUITY_PROMPT.format(
                clause_text=clause_text,
                law_context=law_context
            )
        }],
        temperature=0.1,
        max_tokens=400
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        if found_vague and not result.get("vague_phrases"):
            result["vague_phrases"] = found_vague
            result["is_ambiguous"] = True
        return result
    except:
        return {
            "is_ambiguous": len(found_vague) > 0,
            "vague_phrases": found_vague,
            "how_exploited": "Unable to parse",
            "suggested_fix": "Specify exact measurable terms",
            "indian_law_requirement": "Indian Contract Act requires clear terms"
        }


if __name__ == "__main__":
    test = "Payment of Rs. 10,000 will be given if work done is satisfactory as determined by management."
    result = analyze_ambiguity(test)
    print(json.dumps(result, indent=2))