import json
import torch
import numpy as np
import faiss
import os
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

CLASSIFIER_REPO = "Caraxes22/Lexiguard-LegalBert"

GOOD_SOURCES = {
    "mratanusarkar/Indian-Laws",
    "ShreyasP123/Legal-Dataset-for-india",
    "nisaar/Lawyer_GPT_India",
    "manual"
}
BAD_LAW_NAMES = {"Indian_Law", "Indian_Law_QA", "Indian_Legal"}

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_models = {}


def load_models():
    global _models
    if _models:
        return _models

    hf_token = os.environ.get("HF_TOKEN")

    print("Loading Legal-BERT classifier...")
    clf_tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_REPO, token=hf_token)
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_REPO, token=hf_token)
    clf_model.eval()

    print("Loading label map...")
    label_map_path = hf_hub_download(
        repo_id=CLASSIFIER_REPO,
        filename="label_map.json",
        repo_type="model",
        token=hf_token
    )
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    print("Loading FAISS index...")
    faiss_index = faiss.read_index("faiss_index/indian_laws.index")

    print("Loading metadata...")
    with open("faiss_index/indian_laws_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Loading MiniLM embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Building BM25 index...")
    corpus = [chunk.get("text", "") for chunk in metadata]
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    _models = {
        "clf_tokenizer": clf_tokenizer,
        "clf_model": clf_model,
        "id2label": id2label,
        "faiss_index": faiss_index,
        "metadata": metadata,
        "embedder": embedder,
        "bm25": bm25
    }
    print("All models loaded!")
    return _models


def classify_clause(clause_text, clf_tokenizer, clf_model, id2label):
    inputs = clf_tokenizer(
        clause_text, truncation=True, padding=True,
        max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = clf_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    top_idx = torch.argsort(probs, descending=True)[0]
    clause_type = id2label[top_idx.item()]
    confidence = round(probs[top_idx].item(), 4)
    return clause_type, confidence


def is_retrieval_relevant(query, retrieved_laws):
    if not retrieved_laws:
        return False
    combined = " ".join([l['text'] for l in retrieved_laws]).lower()
    query_words = set(query.lower().split())
    law_words = set(combined.split())
    overlap = len(query_words & law_words)
    return overlap > 3


def hybrid_search(query, faiss_index, metadata, embedder, bm25, top_k=3):
    # Dense — FAISS
    query_embedding = embedder.encode(
        [query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_index.search(query_embedding, top_k * 10)

    dense_results = {}
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:
            dense_results[int(idx)] = 1 / (rank + 1)

    # Sparse — BM25
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
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
        chunk = metadata[idx]
        if chunk.get("source", "") not in GOOD_SOURCES:
            continue
        if chunk.get("law_name", "") in BAD_LAW_NAMES:
            continue
        if len(chunk.get("text", "").split()) < 15:
            continue
        results.append({
            "law": chunk.get("law_name", ""),
            "section": chunk.get("section_number", ""),
            "text": chunk.get("text", "")[:300]
        })
        if len(results) == top_k:
            break

    # Fallback if nothing passed filters
    if not results:
        for idx in top_indices[:top_k]:
            chunk = metadata[idx]
            results.append({
                "law": chunk.get("law_name", "Unknown"),
                "section": chunk.get("section_number", ""),
                "text": chunk.get("text", "")[:300]
            })

    return results


def retrieve_laws(clause_text, clause_type, faiss_index, metadata, embedder, bm25, top_k=3):
    query = f"{clause_type}: {clause_text}"
    results = hybrid_search(query, faiss_index, metadata, embedder, bm25, top_k)

    # CRAG — if not relevant retry with broader query
    if not is_retrieval_relevant(clause_text, results):
        broader_query = clause_text + " Indian law rights obligations explanation"
        results = hybrid_search(broader_query, faiss_index, metadata, embedder, bm25, top_k)

    return results


def generate_explanation(clause_text, clause_type, retrieved_laws):
    law_context = ""
    for i, law in enumerate(retrieved_laws, 1):
        law_context += f"{i}. {law['law']} — {law['section']}\n{law['text']}\n\n"

    prompt = f"""You are a legal assistant explaining Indian contracts to a non-lawyer.

Clause Type: {clause_type}

Clause Text:
{clause_text}

Relevant Indian Laws:
{law_context}

Explain this clause clearly. Cover:
1. What this clause means in plain simple English
2. What the person signing has to do or give up
3. Which Indian law applies and why
4. Any risks to watch out for

Be concise. No legal jargon. Write like you are explaining to a college student."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


def explain_clause(clause_text):
    models = load_models()

    clause_type, confidence = classify_clause(
        clause_text,
        models["clf_tokenizer"],
        models["clf_model"],
        models["id2label"]
    )

    retrieved_laws = retrieve_laws(
        clause_text,
        clause_type,
        models["faiss_index"],
        models["metadata"],
        models["embedder"],
        models["bm25"]
    )

    explanation = generate_explanation(clause_text, clause_type, retrieved_laws)

    return {
        "clause_type": clause_type,
        "confidence": confidence,
        "relevant_laws": retrieved_laws,
        "explanation": explanation
    }


if __name__ == "__main__":
    test = "The employee shall be paid Rs. 10,000 if work done is satisfactory as determined by management."
    print("Testing explanation agent...")
    result = explain_clause(test)
    print(f"\nClause Type: {result['clause_type']} ({result['confidence']:.1%} confidence)")
    print(f"\nRelevant Laws:")
    for law in result['relevant_laws']:
        print(f"  - {law['law']} — {law['section']}")
    print(f"\nExplanation:\n{result['explanation']}")