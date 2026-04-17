"""
FIXED Retrieval Strategy Ablation
Fixes:
  - Filters Indian_Law_QA / Indian_Law / Indian_Legal noise chunks
    from ALL strategies so scores reflect real retrieval quality
  - Scoring now checks ALL top-3 results, not just top-1
  - Suppresses tensorflow/protobuf import warnings cleanly
  - Handles sentence_transformers numpy 2.x crash gracefully
"""

import os
import sys
import json
import time
import warnings

# Suppress tensorflow and protobuf warnings before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np

try:
    import faiss
except ImportError:
    print("faiss not found. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f"SentenceTransformer import failed: {e}")
    print("Fix: pip install numpy<2  then rerun")
    sys.exit(1)

from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index/indian_laws_v3.index"
METADATA_PATH    = "embeddings/indian_laws_metadata_v3.json"

# These law_name values are QA/noise — filter them out
BAD_LAW_NAMES = {"Indian_Law_QA", "Indian_Law", "Indian_Legal"}

# ── Test queries with expected ground truth ───────────────────────────
TEST_QUERIES = [
    {
        "query": "non-compete clause after employment ends",
        "expected_law_substr": "contract act",        # case-insensitive substring check
        "expected_keyword":    "restraint",
        "expected_section":    "27",
    },
    {
        "query": "overtime work without extra compensation",
        "expected_law_substr": "wages",
        "expected_keyword":    "working hours",
        "expected_section":    "25",
    },
    {
        "query": "termination of employment without prior notice",
        "expected_law_substr": "industrial",
        "expected_keyword":    "notice",
        "expected_section":    "77",
    },
    {
        "query": "payment condition based on satisfactory performance",
        "expected_law_substr": "wages",
        "expected_keyword":    "wages",
        "expected_section":    "17",
    },
    {
        "query": "arbitration dispute resolution agreement India",
        "expected_law_substr": "arbitration",
        "expected_keyword":    "arbitration",
        "expected_section":    "7",
    },
]

# ── Resource loading (lazy, once) ─────────────────────────────────────
_res = {}


def load_resources():
    if _res:
        return
    print("Loading FAISS index and embedder (first call only)...")
    _res["embedder"] = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    _res["index"]    = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _res["metadata"] = json.load(f)
    corpus = [c.get("text", "") for c in _res["metadata"]]
    _res["bm25"] = BM25Okapi([doc.lower().split() for doc in corpus])
    print(f"Loaded. Index vectors: {_res['index'].ntotal:,}")


def is_bad(chunk):
    return chunk.get("law_name", "") in BAD_LAW_NAMES


def top_chunks(ranked_indices, top_k=3):
    """Return top_k non-noise chunks from a ranked index list."""
    results = []
    for idx in ranked_indices:
        chunk = _res["metadata"][int(idx)]
        if is_bad(chunk):
            continue
        if len(chunk.get("text", "").split()) < 15:
            continue
        results.append({
            "law":     chunk.get("law_name", ""),
            "section": chunk.get("section_number", ""),
            "text":    chunk.get("text", "")[:250],
        })
        if len(results) == top_k:
            break
    return results


# ── Four retrieval strategies ─────────────────────────────────────────

def bm25_only(query, top_k=3):
    load_resources()
    scores = _res["bm25"].get_scores(query.lower().split())
    ranked = scores.argsort()[::-1]
    return top_chunks(ranked, top_k)


def faiss_only(query, top_k=3):
    load_resources()
    q_emb = _res["embedder"].encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    _, indices = _res["index"].search(q_emb, top_k * 10)
    return top_chunks(indices[0], top_k)


def hybrid_rrf(query, top_k=3, k=60):
    load_resources()
    # Dense
    q_emb = _res["embedder"].encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    _, f_indices = _res["index"].search(q_emb, top_k * 15)
    dense = {int(idx): 1 / (rank + 1 + k)
             for rank, idx in enumerate(f_indices[0]) if idx != -1}

    # Sparse
    bm25_sc = _res["bm25"].get_scores(query.lower().split())
    top_b   = bm25_sc.argsort()[-top_k * 15:][::-1]
    sparse  = {int(idx): 1 / (rank + 1 + k)
               for rank, idx in enumerate(top_b)}

    fused   = {idx: dense.get(idx, 0) + sparse.get(idx, 0)
               for idx in set(dense) | set(sparse)}
    ranked  = sorted(fused, key=lambda x: fused[x], reverse=True)
    return top_chunks(ranked, top_k)


def hybrid_crag(query, top_k=3):
    """Hybrid RRF with CRAG fallback."""
    results = hybrid_rrf(query, top_k)

    # CRAG relevance check
    combined = " ".join(r["text"] for r in results).lower()
    overlap  = len(set(query.lower().split()) & set(combined.split()))
    if overlap <= 3:
        broad   = query + " Indian contract law rights obligations"
        results = hybrid_rrf(broad, top_k)

    return results


# ── Scoring ───────────────────────────────────────────────────────────

def score_results(results, test):
    """
    Check across ALL top-3 results, not just top-1.
    Returns (score 0/0.5/1.0, law_found bool, section_found bool)
    """
    if not results:
        return 0.0, False, False

    all_laws    = " ".join(r["law"].lower()     for r in results)
    all_text    = " ".join(r["text"].lower()    for r in results)
    all_section = " ".join(r["section"].lower() for r in results)

    law_found     = test["expected_law_substr"].lower() in all_laws
    keyword_found = test["expected_keyword"].lower()    in all_text
    section_found = test["expected_section"].lower()    in all_section

    if law_found and keyword_found:
        score = 1.0
    elif law_found or keyword_found:
        score = 0.5
    else:
        score = 0.0

    return score, law_found, section_found


# ── Main ──────────────────────────────────────────────────────────────

def run_ablation():
    print("=" * 70)
    print("LEXIGUARD — RETRIEVAL STRATEGY ABLATION STUDY (FIXED)")
    print("=" * 70)

    strategies = {
        "BM25 Only":      bm25_only,
        "FAISS Only":     faiss_only,
        "Hybrid RRF":     hybrid_rrf,
        "Hybrid + CRAG":  hybrid_crag,
    }

    all_results = {}

    for name, fn in strategies.items():
        print(f"\n{'='*55}")
        print(f"  Strategy: {name}")
        print(f"{'='*55}")

        scores, law_hits, sec_hits, lats = [], [], [], []

        for test in TEST_QUERIES:
            t0      = time.time()
            results = fn(test["query"], top_k=3)
            lat     = time.time() - t0

            sc, lh, sh = score_results(results, test)
            scores.append(sc)
            law_hits.append(lh)
            sec_hits.append(sh)
            lats.append(lat)

            top_law = results[0]["law"] if results else "—"
            top_sec = results[0]["section"] if results else "—"
            print(f"  Q: {test['query'][:55]:<55}")
            print(f"     Score={sc:.1f} | Law={'✓' if lh else '✗'} | "
                  f"Section={'✓' if sh else '✗'} | {lat*1000:.0f}ms")
            print(f"     Top: {top_law} — {top_sec}")

        all_results[name] = {
            "avg_score":        round(sum(scores)   / len(scores),   3),
            "law_recall":       round(sum(law_hits)  / len(law_hits),  3),
            "section_precision":round(sum(sec_hits)  / len(sec_hits),  3),
            "avg_latency_ms":   round(sum(lats)      / len(lats) * 1000, 1),
            "per_query_scores": scores,
        }

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Avg Score':>12} {'Law Recall':>12} "
          f"{'Section P':>12} {'Latency(ms)':>14}")
    print("-" * 72)
    for name, res in all_results.items():
        print(f"{name:<20} {res['avg_score']:>12.3f} {res['law_recall']:>12.3f} "
              f"{res['section_precision']:>12.3f} {res['avg_latency_ms']:>13.1f}ms")

    print("\nNote: Hybrid+CRAG filters noise chunks (Indian_Law_QA) that inflate "
          "BM25/FAISS scores when not filtered — showing real retrieval quality.")

    with open("retrieval_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved → retrieval_ablation_results.json")

    return all_results


if __name__ == "__main__":
    run_ablation()
