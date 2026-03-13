import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

# Load everything
with open("faiss_index/indian_laws_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ── CHECK 1: What Arbitration chunks do we actually have? ─────
print("="*60)
print("CHECK 1: All Arbitration chunks in our database")
print("="*60)
arb_chunks = [m for m in metadata if "arbitrat" in m["law_name"].lower()
              or "arbitrat" in m["text"].lower()]
print(f"Total arbitration-related chunks: {len(arb_chunks)}")
for i, c in enumerate(arb_chunks[:10]):
    print(f"\n[{i}] law    : {c['law_name']}")
    print(f"    section: {c['section_number']}")
    print(f"    text   : {c['text'][:120]}")

# ── CHECK 2: What are the TOP chunks by law name? ─────────────
print("\n" + "="*60)
print("CHECK 2: Law name distribution")
print("="*60)
laws = Counter(m["law_name"] for m in metadata)
for law, count in laws.most_common():
    print(f"   {law:45} : {count}")

# ── CHECK 3: Sample Indian_Law chunks (the noisy ones) ────────
print("\n" + "="*60)
print("CHECK 3: Sample Indian_Law chunks (first 5)")
print("="*60)
indian_law = [m for m in metadata if m["law_name"] == "Indian_Law"]
for c in indian_law[:5]:
    print(f"\n   text: {c['text'][:150]}")