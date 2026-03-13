import json
import os
import numpy as np
import faiss

print("="*60)
print("   LEXIGUARD - FINAL STATE VERIFICATION")
print("="*60)

# ── CHECK ALL KEY FILES ───────────────────────────────────────
files = {
    "Training Data": [
        "data/processed/combined_dataset.json",
        "data/processed/cuad_processed.json",
        "data/processed/ledgar_processed.json",
    ],
    "Knowledge Base Chunks": [
        "data/processed/indian_laws_chunks_final.json",
        "data/processed/indian_laws_chunks.json",
    ],
    "Embeddings": [
        "embeddings/indian_laws_embeddings.npy",
        "embeddings/indian_laws_metadata.json",
    ],
    "FAISS Index": [
        "faiss_index/indian_laws.index",
        "faiss_index/indian_laws_metadata.json",
    ],
}

all_ok = True
for category, paths in files.items():
    print(f"\n📁 {category}:")
    for path in paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"   ✅ {path:52} {size:.1f} MB")
        else:
            print(f"   ❌ MISSING: {path}")
            all_ok = False

# ── VERIFY EMBEDDINGS MATCH CHUNKS ───────────────────────────
print(f"\n{'='*60}")
print("   CONSISTENCY CHECK")
print("="*60)

# Load chunks
with open("data/processed/indian_laws_chunks_final.json",
          "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embeddings
emb = np.load("embeddings/indian_laws_embeddings.npy")

# Load metadata
with open("faiss_index/indian_laws_metadata.json",
          "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load FAISS
index = faiss.read_index("faiss_index/indian_laws.index")

print(f"\n   Chunks in final JSON    : {len(chunks):,}")
print(f"   Embedding vectors       : {emb.shape[0]:,}")
print(f"   Embedding dimensions    : {emb.shape[1]}")
print(f"   Metadata rows           : {len(metadata):,}")
print(f"   FAISS index vectors     : {index.ntotal:,}")

# Check all match
match = (len(chunks) == emb.shape[0] == len(metadata)
         == index.ntotal)
print(f"\n   All counts match        : {'✅ YES' if match else '❌ NO'}")

if not match:
    print(f"   ⚠️  MISMATCH DETECTED!")
    all_ok = False

# ── QUICK SEARCH TEST ─────────────────────────────────────────
print(f"\n{'='*60}")
print("   QUICK SEARCH TEST")
print("="*60)

from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

query = "arbitration agreement enforceable India"
q_emb = model.encode(
    [query], normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

scores, indices = index.search(q_emb, 3)
print(f"\n   Query: {query}")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
    m = metadata[idx]
    print(f"   [{rank+1}] score={score:.3f} | "
          f"law={m['law_name'][:25]:25} | "
          f"text={m['text'][:55]}")

# ── FINAL VERDICT ─────────────────────────────────────────────
print(f"\n{'='*60}")
if all_ok and match:
    print("✅ EVERYTHING IS CONSISTENT AND READY!")
    print("✅ Embeddings match chunks match FAISS index")
    print("✅ Ready to proceed to Step 14: LoRA Fine-tuning")
else:
    print("❌ ISSUES FOUND — needs fixing before Step 14")
print("="*60)