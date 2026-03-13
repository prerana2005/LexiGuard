import json
import os
import numpy as np
import faiss
import time

# ── CONFIG ────────────────────────────────────────────────────
EMB_PATH    = "embeddings/indian_laws_embeddings.npy"
META_PATH   = "embeddings/indian_laws_metadata.json"
INDEX_DIR   = "faiss_index"
INDEX_PATH  = os.path.join(INDEX_DIR, "indian_laws.index")
META_OUT    = os.path.join(INDEX_DIR, "indian_laws_metadata.json")

os.makedirs(INDEX_DIR, exist_ok=True)

# ── LOAD EMBEDDINGS ───────────────────────────────────────────
print("📂 Loading embeddings...")
embeddings = np.load(EMB_PATH).astype("float32")
print(f"   Shape         : {embeddings.shape}")
print(f"   Dimensions    : {embeddings.shape[1]}")
print(f"   Total vectors : {embeddings.shape[0]:,}")

# Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"   Metadata rows : {len(metadata):,}")

# ── BUILD FAISS INDEX ─────────────────────────────────────────
print(f"\n⚙️  Building FAISS index...")
dim = embeddings.shape[1]  # 768

# IndexFlatIP = Inner Product (cosine similarity for normalized vectors)
index = faiss.IndexFlatIP(dim)

# Wrap with IDMap so we can retrieve by original ID
index = faiss.IndexIDMap(index)

# Add embeddings with their IDs
ids = np.arange(len(embeddings)).astype("int64")

print(f"   Adding {len(embeddings):,} vectors...")
start = time.time()
index.add_with_ids(embeddings, ids)
elapsed = time.time() - start

print(f"   ✅ Added in {elapsed:.2f} seconds")
print(f"   Total in index: {index.ntotal:,}")

# ── SAVE INDEX ────────────────────────────────────────────────
print(f"\n💾 Saving index → {INDEX_PATH}")
faiss.write_index(index, INDEX_PATH)

print(f"💾 Saving metadata → {META_OUT}")
import shutil
shutil.copy(META_PATH, META_OUT)

size = os.path.getsize(INDEX_PATH) / (1024*1024)
print(f"   Index file size: {size:.1f} MB")

# ── TEST SEARCH ───────────────────────────────────────────────
print(f"\n🔍 Testing FAISS search...")
loaded_index = faiss.read_index(INDEX_PATH)
print(f"   Index loaded! Total vectors: {loaded_index.ntotal:,}")

# Test queries
test_queries = [
    "What is arbitration agreement in India?",
    "Consumer protection rights for defective products",
    "Data protection and privacy obligations",
    "Real estate promoter obligations RERA",
    "Employee wages and working hours"
]

# We need to encode queries — load model
print("\n   Loading model for query encoding...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "nlpaueb/legal-bert-base-uncased",
    device="cuda"
)

print("\n   Running test searches (Top 3 results each):\n")
for query in test_queries:
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = loaded_index.search(q_emb, 3)

    print(f"   Query: {query}")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = metadata[idx]
        print(f"   [{rank+1}] score={score:.3f} | "
              f"law={chunk['law_name'][:25]} | "
              f"text={chunk['text'][:60]}")
    print()

print("="*55)
print(f"✅ FAISS index built and tested successfully!")
print(f"📁 Index saved to : {INDEX_PATH}")
print(f"📁 Metadata saved : {META_OUT}")
print("="*55)