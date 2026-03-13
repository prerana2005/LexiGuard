import json
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

# ── LOAD ALL CHUNKS ───────────────────────────────────────────
print("📂 Loading chunks...")
with open("data/processed/indian_laws_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"   Total before filter: {len(chunks):,}")

# ── QUALITY FILTER ────────────────────────────────────────────
# Remove low quality generic responses
BAD_PHRASES = [
    "the given context does not",
    "the context provided doesn",
    "context does not specify",
    "context provided doesn",
    "does not provide information",
    "the text does not",
    "cannot be determined from",
    "not mentioned in the",
    "no information available",
    "the passage does not",
]

def is_good_chunk(text):
    text_lower = text.lower()
    # Filter bad phrases
    for phrase in BAD_PHRASES:
        if phrase in text_lower:
            return False
    # Must be long enough
    if len(text.split()) < 30:
        return False
    # Must not be too repetitive
    words = text_lower.split()
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:
        return False
    return True

good_chunks = [c for c in chunks if is_good_chunk(c["text"])]
bad_count   = len(chunks) - len(good_chunks)

print(f"   Filtered out       : {bad_count:,} low quality chunks")
print(f"   Remaining chunks   : {len(good_chunks):,}")

# Show source breakdown after filtering
from collections import Counter
sources = Counter(c.get("source","unknown") for c in good_chunks)
laws    = Counter(c["law_name"] for c in good_chunks)

print(f"\n   Sources after filter:")
for src, count in sources.most_common():
    print(f"      {src[:45]:45} : {count:,}")

print(f"\n   Law names after filter:")
for law, count in laws.most_common(10):
    print(f"      {law[:45]:45} : {count:,}")

# Save filtered chunks
out_path = "data/processed/indian_laws_chunks_filtered.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(good_chunks, f, indent=2, ensure_ascii=False)
print(f"\n💾 Saved filtered chunks to {out_path}")

# ── RE-GENERATE EMBEDDINGS ────────────────────────────────────
print(f"\n🤖 Re-generating embeddings for {len(good_chunks):,} chunks...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer("nlpaueb/legal-bert-base-uncased", device=device)

texts      = [c["text"] for c in good_chunks]
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"   Shape: {embeddings.shape}")

# Save new embeddings
np.save("embeddings/indian_laws_embeddings.npy", embeddings)

# Save new metadata
metadata = []
for i, chunk in enumerate(good_chunks):
    metadata.append({
        "id"            : i,
        "law_name"      : chunk["law_name"],
        "section_number": chunk["section_number"],
        "topic"         : chunk["topic"],
        "text"          : chunk["text"],
        "source"        : chunk.get("source", "unknown")
    })
with open("embeddings/indian_laws_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ── REBUILD FAISS INDEX ───────────────────────────────────────
print(f"\n⚙️  Rebuilding FAISS index...")
os.makedirs("faiss_index", exist_ok=True)
emb_f32 = embeddings.astype("float32")
dim     = emb_f32.shape[1]
index   = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
ids     = np.arange(len(emb_f32)).astype("int64")
index.add_with_ids(emb_f32, ids)
print(f"   Total vectors: {index.ntotal:,}")

faiss.write_index(index, "faiss_index/indian_laws.index")
import shutil
shutil.copy(
    "embeddings/indian_laws_metadata.json",
    "faiss_index/indian_laws_metadata.json"
)
print(f"   ✅ Index saved!")

# ── TEST SEARCH ───────────────────────────────────────────────
print(f"\n🔍 Testing improved search...")
test_queries = [
    "What is arbitration agreement in India?",
    "Consumer protection rights for defective products",
    "Data protection and privacy obligations",
    "Real estate promoter obligations RERA",
    "Employee wages and working hours"
]

for query in test_queries:
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(q_emb, 3)
    print(f"\n   Query: {query}")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = metadata[idx]
        print(f"   [{rank+1}] score={score:.3f} | "
              f"law={chunk['law_name'][:25]:25} | "
              f"text={chunk['text'][:70]}")

print("\n" + "="*55)
print("✅ Knowledge base fixed and FAISS rebuilt!")
print("="*55)