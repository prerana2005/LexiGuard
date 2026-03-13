import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# ── CLEAR GPU MEMORY FIRST ────────────────────────────────────
torch.cuda.empty_cache()
print(f"GPU memory freed!")

# ── LOAD FILTERED CHUNKS ──────────────────────────────────────
print("📂 Loading filtered chunks...")
with open("data/processed/indian_laws_chunks_filtered.json",
          "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"   Total chunks : {len(chunks):,}")

texts = [c["text"] for c in chunks]

# ── LOAD MODEL ON CPU FIRST ───────────────────────────────────
print("\n🤖 Loading model...")
device = "cpu"
print(f"   Device: {device}")

model = SentenceTransformer(
    "nlpaueb/legal-bert-base-uncased",
    device=device
)

# ── GENERATE EMBEDDINGS WITH SMALL BATCH ──────────────────────
print(f"\n⚙️  Generating embeddings (batch_size=8)...")
print("   ⏳ ~5-8 mins with small batch...\n")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

embeddings = model.encode(
    texts,
    batch_size=8,          # ← reduced from 32 to 8
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"\n✅ Embeddings done!")
print(f"   Shape: {embeddings.shape}")

# ── SAVE ──────────────────────────────────────────────────────
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/indian_laws_embeddings.npy", embeddings)

metadata = [
    {
        "id"            : i,
        "law_name"      : c["law_name"],
        "section_number": c["section_number"],
        "topic"         : c["topic"],
        "text"          : c["text"],
        "source"        : c.get("source","unknown")
    }
    for i, c in enumerate(chunks)
]
with open("embeddings/indian_laws_metadata.json", "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"💾 Embeddings saved: {os.path.getsize('embeddings/indian_laws_embeddings.npy')/(1024*1024):.1f} MB")

# ── BUILD FAISS ───────────────────────────────────────────────
print(f"\n⚙️  Building FAISS index...")
os.makedirs("faiss_index", exist_ok=True)
emb   = embeddings.astype("float32")
index = faiss.IndexIDMap(faiss.IndexFlatIP(emb.shape[1]))
index.add_with_ids(emb, np.arange(len(emb)).astype("int64"))
faiss.write_index(index, "faiss_index/indian_laws.index")

import shutil
shutil.copy("embeddings/indian_laws_metadata.json",
            "faiss_index/indian_laws_metadata.json")
print(f"   Total vectors: {index.ntotal:,} ✅")

# ── FREE GPU BEFORE SEARCH TEST ───────────────────────────────
del model
torch.cuda.empty_cache()

# ── TEST SEARCH ───────────────────────────────────────────────
print(f"\n🔍 Testing search...")
model2 = SentenceTransformer(
    "nlpaueb/legal-bert-base-uncased",
    device=device
)

queries = [
    "What is arbitration agreement in India?",
    "Consumer protection rights for defective products",
    "Data protection and privacy DPDP",
    "Real estate promoter obligations RERA",
    "Employee wages and working hours",
    "Confidentiality clause obligations",
]

for query in queries:
    q_emb = model2.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
    scores, indices = index.search(q_emb, 3)

    print(f"\n   Q: {query}")
    for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0])):
        m = metadata[idx]
        print(f"   [{rank+1}] {score:.3f} | "
              f"{m['law_name'][:22]:22} | "
              f"{m['text'][:65]}")

print("\n" + "="*55)
print("✅ FAISS index ready!")
print("="*55)