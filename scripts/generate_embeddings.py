import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────
MODEL_NAME  = "nlpaueb/legal-bert-base-uncased"
CHUNKS_PATH = "data/processed/indian_laws_chunks.json"
OUTPUT_DIR  = "embeddings"
BATCH_SIZE  = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────
print("📂 Loading chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
print(f"   Total chunks  : {len(chunks):,}")
print(f"   Sample text   : {texts[0][:80]}")

# ── LOAD MODEL ────────────────────────────────────────────────
print(f"\n🤖 Loading model: {MODEL_NAME}")
print("   ⏳ First run downloads ~400MB — please wait...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device        : {device}")

model = SentenceTransformer(MODEL_NAME, device=device)
print(f"   ✅ Model loaded!")
print(f"   Embedding dim : {model.get_sentence_embedding_dimension()}")

# ── GENERATE EMBEDDINGS ───────────────────────────────────────
print(f"\n⚙️  Generating embeddings...")
print(f"   Batch size    : {BATCH_SIZE}")
print(f"   Total batches : {len(texts) // BATCH_SIZE + 1}")
print("   ⏳ ~3-5 mins on GPU...\n")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"\n✅ Embeddings generated!")
print(f"   Shape         : {embeddings.shape}")
print(f"   Dimensions    : {embeddings.shape[1]}")
print(f"   Total vectors : {embeddings.shape[0]:,}")

# ── SAVE EMBEDDINGS ───────────────────────────────────────────
emb_path  = os.path.join(OUTPUT_DIR, "indian_laws_embeddings.npy")
meta_path = os.path.join(OUTPUT_DIR, "indian_laws_metadata.json")

print(f"\n💾 Saving embeddings  → {emb_path}")
np.save(emb_path, embeddings)

print(f"💾 Saving metadata    → {meta_path}")
metadata = []
for i, chunk in enumerate(chunks):
    metadata.append({
        "id"            : i,
        "law_name"      : chunk["law_name"],
        "section_number": chunk["section_number"],
        "topic"         : chunk["topic"],
        "text"          : chunk["text"],
        "source"        : chunk.get("source", "unknown")
    })

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ── VERIFY ────────────────────────────────────────────────────
saved = np.load(emb_path)
print(f"\n✅ Verification:")
print(f"   Saved shape   : {saved.shape}")
print(f"   File size     : {os.path.getsize(emb_path)/(1024*1024):.1f} MB")
print(f"   Metadata rows : {len(metadata):,}")

# Quick similarity test
print(f"\n🔍 Quick similarity test:")
from numpy.linalg import norm
q = embeddings[0]
sims = embeddings @ q
top3 = np.argsort(sims)[::-1][1:4]
print(f"   Query : {texts[0][:60]}")
print(f"   Top 3 similar chunks:")
for idx in top3:
    print(f"      [{idx}] score={sims[idx]:.3f} | {texts[idx][:60]}")

print(f"\n🎉 Embeddings ready for FAISS indexing!")