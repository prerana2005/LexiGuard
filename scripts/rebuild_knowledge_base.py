import json
import os
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from collections import Counter

print("📂 Loading all chunks...")
with open("data/processed/indian_laws_chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)
print(f"   Total: {len(all_chunks):,}")

# ── STRATEGY: Keep only HIGH QUALITY sources ──────────────────
# viber1 = generic Indian constitution QA → REMOVE
# ShreyasP123 = actual legal document chunks → KEEP
# nisaar = Indian law QA → KEEP
# manual = our written law sections → KEEP (boost these!)

GOOD_SOURCES = [
    "ShreyasP123/Legal-Dataset-for-india",
    "nisaar/Lawyer_GPT_India",
    "manual"
]

# Also keep viber1 only if it mentions specific law keywords
LAW_KEYWORDS = [
    "contract act", "arbitration", "consumer protection",
    "rera", "real estate", "it act", "information technology",
    "dpdp", "data protection", "labour code", "wages",
    "section", "act 1872", "act 1996", "act 2000",
    "act 2016", "act 2019", "act 2020", "act 2023",
    "indemnif", "termination", "confidential", "liability",
    "intellectual property", "governing law", "jurisdiction",
    "penalty", "offence", "court", "tribunal", "arbitrat"
]

def is_law_specific(text):
    text_lower = text.lower()
    return any(kw in text_lower for kw in LAW_KEYWORDS)

def is_good_quality(text):
    bad = [
        "does not include article",
        "context does not explicitly",
        "content provided does not",
        "the given context does not",
        "context provided doesn",
        "does not specify",
        "cannot be determined",
        "not mentioned in",
        "the passage does not",
        "article 272",
        "part ixa",
        "mizoram",
        "municipalities",
    ]
    text_lower = text.lower()
    for b in bad:
        if b in text_lower:
            return False
    if len(text.split()) < 25:
        return False
    return True

# Filter chunks
good_chunks = []
stats = {"kept_manual": 0, "kept_shreyas": 0,
         "kept_nisaar": 0, "kept_viber1": 0, "removed": 0}

for chunk in all_chunks:
    source = chunk.get("source", "")
    text   = chunk["text"]

    if not is_good_quality(text):
        stats["removed"] += 1
        continue

    if source == "manual":
        good_chunks.append(chunk)
        stats["kept_manual"] += 1

    elif source == "nisaar/Lawyer_GPT_India":
        good_chunks.append(chunk)
        stats["kept_nisaar"] += 1

    elif source == "ShreyasP123/Legal-Dataset-for-india":
        good_chunks.append(chunk)
        stats["kept_shreyas"] += 1

    elif source == "viber1/indian-law-dataset":
        # Only keep viber1 if it's about specific laws
        if is_law_specific(text):
            good_chunks.append(chunk)
            stats["kept_viber1"] += 1
        else:
            stats["removed"] += 1

print(f"\n📊 Filtering results:")
print(f"   Manual law sections  : {stats['kept_manual']:,}")
print(f"   ShreyasP123 legal    : {stats['kept_shreyas']:,}")
print(f"   Nisaar lawyer QA     : {stats['kept_nisaar']:,}")
print(f"   Viber1 (law-specific): {stats['kept_viber1']:,}")
print(f"   Removed (low quality): {stats['removed']:,}")
print(f"   ─────────────────────────────")
print(f"   Total kept           : {len(good_chunks):,}")

# Show law breakdown
laws = Counter(c["law_name"] for c in good_chunks)
print(f"\n   Law names breakdown:")
for law, count in laws.most_common():
    print(f"      {law[:45]:45} : {count:,}")

# Save
out = "data/processed/indian_laws_chunks_filtered.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(good_chunks, f, indent=2, ensure_ascii=False)
print(f"\n💾 Saved {len(good_chunks):,} chunks to {out}")

# ── GENERATE EMBEDDINGS ───────────────────────────────────────
print(f"\n🤖 Generating embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = SentenceTransformer("nlpaueb/legal-bert-base-uncased",
                              device=device)
texts  = [c["text"] for c in good_chunks]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"   Shape: {embeddings.shape}")

np.save("embeddings/indian_laws_embeddings.npy", embeddings)

metadata = [
    {
        "id"            : i,
        "law_name"      : c["law_name"],
        "section_number": c["section_number"],
        "topic"         : c["topic"],
        "text"          : c["text"],
        "source"        : c.get("source", "unknown")
    }
    for i, c in enumerate(good_chunks)
]
with open("embeddings/indian_laws_metadata.json", "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ── REBUILD FAISS ─────────────────────────────────────────────
print(f"\n⚙️  Rebuilding FAISS index...")
emb   = embeddings.astype("float32")
index = faiss.IndexIDMap(faiss.IndexFlatIP(emb.shape[1]))
index.add_with_ids(emb, np.arange(len(emb)).astype("int64"))
faiss.write_index(index, "faiss_index/indian_laws.index")

import shutil
shutil.copy("embeddings/indian_laws_metadata.json",
            "faiss_index/indian_laws_metadata.json")
print(f"   Total vectors: {index.ntotal:,} ✅")

# ── TEST SEARCH ───────────────────────────────────────────────
print(f"\n🔍 Final search test:")
queries = [
    "What is arbitration agreement in India?",
    "Consumer protection rights for defective products",
    "Data protection and privacy DPDP",
    "Real estate promoter obligations RERA",
    "Employee wages and working hours labour code",
    "Confidentiality clause in contract",
]

for query in queries:
    q_emb          = model.encode(
        [query], normalize_embeddings=True,
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
print("✅ Knowledge base rebuilt successfully!")
print("="*55)