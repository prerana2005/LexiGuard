import json
import os
import re
import numpy as np
import faiss
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from collections import Counter

torch.cuda.empty_cache()
os.makedirs("data/processed", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

all_chunks = []

# ══════════════════════════════════════════════════════════════
# DATASET 1: mratanusarkar/Indian-Laws (34,200 actual Act rows)
# ══════════════════════════════════════════════════════════════
print("📥 Loading mratanusarkar/Indian-Laws...")
try:
    ds = load_dataset("mratanusarkar/Indian-Laws")
    before = len(all_chunks)

    TARGET_ACTS = [
        "arbitration", "contract act", "information technology",
        "consumer protection", "real estate", "rera",
        "data protection", "labour", "wages", "employment",
        "companies act", "intellectual property", "copyright",
        "trademark", "patent", "insolvency", "bankruptcy",
        "negotiable instruments", "transfer of property",
        "specific relief", "limitation act", "civil procedure",
        "criminal procedure", "evidence act", "indian penal"
    ]

    def is_target_act(title):
        title_lower = title.lower()
        return any(kw in title_lower for kw in TARGET_ACTS)

    skipped = 0
    for item in ds["train"]:
        act_title = str(item.get("act_title", "")).strip()
        section   = str(item.get("section", "")).strip()
        law_text  = str(item.get("law", "")).strip()

        if not is_target_act(act_title):
            skipped += 1
            continue
        if len(law_text.split()) < 20:
            continue

        # Chunk into 150 word pieces
        words = law_text.split()
        for i in range(0, len(words), 150):
            chunk_text = " ".join(words[i:i+150])
            if len(chunk_text.split()) < 20:
                continue
            all_chunks.append({
                "law_name"       : act_title[:60],
                "section_number" : f"Section {section}",
                "topic"          : act_title[:80],
                "text"           : chunk_text,
                "source"         : "mratanusarkar/Indian-Laws"
            })

    print(f"   ✅ Chunks added : {len(all_chunks) - before:,}")
    print(f"   Skipped acts   : {skipped:,}")

except Exception as e:
    print(f"   ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════
# DATASET 2: Keep ShreyasP123 (good quality legal chunks)
# ══════════════════════════════════════════════════════════════
print("\n📥 Loading ShreyasP123/Legal-Dataset-for-india...")
try:
    ds2    = load_dataset("ShreyasP123/Legal-Dataset-for-india")
    before = len(all_chunks)

    BAD = ["does not", "context", "not specified", "not mentioned",
           "not provided", "cannot be"]

    for item in ds2["train"]:
        text = str(item.get("text", "")).strip()
        if len(text.split()) < 25:
            continue
        text_lower = text.lower()
        if any(b in text_lower for b in BAD):
            continue
        all_chunks.append({
            "law_name"       : "Indian_Legal",
            "section_number" : str(item.get("chunk_id", "General")),
            "topic"          : text[:80],
            "text"           : text[:800],
            "source"         : "ShreyasP123/Legal-Dataset-for-india"
        })
    print(f"   ✅ Chunks added : {len(all_chunks) - before:,}")

except Exception as e:
    print(f"   ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════
# DATASET 3: Keep our manual law files (high quality)
# ══════════════════════════════════════════════════════════════
print("\n📥 Loading manual law files...")
law_dir = "data/indian_laws"
before  = len(all_chunks)

for filename in sorted(os.listdir(law_dir)):
    if not filename.endswith(".txt"):
        continue
    law_name = filename.replace(".txt", "")
    with open(os.path.join(law_dir, filename),
              "r", encoding="utf-8") as f:
        text = f.read()

    sections = re.split(
        r'(Section\s+\d+[A-Z]?\s*[-–]?\s*[^\n]*)', text)
    current_sec   = "General"
    current_topic = "General"

    for part in sections:
        part = part.strip()
        if not part:
            continue
        sec_match = re.match(
            r'Section\s+(\d+[A-Z]?)\s*[-–]?\s*(.*)',
            part, re.IGNORECASE)
        if sec_match:
            current_sec   = f"Section {sec_match.group(1)}"
            current_topic = sec_match.group(2).strip()[:80]
        else:
            words = part.split()
            for i in range(0, len(words), 150):
                chunk = " ".join(words[i:i+150])
                if len(chunk.split()) >= 20:
                    all_chunks.append({
                        "law_name"       : law_name,
                        "section_number" : current_sec,
                        "topic"          : current_topic,
                        "text"           : chunk,
                        "source"         : "manual"
                    })

print(f"   ✅ Chunks added : {len(all_chunks) - before:,}")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"📊 Total chunks : {len(all_chunks):,}")
sources = Counter(c["source"] for c in all_chunks)
laws    = Counter(c["law_name"] for c in all_chunks)
print(f"\nBy source:")
for src, count in sources.most_common():
    print(f"   {src[:45]:45} : {count:,}")
print(f"\nTop law names:")
for law, count in laws.most_common(15):
    print(f"   {law[:50]:50} : {count:,}")

# Save chunks
out = "data/processed/indian_laws_chunks_final.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)
print(f"\n💾 Saved to {out}")

# ══════════════════════════════════════════════════════════════
# EMBEDDINGS with all-MiniLM-L6-v2 (better + faster)
# ══════════════════════════════════════════════════════════════
print(f"\n🤖 Generating embeddings with all-MiniLM-L6-v2...")
print("   ⏳ Downloading ~80MB model (much smaller than legal-bert)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device : {device}")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)
print(f"   ✅ Model loaded! Dim: {model.get_sentence_embedding_dimension()}")

texts = [c["text"] for c in all_chunks]
print(f"   Encoding {len(texts):,} chunks...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"   Shape : {embeddings.shape}")

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
    for i, c in enumerate(all_chunks)
]
with open("embeddings/indian_laws_metadata.json", "w",
          encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════
# BUILD FAISS
# ══════════════════════════════════════════════════════════════
print(f"\n⚙️  Building FAISS index...")
emb   = embeddings.astype("float32")
index = faiss.IndexIDMap(faiss.IndexFlatIP(emb.shape[1]))
index.add_with_ids(emb, np.arange(len(emb)).astype("int64"))
faiss.write_index(index, "faiss_index/indian_laws.index")

import shutil
shutil.copy("embeddings/indian_laws_metadata.json",
            "faiss_index/indian_laws_metadata.json")
print(f"   Total vectors : {index.ntotal:,} ✅")

# ══════════════════════════════════════════════════════════════
# TEST SEARCH
# ══════════════════════════════════════════════════════════════
print(f"\n🔍 Final search quality test:")
queries = [
    "What is an arbitration agreement?",
    "Consumer protection rights defective products",
    "Data protection privacy DPDP obligations",
    "Real estate promoter obligations RERA registration",
    "Employee wages working hours labour code",
    "Confidentiality clause breach of contract",
    "IT Act electronic signature digital",
    "Indian Contract Act offer acceptance valid contract",
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
              f"{m['law_name'][:28]:28} | "
              f"{m['text'][:60]}")

print("\n" + "="*55)
print("✅ Knowledge base rebuilt with better data + model!")
print("="*55)