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
# SOURCE 1: mratanusarkar/Indian-Laws (best quality — actual Acts)
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

    for item in ds["train"]:
        act_title = str(item.get("act_title", "")).strip()
        section   = str(item.get("section",   "")).strip()
        law_text  = str(item.get("law",       "")).strip()

        if not any(kw in act_title.lower() for kw in TARGET_ACTS):
            continue
        if len(law_text.split()) < 20:
            continue

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
except Exception as e:
    print(f"   ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════
# SOURCE 2: ShreyasP123 (legal document chunks)
# ══════════════════════════════════════════════════════════════
print("\n📥 Loading ShreyasP123/Legal-Dataset-for-india...")
try:
    ds2    = load_dataset("ShreyasP123/Legal-Dataset-for-india")
    before = len(all_chunks)

    BAD = ["does not", "context", "not specified",
           "not mentioned", "not provided", "cannot be"]

    for item in ds2["train"]:
        text = str(item.get("text", "")).strip()
        if len(text.split()) < 25:
            continue
        if any(b in text.lower() for b in BAD):
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
# SOURCE 3: viber1 — law-specific QA only
# ══════════════════════════════════════════════════════════════
print("\n📥 Loading viber1/indian-law-dataset (filtered)...")
try:
    ds3    = load_dataset("viber1/indian-law-dataset")
    before = len(all_chunks)

    LAW_KEYWORDS = [
        "arbitration", "contract", "consumer protection",
        "real estate", "rera", "information technology",
        "it act", "data protection", "dpdp", "labour",
        "wages", "employment", "copyright", "trademark",
        "patent", "insolvency", "company", "negligence",
        "indemnif", "termination", "confidential", "liability",
        "intellectual property", "governing law", "jurisdiction",
        "penalty", "offence", "tribunal", "section",
        "agreement", "clause", "provision", "act 1872",
        "act 1996", "act 2000", "act 2016", "act 2019",
        "act 2020", "act 2023", "enforce", "breach",
        "damages", "remedy", "injunction", "specific performance"
    ]

    BAD_PHRASES = [
        "does not include article",
        "context does not explicitly",
        "content provided does not",
        "the given context does not",
        "context provided doesn",
        "cannot be determined",
        "not mentioned in the",
        "the passage does not",
        "article 272",
        "part ixa",
        "union territory of mizoram",
        "governor is required to",
        "administrator of the union",
    ]

    for item in ds3["train"]:
        response = str(item.get("Response", "")).strip()
        instr    = str(item.get("Instruction", "")).strip()

        if len(response.split()) < 30:
            continue

        resp_lower = response.lower()

        # Skip bad quality
        if any(b in resp_lower for b in BAD_PHRASES):
            continue

        # Only keep law-specific content
        combined = (instr + " " + response).lower()
        if not any(kw in combined for kw in LAW_KEYWORDS):
            continue

        all_chunks.append({
            "law_name"       : "Indian_Law",
            "section_number" : "QA",
            "topic"          : instr[:80],
            "text"           : response[:800],
            "source"         : "viber1/indian-law-dataset"
        })

    print(f"   ✅ Chunks added : {len(all_chunks) - before:,}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════
# SOURCE 4: nisaar/Lawyer_GPT_India (lawyer QA)
# ══════════════════════════════════════════════════════════════
print("\n📥 Loading nisaar/Lawyer_GPT_India...")
try:
    ds4    = load_dataset("nisaar/Lawyer_GPT_India")
    before = len(all_chunks)

    for item in ds4["train"]:
        question = str(item.get("question", "")).strip()
        answer   = str(item.get("answer",   "")).strip()
        if len(answer.split()) < 30:
            continue
        all_chunks.append({
            "law_name"       : "Indian_Law_QA",
            "section_number" : "QA",
            "topic"          : question[:80],
            "text"           : answer[:800],
            "source"         : "nisaar/Lawyer_GPT_India"
        })
    print(f"   ✅ Chunks added : {len(all_chunks) - before:,}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ══════════════════════════════════════════════════════════════
# SOURCE 5: Manual law files (our written sections)
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

    sections      = re.split(
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
print(f"\n{'='*60}")
print(f"📊 TOTAL CHUNKS : {len(all_chunks):,}")
sources = Counter(c["source"] for c in all_chunks)
laws    = Counter(c["law_name"] for c in all_chunks)

print(f"\nBy source:")
for src, count in sources.most_common():
    print(f"   {src[:48]:48} : {count:,}")

print(f"\nTop law names:")
for law, count in laws.most_common(15):
    print(f"   {law[:52]:52} : {count:,}")

# Save
out = "data/processed/indian_laws_chunks_final.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

# Also update default
import shutil
shutil.copy(out, "data/processed/indian_laws_chunks.json")
print(f"\n💾 Saved {len(all_chunks):,} chunks to {out}")

# ══════════════════════════════════════════════════════════════
# EMBEDDINGS with all-MiniLM-L6-v2
# ══════════════════════════════════════════════════════════════
print(f"\n🤖 Generating embeddings with all-MiniLM-L6-v2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device : {device}")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device
)
print(f"   ✅ Model loaded! Dim: "
      f"{model.get_sentence_embedding_dimension()}")

texts = [c["text"] for c in all_chunks]
print(f"   Encoding {len(texts):,} chunks...\n")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"\n   Shape : {embeddings.shape}")

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
with open("embeddings/indian_laws_metadata.json",
          "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════
# BUILD FAISS
# ══════════════════════════════════════════════════════════════
print(f"\n⚙️  Building FAISS index...")
emb   = embeddings.astype("float32")
index = faiss.IndexIDMap(faiss.IndexFlatIP(emb.shape[1]))
index.add_with_ids(emb, np.arange(len(emb)).astype("int64"))
faiss.write_index(index, "faiss_index/indian_laws.index")
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
    "Real estate promoter obligations RERA",
    "Employee wages working hours labour code",
    "Confidentiality clause breach of contract",
    "IT Act electronic signature digital",
    "Indian Contract Act offer acceptance",
    "Intellectual property copyright infringement",
    "Company director duties liabilities",
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

print("\n" + "="*60)
print("✅ Full combined knowledge base ready!")
print("="*60)