from datasets import load_dataset
import json
import os

os.makedirs("data/indian_laws", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

all_chunks = []

# ── DATASET 1: viber1/indian-law-dataset ──────────────────────
# Keys: Instruction, Response
print("📥 Processing viber1/indian-law-dataset...")
try:
    ds1 = load_dataset("viber1/indian-law-dataset")
    before = len(all_chunks)
    for item in ds1["train"]:
        instruction = str(item.get("Instruction", "")).strip()
        response    = str(item.get("Response", "")).strip()
        if len(response) > 100:
            all_chunks.append({
                "law_name"       : "Indian_Law",
                "section_number" : "QA",
                "topic"          : instruction[:80],
                "text"           : response[:1000],
                "source"         : "viber1/indian-law-dataset"
            })
    print(f"   ✅ Chunks collected: {len(all_chunks) - before}")
    print(f"   Sample instruction : {ds1['train'][0]['Instruction'][:80]}")
    print(f"   Sample response    : {ds1['train'][0]['Response'][:80]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ── DATASET 2: nisaar/Lawyer_GPT_India ────────────────────────
# Keys: question, answer
print("\n📥 Processing nisaar/Lawyer_GPT_India...")
try:
    ds2 = load_dataset("nisaar/Lawyer_GPT_India")
    before = len(all_chunks)
    for item in ds2["train"]:
        question = str(item.get("question", "")).strip()
        answer   = str(item.get("answer", "")).strip()
        if len(answer) > 100:
            all_chunks.append({
                "law_name"       : "Indian_Law_QA",
                "section_number" : "QA",
                "topic"          : question[:80],
                "text"           : answer[:1000],
                "source"         : "nisaar/Lawyer_GPT_India"
            })
    print(f"   ✅ Chunks collected: {len(all_chunks) - before}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ── DATASET 3: ShreyasP123/Legal-Dataset-for-india ────────────
# Keys: chunk_id, text
print("\n📥 Processing ShreyasP123/Legal-Dataset-for-india...")
try:
    ds3 = load_dataset("ShreyasP123/Legal-Dataset-for-india")
    before = len(all_chunks)
    for item in ds3["train"]:
        text = str(item.get("text", "")).strip()
        if len(text) > 100:
            all_chunks.append({
                "law_name"       : "Indian_Legal",
                "section_number" : str(item.get("chunk_id", "General")),
                "topic"          : text[:80],
                "text"           : text[:1000],
                "source"         : "ShreyasP123/Legal-Dataset-for-india"
            })
    print(f"   ✅ Chunks collected: {len(all_chunks) - before}")
    print(f"   Sample text        : {ds3['train'][0]['text'][:100]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ── ALSO ADD OUR MANUALLY WRITTEN LAW FILES ───────────────────
print("\n📥 Adding our manually written Indian law files...")
law_dir = "data/indian_laws"
import re
before = len(all_chunks)

for filename in sorted(os.listdir(law_dir)):
    if not filename.endswith(".txt"):
        continue
    law_name = filename.replace(".txt", "")
    with open(os.path.join(law_dir, filename), "r", encoding="utf-8") as f:
        text = f.read()

    # Split by Section
    lines      = text.split("\n")
    clean_text = "\n".join(lines[4:])
    sections   = re.split(r'(Section\s+\d+[A-Z]?\s*[-–]?\s*[^\n]*)', clean_text)

    current_sec   = "General"
    current_topic = "General"
    buffer        = []

    def save_buffer(sec, topic, buf, law):
        content = " ".join(buf).strip()
        content = re.sub(r'\s+', ' ', content)
        if len(content.split()) > 20:
            all_chunks.append({
                "law_name"       : law,
                "section_number" : sec,
                "topic"          : topic[:80],
                "text"           : content[:1000],
                "source"         : "manual"
            })

    for part in sections:
        part = part.strip()
        if not part:
            continue
        sec_match = re.match(r'Section\s+(\d+[A-Z]?)\s*[-–]?\s*(.*)', part, re.IGNORECASE)
        if sec_match:
            if buffer:
                save_buffer(current_sec, current_topic, buffer, law_name)
                buffer = []
            current_sec   = f"Section {sec_match.group(1)}"
            current_topic = sec_match.group(2).strip()[:80]
            buffer        = [part]
        else:
            words = part.split()
            for i in range(0, len(words), 150):
                sub = " ".join(words[i:i+150])
                if len(sub.split()) > 20:
                    all_chunks.append({
                        "law_name"       : law_name,
                        "section_number" : current_sec,
                        "topic"          : current_topic[:80],
                        "text"           : sub,
                        "source"         : "manual"
                    })

    if buffer:
        save_buffer(current_sec, current_topic, buffer, law_name)

print(f"   ✅ Chunks from manual files: {len(all_chunks) - before}")

# ── SAVE ──────────────────────────────────────────────────────
out_path = "data/processed/indian_laws_chunks.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n{'='*55}")
print(f"✅ Total chunks saved : {len(all_chunks)}")
print(f"💾 Saved to           : {out_path}")

print("\n📊 Breakdown by source:")
from collections import Counter
sources = Counter(c["source"] for c in all_chunks)
for src, count in sources.most_common():
    print(f"   {src:45}: {count}")

print("\n📄 Sample chunks:")
for i in [0, 100, 500]:
    if i < len(all_chunks):
        s = all_chunks[i]
        print(f"\n  [{i}] law    : {s['law_name']}")
        print(f"       section: {s['section_number']}")
        print(f"       text   : {s['text'][:100]}")