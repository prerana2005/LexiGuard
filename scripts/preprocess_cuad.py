import json
import os

print("📥 Loading CUAD_v1.json...")
with open("data/raw/CUAD_v1.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

contracts = raw["data"]
print(f"Total contracts : {len(contracts)}")

os.makedirs("data/processed", exist_ok=True)

processed = []
skipped = 0

for doc in contracts:
    title = doc["title"]
    for para in doc["paragraphs"]:
        context = para["context"].strip()
        for qa in para["qas"]:
            question = qa["question"].strip()
            answers = qa.get("answers", [])

            # Only keep QAs that have actual answers (not unanswerable)
            if answers:
                answer_text = answers[0]["text"].strip()
                if len(answer_text) > 10 and len(context) > 50:
                    processed.append({
                        "clause_text": context,
                        "label": question,
                        "answer": answer_text,
                        "source": title
                    })
            else:
                skipped += 1

print(f"Processed records : {len(processed)}")
print(f"Skipped (no answer): {skipped}")

# Save
out_path = "data/processed/cuad_processed.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved to {out_path}")
print(f"\n📄 Sample record:")
s = processed[0]
print(f"  clause_text : {s['clause_text'][:150]}")
print(f"  label       : {s['label']}")
print(f"  answer      : {s['answer'][:100]}")
print(f"  source      : {s['source']}")