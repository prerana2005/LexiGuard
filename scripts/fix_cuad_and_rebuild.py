import json
import random
from collections import defaultdict, Counter

print("=" * 60)
print("Step 1: Re-extracting CUAD from raw file")
print("=" * 60)

with open("data/raw/CUAD_v1.json", "r", encoding="utf-8") as f:
    cuad_raw = json.load(f)

def clean_label(question):
    label = question
    if '"' in question:
        start = question.find('"') + 1
        end = question.find('"', start)
        if end > start:
            label = question[start:end]
    label = label.strip().rstrip("?").strip()
    return label

fixed_records = []
skipped = 0

for article in cuad_raw["data"]:
    for para in article["paragraphs"]:
        for qa in para["qas"]:
            if qa["is_impossible"] or not qa["answers"]:
                skipped += 1
                continue

            answer_text = qa["answers"][0]["text"].strip()

            if len(answer_text.split()) < 5:
                skipped += 1
                continue

            label = clean_label(qa["question"])

            fixed_records.append({
                "clause_text": answer_text,
                "label":       label,
                "source":      "CUAD"
            })

print(f"Fixed CUAD records:  {len(fixed_records)}")
print(f"Skipped (no answer): {skipped}")

lengths = [len(r["clause_text"].split()) for r in fixed_records]
print(f"Avg clause length:   {sum(lengths) // len(lengths)} words")
print(f"Max clause length:   {max(lengths)} words")
print(f"Min clause length:   {min(lengths)} words")

print("\n--- Sample fixed CUAD records ---")
for r in random.sample(fixed_records, 3):
    print(f"\n  Label: {r['label']}")
    print(f"  Text:  {r['clause_text'][:200]}")

print("\n" + "=" * 60)
print("Step 2: Loading clean LEDGAR (named labels)")
print("=" * 60)

with open("data/processed/ledgar_processed_named.json", "r", encoding="utf-8") as f:
    ledgar_records = json.load(f)

print(f"LEDGAR records: {len(ledgar_records)}")

# Verify no numeric labels remain
numeric_check = [r for r in ledgar_records if str(r["label"]).isdigit()]
print(f"Numeric labels remaining: {len(numeric_check)} (should be 0)")

print("\n" + "=" * 60)
print("Step 3: Combining and balancing")
print("=" * 60)

all_records = fixed_records + ledgar_records
print(f"Total combined:  {len(all_records)}")
print(f"Unique labels:   {len(set(r['label'] for r in all_records))}")

# Balance — cap at 150, oversample below 50
grouped = defaultdict(list)
for r in all_records:
    grouped[r["label"]].append(r)

balanced = []
for label, records in grouped.items():
    if len(records) < 50:
        oversampled = (records * (50 // len(records) + 1))[:50]
        balanced.extend(oversampled)
    elif len(records) > 150:
        balanced.extend(random.sample(records, 150))
    else:
        balanced.extend(records)

random.shuffle(balanced)

label_counts = Counter(r["label"] for r in balanced)
print(f"Balanced dataset:  {len(balanced)} records")
print(f"Unique labels:     {len(label_counts)}")
print(f"Min samples/label: {min(label_counts.values())}")
print(f"Max samples/label: {max(label_counts.values())}")
print(f"Avg samples/label: {sum(label_counts.values()) // len(label_counts)}")

print("\n--- Sample records from fixed dataset ---")
for r in random.sample(balanced, 5):
    print(f"\n  Source: {r['source']}")
    print(f"  Label:  {r['label']}")
    print(f"  Text:   {r['clause_text'][:150]}")

print("\n" + "=" * 60)
print("Step 4: Saving fixed dataset")
print("=" * 60)

out_path = "data/processed/combined_dataset_fixed.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(balanced, f, ensure_ascii=False, indent=2)

print(f"Saved to {out_path}")

# Final sanity check
with open(out_path, "r", encoding="utf-8") as f:
    verify = json.load(f)

numeric_labels = [r for r in verify if str(r["label"]).isdigit()]
print(f"Verified records:       {len(verify)}")
print(f"Numeric labels in final: {len(numeric_labels)} (should be 0)")
print(f"\nAll done! Use '{out_path}' for Legal-BERT training.")