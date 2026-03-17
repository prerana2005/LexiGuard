import json
from collections import Counter

print("=" * 60)
print("Verifying combined_dataset_fixed.json")
print("=" * 60)

with open("data/processed/combined_dataset_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Basic stats
labels       = [r["label"] for r in data]
sources      = [r["source"] for r in data]
lengths      = [len(r["clause_text"].split()) for r in data]
numeric      = [l for l in labels if str(l).isdigit()]
truncated    = [r for r in data if len(r["clause_text"].split()) == 400]

print(f"Total records:        {len(data)}")
print(f"Unique labels:        {len(set(labels))}")
print(f"Numeric labels:       {len(numeric)}  (should be 0)")
print(f"Truncated (400w):     {len(truncated)}  (should be ~0)")
print(f"Avg clause length:    {sum(lengths) // len(lengths)} words")
print(f"Min clause length:    {min(lengths)} words")
print(f"Max clause length:    {max(lengths)} words")

print("\n--- Source breakdown ---")
for source, count in Counter(sources).most_common():
    print(f"  {source}: {count} records")

print("\n--- Label balance check ---")
label_counts = Counter(labels)
print(f"  Min samples/label:  {min(label_counts.values())}")
print(f"  Max samples/label:  {max(label_counts.values())}")
print(f"  Avg samples/label:  {sum(label_counts.values()) // len(label_counts)}")

print("\n--- Sample records (one per source) ---")
seen = set()
for r in data:
    if r["source"] not in seen:
        seen.add(r["source"])
        print(f"\n  Source: {r['source']}")
        print(f"  Label:  {r['label']}")
        print(f"  Text:   {r['clause_text'][:200]}")

print("\n--- Top 10 labels ---")
for label, count in label_counts.most_common(10):
    print(f"  {count:>5}  {label}")

print("\n--- Bottom 10 labels ---")
for label, count in label_counts.most_common()[:-11:-1]:
    print(f"  {count:>5}  {label}")