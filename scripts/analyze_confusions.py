import json
import random
from collections import defaultdict, Counter

with open("data/processed/combined_dataset_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Before merge: {len(data)} records, {len(set(r['label'] for r in data))} labels")

# SAFE merges only — where text content is genuinely the same
MERGE_MAP = {
    # Governing law — identical clause, just singular/plural naming
    "Governing Laws":   "Governing Law",

    # Date labels — all just dates in contracts
    "Effective Date":   "Agreement Date",
    "Expiration Date":  "Agreement Date",

    # License family — all describe license grants with different scope
    "Non-Transferable License":         "License Grant",
    "Irrevocable Or Perpetual License": "License Grant",

    # Renewal/termination notice — same concept
    "Notice Period To Terminate Renewal": "Renewal Term",
}

# Apply merges
for record in data:
    record["label"] = MERGE_MAP.get(record["label"], record["label"])

print(f"After merge:  {len(data)} records, {len(set(r['label'] for r in data))} labels")

# Re-balance after merging
# (merged labels now have more samples — need to re-cap)
grouped = defaultdict(list)
for r in data:
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
print(f"\nBalanced:      {len(balanced)} records")
print(f"Unique labels: {len(label_counts)}")
print(f"Min/Max/Avg:   {min(label_counts.values())} / {max(label_counts.values())} / {sum(label_counts.values())//len(label_counts)}")

# Verify merges worked
print("\n--- Verifying merges ---")
for old, new in MERGE_MAP.items():
    old_count = sum(1 for r in balanced if r["label"] == old)
    new_count = sum(1 for r in balanced if r["label"] == new)
    print(f"  '{old}' → '{new}': old={old_count} (should be 0), new={new_count}")

# Save
out_path = "data/processed/combined_dataset_merged.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(balanced, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {out_path}")