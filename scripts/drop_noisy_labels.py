import json
import random
from collections import defaultdict, Counter

with open("data/processed/combined_dataset_merged.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Before: {len(data)} records, {len(set(r['label'] for r in data))} labels")

# Labels where LEDGAR assigned wrong text content
# Confirmed by looking at actual samples in analyze_confusions.py
NOISY_LABELS = {
    "Applicable Laws",        # samples showed choice-of-law text but labelled wrong
    "Works Councils",         # EU labor concept, text doesn't match
    "Records",                # too few samples + generic text
    "Authorizations",         # only 7 samples, consistently wrong
    "Competitive Restriction Exception",  # too few + confused with Non-Compete
    "Interests",              # generic financial term, text all over the place
    "General",                # literally a catch-all — meaningless label
    "Miscellaneous",          # same as General — catch-all
    "Construction",           # legal construction rules, confused with other labels
    "Operations",             # too generic, overlaps with many labels
}

# Filter out noisy labels
clean_data = [r for r in data if r["label"] not in NOISY_LABELS]

print(f"Dropped {len(data) - len(clean_data)} records from {len(NOISY_LABELS)} noisy labels")
print(f"After:  {len(clean_data)} records, {len(set(r['label'] for r in clean_data))} labels")

# Re-balance
grouped = defaultdict(list)
for r in clean_data:
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

# Show remaining problem labels (< 50 real samples)
print("\n--- Labels with few real samples ---")
real_counts = Counter(r["label"] for r in clean_data)
for label, count in sorted(real_counts.items(), key=lambda x: x[1]):
    if count < 60:
        print(f"  {count:>5}  {label}")

# Save
out_path = "data/processed/combined_dataset_clean.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(balanced, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {out_path}")