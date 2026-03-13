from datasets import load_dataset
import json
import os

print("📥 Loading LEDGAR dataset from HuggingFace...")

# LEDGAR is available as a standard parquet dataset — no scripts needed
dataset = load_dataset("lex_glue", "ledgar")

print(f"✅ Loaded!")
print(f"   Train : {len(dataset['train'])}")
print(f"   Valid : {len(dataset['validation'])}")
print(f"   Test  : {len(dataset['test'])}")

# Preview structure
sample = dataset['train'][0]
print(f"\n🔍 Keys    : {list(sample.keys())}")
print(f"   Text   : {sample['text'][:150]}")
print(f"   Label  : {sample['label']}")

# Save processed
os.makedirs("data/processed", exist_ok=True)

processed = []
for split in ['train', 'validation', 'test']:
    for item in dataset[split]:
        text = item['text'].strip()
        label = item['label']
        if len(text) > 20:
            processed.append({
                "clause_text": text,
                "label": str(label),
                "source": "LEDGAR"
            })

out_path = "data/processed/ledgar_processed.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(processed)} records to {out_path}")
print(f"\n📄 Sample:")
print(f"   clause_text : {processed[0]['clause_text'][:150]}")
print(f"   label       : {processed[0]['label']}")