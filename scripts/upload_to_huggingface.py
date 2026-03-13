from huggingface_hub import HfApi
import os

api      = HfApi()
REPO_ID  = "caraxes22/LexiGuard-datasets"
REPO_TYPE = "dataset"

print("🚀 Uploading LexiGuard datasets to HuggingFace...")
print(f"   Repo: {REPO_ID}\n")

# ── FILES TO UPLOAD ───────────────────────────────────────────
files = [
    # (local path, path in HF repo)

    # Processed datasets
    ("data/processed/combined_dataset.json",
     "processed/combined_dataset.json"),

    ("data/processed/indian_laws_chunks_final.json",
     "processed/indian_laws_chunks_final.json"),

    # Embeddings
    ("embeddings/indian_laws_embeddings.npy",
     "embeddings/indian_laws_embeddings.npy"),

    ("embeddings/indian_laws_metadata.json",
     "embeddings/indian_laws_metadata.json"),

    # FAISS index
    ("faiss_index/indian_laws.index",
     "faiss_index/indian_laws.index"),

    ("faiss_index/indian_laws_metadata.json",
     "faiss_index/indian_laws_metadata.json"),

    # Indian law text files
    ("data/indian_laws/Arbitration_Act_1996.txt",
     "indian_laws/Arbitration_Act_1996.txt"),

    ("data/indian_laws/Consumer_Protection_Act_2019.txt",
     "indian_laws/Consumer_Protection_Act_2019.txt"),

    ("data/indian_laws/DPDP_Act_2023.txt",
     "indian_laws/DPDP_Act_2023.txt"),

    ("data/indian_laws/IT_Act_2000.txt",
     "indian_laws/IT_Act_2000.txt"),

    ("data/indian_laws/Indian_Contract_Act_1872.txt",
     "indian_laws/Indian_Contract_Act_1872.txt"),

    ("data/indian_laws/Labour_Codes_2020.txt",
     "indian_laws/Labour_Codes_2020.txt"),

    ("data/indian_laws/RERA_2016.txt",
     "indian_laws/RERA_2016.txt"),
]

# ── UPLOAD EACH FILE ──────────────────────────────────────────
total  = len(files)
failed = []

for i, (local_path, hf_path) in enumerate(files, 1):
    if not os.path.exists(local_path):
        print(f"   [{i}/{total}] ⚠️  SKIPPED (not found): {local_path}")
        continue

    size = os.path.getsize(local_path) / (1024*1024)
    print(f"   [{i}/{total}] 📤 Uploading {local_path} ({size:.1f} MB)...")

    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"           ✅ Done!")
    except Exception as e:
        print(f"           ❌ Failed: {e}")
        failed.append(local_path)

# ── SUMMARY ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"✅ Uploaded : {total - len(failed)}/{total} files")
if failed:
    print(f"❌ Failed   : {len(failed)} files")
    for f in failed:
        print(f"   - {f}")
else:
    print(f"🎉 All files uploaded successfully!")

print(f"\n🔗 View at: https://huggingface.co/datasets/{REPO_ID}")
print("="*55)