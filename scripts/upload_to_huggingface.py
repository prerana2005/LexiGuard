from huggingface_hub import HfApi, login
import os

HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
login(token=HF_TOKEN)

api = HfApi()

print("Uploading combined_dataset_clean.json...")
api.upload_file(
    path_or_fileobj = "data/processed/combined_dataset_clean.json",
    path_in_repo    = "processed/combined_dataset_clean.json",
    repo_id         = "Caraxes22/LexiGuard-datasets",
    repo_type       = "dataset",
    token           = HF_TOKEN
)
print("✅ Done!")