from huggingface_hub import hf_hub_download

# Download v3 FAISS index
hf_hub_download(
    repo_id="Caraxes22/Lexiguard_dataset_v2",
    repo_type="dataset",
    filename="faiss_index/indian_laws_v3.index",
    local_dir="."
)

# Download v3 metadata
hf_hub_download(
    repo_id="Caraxes22/Lexiguard_dataset_v2",
    repo_type="dataset",
    filename="embeddings/indian_laws_metadata_v3.json",
    local_dir="."
)

print("Downloaded v3 files!")