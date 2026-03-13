# LexiGuard — AI-Powered Legal Assistant

An intelligent legal AI system for understanding, analyzing, 
and explaining legal clauses with Indian law context.

## Project Structure
```
LexiGuard/
├── data/
│   ├── raw/                  # Raw downloaded datasets
│   ├── processed/            # Cleaned and processed data
│   └── indian_laws/          # Indian law text files
├── embeddings/               # Vector embeddings
├── faiss_index/              # FAISS search index
├── models/                   # Fine-tuned model weights
├── scripts/                  # All pipeline scripts
├── evaluation/               # Evaluation results
└── logs/                     # Training logs
```

## Pipeline

### Data Collection & Processing
- **CUAD Dataset** — 6,478 labeled contract clauses from SEC filings
- **LEDGAR Dataset** — 80,000 legal provisions with clause types
- **Indian Laws** — 17,228 chunks from multiple sources:
  - Arbitration & Conciliation Act 1996
  - Consumer Protection Act 2019
  - Real Estate (RERA) Act 2016
  - Information Technology Act 2000
  - Indian Contract Act 1872
  - Code on Wages 2019
  - DPDP Act 2023
  - Companies Act 2013
  - Copyright Act 1957
  - And more...

### Vector Search
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: FAISS (17,228 vectors, 384 dimensions)
- Semantic search across all Indian law chunks

### Fine-Tuning
- Base model: LLaMA-3.2-3B
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Task: Legal clause classification + explanation

## Scripts

| Script | Purpose |
|--------|---------|
| `download_cuad.py` | Download CUAD dataset |
| `preprocess_cuad.py` | Clean and format CUAD |
| `download_ledgar.py` | Download LEDGAR dataset |
| `combine_datasets.py` | Merge all training data |
| `fix_combined_dataset.py` | Clean labels and truncate |
| `download_indian_laws.py` | Scrape Indian law texts |
| `write_indian_laws.py` | Write manual law sections |
| `build_final_combined_kb.py` | Build complete knowledge base |
| `generate_embeddings.py` | Generate MiniLM embeddings |
| `build_faiss_index.py` | Build FAISS vector index |
| `verify_final_state.py` | Verify all files consistent |

## Setup
```bash
# Create virtual environment
python -m venv lexiguard
lexiguard\Scripts\activate

# Install dependencies
pip install torch transformers datasets sentence-transformers
pip install faiss-cpu peft trl accelerate bitsandbytes
pip install scikit-learn pandas kaggle huggingface-hub
```

## Team
- LexiGuard — Legal AI Project