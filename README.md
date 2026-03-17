# LexiGuard — AI-Powered Legal Assistant

An intelligent legal AI system for understanding, analyzing, and explaining legal clauses with Indian law context. LexiGuard combines Legal-BERT clause classification, FAISS semantic search over Indian law, and LLaMA-based explanation generation.

---

## 🏆 Results

| Component | Details |
|-----------|---------|
| Classifier | Legal-BERT fine-tuned, 122 clause types |
| Test Accuracy | **94.65%** |
| Test F1 Macro | **0.9417** |
| Knowledge Base | 17,228 Indian law chunks |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| Explanation Model | LLaMA 3.2 3B Instruct |

---

## 📦 Models & Datasets

| Resource | Location |
|----------|----------|
| Classifier model | [Caraxes22/Lexiguard-LegalBert](https://huggingface.co/Caraxes22/Lexiguard-LegalBert) |
| Datasets + embeddings + FAISS index | [Caraxes22/LexiGuard-datasets](https://huggingface.co/datasets/Caraxes22/LexiGuard-datasets) |

### Download all data (run once):
```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Caraxes22/LexiGuard-datasets',
    repo_type='dataset',
    local_dir='.'
)
print('All files downloaded!')
"
```

---

## 🏗️ Project Structure
```
LexiGuard/
├── data/
│   ├── raw/                        # Raw downloaded datasets
│   ├── processed/                  # Cleaned and processed data
│   │   ├── combined_dataset_clean.json   # Final training data (16,648 records, 122 labels)
│   │   ├── ledgar_processed_named.json   # LEDGAR with named labels
│   │   └── indian_laws_chunks_final.json # 17,228 Indian law chunks
│   └── indian_laws/                # Indian law text files (7 Acts)
├── embeddings/                     # MiniLM vector embeddings (25.2 MB)
├── faiss_index/                    # FAISS search index (25.4 MB)
├── scripts/                        # All pipeline scripts
└── notebooks/                      # Kaggle training notebooks
```

---

## ⚙️ Pipeline

### Step 1 — Data Collection
```
download_cuad.py          → Download raw CUAD dataset (SEC contracts)
download_ledgar.py        → Download LEDGAR from HuggingFace
download_indian_laws.py   → Scrape Indian law text from Wikipedia
write_indian_laws.py      → Write manual sections for 4 key Acts
```

### Step 2 — Data Processing
```
fix_ledgar.py             → Convert numeric LEDGAR labels → named labels
fix_cuad_and_rebuild.py   → Re-extract actual CUAD answer text (fix context bug)
analyze_confusions.py     → Merge overlapping labels (138 → 132)
drop_noisy_labels.py      → Drop noisy/generic labels (132 → 122)
```

### Step 3 — Knowledge Base
```
build_final_combined_kb.py → Build 17,228 chunk knowledge base
                             + Generate MiniLM embeddings
                             + Build FAISS index
```

### Step 4 — Training (Kaggle T4)
```
Legal-BERT fine-tuned on combined_dataset_clean.json
122 clause types, 10 epochs, batch size 16, lr=3e-5
Result: 94.65% test accuracy, 0.9417 F1 Macro
```

### Step 5 — Explanation Agent
```
explanation_agent.py → Full RAG pipeline:
  clause text → Legal-BERT classify → FAISS retrieve → LLaMA explain
```

---

## 📊 Training Data

| Source | Records | Labels | Notes |
|--------|---------|--------|-------|
| CUAD | 5,328 | 41 | SEC contract clauses (answer text extracted) |
| LEDGAR | 80,000 | 100 | Legal provisions with named labels |
| **Combined (balanced)** | **16,648** | **122** | Capped 50–150 per label |

### Indian Law Knowledge Base

| Source | Chunks | Type |
|--------|--------|------|
| mratanusarkar/Indian-Laws | 6,029 | Actual bare Act text |
| viber1/indian-law-dataset | 5,701 | Law-specific QA pairs |
| ShreyasP123/Legal-Dataset | 1,333 | Legal document chunks |
| nisaar/Lawyer_GPT_India | 149 | Lawyer QA pairs |
| Manual (7 Acts) | 134 | Hand-written sections |
| **Total** | **17,228** | |

---

## 🔍 Explanation Agent

The explanation agent combines all three components into a single pipeline:
```python
from explanation_agent import load_models, lexiguard_explain
import os

# Load all models
clf_tokenizer, clf_model, id2label, faiss_index, metadata, \
    embedder, llm_pipe, llm_tokenizer = load_models(os.environ["HF_TOKEN"])

# Explain a clause
result = lexiguard_explain(
    "Either party may terminate this Agreement upon 30 days written notice.",
    clf_tokenizer, clf_model, id2label,
    faiss_index, metadata, embedder,
    llm_pipe, llm_tokenizer
)

# Output:
# 🏷️  Clause Type:  Termination For Convenience (97.6% confidence)
# 📚 Relevant Laws: Indian Contract Act, 1872 — Section 73 ...
# 📝 Explanation:   This clause allows either party to end the agreement...
```

---

## 🛠️ Setup
```bash
# Create virtual environment
python -m venv lexiguard
lexiguard\Scripts\activate

# Install dependencies
pip install torch transformers datasets sentence-transformers
pip install faiss-cpu peft trl accelerate bitsandbytes
pip install scikit-learn pandas huggingface-hub
```

---

## 📜 Scripts Reference

| Script | Purpose |
|--------|---------|
| `download_cuad.py` | Download raw CUAD dataset |
| `download_ledgar.py` | Download LEDGAR from HuggingFace |
| `download_indian_laws.py` | Scrape Indian law texts from Wikipedia |
| `write_indian_laws.py` | Write manual Indian Act sections |
| `fix_ledgar.py` | Convert LEDGAR numeric → named labels |
| `fix_cuad_and_rebuild.py` | Re-extract CUAD answer text + rebuild dataset |
| `preprocess_cuad.py` | CUAD preprocessing utilities |
| `analyze_confusions.py` | Analyze and merge overlapping labels |
| `drop_noisy_labels.py` | Drop noisy/generic labels |
| `build_final_combined_kb.py` | Build knowledge base + embeddings + FAISS |
| `build_faiss_index.py` | Build FAISS index from existing embeddings |
| `explanation_agent.py` | Full RAG pipeline — classify + retrieve + explain |
| `upload_to_huggingface.py` | Upload datasets and models to HuggingFace |
| `verify_progress.py` | Verify all files are consistent |
| `Dataset_verify.py` | Verify dataset stats and label balance |

---

## 👥 Team
LexiGuard — Legal AI Project

**Person 1 (Vineet)** — Data Pipeline & Model Training
- Dataset collection, cleaning, and processing
- Indian Law Knowledge Base (17,228 chunks)
- MiniLM embeddings + FAISS index
- Legal-BERT classifier (94.65% accuracy)
- Explanation Agent (RAG pipeline)