# LexiGuard — AI-Powered Indian Contract Analysis System

LexiGuard is an end-to-end AI system for analyzing legal contracts under Indian law. It combines Legal-BERT classification, Hybrid Retrieval-Augmented Generation (RAG), and a multi-agent architecture to detect risks, ambiguities, and legal violations at the clause level.

---

## 🏆 Key Results

| Component | Details |
|----------|--------|
| Clause Classifier | Legal-BERT (122 classes) |
| Test Accuracy | **94.65%** |
| F1 Macro | **0.9417** |
| Knowledge Base | **4,836 verified statutory chunks** |
| Context Relevance | **0.67 (↑ from 0.21)** |
| Groundedness | **0.86** |
| Answer Relevance | **0.89** |
| Best Compliance Model | LLaMA 3.1 8B (F1 = **1.000**) |

---

## 🧠 System Overview

LexiGuard processes contracts through five stages:

1. Document Ingestion  
   PDF parsing via `pdfplumber` and OCR via Groq Vision API  

2. Clause Segmentation  
   Regex-based clause splitting with fallback strategies  

3. Clause Classification  
   Legal-BERT predicts clause type across 122 categories  

4. Hybrid RAG Retrieval  
   BM25 (keyword search) + FAISS (semantic search) + RRF + CRAG  

5. Multi-Agent Analysis  
   Risk, Ambiguity, Compliance, and Explanation agents  

---

## 🤖 Agent Architecture

| Agent | Function | Model |
|------|---------|------|
| Risk Agent | Detects high/medium/low risk clauses | LLaMA 3.1 8B |
| Ambiguity Agent | Identifies vague or exploitable terms | LLaMA 3.1 8B |
| Compliance Agent | Detects legal violations with statute references | LLaMA 3.1 8B |
| Explanation Agent | Generates plain-English explanations | LLaMA 3.3 70B |

---

## 🔍 Hybrid RAG Pipeline

- Knowledge base: 4,836 high-quality Indian law chunks  
- Removed ~12,000 noisy entries to improve retrieval  
- Retrieval methods:
  - BM25 (keyword matching)
  - FAISS (semantic similarity)
  - RRF fusion for ranking
  - CRAG fallback for low-confidence queries  

Hybrid + CRAG improves robustness and achieves higher law recall (~0.60).

---

## 📊 Evaluation

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| Context Relevance | 0.21 | 0.67 | ↑ 3.2× |
| Groundedness | 0.80 | 0.86 | ↑ +0.06 |
| Answer Relevance | 0.83 | 0.89 | ↑ +0.06 |

**Insight:** Improvements are driven by better data quality and retrieval strategy rather than model size.

---

## 🧪 Ablation Studies

### LLM Backbone (Agents)

| Model | Risk Accuracy | Violation F1 | Latency |
|------|--------------|-------------|--------|
| LLaMA 3.1 8B | 85.7% | **1.000** | 1.08s |
| LLaMA 3.3 70B | **100%** | 0.857 | 1.35s |

**Conclusion:** Smaller models perform better for structured compliance detection due to better prompt adherence.

---

### Retrieval Strategy

| Strategy | Law Recall | Notes |
|----------|-----------|------|
| BM25 Only | 0.20 | Keyword mismatch issues |
| FAISS Only | 0.60 | Good semantic retrieval |
| Hybrid RRF | 0.20 | Noise reduces performance |
| Hybrid + CRAG | **0.60** | Best robustness |

**Insight:** Hybrid + CRAG handles ambiguous legal queries more effectively.

---

## 📦 Dataset

- Combined CUAD + LEDGAR datasets  
- Final dataset: **16,648 records, 122 clause types**  
- Class-balanced (50–150 samples per class)  

### Key Fixes
- CUAD bug: incorrect full-text extraction → fixed  
- LEDGAR labels: numeric → mapped to names  
- Removed noisy and irrelevant labels  

---

## 🛠️ Tech Stack

- PDF Parsing: pdfplumber  
- OCR: Groq Vision API  
- Embeddings: all-MiniLM-L6-v2  
- Vector Search: FAISS  
- Keyword Search: BM25  
- Backend: FastAPI  
- Frontend: Streamlit  
- Models: Legal-BERT + LLaMA  

---

## ⚙️ Setup

```bash
# Create virtual environment
python -m venv lexiguard
source lexiguard/bin/activate  # or lexiguard\Scripts\activate (Windows)

# Install dependencies
pip install torch transformers datasets sentence-transformers
pip install faiss-cpu peft trl accelerate bitsandbytes
pip install scikit-learn pandas huggingface-hub
```
