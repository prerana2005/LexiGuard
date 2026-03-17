
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import json
import torch
import numpy as np
import faiss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download, login

# ── Config ─────────────────────────────────────────────────────
CLASSIFIER_REPO = "Caraxes22/Lexiguard-LegalBert"
DATASET_REPO    = "Caraxes22/LexiGuard-datasets"
LLAMA_REPO      = "meta-llama/Llama-3.2-3B-Instruct"

GOOD_SOURCES  = {
    "mratanusarkar/Indian-Laws",
    "ShreyasP123/Legal-Dataset-for-india",
    "nisaar/Lawyer_GPT_India",
    "manual"
}
BAD_LAW_NAMES = {"Indian_Law", "Indian_Law_QA", "Indian_Legal"}

# ── Load models ────────────────────────────────────────────────
def load_models(hf_token):
    login(token=hf_token)

    print("Loading Legal-BERT classifier...")
    clf_tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_REPO, token=hf_token)
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_REPO, token=hf_token)
    clf_model.eval()
    if torch.cuda.is_available():
        clf_model = clf_model.cuda()

    label_map_path = hf_hub_download(
        repo_id=CLASSIFIER_REPO, filename="label_map.json",
        repo_type="model", token=hf_token)
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    print("Loading FAISS index...")
    index_path = hf_hub_download(
        repo_id=DATASET_REPO, filename="faiss_index/indian_laws.index",
        repo_type="dataset", token=hf_token)
    faiss_index = faiss.read_index(index_path)

    metadata_path = hf_hub_download(
        repo_id=DATASET_REPO, filename="faiss_index/indian_laws_metadata.json",
        repo_type="dataset", token=hf_token)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Loading MiniLM embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading LLaMA 3.2 3B...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLAMA_REPO, token=hf_token)
    llm_model     = AutoModelForCausalLM.from_pretrained(
        LLAMA_REPO, token=hf_token,
        torch_dtype=torch.float16, device_map="auto")
    llm_pipe = pipeline(
        "text-generation", model=llm_model, tokenizer=llm_tokenizer)

    print("All models loaded!")
    return clf_tokenizer, clf_model, id2label, faiss_index, metadata, embedder, llm_pipe, llm_tokenizer

# ── Classify ───────────────────────────────────────────────────
def classify_clause(clause_text, clf_tokenizer, clf_model, id2label, top_n=3):
    inputs = clf_tokenizer(
        clause_text, truncation=True, padding=True,
        max_length=512, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clf_model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1)[0]
    top_indices = torch.argsort(probs, descending=True)[:top_n]
    top_preds   = [(id2label[i.item()], round(probs[i].item(), 4))
                   for i in top_indices]
    return top_preds[0][0], top_preds[0][1], top_preds

# ── Retrieve ───────────────────────────────────────────────────
def retrieve_indian_law(query_text, clause_type, faiss_index,
                        metadata, embedder, top_k=5, confidence=1.0):
    if confidence >= 0.70:
        search_query = f"{clause_type}: {query_text}"
    else:
        search_query = query_text

    query_embedding = embedder.encode(
        [search_query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_index.search(query_embedding, top_k * 20)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk    = metadata[idx]
        source   = chunk.get("source",   "")
        law_name = chunk.get("law_name", "")
        text     = chunk.get("text",     "")
        if source not in GOOD_SOURCES:
            continue
        if law_name in BAD_LAW_NAMES:
            continue
        if len(text.split()) < 15:
            continue
        results.append({
            "law":     law_name,
            "section": chunk.get("section_number", ""),
            "topic":   chunk.get("topic", ""),
            "text":    text,
            "score":   round(float(score), 4)
        })
        if len(results) == top_k:
            break

    # Fallback
    if not results:
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                chunk = metadata[idx]
                results.append({
                    "law":     chunk.get("law_name", "Unknown"),
                    "section": chunk.get("section_number", ""),
                    "topic":   chunk.get("topic", ""),
                    "text":    chunk.get("text", ""),
                    "score":   round(float(score), 4)
                })
            if len(results) == top_k:
                break

    return results

# ── Explain ────────────────────────────────────────────────────
def generate_explanation(clause_text, clause_type,
                         retrieved_laws, llm_pipe, llm_tokenizer):
    law_context = ""
    for i, law in enumerate(retrieved_laws, 1):
        law_context += f"\n{i}. {law['law']} — {law['section']} {law['topic']}\n"
        law_context += f"   {law['text'][:300]}\n"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are LexiGuard, an AI legal assistant specializing in Indian law.
Explain legal clauses clearly and relate them to relevant Indian laws.
Always complete your full response including all risk flags.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
CLAUSE TYPE: {clause_type}

CLAUSE TEXT:
{clause_text}

RELEVANT INDIAN LAW SECTIONS:
{law_context}

Please provide:
1. Plain English explanation
2. Rights and obligations created
3. Relevant Indian law that applies
4. Risk flags to watch out for
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>"""

    output = llm_pipe(
        prompt,
        max_new_tokens     = 1000,
        max_length         = None,
        temperature        = 0.3,
        do_sample          = True,
        pad_token_id       = llm_tokenizer.eos_token_id,
        eos_token_id       = llm_tokenizer.eos_token_id,
        repetition_penalty = 1.1,
    )

    full_output    = output[0]["generated_text"]
    response_start = full_output.find("<|start_header_id|>assistant<|end_header_id|>")
    if response_start != -1:
        response = full_output[response_start:].replace(
            "<|start_header_id|>assistant<|end_header_id|>", "").strip()
    else:
        response = full_output[len(prompt):].strip()

    return response

# ── Main pipeline ──────────────────────────────────────────────
def lexiguard_explain(clause_text, clf_tokenizer, clf_model, id2label,
                      faiss_index, metadata, embedder, llm_pipe, llm_tokenizer):
    print("=" * 60)
    print("LEXIGUARD — Legal Clause Explanation")
    print("=" * 60)

    print("\n📋 Step 1: Classifying clause...")
    clause_type, confidence, top_preds = classify_clause(
        clause_text, clf_tokenizer, clf_model, id2label)
    print(f"   Clause Type: {clause_type} ({confidence:.1%})")
    if confidence < 0.70:
        print(f"   ⚠️  Low confidence — top alternatives:")
        for label, score in top_preds[1:]:
            print(f"      • {label} ({score:.1%})")

    print("\n🔍 Step 2: Retrieving relevant Indian law...")
    retrieved_laws = retrieve_indian_law(
        clause_text, clause_type, faiss_index,
        metadata, embedder, top_k=5, confidence=confidence)
    for law in retrieved_laws:
        print(f"   [{law['score']:.3f}] {law['law']} — {law['section']}")

    print("\n💡 Step 3: Generating explanation...")
    explanation = generate_explanation(
        clause_text, clause_type, retrieved_laws, llm_pipe, llm_tokenizer)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"\n🏷️  Clause Type:  {clause_type} ({confidence:.1%} confidence)")
    if confidence < 0.70:
        print(f"   ⚠️  Low confidence — alternatives:")
        for label, score in top_preds[1:]:
            print(f"      • {label} ({score:.1%})")
    print(f"\n📚 Relevant Laws:")
    for law in retrieved_laws:
        print(f"   • {law['law']} — {law['section']}")
    print(f"\n📝 Explanation:\n{explanation}")
    print("=" * 60)

    return {
        "clause_type":     clause_type,
        "confidence":      confidence,
        "top_predictions": top_preds,
        "retrieved_laws":  retrieved_laws,
        "explanation":     explanation
    }
