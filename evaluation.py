import json
import re
import time
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Import your actual agents
from compliance_agent import compliance_agent, search_relevant_laws

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── LLM Judge ──────────────────────────────────────────────────
def llm_judge(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    text = response.choices[0].message.content.strip()
    matches = re.findall(r'\d+', text)
    if matches:
        return min(max(float(matches[0]) / 9.0, 0.0), 1.0)
    return 0.0


# ── 3 Metrics ──────────────────────────────────────────────────
def context_relevance(clause, retrieved_laws):
    law_text = "\n".join([f"{l['law_name']} - {l['section_number']}: {l['text'][:400]}"
                          for l in retrieved_laws])
    prompt = f"""Rate how relevant the retrieved Indian law sections are to this contract clause.
Score from 1-9. Single digit only.

Clause: {clause}
Retrieved Laws: {law_text}

Score:"""
    return llm_judge(prompt)


def groundedness(retrieved_laws, verdict):
    law_text = "\n".join([f"{l['law_name']} - {l['section_number']}: {l['text'][:200]}"
                          for l in retrieved_laws])
    prompt = f"""Rate how well the compliance verdict is supported by the retrieved law sections.
Score from 1-9. Single digit only.

Retrieved Laws: {law_text}
Compliance Verdict: {verdict}

Score:"""
    return llm_judge(prompt)


def answer_relevance(clause, verdict):
    prompt = f"""Rate how relevant and useful the compliance verdict is for this contract clause.
Score from 1-9. Single digit only.

Clause: {clause}
Verdict: {verdict}

Score:"""
    return llm_judge(prompt)


# ── Run Evaluation ─────────────────────────────────────────────
test_clauses = [
    "Company may terminate this agreement at any time without notice or reason.",
    "Freelancer shall not work for any company in any industry for 5 years after termination.",
    "Payment shall be made if work done is satisfactory as determined by management.",
    "The employee shall maintain confidentiality of all company information indefinitely.",
    "Tenant shall pay damages at two times the rent for any period of overstay.",
    "Either party may terminate with one month written notice.",
    "Employee agrees to work additional hours as required by the employer without extra compensation.",
]

results = []

print("Running LexiGuard RAG Evaluation...")
print("=" * 60)

for i, clause in enumerate(test_clauses):
    print(f"\nClause {i+1}/{len(test_clauses)}: {clause[:60]}...")

    # Get compliance result
    result = compliance_agent(clause)
    retrieved_laws = search_relevant_laws(clause, top_k=7)

    verdict = f"Violation: {result['violation']}. {result['explanation']}"

    # Score all 3 metrics
    cr = context_relevance(clause, retrieved_laws)
    gr = groundedness(retrieved_laws, verdict)
    ar = answer_relevance(clause, verdict)

    results.append({
        "clause": clause[:80],
        "violation_found": result['violation'],
        "law_cited": result['law_violated'],
        "context_relevance": round(cr, 2),
        "groundedness": round(gr, 2),
        "answer_relevance": round(ar, 2)
    })

    print(f"  Context Relevance: {cr:.2f}")
    print(f"  Groundedness:      {gr:.2f}")
    print(f"  Answer Relevance:  {ar:.2f}")

    # Small delay to avoid rate limit
    time.sleep(3)

# ── Print Summary ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

avg_cr = sum(r['context_relevance'] for r in results) / len(results)
avg_gr = sum(r['groundedness'] for r in results) / len(results)
avg_ar = sum(r['answer_relevance'] for r in results) / len(results)

print(f"Average Context Relevance : {avg_cr:.2f}  (target > 0.70)")
print(f"Average Groundedness      : {avg_gr:.2f}  (target > 0.80)")
print(f"Average Answer Relevance  : {avg_ar:.2f}  (target > 0.70)")

print("\nPer-clause Results:")
print(f"{'Clause':<45} {'CR':>5} {'GR':>5} {'AR':>5} {'Violation':>10}")
print("-" * 75)
for r in results:
    print(f"{r['clause']:<45} {r['context_relevance']:>5.2f} {r['groundedness']:>5.2f} {r['answer_relevance']:>5.2f} {str(r['violation_found']):>10}")

# Save to JSON for research paper
with open("evaluation_results.json", "w") as f:
    json.dump({
        "averages": {
            "context_relevance": round(avg_cr, 2),
            "groundedness": round(avg_gr, 2),
            "answer_relevance": round(avg_ar, 2)
        },
        "per_clause": results
    }, f, indent=2)

print("\nResults saved to evaluation_results.json")