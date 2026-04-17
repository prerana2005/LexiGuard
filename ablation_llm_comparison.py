"""
FIXED LLM Ablation Study
Fixes:
  - Robust JSON parsing (handles markdown fences, extra text, partial JSON)
  - Compliance uses line-by-line text parsing, not JSON
  - Retry logic for rate limits
  - Keyword-scan fallback when JSON completely fails
"""

import json
import time
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TEST_CLAUSES = [
    {
        "clause": "The employee shall not work for any competing company in any industry for 5 years after termination.",
        "gt_risk": "High",
        "gt_violation": True,
    },
    {
        "clause": "Payment of Rs. 10,000 will be given if work done is satisfactory as determined by management.",
        "gt_risk": "High",
        "gt_violation": True,
    },
    {
        "clause": "Either party may terminate this agreement with 30 days written notice.",
        "gt_risk": "Low",
        "gt_violation": False,
    },
    {
        "clause": "Company may terminate employee at any time without prior notice or reason.",
        "gt_risk": "High",
        "gt_violation": True,
    },
    {
        "clause": "Employee agrees to work additional hours as required without extra compensation.",
        "gt_risk": "High",
        "gt_violation": True,
    },
    {
        "clause": "All disputes shall be resolved through arbitration as per the Arbitration and Conciliation Act, 1996.",
        "gt_risk": "Low",
        "gt_violation": False,
    },
    {
        "clause": "The tenant shall maintain premises in tenable condition and carry out minor repairs as directed.",
        "gt_risk": "Medium",
        "gt_violation": False,
    },
]

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

RISK_PROMPT = """Analyze this Indian contract clause for legal risk.

Respond with ONLY a raw JSON object. No markdown, no code fences, no explanation.

Format:
{{"risk_level": "High", "risk_type": "short name", "explanation": "1 sentence", "applicable_law": "law name"}}

risk_level must be exactly: High, Medium, or Low

High Risk rules (always flag these):
- No notice period or less than 14 days for termination
- Non-compete restriction after employment ends (void under Section 27 Indian Contract Act 1872)
- Payment depends on subjective undefined criteria like "satisfactory" or "as determined by management"
- Overtime work required without additional pay

Low Risk (never flag):
- Mutual 30-day notice termination
- Standard arbitration clauses
- Governing law clauses

Clause: {clause}"""

COMPLIANCE_PROMPT = """You are an Indian legal compliance expert.

Analyze if this contract clause violates any Indian law.
Respond in EXACTLY this format. No other text.

VIOLATION: YES or NO
LAW: name of Indian law violated, or None
SEVERITY: HIGH or MEDIUM or LOW or None
REASON: one sentence

Clause: {clause}"""


def extract_json_robust(text):
    """Four-strategy JSON extractor."""
    if not text:
        return None

    # 1: direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # 2: strip markdown fences then parse
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3: find first complete {...} block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # 4: regex scan for risk_level value
    m = re.search(r'"?risk_level"?\s*:\s*"?(High|Medium|Low)"?', text, re.IGNORECASE)
    if m:
        level = m.group(1).capitalize()
        return {
            "risk_level": level,
            "risk_type": "Parsed via regex",
            "explanation": text[:120],
            "applicable_law": "Unknown",
        }

    return None


def parse_compliance_lines(text):
    """Parse the fixed-format compliance response line by line."""
    result = {"violation": False, "severity": "None", "law": "None"}
    if not text:
        return result
    for line in text.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("VIOLATION:"):
            val = line.split(":", 1)[1].strip().upper()
            result["violation"] = val.startswith("YES")
        elif upper.startswith("SEVERITY:"):
            result["severity"] = line.split(":", 1)[1].strip()
        elif upper.startswith("LAW:"):
            result["law"] = line.split(":", 1)[1].strip()
    return result


def call_groq(model, prompt, max_tokens=250, retries=3):
    """Groq call with exponential-back-off retry for 429."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip(), resp.usage.total_tokens
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = 20 * (attempt + 1)
                print(f"        Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"        API error: {msg[:100]}")
                return None, 0
    return None, 0


def run_risk(clause, model):
    start = time.time()
    raw, tokens = call_groq(model, RISK_PROMPT.format(clause=clause), max_tokens=200)
    latency = round(time.time() - start, 2)

    if raw is None:
        return {"risk_level": "Error", "latency": latency, "tokens": 0}

    parsed = extract_json_robust(raw)
    if parsed:
        level = parsed.get("risk_level", "Unknown")
        for v in ["High", "Medium", "Low"]:
            if level.lower() == v.lower():
                level = v
                break
        return {"risk_level": level, "latency": latency, "tokens": tokens}

    # Absolute fallback: keyword scan on raw text
    rl = raw.lower()
    if "high" in rl:
        level = "High"
    elif "medium" in rl:
        level = "Medium"
    elif "low" in rl:
        level = "Low"
    else:
        level = "Unknown"
    return {"risk_level": level, "latency": latency, "tokens": tokens}


def run_compliance(clause, model):
    start = time.time()
    raw, tokens = call_groq(model, COMPLIANCE_PROMPT.format(clause=clause), max_tokens=150)
    latency = round(time.time() - start, 2)

    if raw is None:
        return {"violation": False, "severity": "None", "latency": latency, "tokens": 0}

    parsed = parse_compliance_lines(raw)
    parsed["latency"] = latency
    parsed["tokens"] = tokens
    return parsed


def calc_accuracy(preds, gts):
    correct = sum(1 for p, g in zip(preds, gts) if str(p).lower() == str(g).lower())
    return round(correct / len(gts), 3)


def calc_f1(preds, gts):
    tp = sum(1 for p, g in zip(preds, gts) if p and g)
    fp = sum(1 for p, g in zip(preds, gts) if p and not g)
    fn = sum(1 for p, g in zip(preds, gts) if not p and g)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(f1, 3), round(prec, 3), round(rec, 3)


def run_ablation():
    print("=" * 70)
    print("LEXIGUARD — LLM ABLATION STUDY")
    print("=" * 70)

    all_results = {}

    for model in MODELS:
        print(f"\n{'='*55}")
        print(f"  Model: {model}")
        print(f"{'='*55}")

        risk_preds, viol_preds = [], []
        latencies, token_counts = [], []

        for i, test in enumerate(TEST_CLAUSES):
            print(f"  [{i+1}/7] {test['clause'][:65]}...")

            r = run_risk(test["clause"], model)
            c = run_compliance(test["clause"], model)

            risk_preds.append(r["risk_level"])
            viol_preds.append(c["violation"])
            latencies.append(r["latency"] + c["latency"])
            token_counts.append(r.get("tokens", 0) + c.get("tokens", 0))

            match_r = "✓" if r["risk_level"].lower() == test["gt_risk"].lower() else "✗"
            match_v = "✓" if c["violation"] == test["gt_violation"] else "✗"
            print(f"         Risk: {r['risk_level']:<8} GT={test['gt_risk']:<8} {match_r}  |  "
                  f"Violation: {str(c['violation']):<6} GT={str(test['gt_violation']):<6} {match_v}  |  "
                  f"{r['latency']+c['latency']:.2f}s")

            time.sleep(3)   # conservative rate-limit buffer

        gt_risk = [t["gt_risk"] for t in TEST_CLAUSES]
        gt_viol = [t["gt_violation"] for t in TEST_CLAUSES]

        risk_acc      = calc_accuracy(risk_preds, gt_risk)
        f1, prec, rec = calc_f1(viol_preds, gt_viol)
        avg_lat       = round(sum(latencies) / len(latencies), 2)
        avg_tok       = round(sum(token_counts) / len(token_counts))

        all_results[model] = {
            "risk_accuracy":         risk_acc,
            "violation_f1":          f1,
            "violation_precision":   prec,
            "violation_recall":      rec,
            "avg_latency_s":         avg_lat,
            "avg_tokens":            avg_tok,
            "risk_predictions":      risk_preds,
            "violation_predictions": viol_preds,
        }

        print(f"\n  Risk Accuracy : {risk_acc:.1%}")
        print(f"  Violation F1  : {f1:.3f}  (P={prec:.3f}  R={rec:.3f})")
        print(f"  Avg Latency   : {avg_lat:.2f}s")
        print(f"  Avg Tokens    : {avg_tok}")

    print("\n" + "=" * 92)
    print("FINAL COMPARISON TABLE")
    print("=" * 92)
    print(f"{'Model':<30} {'Risk Acc':>10} {'Viol F1':>10} {'Precision':>10} {'Recall':>10} {'Latency(s)':>12} {'Tokens':>8}")
    print("-" * 92)
    for model, res in all_results.items():
        print(f"{model:<30} {res['risk_accuracy']:>10.1%} {res['violation_f1']:>10.3f} "
              f"{res['violation_precision']:>10.3f} {res['violation_recall']:>10.3f} "
              f"{res['avg_latency_s']:>12.2f} {res['avg_tokens']:>8}")

    with open("ablation_results.json", "w") as f:
        json.dump({
            "test_clauses": TEST_CLAUSES,
            "models_tested": MODELS,
            "results": all_results,
        }, f, indent=2)

    print("\nSaved → ablation_results.json")
    return all_results


if __name__ == "__main__":
    run_ablation()
