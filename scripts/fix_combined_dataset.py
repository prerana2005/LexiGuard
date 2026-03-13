import json
import re
from collections import Counter

LEDGAR_LABELS = {
    0:"Adjustments", 1:"Agreements", 2:"Amendments", 3:"Announcements",
    4:"Arbitration", 5:"Assignments", 6:"Assigns", 7:"Authority",
    8:"Authorizations", 9:"Base Salary", 10:"Benefits", 11:"Binding Effects",
    12:"Books", 13:"Brokers", 14:"Capitalization", 15:"Change In Control",
    16:"Closings", 17:"Competitive Restrictions", 18:"Compliance With Laws",
    19:"Confidentiality", 20:"Consent To Jurisdiction", 21:"Consents",
    22:"Construction", 23:"Cooperation", 24:"Costs", 25:"Counterparts",
    26:"Death", 27:"Defined Terms", 28:"Definitions", 29:"Disability",
    30:"Disclosures", 31:"Duties", 32:"Effective Dates", 33:"Effectiveness",
    34:"Employment", 35:"Enforceability", 36:"Entire Agreements",
    37:"Erisa", 38:"Existence", 39:"Expenses", 40:"Fees",
    41:"Financial Statements", 42:"Forfeitures", 43:"Further Assurances",
    44:"General", 45:"Governing Laws", 46:"Grants", 47:"Headings",
    48:"Indemnifications", 49:"Indemnity", 50:"Insurance",
    51:"Integration", 52:"Intellectual Property", 53:"Interests",
    54:"Interpretations", 55:"Jurisdictions", 56:"Liens",
    57:"Limitations Of Liability", 58:"Litigation", 59:"Miscellaneous",
    60:"Modifications", 61:"No Conflicts", 62:"No Defaults",
    63:"No Waivers", 64:"Non-Competition", 65:"Non-Disparagement",
    66:"Non-Solicitation", 67:"Notices", 68:"Operations",
    69:"Organizations", 70:"Payments", 71:"Positions",
    72:"Powers Of Attorney", 73:"Publicity", 74:"Qualifications",
    75:"Records", 76:"Releases", 77:"Remedies", 78:"Representations",
    79:"Resignations", 80:"Restrictions", 81:"Results Of Operations",
    82:"Sanctions", 83:"Securities", 84:"Severability",
    85:"Solvency", 86:"Specific Performance", 87:"Submission To Jurisdiction",
    88:"Subsidiaries", 89:"Successors", 90:"Survival",
    91:"Tax Withholdings", 92:"Taxes", 93:"Terminations",
    94:"Terms", 95:"Titles", 96:"Transactions With Affiliates",
    97:"Vesting", 98:"Waivers", 99:"Warranties"
}

MAX_WORDS = 400

def extract_cuad_label(raw_label):
    """Extract clean clause type from CUAD question format."""
    # Pattern: 'Highlight the parts ... related to "Clause Type" ...'
    match = re.search(r'"([^"]+)"', raw_label)
    if match:
        return match.group(1).strip()
    return raw_label.strip()[:60]

def truncate_text(text, max_words=MAX_WORDS):
    """Truncate text to max_words words."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text

print("📂 Loading combined dataset...")
with open("data/processed/combined_dataset.json", "r", encoding="utf-8") as f:
    combined = json.load(f)
print(f"   Original records: {len(combined):,}")

fixed = []
cuad_truncated   = 0
label_fixed_cuad = 0
label_fixed_ledg = 0

for item in combined:
    source      = item.get("source", "")
    clause_text = item.get("clause_text", "").strip()
    label       = item.get("label", "").strip()

    # Fix 1: Truncate long clause texts
    words = clause_text.split()
    if len(words) > MAX_WORDS:
        clause_text = truncate_text(clause_text, MAX_WORDS)
        cuad_truncated += 1

    # Fix 2: Clean CUAD labels
    if source == "CUAD":
        clean_label = extract_cuad_label(label)
        if clean_label != label:
            label = clean_label
            label_fixed_cuad += 1

    # Fix 3: Convert LEDGAR numeric labels
    if source == "LEDGAR":
        try:
            label_id = int(label)
            label    = LEDGAR_LABELS.get(label_id, f"Label_{label_id}")
            label_fixed_ledg += 1
        except ValueError:
            pass  # Already a string label

    fixed.append({
        "clause_text": clause_text,
        "label"      : label,
        "source"     : source
    })

# Save fixed dataset
out_path = "data/processed/combined_dataset.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(fixed, f, indent=2, ensure_ascii=False)

print(f"\n✅ Fixes applied:")
print(f"   Texts truncated      : {cuad_truncated:,}")
print(f"   CUAD labels cleaned  : {label_fixed_cuad:,}")
print(f"   LEDGAR labels fixed  : {label_fixed_ledg:,}")

# Verify
print(f"\n📊 After fix stats:")
clause_lens = [len(r["clause_text"].split()) for r in fixed]
labels      = Counter(r["label"] for r in fixed)
print(f"   Total records        : {len(fixed):,}")
print(f"   Avg clause length    : {sum(clause_lens)//len(clause_lens)} words")
print(f"   Max clause length    : {max(clause_lens)} words")
print(f"   Unique labels        : {len(labels)}")
print(f"\n   Top 10 labels:")
for label, count in labels.most_common(10):
    print(f"      {label:35} : {count:,}")

print(f"\n📄 Sample CUAD record:")
cuad_samples = [r for r in fixed if r["source"] == "CUAD"]
s = cuad_samples[0]
print(f"   label       : {s['label']}")
print(f"   clause_text : {s['clause_text'][:120]}")

print(f"\n📄 Sample LEDGAR record:")
ledg_samples = [r for r in fixed if r["source"] == "LEDGAR"]
s = ledg_samples[0]
print(f"   label       : {s['label']}")
print(f"   clause_text : {s['clause_text'][:120]}")

print(f"\n✅ Saved clean dataset to {out_path}")