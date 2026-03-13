import json
import os
import random

# LEDGAR label map (100 clause types)
LEDGAR_LABELS = {
    0: "Adjustments", 1: "Agreements", 2: "Amendments", 3: "Announcements",
    4: "Arbitration", 5: "Assignments", 6: "Assigns", 7: "Authority",
    8: "Authorizations", 9: "Base Salary", 10: "Benefits", 11: "Binding Effects",
    12: "Books", 13: "Brokers", 14: "Capitalization", 15: "Change In Control",
    16: "Closings", 17: "Competitive Restrictions", 18: "Compliance With Laws",
    19: "Confidentiality", 20: "Consent To Jurisdiction", 21: "Consents",
    22: "Construction", 23: "Cooperation", 24: "Costs", 25: "Counterparts",
    26: "Death", 27: "Defined Terms", 28: "Definitions", 29: "Disability",
    30: "Disclosures", 31: "Duties", 32: "Effective Dates", 33: "Effectiveness",
    34: "Employment", 35: "Enforceability", 36: "Entire Agreements",
    37: "Erisa", 38: "Existence", 39: "Expenses", 40: "Fees",
    41: "Financial Statements", 42: "Forfeitures", 43: "Further Assurances",
    44: "General", 45: "Governing Laws", 46: "Grants", 47: "Headings",
    48: "Indemnifications", 49: "Indemnity", 50: "Insurance",
    51: "Integration", 52: "Intellectual Property", 53: "Interests",
    54: "Interpretations", 55: "Jurisdictions", 56: "Liens",
    57: "Limitations Of Liability", 58: "Litigation", 59: "Miscellaneous",
    60: "Modifications", 61: "No Conflicts", 62: "No Defaults",
    63: "No Waivers", 64: "Non-Competition", 65: "Non-Disparagement",
    66: "Non-Solicitation", 67: "Notices", 68: "Operations",
    69: "Organizations", 70: "Payments", 71: "Positions",
    72: "Powers Of Attorney", 73: "Publicity", 74: "Qualifications",
    75: "Records", 76: "Releases", 77: "Remedies", 78: "Representations",
    79: "Resignations", 80: "Restrictions", 81: "Results Of Operations",
    82: "Sanctions", 83: "Securities", 84: "Severability",
    85: "Solvency", 86: "Specific Performance", 87: "Submission To Jurisdiction",
    88: "Subsidiaries", 89: "Successors", 90: "Survival",
    91: "Tax Withholdings", 92: "Taxes", 93: "Terminations",
    94: "Terms", 95: "Titles", 96: "Transactions With Affiliates",
    97: "Vesting", 98: "Waivers", 99: "Warranties"
}

print("📂 Loading CUAD processed data...")
with open("data/processed/cuad_processed.json", "r", encoding="utf-8") as f:
    cuad_data = json.load(f)
print(f"   CUAD records   : {len(cuad_data)}")

print("📂 Loading LEDGAR processed data...")
with open("data/processed/ledgar_processed.json", "r", encoding="utf-8") as f:
    ledgar_data = json.load(f)
print(f"   LEDGAR records : {len(ledgar_data)}")

# Normalize CUAD
cuad_final = []
for item in cuad_data:
    cuad_final.append({
        "clause_text": item["clause_text"].strip(),
        "label"      : item["label"].strip(),
        "source"     : "CUAD"
    })

# Normalize LEDGAR — convert numeric label to name
ledgar_final = []
for item in ledgar_data:
    label_id = int(item["label"])
    label_name = LEDGAR_LABELS.get(label_id, f"Label_{label_id}")
    ledgar_final.append({
        "clause_text": item["clause_text"].strip(),
        "label"      : label_name,
        "source"     : "LEDGAR"
    })

# Combine and shuffle
combined = cuad_final + ledgar_final
random.seed(42)
random.shuffle(combined)

print(f"\n✅ Combined total  : {len(combined)} records")
print(f"   CUAD portion   : {len(cuad_final)}")
print(f"   LEDGAR portion : {len(ledgar_final)}")

# Save
out_path = "data/processed/combined_dataset.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"\n💾 Saved to {out_path}")

# Show samples
print("\n📄 Sample records:")
for i in range(3):
    s = combined[i]
    print(f"\n  [{i+1}] Source : {s['source']}")
    print(f"       Label  : {s['label']}")
    print(f"       Text   : {s['clause_text'][:120]}")

print("\n✅ Dataset combination complete!")