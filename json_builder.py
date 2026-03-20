import uuid
import json
from datetime import datetime
from pdf_parser import extract_text_from_pdf, split_into_clauses

def build_contract_json(source, source_type="pdf"):
    if source_type == "pdf":
        extracted = extract_text_from_pdf(source)
        full_text = extracted['full_text']
        filename = source
    else:
        full_text = source
        filename = "image_upload"

    clauses = split_into_clauses(full_text)

    contract = {
        "contract_id": str(uuid.uuid4()),
        "metadata": {
            "source_type": source_type,
            "filename": filename,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_clauses": len(clauses)
        },
        "raw_text": full_text,
        "parties": {
            "party_1": "",
            "party_2": ""
        },
        "contract_type": "Unknown",
        "effective_date": "",
        "clauses": clauses,
        "analysis": {},
        "overall_score": None,
        "risk_band": None
    }
    return contract


if __name__ == "__main__":
    result = build_contract_json("rental-agreement.pdf")
    print(f"Contract ID: {result['contract_id']}")
    print(f"Total clauses: {result['metadata']['total_clauses']}")
    print(f"First clause: {result['clauses'][0]['clause_id']}")

    with open("test_output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to test_output.json")