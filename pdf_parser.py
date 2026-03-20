import pdfplumber
import re


def extract_heading(clause_text):
    first_line = clause_text.split('\n')[0].strip()

    # Remove leading number like "1." or "2."
    clean = re.sub(r'^\d+[\.\)]\s*', '', first_line).strip()
    
    # Remove common filler starts
    filler = [
        r'^That\s+the\s+', r'^That\s+', r'^The\s+', r'^this\s+',
        r'^In\s+case\s+', r'^during\s+the\s+', r'^all\s+the\s+',
        r'^no\s+structural\s+'
    ]
    for f in filler:
        clean = re.sub(f, '', clean, flags=re.IGNORECASE).strip()

    # Capitalize first letter
    if clean:
        clean = clean[0].upper() + clean[1:]

    # If short enough use directly
    if 3 < len(clean) < 50:
        # Cut at first comma or semicolon
        for sep in [',', ';', ' shall', ' will', ' may']:
            if sep in clean.lower():
                clean = clean[:clean.lower().index(sep)].strip()
                break
        return clean if len(clean) > 3 else first_line[:40]

    # Take first 5 words
    words = clean.split()[:5]
    return " ".join(words)


def extract_text_from_pdf(pdf_path):
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                all_text.append({"page": page_num, "text": text.strip()})

    full_text = "\n".join([p["text"] for p in all_text])
    return {
        "full_text": full_text,
        "pages": all_text,
        "total_pages": len(all_text)
    }


def split_into_clauses(full_text):
    clauses = []

    # Method 1 — split on numbered sections like 1. or 2. or Section 1
    pattern = r'(?=(?:Section|Clause|CLAUSE|SECTION)?\s*\d+[\.\)]\s+[A-Z])'
    parts = re.split(pattern, full_text)
    parts = [p.strip() for p in parts if len(p.strip()) > 30]

    if len(parts) > 2:
        for i, part in enumerate(parts):
            clauses.append({
                "clause_id": f"C{str(i+1).zfill(3)}",
                "clause_number": str(i+1),
                "clause_heading": extract_heading(part),
                "clause_text": part,
                "page_number": 1
            })
    else:
        # Method 2 — fallback split by double newline
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
        for i, para in enumerate(paragraphs):
            clauses.append({
                "clause_id": f"C{str(i+1).zfill(3)}",
                "clause_number": str(i+1),
                "clause_heading": extract_heading(para),
                "clause_text": para,
                "page_number": 1
            })

    return clauses


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = extract_text_from_pdf(pdf_path)

    print(f"Total pages: {result['total_pages']}")
    print(f"Text length: {len(result['full_text'])} characters")

    clauses = split_into_clauses(result['full_text'])
    print(f"\nTotal clauses found: {len(clauses)}")
    for c in clauses:
        print(f"{c['clause_id']}: {c['clause_heading']}")