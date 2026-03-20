from dotenv import load_dotenv
import os
load_dotenv()
from groq import Groq
from rag_pipeline import search_relevant_laws
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compliance_agent(clause_text):
    # Step 1: Get relevant laws from FAISS
    relevant_laws = search_relevant_laws(clause_text, top_k=3)
    
    # Step 2: Build law context
    law_context = ""
    for i, law in enumerate(relevant_laws):
        law_context += f"\nLaw {i+1}: {law['law_name']}"
        law_context += f"\nSection: {law['section_number']}"
        law_context += f"\nText: {law['text']}\n"
    
    # Step 3: Build prompt
    prompt = f"""You are an expert Indian legal compliance checker.

A contract clause needs to be checked against Indian laws.

CONTRACT CLAUSE:
"{clause_text}"

RELEVANT INDIAN LAWS:
{law_context}

Analyze if this contract clause violates any of the above Indian laws.

Respond in this exact format:
VIOLATION: YES or NO
LAW VIOLATED: (name of law, or "None")
SECTION: (section number, or "None")
EXPLANATION: (2-3 sentences explaining why it violates or does not violate)
SEVERITY: HIGH, MEDIUM, or LOW (or "None" if no violation)
"""
    
    # Step 4: Call Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.choices[0].message.content
    
    # Step 5: Parse response
    result = {
        "clause": clause_text,
        "violation": False,
        "law_violated": "None",
        "section": "None",
        "explanation": "",
        "severity": "None",
        "relevant_laws_searched": [
            f"{l['law_name']} - {l['section_number']}"
            for l in relevant_laws
        ]
    }
    
    lines = response_text.strip().split('\n')
    for line in lines:
        if line.startswith("VIOLATION:"):
            result["violation"] = "YES" in line.upper()
        elif line.startswith("LAW VIOLATED:"):
            result["law_violated"] = line.replace("LAW VIOLATED:", "").strip()
        elif line.startswith("SECTION:"):
            result["section"] = line.replace("SECTION:", "").strip()
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("SEVERITY:"):
            result["severity"] = line.replace("SEVERITY:", "").strip()
    
    return result


if __name__ == "__main__":
    test_clause = "The company may terminate the employee at any time without prior notice or reason."
    
    print("Running Compliance Agent...")
    print(f"Clause: {test_clause}\n")
    
    result = compliance_agent(test_clause)
    
    print("=== COMPLIANCE RESULT ===")
    print(f"Violation Found: {result['violation']}")
    print(f"Law Violated: {result['law_violated']}")
    print(f"Section: {result['section']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Severity: {result['severity']}")
    print(f"Laws Searched: {result['relevant_laws_searched']}")