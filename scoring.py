def calculate_risk_score(risk_result, ambiguity_result, compliance_result, explanation_result=None):
    """
    Takes results from all 4 agents and calculates overall danger score 0-100
    
    Risk Agent result expected keys: 'risk_level' (HIGH/MEDIUM/LOW)
    Ambiguity Agent result expected keys: 'has_ambiguity' (True/False)
    Compliance Agent result expected keys: 'violation' (True/False), 'severity' (HIGH/MEDIUM/LOW)
    """
    
    score = 0
    flags = []

    # --- Compliance Agent scoring ---
    if compliance_result.get('violation') == True:
        severity = compliance_result.get('severity', 'LOW').upper()
        if severity == 'HIGH':
            score += 8
            flags.append("Non-compliant with Indian law (HIGH)")
        elif severity == 'MEDIUM':
            score += 5
            flags.append("Non-compliant with Indian law (MEDIUM)")
        else:
            score += 3
            flags.append("Non-compliant with Indian law (LOW)")

    # --- Risk Agent scoring ---
    risk_level = risk_result.get('risk_level', 'LOW').upper()
    if risk_level == 'HIGH':
        score += 10
        flags.append("High risk clause")
    elif risk_level == 'MEDIUM':
        score += 5
        flags.append("Medium risk clause")
    else:
        score += 2
        flags.append("Low risk clause")

    # --- Ambiguity Agent scoring ---
    if ambiguity_result.get('is_ambiguous') == True:
        score += 7
        flags.append("Contains vague/ambiguous language")

    # --- Cap score at 100 ---
    score = min(score, 100)

    # --- Determine color/label ---
    if score <= 25:
        label = "GREEN"
        verdict = "Safe to sign"
    elif score <= 50:
        label = "YELLOW"
        verdict = "Review carefully before signing"
    elif score <= 75:
        label = "ORANGE"
        verdict = "Risky — consult a lawyer"
    else:
        label = "RED"
        verdict = "Do NOT sign — high risk"

    return {
        "score": score,
        "label": label,
        "verdict": verdict,
        "flags": flags
    }


# Test it
if __name__ == "__main__":
    # Simulate results from all 3 agents
    mock_compliance = {
        "violation": True,
        "severity": "HIGH"
    }
    mock_risk = {
        "risk_level": "HIGH"
    }
    mock_ambiguity = {
        "has_ambiguity": True
    }

    result = calculate_risk_score(mock_risk, mock_ambiguity, mock_compliance)

    print("=== RISK SCORE RESULT ===")
    print(f"Score: {result['score']}/100")
    print(f"Label: {result['label']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Flags: {result['flags']}")