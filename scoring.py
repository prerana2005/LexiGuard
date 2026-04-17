"""
BUG FIX: scoring.py uses ambiguity_result.get('has_ambiguity')
but ambiguity_agent.py returns 'is_ambiguous'
This mismatch means ambiguity never contributes to score.
Fix: normalize the key in scoring.py
"""

# In scoring.py, change line:
# if ambiguity_result.get('has_ambiguity') == True:
# TO:
# if ambiguity_result.get('is_ambiguous') or ambiguity_result.get('has_ambiguity'):

def calculate_risk_score(risk_result, ambiguity_result, compliance_result, explanation_result=None):
    score = 0
    flags = []

    # Compliance scoring
    if compliance_result.get('violation') == True:
        severity = compliance_result.get('severity', 'LOW').upper()
        if severity == 'HIGH':
            score += 9
            flags.append("Non-compliant with Indian law (HIGH)")
        elif severity == 'MEDIUM':
            score += 6
            flags.append("Non-compliant with Indian law (MEDIUM)")
        else:
            score += 3
            flags.append("Non-compliant with Indian law (LOW)")

    # Risk scoring
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

    # FIX: support both key names
    is_ambiguous = (
        ambiguity_result.get('is_ambiguous') or
        ambiguity_result.get('has_ambiguity')
    )
    if is_ambiguous:
        score += 7
        flags.append("Contains vague/ambiguous language")

    score = min(score, 100)

    if score <= 10:
        label = "GREEN"
        verdict = "Safe to sign"
    elif score <= 18:
        label = "YELLOW"
        verdict = "Review carefully before signing"
    elif score <= 24:
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
