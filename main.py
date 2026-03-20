from fastapi import FastAPI
from pydantic import BaseModel
from compliance_agent import compliance_agent
from scoring import calculate_risk_score

app = FastAPI(title="LexiGuard API")

# Input model
class ClauseInput(BaseModel):
    clause_text: str

# Input model for full analysis
class FullAnalysisInput(BaseModel):
    clause_text: str
    risk_level: str = "LOW"        # Person 3 will send this
    has_ambiguity: bool = False     # Person 3 will send this

# ---- Route 1: Health check ----
@app.get("/")
def home():
    return {"message": "LexiGuard API is running!"}

# ---- Route 2: Compliance Agent ----
@app.post("/analyze/compliance")
def analyze_compliance(data: ClauseInput):
    result = compliance_agent(data.clause_text)
    return result

# ---- Route 3: Full analysis (compliance + scoring) ----
@app.post("/analyze/full")
def analyze_full(data: FullAnalysisInput):
    # Run compliance agent
    compliance_result = compliance_agent(data.clause_text)
    
    # Mock risk and ambiguity from Person 3's agents
    risk_result = {"risk_level": data.risk_level}
    ambiguity_result = {"has_ambiguity": data.has_ambiguity}
    
    # Calculate score
    score_result = calculate_risk_score(
        risk_result,
        ambiguity_result,
        compliance_result
    )
    
    return {
        "clause": data.clause_text,
        "compliance": compliance_result,
        "score": score_result
    }