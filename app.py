import streamlit as st
import requests
import tempfile
import os
from pdf_parser import extract_text_from_pdf, split_into_clauses
from risk_agent import analyze_risk
from ambiguity_agent import analyze_ambiguity
from explanation_agent import explain_clause
from ocr_pipeline import extract_text_from_image
from chatbot import answer_query

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="LexiGuard", page_icon="⚖️", layout="wide")

st.title("⚖️ LexiGuard")
st.caption("AI-Powered Indian Contract Analyzer")
st.divider()

# Initialize session state
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

upload_type = st.radio("Upload contract as:", ["PDF", "Image/Photo"])

if upload_type == "PDF":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
else:
    uploaded_file = st.file_uploader("Upload image of contract", type=["jpg", "jpeg", "png"])

if uploaded_file and st.button("Analyze Contract", type="primary"):

    # Reset everything on new upload
    st.session_state.analysis_done = False
    st.session_state.chat_history = []

    suffix = ".pdf" if upload_type == "PDF" else f".{uploaded_file.name.split('.')[-1]}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading contract..."):
        if upload_type == "PDF":
            extracted = extract_text_from_pdf(tmp_path)
            full_text = extracted['full_text']
        else:
            full_text = extract_text_from_image(tmp_path)
        os.unlink(tmp_path)

    if not full_text or len(full_text.strip()) < 50:
        st.error("Could not extract text from the uploaded file. Please try a clearer image or PDF.")
        st.stop()

    with st.spinner("Splitting into clauses..."):
        clauses = split_into_clauses(full_text)

    if not clauses:
        st.error("No clauses found. Please check the uploaded file.")
        st.stop()

    st.success(f"Found {len(clauses)} clauses. Analyzing...")

    results = {}
    progress = st.progress(0)
    status = st.empty()

    for i, clause in enumerate(clauses):
        cid = clause['clause_id']
        status.text(f"Analyzing clause {i+1} of {len(clauses)}: {clause['clause_heading']}")
        results[cid] = {}

        # Risk Agent
        results[cid]['risk'] = analyze_risk(clause['clause_text'])

        # Ambiguity Agent
        results[cid]['ambiguity'] = analyze_ambiguity(clause['clause_text'])

        # Compliance Agent
        try:
            resp = requests.post(
                f"{API_BASE}/analyze/compliance",
                json={"clause_text": clause['clause_text']},
                timeout=30
            )
            results[cid]['compliance'] = resp.json()
        except:
            results[cid]['compliance'] = {
                "violation": False,
                "explanation": "Compliance API unavailable. Make sure uvicorn is running."
            }

        progress.progress((i + 1) / len(clauses))

    status.empty()
    progress.empty()

    # Store everything in session state
    st.session_state.full_text = full_text
    st.session_state.clauses = clauses
    st.session_state.results = results
    st.session_state.analysis_done = True
    st.rerun()

# ── Show results only if analysis is done ──────────────────────
if st.session_state.analysis_done:

    full_text = st.session_state.full_text
    clauses = st.session_state.clauses
    results = st.session_state.results

    # Calculate overall score
    total_score = 0
    for cid, res in results.items():
        risk_level = res.get('risk', {}).get('risk_level', 'Low')
        is_ambiguous = res.get('ambiguity', {}).get('is_ambiguous', False)
        has_violation = res.get('compliance', {}).get('violation', False)

        clause_score = 0
        if risk_level == "High": clause_score += 10
        elif risk_level == "Medium": clause_score += 5
        else: clause_score += 2
        if is_ambiguous: clause_score += 3
        if has_violation: clause_score += 5

        total_score += clause_score

    max_score = len(clauses) * 18
    overall = min(int((total_score / max(max_score, 1)) * 100), 100)

    if overall <= 30: band = "🟢 Mostly Safe"
    elif overall <= 60: band = "🟡 Review Carefully"
    elif overall <= 80: band = "🟠 Significant Risks"
    else: band = "🔴 Do Not Sign"

    high_risk = sum(1 for r in results.values() if r.get('risk', {}).get('risk_level') == 'High')
    medium_risk = sum(1 for r in results.values() if r.get('risk', {}).get('risk_level') == 'Medium')
    ambiguous = sum(1 for r in results.values() if r.get('ambiguity', {}).get('is_ambiguous', False))
    violations = sum(1 for r in results.values() if r.get('compliance', {}).get('violation', False))

    # Dashboard
    st.divider()
    st.subheader("Overall Risk Assessment")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risk Score", f"{overall}/100")
    col2.metric("Risk Band", band)
    col3.metric("🔴 High Risk", high_risk)
    col4.metric("⚠️ Ambiguous", ambiguous)
    col5.metric("❌ Violations", violations)

    if overall <= 30:
        st.success(f"This contract is relatively safe to sign. Score: {overall}/100")
    elif overall <= 60:
        st.warning(f"This contract has some concerns. Review carefully before signing. Score: {overall}/100")
    elif overall <= 80:
        st.error(f"This contract has significant risks. Consider negotiating clauses. Score: {overall}/100")
    else:
        st.error(f"⚠️ Do NOT sign this contract without consulting a lawyer. Score: {overall}/100")

    st.divider()
    st.subheader("Clause by Clause Analysis")

    filter_option = st.selectbox(
        "Filter clauses:",
        ["All Clauses", "High Risk Only", "Ambiguous Only", "Violations Only"]
    )

    for clause in clauses:
        cid = clause['clause_id']
        res = results.get(cid, {})

        risk = res.get('risk', {})
        ambiguity = res.get('ambiguity', {})
        compliance = res.get('compliance', {})

        risk_level = risk.get('risk_level', 'Low')
        is_ambiguous = ambiguity.get('is_ambiguous', False)
        has_violation = compliance.get('violation', False)

        if filter_option == "High Risk Only" and risk_level != "High":
            continue
        if filter_option == "Ambiguous Only" and not is_ambiguous:
            continue
        if filter_option == "Violations Only" and not has_violation:
            continue

        badge = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
        amb = " ⚠️" if is_ambiguous else ""
        vio = " ❌" if has_violation else ""
        label = f"{badge}{amb}{vio}  {cid} — {clause['clause_heading']}"

        with st.expander(label):
            st.markdown("**Clause Text:**")
            st.info(clause['clause_text'])

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔍 Risk Analysis**")
                color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
                st.markdown(f":{color}[**{risk_level} Risk** — {risk.get('risk_type', 'N/A')}]")
                st.markdown(f"**Why risky:** {risk.get('explanation', 'N/A')}")
                st.markdown(f"**Impact:** {risk.get('impact', 'N/A')}")

            with col2:
                if is_ambiguous:
                    st.markdown("**⚠️ Ambiguity Detected**")
                    vague = ", ".join(ambiguity.get('vague_phrases', []))
                    st.markdown(f"**Vague terms:** `{vague}`")
                    st.markdown(f"**How it can be misused:** {ambiguity.get('how_exploited', 'N/A')}")
                    st.markdown(f"**Suggested fix:** _{ambiguity.get('suggested_fix', 'N/A')}_")
                else:
                    st.markdown("**✅ No Ambiguity Detected**")
                    st.markdown("No vague or undefined terms found in this clause.")

            if has_violation:
                st.divider()
                st.markdown("**❌ Compliance Violation Found**")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"**Law Violated:** {compliance.get('law_violated', 'N/A')}")
                    st.markdown(f"**Section:** {compliance.get('section', 'N/A')}")
                with col4:
                    st.markdown(f"**Explanation:** {compliance.get('explanation', 'N/A')}")
            else:
                st.divider()
                st.markdown("**✅ No Compliance Violation**")

            st.divider()
            st.markdown("**📖 Plain English Explanation**")
            if st.button(f"Generate Explanation", key=f"explain_{cid}"):
                with st.spinner("Generating explanation..."):
                    try:
                        exp = explain_clause(clause['clause_text'])
                        st.markdown(f"**Clause Type:** {exp.get('clause_type', 'N/A')} ({exp.get('confidence', 0):.1%} confidence)")
                        relevant_laws = exp.get('relevant_laws', [])
                        if relevant_laws:
                            laws_text = ", ".join([f"{l['law']} — {l['section']}" for l in relevant_laws])
                            st.markdown(f"**Relevant Laws:** {laws_text}")
                        st.markdown(exp.get('explanation', 'N/A'))
                    except Exception as e:
                        st.error("Could not generate explanation. Try again.")

    # ── Chatbot ────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask About Your Contract")
    st.caption("Ask any question about this contract and get answers grounded in Indian law")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about your contract...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_query(user_query, full_text)
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})