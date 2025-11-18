import streamlit as st

st.title("Stroke Pipeline Demo (Enhanced Mock Version)")
st.write("This demo simulates the extraction–validation–prediction workflow using three different example notes.")

# =========================
# 0) Example Notes
# =========================
example_notes = {
    "Example Note 1 (Clean Case)": """
72-year-old male with history of hypertension.
Presented with right arm weakness, onset 50 minutes ago.
NIHSS measured as 6 by neurology resident.
CT ASPECTS is reported as 9.
""",
    
    "Example Note 2 (Incorrect ASPECTS)": """
68-year-old female with diabetes mellitus.
Presented with aphasia and left-sided weakness for 1 hour.
NIHSS recorded as 12.
CT ASPECTS documented as '19' in note (likely typo).
Atrial fibrillation absent.
""",

    "Example Note 3 (Missing Variables)": """
75-year-old male presented with dysarthria.
No past medical history listed.
NIHSS reported as 5.
ASPECTS not mentioned.
Atrial fibrillation unknown.
"""
}

# =========================
# 1) Extraction Results (Mock)
# =========================
extraction_results = {
    "Example Note 1 (Clean Case)": {
        "NIHSS": 6,
        "Hypertension": "yes",
        "Diabetes": "no",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 9
    },

    "Example Note 2 (Incorrect ASPECTS)": {
        "NIHSS": 12,
        "Hypertension": "no",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 19    # <- intentionally wrong
    },

    "Example Note 3 (Missing Variables)": {
        "NIHSS": 5,
        "Hypertension": "unknown",
        "Diabetes": "unknown",
        "Atrial_Fibrillation": "unknown",
        "ASPECTS": "missing"
    }
}

# =========================
# 2) Validation Logic (Mock)
# =========================
def validate_data(selected_note):
    extracted = extraction_results[selected_note]
    val = {}

    # Rule-based
    rule_msg = []
    if isinstance(extracted["ASPECTS"], int) and not (0 <= extracted["ASPECTS"] <= 10):
        rule_msg.append("ASPECTS out of valid range (0–10).")
    if extracted["NIHSS"] < 0 or extracted["NIHSS"] > 42:
        rule_msg.append("NIHSS out of valid range (0–42).")
    if extracted["ASPECTS"] == "missing":
        rule_msg.append("ASPECTS missing from note.")
    if not rule_msg:
        rule_msg = ["No rule-based issues detected."]
    
    val["Rule-based"] = rule_msg

    # RAG Verification (mock)
    if selected_note == "Example Note 2 (Incorrect ASPECTS)":
        val["RAG"] = [
            "Retrieved text: 'CT ASPECTS likely around 9 based on radiology context.'",
            "Model suggestion: ASPECTS=9"
        ]
    elif selected_note == "Example Note 3 (Missing Variables)":
        val["RAG"] = [
            "Retrieved text: 'Past medical history not mentioned.'",
            "Could not confirm hypertension/AFib."
        ]
    else:
        val["RAG"] = ["Retrieved segments confirm all extracted values."]

    # Cosine similarity flagging (mock)
    if selected_note == "Example Note 2 (Incorrect ASPECTS)":
        val["Flag"] = "FLAGGED (Similarity score = 0.42)"
    elif selected_note == "Example Note 3 (Missing Variables)":
        val["Flag"] = "FLAGGED (Low confidence: score = 0.38)"
    else:
        val["Flag"] = "Not flagged (score = 0.91)"

    # HITL
    if selected_note == "Example Note 2 (Incorrect ASPECTS)":
        val["HITL"] = "Reviewer corrected ASPECTS from 19 → 9."
    elif selected_note == "Example Note 3 (Missing Variables)":
        val["HITL"] = "Reviewer marked missing fields as 'unknown'."
    else:
        val["HITL"] = "No corrections needed."

    return val


def final_prediction(selected_note):
    if selected_note == "Example Note 1 (Clean Case)":
        return 0.21
    elif selected_note == "Example Note 2 (Incorrect ASPECTS)":
        return 0.44
    else:
        return 0.33


# =========================
# UI Starts Here
# =========================
selected_note = st.selectbox("Select an example patient note:", list(example_notes.keys()))

with st.expander("1. Clinical Note"):
    st.text_area("Clinical Text", example_notes[selected_note], height=200)

with st.expander("2. Extraction Output (Mock)"):
    st.json(extraction_results[selected_note])

with st.expander("3. Validation Steps (Mock)"):
    results = validate_data(selected_note)
    st.markdown("### Rule-based Validation")
    st.write(results["Rule-based"])
    st.markdown("### RAG Verification")
    st.write(results["RAG"])
    st.markdown("### Cosine Similarity Flagging")
    st.write(results["Flag"])
    st.markdown("### Human-in-the-Loop Review")
    st.write(results["HITL"])

with st.expander("4. Prediction (Mock)"):
    prob = final_prediction(selected_note)
    st.metric("Predicted Poor Outcome Probability", f"{prob:.2f}")
