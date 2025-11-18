import streamlit as st

st.title("Stroke Pipeline Demo (Step-by-Step Mock)")

# Example data
example_note = """
73-year-old female with history of hypertension and atrial fibrillation.
Presented with slurred speech and right-sided weakness onset 2 hours ago.
Initial NIHSS reported as 7 by neurology resident.
CT shows ASPECTS 9.
tPA not administered due to contraindication.
"""

extracted_json = {
    "NIHSS": 7,
    "Hypertension": "yes",
    "Atrial_Fibrillation": "yes",
    "Diabetes": "no",
    "ASPECTS": 9,
    "tPA": "no"
}

validation_steps = {
    "Rule-based": "No abnormal values found.",
    "RAG Verification": "Relevant retrieved segments confirm data.",
    "Cosine Flagging": "Similarity score=0.91 (not flagged).",
    "HITL": "Human reviewer confirms accuracy."
}

prediction_result = {
    "Poor_outcome_probability": 0.32
}

# -----------------------------
# Step 1
# -----------------------------
with st.expander("1. Clinical Note"):
    st.text_area("Input Note:", example_note, height=200)

# -----------------------------
# Step 2
# -----------------------------
with st.expander("2. Extraction"):
    st.json(extracted_json)

# -----------------------------
# Step 3
# -----------------------------
with st.expander("3. Validation Steps"):
    for step, desc in validation_steps.items():
        st.markdown(f"### {step}")
        st.write(desc)

# -----------------------------
# Step 4
# -----------------------------
with st.expander("4. Prediction"):
    st.metric(label="Predicted Poor Outcome Probability", value=f"{prediction_result['Poor_outcome_probability']:.2f}")
