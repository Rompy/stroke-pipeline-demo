import streamlit as st

st.title("Stroke Pipeline Demo (Mock)")
st.write("This demo illustrates the workflow of the extraction–validation–prediction pipeline using example data only.")

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
    "Rule-based": "No abnormal values detected.",
    "RAG Verification": "Retrieved segments confirm hypertension and AFib.",
    "Cosine Flagging": "Similarity score = 0.91 → Not flagged.",
    "HITL": "Human reviewer confirmed accuracy."
}

prediction_result = {
    "Poor_outcome_probability": 0.32
}

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["1. Clinical Note", "2. Extraction", "3. Validation", "4. Prediction"])

with tab1:
    st.subheader("Clinical Note Input (Example)")
    st.text_area("Example note", example_note, height=200)

with tab2:
    st.subheader("Extraction Result (Mock Output)")
    st.json(extracted_json)

with tab3:
    st.subheader("Validation Steps (Mock Output)")
    for step, desc in validation_steps.items():
        st.markdown(f"### {step}")
        st.write(desc)

with tab4:
    st.subheader("Prediction Result (Mock Output)")
    st.metric("Predicted Poor Outcome Probability", f"{prediction_result['Poor_outcome_probability']:.2f}")
