import streamlit as st

st.title("Stroke Pipeline Demo (Mock)")

# Sidebar selection
menu = st.sidebar.selectbox(
    "Select Step",
    ["Clinical Note", "Extraction", "Validation", "Prediction"]
)

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
    "RAG Verification": "Retrieved segments support extracted values.",
    "Cosine Flagging": "Score=0.91 → Not flagged.",
    "HITL": "Reviewer validated all variables."
}

prediction_result = {
    "Poor_outcome_probability": 0.32
}

# Render based on selection
if menu == "Clinical Note":
    st.header("1. Clinical Note")
    st.text_area("Note:", example_note, height=200)

elif menu == "Extraction":
    st.header("2. Extraction Result")
    st.json(extracted_json)

elif menu == "Validation":
    st.header("3. Validation Steps")
    for step, desc in validation_steps.items():
        st.markdown(f"### {step}")
        st.write(desc)

elif menu == "Prediction":
    st.header("4. Prediction Result")
    st.metric("Poor Outcome Probability", f"{prediction_result['Poor_outcome_probability']:.2f}")
