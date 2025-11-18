import streamlit as st
import pandas as pd

# 기본 설정
st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")
st.title("Enhanced Stroke Pipeline Demo (Mock Version)")
st.write("This demo simulates a multimodal extraction–validation–prediction pipeline using text and imaging sources.")

# ======================================
# 0) Example Inputs: Neurology Note, Radiology Report, ASPECTS Images
# ======================================

neurology_notes = {
    "Example Case 1": """
73F with history of hypertension, presented with slurred speech and right-sided weakness.
Neurology exam: NIHSS = 6. Mild facial palsy, right arm drift.
""",

    "Example Case 2": """
68F with diabetes mellitus presented with aphasia.
NIHSS recorded as 12. Motor weakness in left arm. No AFib documented.
""",

    "Example Case 3": """
75M with dysarthria. Past history unknown.
NIHSS = 5. Unable to assess AFib due to incomplete history.
"""
}

radiology_reports = {
    "Example Case 1": """
Non-contrast CT: ASPECTS = 9. No evidence of hemorrhage or LVO.
""",

    "Example Case 2": """
CT ASPECTS documented as '19' in note (likely typo). No hemorrhage.
""",

    "Example Case 3": """
ASPECTS not documented in CT report. MRI recommended for further evaluation.
"""
}

# ← 반드시 /images 폴더에 넣어야 함
aspect_images = {
    "Example Case 1": "images/aspects1.png",
    "Example Case 2": "images/aspects2.png",
    "Example Case 3": "images/aspects3.png"
}

# Image-based ASPECTS Ground Truth
aspect_ground_truth = {
    "Example Case 1": 5,
    "Example Case 2": 6,
    "Example Case 3": 10
}

# ======================================
# 1) Extraction Mock Output
# ======================================

extraction_results = {
    "Example Case 1": {
        "NIHSS": 6,
        "Hypertension": "yes",
        "Diabetes": "no",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 9
    },

    "Example Case 2": {
        "NIHSS": 12,
        "Hypertension": "no",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 19
    },

    "Example Case 3": {
        "NIHSS": 5,
        "Hypertension": "unknown",
        "Diabetes": "unknown",
        "Atrial_Fibrillation": "unknown",
        "ASPECTS": "missing"
    }
}

# ======================================
# 2) Validation Function (Rule-Based, RAG, Cosine, HITL)
# ======================================

def validate_data(selected, extracted):
    val = {}

    # --- Rule-based ----
    rule_msgs = []

    # ASPECTS Checks
    if extracted["ASPECTS"] == "missing":
        rule_msgs.append("❗ ASPECTS missing from extraction.")
    elif isinstance(extracted["ASPECTS"], int) and (extracted["ASPECTS"] < 0 or extracted["ASPECTS"] > 10):
        rule_msgs.append("❗ ASPECTS value out of expected range (0–10).")

    # NIHSS Checks
    if extracted["NIHSS"] < 0 or extracted["NIHSS"] > 42:
        rule_msgs.append("❗ NIHSS out of normal clinical range (0–42).")

    if not rule_msgs:
        rule_msgs.append("✔ No rule-based issues detected.")

    val["Rule-based"] = rule_msgs

    # --- RAG verification (Mock) ---
    if selected == "Example Case 2":
        val["RAG"] = [
            "Retrieved snippet: 'CT interpretation suggests ASPECTS ~6-9'.",
            "Model suggestion: ASPECTS likely 6–9, not 19."
        ]
    elif selected == "Example Case 3":
        val["RAG"] = [
            "Retrieved snippet: 'ASPECTS not documented'.",
            "Unable to confirm AFib/HTN from text."
        ]
    else:
        val["RAG"] = [
            "Retrieved segments confirm NIHSS, HTN, and ASPECTS values."
        ]

    # --- Cosine Similarity Flagging (Mock) ---
    if selected == "Example Case 1":
        val["Flag"] = "✔ Not flagged (similarity score = 0.92)"
    elif selected == "Example Case 2":
        val["Flag"] = "❗ FLAGGED (score = 0.42) – possible ASPECTS error"
    else:
        val["Flag"] = "❗ FLAGGED (low confidence score = 0.38)"

    # --- HITL Correction ---
    if selected == "Example Case 2":
        val["HITL"] = "Reviewer: corrected ASPECTS from 19 → 6."
    elif selected == "Example Case 3":
        val["HITL"] = "Reviewer: confirmed missing variables as 'unknown'."
    else:
        val["HITL"] = "No corrections required."

    return val

# ======================================
# UI Start
# ======================================

selected = st.selectbox("Select Example Case", list(neurology_notes.keys()))

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📝 Neurology Note")
    st.text_area("Note", neurology_notes[selected], height=250)

with col2:
    st.subheader("🖼️ ASPECTS CT Image")
    st.image(aspect_images[selected], caption=f"ASPECTS Ground Truth = {aspect_ground_truth[selected]}")

with col3:
    st.subheader("📄 Radiology Report")
    st.text_area("Radiology", radiology_reports[selected], height=250)

# ======================================
# Extraction
# ======================================

st.markdown("---")
with st.expander("1. Extraction Output (Mock)"):
    extracted = extraction_results[selected]
    st.json(extracted)
    st.markdown(f"**ASPECTS from Image (Ground Truth):** {aspect_ground_truth[selected]}")

# ======================================
# Validation
# ======================================

with st.expander("2. Validation Steps"):
    results = validate_data(selected, extracted)

    st.markdown("### 🔎 Rule-based Validation")
    for m in results["Rule-based"]:
        st.write(m)

    st.markdown("---")
    st.markdown("### 📚 RAG Verification")
    for m in results["RAG"]:
        st.write("- " + m)

    st.markdown("---")
    st.markdown("### 📌 Vector Similarity (Cosine)")
    st.write(results["Flag"])

    st.markdown("---")
    st.markdown("### 🧑‍⚕️ Human-in-the-loop Review")
    st.write(results["HITL"])

    st.markdown("---")
    st.markdown("### 🧩 Cross-check with ASPECTS Image")
    true_score = aspect_ground_truth[selected]
    if extracted["ASPECTS"] == true_score:
        st.success("ASPECTS matches image-derived score.")
    else:
        st.error(f"Mismatch detected: Extracted={extracted['ASPECTS']} vs Image={true_score}")

# ======================================
# Prediction
# ======================================

with st.expander("3. Prediction (Mock)"):
    if selected == "Example Case 1":
        prob = 0.22
    elif selected == "Example Case 2":
        prob = 0.44
    else:
        prob = 0.33
    
    st.metric("Predicted Poor Outcome Probability", f"{prob:.2f}")

# ======================================
# CSV Export
# ======================================

final_df = pd.DataFrame([{
    "Case": selected,
    "NIHSS": extracted["NIHSS"],
    "Hypertension": extracted["Hypertension"],
    "Diabetes": extracted["Diabetes"],
    "Atrial_Fibrillation": extracted["Atrial_Fibrillation"],
    "ASPECTS_Extracted": extracted["ASPECTS"],
    "ASPECTS_Image_GT": true_score,
    "Corrected_ASPECTS": results["HITL"],
    "Poor_Outcome_Prob": prob
}])

st.download_button(
    label="⬇️ Download Final Structured Data (CSV)",
    data=final_df.to_csv(index=False),
    mime="text/csv",
    file_name=f"{selected}_structured_output.csv"
)
