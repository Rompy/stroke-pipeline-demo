import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")
st.title("🧠 Stroke Pipeline Demo")
st.write("A multimodal extraction–validation–correction–prediction pipeline for stroke outcome research.")

# ===============================================================
# CSS: Card-shaped UI + Highlight blocks
# ===============================================================

def highlight_red(text):
    return f"""
    <div style='background-color:#ffe6e6;padding:10px;border-radius:8px;
                border-left:6px solid #ff4d4d;margin-bottom:10px;'>
        <span style='color:#b30000;font-weight:600;'>{text}</span>
    </div>
    """

def highlight_green(text):
    return f"""
    <div style='background-color:#e8ffe6;padding:10px;border-radius:8px;
                border-left:6px solid #00b33c;margin-bottom:10px;'>
        <span style='color:#006622;font-weight:600;'>{text}</span>
    </div>
    """

card_style = """
<div style="
    background-color:#ffffff;
    padding:20px;
    border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.10);
    margin-bottom:25px;
    color:#000000;">
"""

step_badge = lambda x: f"<div style='background:#0047AB;color:white;padding:6px 12px;border-radius:6px;display:inline-block;margin-bottom:10px;font-weight:600;'>{x}</div>"

# ===============================================================
# 0) Neurology Notes – 3 Cases
# ===============================================================

neurology_notes = {
    "Example Case 1":
    """
**Chief Complaint:**  
Right-sided weakness and slurred speech  

**Onset:**  
August 25, 2018 at 21:40 (LKW 21:30)

**Past Medical History:**  
- Hypertension (poorly controlled)  
- Diabetes mellitus  
- No known atrial fibrillation  
- No prior stroke  

**Present Illness:**  
Sudden right arm and leg weakness with dysarthria at home.  

**Vitals:**  
BP 178/92  

**NIHSS:** 9  

**Impression:**  
Acute left MCA stroke; tPA administered at 22:35.
""",

    "Example Case 2":
    """
**Chief Complaint:**  
Aphasia, left-sided heaviness  

**Onset:**  
September 3, 2018 at 19:10

**Past Medical History:**  
- Diabetes mellitus  
- Hypertension  

**Present Illness:**  
Expressive difficulty + left arm heaviness  

**Vitals:**  
BP 162/88  

**NIHSS:** 5  

**Impression:**  
Early ischemic change in right MCA.
""",

    "Example Case 3":
    """
**Chief Complaint:**  
Presyncope  

**Onset:**  
August 24, 2018 at 23:30

**Past Medical History:**  
- Hypertension  
- Diabetes mellitus  

**NIHSS:** 0  

**Impression:**  
No acute lesion; normal CT/MRI.
"""
}

# ===============================================================
# Radiology Reports – detailed MRI findings
# ===============================================================

radiology_reports = {
    "Example Case 1":
    """
MRI Brain shows restricted diffusion in the left insula, frontal operculum,
and parietal cortex. Consistent with acute left MCA infarction. No hemorrhage.
""",

    "Example Case 2":
    """
MRI shows mild cortical diffusion restriction in right MCA distribution,
compatible with early ischemia. No hemorrhage or mass effect.
""",

    "Example Case 3":
    """
Normal MRI brain. No diffusion restriction. No abnormal enhancement.
""",
}

# ===============================================================
# ASPECTS Images (CT examples)
# ===============================================================

aspect_images = {
    "Example Case 1": "images/aspects1.png",
    "Example Case 2": "images/aspects2.png",
    "Example Case 3": "images/aspects3.png"
}

# ===============================================================
# Extraction Results (with intentional hallucinations for 2 cases)
# ===============================================================

extraction_results = {
    "Example Case 1": {
        "Chief_Complaint": "Right-sided weakness, dysarthria",
        "Onset_Time": "2018-08-25 21:40",
        "NIHSS": 9,
        "Hypertension": "yes",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",

        # ❌ intentional hallucinations
        "ASPECTS": 7,                # should be ≈5
        "tPA_Administered": "no",    # wrong (actual: yes)
        "Weakness_Side": "bilateral",# wrong (actual: right)

        "SBP": 178
    },

    "Example Case 2": {
        # ❌ hallucinations
        "Chief_Complaint": "Aphasia, left arm heaviness",
        "Onset_Time": "2018-09-03 19:10",
        "NIHSS": 5,
        "Hypertension": "no",        # wrong (actual: yes)
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 9,                # wrong (actual ≈6)
        "tPA_Administered": "no",
        "Weakness_Side": "left",
        "SBP": 162
    },

    "Example Case 3": {
        # correct extraction
        "Chief_Complaint": "Presyncope, bilateral leg weakness",
        "Onset_Time": "2018-08-24 23:30",
        "NIHSS": 0,
        "Hypertension": "yes",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 10,
        "tPA_Administered": "no",
        "Weakness_Side": "bilateral",
        "SBP": 211
    }
}
# =====================================================================
# 1) VALIDATION LOGIC — 논문 2.3.2 Multi-tiered Framework (정확 반영)
# =====================================================================

def validate_data(selected, extracted, note_text, radiology_text):
    """Returns structured validation results across 4 layers:
       Rule-Based, RAG, Cosine Similarity, HITL recommendation."""
    
    val = {}

    # Combine clinical text for RAG-like checks (semantic mismatch detection)
    full_text = (note_text + " " + radiology_text).lower()

    # ===============================================================
    # 1. RULE-BASED VERIFICATION (syntax, range, format ONLY)
    # ===============================================================
    rule_msgs = []

    # binary value format check
    for field in ["Hypertension", "Diabetes", "Atrial_Fibrillation", "tPA_Administered"]:
        if extracted[field] not in ["yes", "no", "unknown"]:
            rule_msgs.append(f"❗ {field}: Invalid value '{extracted[field]}'. Expected yes/no/unknown.")

    # NIHSS range
    if not (0 <= extracted["NIHSS"] <= 42):
        rule_msgs.append("❗ NIHSS outside valid range (0–42).")

    # ASPECTS range
    if not (0 <= extracted["ASPECTS"] <= 10):
        rule_msgs.append("❗ ASPECTS outside valid range (0–10).")

    # SBP physiological plausibility
    if extracted["SBP"] < 40 or extracted["SBP"] > 300:
        rule_msgs.append("❗ SBP physiologically implausible.")

    if not rule_msgs:
        rule_msgs.append("✔ Passed all rule-based format/range checks.")

    val["Rule"] = rule_msgs

    # ===============================================================
    # 2. RAG VERIFICATION — semantic mismatch vs original note
    # ===============================================================

    rag = []

    if selected == "Example Case 1":
        # tPA mismatch
        if "tpa" in full_text and extracted["tPA_Administered"] != "yes":
            rag.append("❗ RAG: Original note indicates tPA was administered.")

        # weakness side mismatch
        if "right" in full_text and extracted["Weakness_Side"] != "right":
            rag.append("❗ RAG: Weakness side inconsistent with note (expected: right).")

        # ASPECTS vs MRI mismatch
        if "acute" in full_text and extracted["ASPECTS"] > 7:
            rag.append("❗ RAG: ASPECTS too high relative to described left MCA infarction.")

    if selected == "Example Case 2":
        if "early" in full_text and extracted["ASPECTS"] >= 8:
            rag.append("❗ RAG: Early ischemia incompatible with ASPECTS ≥ 8.")

        if extracted["Hypertension"] == "no" and extracted["SBP"] >= 160:
            rag.append("❗ RAG: Semantic mismatch — SBP pattern suggests hypertension.")

    if not rag:
        rag.append("✔ No semantic mismatch detected via RAG-like verification.")

    val["RAG"] = rag

    # ===============================================================
    # 3. COSINE SIMILARITY FLAGGING (population-level pattern anomaly)
    # ===============================================================

    cos = []

    # Mock similarity scores per case (논문 threshold=0.82)
    if selected == "Example Case 1":
        sim = 0.71
    elif selected == "Example Case 2":
        sim = 0.78
    else:
        sim = 0.92

    if sim < 0.82:
        cos.append(f"❗ Cosine similarity = {sim:.2f} (<0.82) → population-level outlier.")
        cos.append("   (Note: This step does *not* perform clinical reasoning; it detects atypical variable combinations.)")
    else:
        cos.append(f"✔ Cosine similarity = {sim:.2f} → within validated population patterns.")

    val["Cosine"] = cos

    # ===============================================================
    # 4. HITL Review Recommendation
    # ===============================================================

    flagged = any("❗" in msg for stage in val.values() for msg in stage)

    if flagged:
        val["HITL"] = (
            "🔎 Record requires clinician review (flagged by RAG or similarity). "
            "This is the only step where clinical reasoning is performed."
        )
    else:
        val["HITL"] = (
            "✔ No major issues detected — eligible for automated acceptance. "
            "(Random 10% still reviewed by clinicians.)"
        )

    return val


# =====================================================================
# 2) HITL ASSISTED CORRECTION MODULE (Mock)
#     - Only applied when validation detects issues
# =====================================================================

def hitl_correction(selected, extracted, validation):
    """Returns corrected structured output based on flagged issues.
       This simulates clinician-driven corrections in HITL step."""

    corrected = extracted.copy()

    # Only correct if flagged
    if "❗" not in str(validation):
        return corrected, False   # no changes

    # Correction rules per case (clearly documented, transparent)
    if selected == "Example Case 1":
        # tPA correction  
        corrected["tPA_Administered"] = "yes"

        # Weakness correction
        corrected["Weakness_Side"] = "right"

        # ASPECT correction
        corrected["ASPECTS"] = 5

    if selected == "Example Case 2":
        # HTN correction
        corrected["Hypertension"] = "yes"

        # ASPECT correction
        corrected["ASPECTS"] = 6

    # Case 3 has no errors → no correction

    return corrected, True
# =====================================================================
# UI START
# =====================================================================

selected = st.selectbox("Select Example Case", list(neurology_notes.keys()))


# =====================================================================
# Display Neurology Note / MRI / ASPECTS Images
# =====================================================================

col1, col2, col3 = st.columns([1.3, 1.3, 1])

with col1:
    st.markdown(step_badge("Source Document 1"))
    st.markdown(card_style + "<h3>📝 Neurology Note</h3>" +
                neurology_notes[selected] + "</div>", unsafe_allow_html=True)

with col2:
    st.markdown(step_badge("Source Document 2"))
    st.markdown(card_style + "<h3>📄 Radiology Report</h3>" +
                radiology_reports[selected] + "</div>", unsafe_allow_html=True)

with col3:
    st.markdown(step_badge("Source Document 3"))
    st.markdown(card_style + "<h3>🖼️ ASPECT CT Image</h3>", unsafe_allow_html=True)
    st.image(aspect_images[selected], use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")


# =====================================================================
# STEP 1: Extraction Output
# =====================================================================

with st.expander("STEP 1 — Extraction Output (Mock)"):
    extracted = extraction_results[selected]
    st.json(extracted)


# =====================================================================
# STEP 2: Validation (Multi-tiered Framework)
# =====================================================================

with st.expander("STEP 2 — Multi-Tiered Validation (Rule → RAG → Cosine → HITL)"):
    validation = validate_data(
        selected,
        extracted,
        neurology_notes[selected],
        radiology_reports[selected]
    )

    # --------------------
    st.subheader("1) 🔎 Rule-Based Verification")
    for m in validation["Rule"]:
        if "❗" in m:
            st.markdown(highlight_red(m), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(m), unsafe_allow_html=True)

    # --------------------
    st.markdown("---")
    st.subheader("2) 📚 RAG Verification (Semantic vs Original Note)")
    for m in validation["RAG"]:
        if "❗" in m:
            st.markdown(highlight_red(m), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(m), unsafe_allow_html=True)

    # --------------------
    st.markdown("---")
    st.subheader("3) 📈 Cosine Similarity Flagging (Population-level anomaly detection)")
    for m in validation["Cosine"]:
        if "❗" in m:
            st.markdown(highlight_red(m), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(m), unsafe_allow_html=True)

    # --------------------
    st.markdown("---")
    st.subheader("4) 🧑‍⚕️ HITL Review Recommendation")
    if "❗" in validation["HITL"]:
        st.markdown(highlight_red(validation["HITL"]), unsafe_allow_html=True)
    else:
        st.markdown(highlight_green(validation["HITL"]), unsafe_allow_html=True)


# =====================================================================
# STEP 3: HITL-Assisted Correction (New Section)
# =====================================================================

st.markdown("---")
with st.expander("STEP 3 — Corrected Structured Output (HITL-assisted)"):

    corrected, changed = hitl_correction(selected, extracted, validation)

    if changed:
        st.markdown(
            "<p style='color:#cc0000;font-weight:700;font-size:18px;'>"
            "⚠️ Validation detected issues — corrected values applied (HITL simulation)</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='color:#008800;font-weight:700;font-size:18px;'>"
            "✔ No issues — extracted data accepted without correction</p>",
            unsafe_allow_html=True
        )

    st.json(corrected)


# =====================================================================
# STEP 4: Prediction (now based on corrected values)
# =====================================================================

with st.expander("STEP 4 — Prediction (Mock Model)"):

    # prediction uses corrected ASPECTS
    if corrected["ASPECTS"] <= 5:
        prob = 0.55
    elif corrected["ASPECTS"] <= 7:
        prob = 0.32
    else:
        prob = 0.10

    st.metric("Predicted Poor Outcome Probability", f"{prob:.2f}")
    st.write("Prediction is based on corrected structured data.")


# =====================================================================
# STEP 5: CSV Export
# =====================================================================

final_df = pd.DataFrame([{**corrected, "Predicted_Poor_Outcome_Probability": prob}])

st.download_button(
    label="⬇️ Download Final Structured Output (CSV)",
    data=final_df.to_csv(index=False),
    mime="text/csv",
    file_name=f"{selected}_corrected_output.csv"
)
