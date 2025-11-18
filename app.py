import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")
st.title("🧠 Stroke Pipeline Demo")
st.write("A multimodal extraction–validation–correction–prediction pipeline for stroke outcome research.")

# ===============================================================
# GLOBAL CSS (Improved Modern UI)
# ===============================================================

st.markdown("""
<style>
.step-card {
    background: #ffffff;
    padding: 20px 25px;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
    margin-top: 15px;
    border-left: 8px solid #0047AB;
}
.section-header {
    padding: 10px 15px;
    background: #0047AB;
    color: white;
    border-radius: 8px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

# Highlight blocks
def h_red(m): 
    return f"<div style='color:#b30000;font-weight:600;margin-bottom:6px;'>❗ {m}</div>"

def h_green(m): 
    return f"<div style='color:#006622;font-weight:600;margin-bottom:6px;'>✔ {m}</div>"

# Simple card for Source documents
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
# FLOW VISUALIZATION — Pipeline Overview
# ===============================================================

st.markdown("""
<div style='display:flex;justify-content:space-between;margin:30px 0 20px 0;'>
    <div style="background:#e7f0ff;padding:14px 22px;border-radius:10px;border-left:6px solid #0047AB;width:22%;text-align:center;">
        <b>1. Extraction</b>
    </div>
    <div style="background:#e7f0ff;padding:14px 22px;border-radius:10px;border-left:6px solid #0047AB;width:22%;text-align:center;">
        <b>2. Validation</b>
    </div>
    <div style="background:#e7f0ff;padding:14px 22px;border-radius:10px;border-left:6px solid #0047AB;width:22%;text-align:center;">
        <b>3. Correction</b>
    </div>
    <div style="background:#e7f0ff;padding:14px 22px;border-radius:10px;border-left:6px solid #0047AB;width:22%;text-align:center;">
        <b>4. Prediction</b>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# SOURCE DOCUMENTS (Neurology Note, MRI, ASPECTS)
# =====================================================================


# =====================================================================
# 0) Neurology Notes – ASPECT 5 / 6 / 10 (original detailed version)
# =====================================================================

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

**Social History:**  
- Smoking 0.5 pack/day × 10 yrs  
- Alcohol 2 drinks/day × 15 yrs  

**Present Illness:**  
Sudden right arm and leg weakness with dysarthria while at home.  
Persistent symptoms; required assistance for ambulation.

**Vitals:** BP 178/92, HR 84, RR 18, Temp 36.8°C  
**Initial NIHSS:** 9  

**Neurological Examination:**  
- Mental: Alert, mild dysarthria  
- CN: Right facial droop  
- Motor: RUE 3/5, RLE 3/5; LUE/LLE 5/5  
- Sensory: Right side decreased light touch  
- Cerebellum: No ataxia  
- Reflexes: Normal  

**Treatment:**  
IV tPA administered at 22:35 (0.9 mg/kg).
""",

    "Example Case 2":
    """
**Chief Complaint:**  
Aphasia and left-sided heaviness  

**Onset:**  
September 3, 2018 at 19:10

**Past Medical History:**  
- Diabetes mellitus  
- Hypertension  
- No AFib  
- No prior stroke  

**Social History:**  
- Non-smoker  
- Occasional alcohol  

**Present Illness:**  
Expressive difficulty + left arm heaviness; symptoms fluctuated then persisted.

**Vitals:** BP 162/88, HR 76, RR 18, Temp 37°C  
**Initial NIHSS:** 5  

**Neurological Examination:**  
- Mental: Mild aphasia  
- Motor: LUE 4+/5, LLE 4/5  
- Sensory: Intact  
- CN & Cerebellum: Normal 
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
- Treated pulmonary TB  
- Chronic hepatitis B  

**Social History:**  
- Smoking 0.5 pack/day × 10 yrs  
- Alcohol 1–2 drinks/day × 20 yrs  

**Present Illness:**  
Dizziness, chills, transient bilateral leg weakness while playing billiards.

**Vitals:** BP 211/90, HR 73, RR 20, Temp 36.7°C  
**Initial NIHSS:** 0  

**Neurological Examination:**  
UE 5/5, LE 4+/5 bilaterally; CN intact; cerebellum intact; reflexes normal.
"""
}


# =====================================================================
# Radiology Reports – original detailed MRI text
# =====================================================================

radiology_reports = {
    "Example Case 1":
    """
MRI BRAIN WITH AND WITHOUT CONTRAST
Technique:
Multiplanar, multisequence MRI of the brain including T1, T2, FLAIR, DWI/ADC, GRE/SWI, and post-contrast imaging. TOF MRA of the intracranial circulation was obtained.
Findings:
DWI shows restricted diffusion involving the left insula, left frontal operculum, and anterior parietal cortex, consistent with an acute infarction in the left MCA territory.
ADC maps confirm low signal corresponding to areas of restricted diffusion.
FLAIR demonstrates mild cortical swelling and subtle hyperintensity in the same regions, compatible with early ischemic change.
No intracranial hemorrhage is noted on GRE/SWI.
Major intracranial arteries: TOF MRA reveals decreased flow-related signal in the proximal left M2/M3 branches, without complete occlusion.
No mass effect significant enough to shift midline; ventricles remain symmetric.
Basal ganglia, thalami, brainstem, and cerebellum are preserved.
No abnormal meningeal or parenchymal enhancement following contrast.
Conclusion:
Findings consistent with acute ischemic infarction in the left MCA territory, with corresponding cortical restricted diffusion and early FLAIR changes. No hemorrhagic transformation.
""",

    "Example Case 2":
    """
MRI BRAIN WITHOUT CONTRAST
Technique:
Multiplanar, multisequence MRI including T1, T2, FLAIR, DWI/ADC, and SWI. TOF intracranial MRA performed.
Findings:
DWI shows punctate to patchy areas of mildly increased signal in the left basal ganglia and parietal opercular regions, suspicious for early acute ischemia.
ADC demonstrates subtle low-signal correlation but less pronounced than in established infarction.
FLAIR shows faint cortical/subcortical hyperintensity without significant swelling.
No hemorrhage on SWI.
Intracranial vasculature: TOF MRA shows mild irregularity of the left M2 segment, without definite large-vessel occlusion.
Ventricles, midline structures, posterior fossa appear normal.
No mass lesion or abnormal enhancement.
Conclusion:
MRI findings suggest early left MCA territory ischemia, with mild cortical diffusion restriction but no hemorrhage or large-vessel occlusion.
""",

    "Example Case 3":
    """
MRI BRAIN WITH AND WITHOUT CONTRAST
Technique:
Multiplanar T1, T2, FLAIR, DWI/ADC, GRE/SWI, and post-contrast sequences. 3D TOF MRA obtained.
Findings:
Parenchyma: No diffusion restriction. No areas of abnormal T2/FLAIR hyperintensity. Gray–white differentiation preserved.
No hemorrhage on GRE/SWI.
No mass lesion, midline shift, or extra-axial collection.
Ventricular system normal in size and configuration.
Posterior fossa (brainstem and cerebellum) unremarkable.
Intracranial circulation: TOF MRA demonstrates normal flow-related signal in bilateral ICA, MCA, ACA, PCA territories. No stenosis or occlusion.
Enhancement: No abnormal parenchymal or leptomeningeal enhancement.
Paranasal sinuses/orbits normal.
Conclusion:
Normal MRI brain. No acute infarction or structural abnormality detected.
"""
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
col1, col2, col3 = st.columns([1.3, 1.3, 1])


# =====================================================================
# Display Neurology Note / MRI / ASPECTS Images
# =====================================================================

with col1:
    st.markdown(
        card_style +
        step_badge("Source Document 1") +
        "<h3>📝 Neurology Note</h3>" +
        neurology_notes[selected] +
        "</div>", unsafe_allow_html=True
    )

with col2:
    st.markdown(
        card_style +
        step_badge("Source Document 2") +
        "<h3>📄 Radiology Report</h3>" +
        radiology_reports[selected] +
        "</div>", unsafe_allow_html=True
    )

with col3:
    st.markdown(
        card_style +
        step_badge("Source Document 3") +
        "<h3>🖼️ ASPECT CT Image</h3>",
        unsafe_allow_html=True
    )
    st.image(aspect_images[selected], use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 1 — Extraction
# =====================================================================

st.markdown("<div class='step-card'><h3>STEP 1 — Extraction Output (Mock)</h3>", unsafe_allow_html=True)
extracted = extraction_results[selected]
st.json(extracted)
st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 2 — Validation (Rule → RAG → Cosine → HITL)
# =====================================================================

st.markdown("<div class='section-header'><h2>🔍 Multi-tiered Validation</h2></div>", unsafe_allow_html=True)
st.markdown("<div class='step-card'>", unsafe_allow_html=True)

validation = validate_data(
    selected,
    extracted,
    neurology_notes[selected],
    radiology_reports[selected]
)

# Rule-based
st.subheader("1) Rule-Based Verification")
for m in validation["Rule"]:
    st.markdown(h_red(m) if "❗" in m else h_green(m), unsafe_allow_html=True)

# RAG
st.markdown("---")
st.subheader("2) RAG Verification")
for m in validation["RAG"]:
    st.markdown(h_red(m) if "❗" in m else h_green(m), unsafe_allow_html=True)

# Cosine
st.markdown("---")
st.subheader("3) Cosine Similarity Flagging")
for m in validation["Cosine"]:
    st.markdown(h_red(m) if "❗" in m else h_green(m), unsafe_allow_html=True)

# HITL
st.markdown("---")
st.subheader("4) HITL Recommendation")
st.markdown(
    h_red(validation["HITL"]) if "❗" in validation["HITL"] else h_green(validation["HITL"]),
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 3 — HITL-Assisted Correction
# =====================================================================

st.markdown("<div class='step-card'><h3>STEP 3 — Corrected Structured Output (HITL)</h3>", unsafe_allow_html=True)

corrected, changed = hitl_correction(selected, extracted, validation)

if changed:
    st.markdown("<p style='color:#cc0000;font-weight:700;'>⚠️ Corrections applied</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color:#008800;font-weight:700;'>✔ No correction needed</p>", unsafe_allow_html=True)

st.json(corrected)
st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 4 — Prediction
# =====================================================================

st.markdown("<div class='step-card'><h3>STEP 4 — Prediction</h3>", unsafe_allow_html=True)

if corrected["ASPECTS"] <= 5:
    prob = 0.55
elif corrected["ASPECTS"] <= 7:
    prob = 0.32
else:
    prob = 0.10

# Gradient Risk Bar
st.markdown(f"""
<div style='height:22px;border-radius:12px;
    background:linear-gradient(90deg, #ff6666 {prob*100}%, #e0e0e0 {prob*100}%);'>
</div>
<p style='font-size:17px;font-weight:600;'>{prob*100:.1f}% predicted poor outcome</p>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 5 — Export
# =====================================================================

final_df = pd.DataFrame([{**corrected, "Predicted_Poor_Outcome_Probability": prob}])

st.download_button(
    label="⬇️ Download Final Structured Output (CSV)",
    data=final_df.to_csv(index=False),
    mime="text/csv",
    file_name=f"{selected}_corrected_output.csv"
)

