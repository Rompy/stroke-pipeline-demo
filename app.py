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


# =====================================================================
# 0) Neurology Notes UPDATED (all PMH variables fully included)
# =====================================================================

neurology_notes = {
    "Example Case 1":
    """
**Demographics:**  
67-year-old female  

**Chief Complaint:**  
Right-sided weakness and slurred speech  

**Onset:**  
August 25, 2018 at 21:40 (LKW 21:30)

**Past Medical History:**  
- Hypertension (poorly controlled)  
- Diabetes mellitus  
- Dyslipidemia  
- No known atrial fibrillation  
- No cardiovascular disease  
- No prior CVA  
- No malignancy  
- No ESRD  

**Social History:**  
- Smoking 0.5 pack/day × 10 yrs  
- Alcohol 2 drinks/day × 15 yrs  

**Present Illness:**  
Sudden right arm and leg weakness with dysarthria while at home.  
Symptoms persisted and required assistance for ambulation.

**Vitals:** BP 178/92, HR 84, RR 18, Temp 36.8°C  
**Initial NIHSS:** 9  

**Neurological Examination:**  
- Mental: Alert, mild dysarthria  
- Motor: RUE 3/5, RLE 3/5; LUE/LLE 5/5  
- Sensory: Right-sided decreased light touch  
- CN: Right facial droop  
- Cerebellum: No ataxia  
- Reflexes: Normal  

**Treatment:**  
IV tPA administered at 22:35 (0.9 mg/kg).  
No intra-arterial intervention performed.
""",

    "Example Case 2":
    """
**Demographics:**  
73-year-old male  

**Chief Complaint:**  
Aphasia and left-sided heaviness  

**Onset:**  
September 3, 2018 at 19:10

**Past Medical History:**  
- Diabetes mellitus  
- Hypertension  
- Dyslipidemia  
- Old silent lacunar infarct  
- No AFib  
- No malignancy  
- No ESRD  
- No cardiovascular disease  

**Social History:**  
- Non-smoker  
- Occasional alcohol  

**Present Illness:**  
Expressive difficulty and left arm heaviness.  
Symptoms fluctuated then persisted.

**Vitals:** BP 162/88, HR 76, RR 18, Temp 37°C  
**Initial NIHSS:** 5  

**Neurological Examination:**  
- Mental: Mild aphasia  
- Motor: LUE 4+/5, LLE 4/5  
- Sensory: Intact  
- Cranial Nerves: Normal  
- Cerebellum: Normal  

**Treatment:**  
No tPA administered.  
No intra-arterial intervention.
""",

    "Example Case 3":
    """
**Demographics:**  
62-year-old male  

**Chief Complaint:**  
Presyncope  

**Onset:**  
August 24, 2018 at 23:30

**Past Medical History:**  
- Hypertension  
- Diabetes mellitus  
- Treated pulmonary TB  
- Chronic hepatitis B  
- No AFib  
- No cardiovascular disease  
- No dyslipidemia  
- No malignancy  
- No ESRD  
- No prior stroke  

**Social History:**  
- Smoking 0.5 pack/day × 10 yrs  
- Alcohol 1–2 drinks/day × 20 yrs  

**Present Illness:**  
Dizziness, chills, transient bilateral leg weakness during activity (billiards).  

**Vitals:** BP 211/90, HR 73, RR 20, Temp 36.7°C  
**Initial NIHSS:** 0  

**Neurological Examination:**  
UE 5/5, LE 4+/5 bilaterally; CN intact; cerebellum intact; reflexes normal.

**Treatment:**  
No tPA, no IA intervention.
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

aspect_images = {
    "Example Case 1": "images/aspects1.png",
    "Example Case 2": "images/aspects2.png",
    "Example Case 3": "images/aspects3.png"
}

# ===============================================================
# Extraction Results — FULL variable set
# ===============================================================

extraction_results = {
    "Example Case 1": {
        "Age": 67,
        "Sex": "female",

        "Hypertension": "yes",
        "Diabetes": "yes",
        "Dyslipidemia": "yes",
        "Cardiovascular_Disease": "no",
        "Atrial_Fibrillation": "no",
        "Old_CVA": "no",
        "Malignancy": "no",
        "ESRD": "no",

        "MRI_Acute_Infarct": "yes",
        "MRI_No_Lesion": "no",
        "MRI_Other_Lesion": "no",

        "NIHSS": 9,
        "ASPECTS": 7,

        "tPA_Administered": "no",          # hallucinated
        "IA_Thrombectomy": "no",

        "Weakness_Side": "bilateral",      # hallucinated
        "SBP": 178
    },

    "Example Case 2": {
        "Age": 73,
        "Sex": "male",

        "Hypertension": "no",              # hallucinated
        "Diabetes": "yes",
        "Dyslipidemia": "yes",
        "Cardiovascular_Disease": "no",
        "Atrial_Fibrillation": "no",
        "Old_CVA": "yes",
        "Malignancy": "no",
        "ESRD": "no",

        "MRI_Acute_Infarct": "yes",
        "MRI_No_Lesion": "no",
        "MRI_Other_Lesion": "no",

        "NIHSS": 5,
        "ASPECTS": 9,                       # hallucinated

        "tPA_Administered": "no",
        "IA_Thrombectomy": "no",

        "Weakness_Side": "left",
        "SBP": 162
    },

    "Example Case 3": {
        "Age": 62,
        "Sex": "male",

        "Hypertension": "yes",
        "Diabetes": "yes",
        "Dyslipidemia": "no",
        "Cardiovascular_Disease": "no",
        "Atrial_Fibrillation": "no",
        "Old_CVA": "no",
        "Malignancy": "no",
        "ESRD": "no",

        "MRI_Acute_Infarct": "no",
        "MRI_No_Lesion": "yes",
        "MRI_Other_Lesion": "no",

        "NIHSS": 0,
        "ASPECTS": 10,

        "tPA_Administered": "no",
        "IA_Thrombectomy": "no",

        "Weakness_Side": "bilateral",
        "SBP": 211
    }
}


# =====================================================================
# VALIDATION LOGIC
# =====================================================================

def validate_data(selected, extracted, note_text, radiology_text):

    full_text = (note_text + " " + radiology_text).lower()
    val = {}
    rule_msgs = []

    # ---- Binary field checking ----
    binary_fields = [
        "Hypertension", "Diabetes", "Dyslipidemia", "Cardiovascular_Disease",
        "Atrial_Fibrillation", "Old_CVA", "Malignancy", "ESRD",
        "MRI_Acute_Infarct", "MRI_No_Lesion", "MRI_Other_Lesion",
        "tPA_Administered", "IA_Thrombectomy"
    ]

    for f in binary_fields:
        if extracted[f] not in ["yes", "no", "unknown"]:
            rule_msgs.append(f"❗ {f}: invalid binary (yes/no/unknown expected).")

    # NIHSS
    if not (0 <= extracted["NIHSS"] <= 42):
        rule_msgs.append("❗ NIHSS outside valid range.")

    # ASPECTS
    if not (0 <= extracted["ASPECTS"] <= 10):
        rule_msgs.append("❗ ASPECTS outside valid range.")

    # SBP
    if extracted["SBP"] < 40 or extracted["SBP"] > 300:
        rule_msgs.append("❗ SBP physiologically implausible.")

    if not rule_msgs:
        rule_msgs.append("✔ Passed all rule-based format checks.")

    val["Rule"] = rule_msgs

    # ---- RAG checks ----
    rag = []

    if selected == "Example Case 1":
        if "tpa" in full_text and extracted["tPA_Administered"] != "yes":
            rag.append("❗ tPA mismatch: note indicates tPA was given.")
        if "right" in full_text and extracted["Weakness_Side"] != "right":
            rag.append("❗ Weakness side mismatch.")
        if extracted["MRI_Acute_Infarct"] == "yes" and extracted["ASPECTS"] > 7:
            rag.append("❗ ASPECT too high for acute MCA infarction.")

    if selected == "Example Case 2":
        if extracted["Hypertension"] == "no" and extracted["SBP"] >= 160:
            rag.append("❗ High BP suggests hypertension.")
        if extracted["MRI_Acute_Infarct"] == "yes" and extracted["ASPECTS"] >= 8:
            rag.append("❗ ASPECT inconsistent with early ischemia severity.")

    if not rag:
        rag.append("✔ No semantic mismatch.")

    val["RAG"] = rag

    # ---- Cosine similarity mock ----
    cos = []
    sim = 0.71 if selected=="Example 1" else (0.78 if selected=="Example 2" else 0.92)

    if sim < 0.82:
        cos.append(f"❗ Cosine similarity {sim:.2f} → atypical pattern")
    else:
        cos.append(f"✔ Cosine similarity {sim:.2f} → typical pattern")

    val["Cosine"] = cos

    flagged = any("❗" in msg for stage in val.values() for msg in stage)
    val["HITL"] = "🔎 Needs manual review." if flagged else "✔ Auto-acceptable."

    return val
# =====================================================================
# 2) HITL ASSISTED CORRECTION MODULE
# =====================================================================

def hitl_correction(selected, extracted, validation):

    corrected = extracted.copy()

    if "❗" not in str(validation):
        return corrected, False

    # ---- Example Case corrections (mock rules) ----
    if selected == "Example Case 1":
        corrected["tPA_Administered"] = "yes"
        corrected["Weakness_Side"] = "right"
        corrected["ASPECTS"] = 5

    if selected == "Example Case 2":
        corrected["Hypertension"] = "yes"
        corrected["ASPECTS"] = 6

    return corrected, True


# =====================================================================
# ===========================  UI START  ===============================
# =====================================================================

selected = st.selectbox("Select Example Case", list(neurology_notes.keys()))
col1, col2, col3 = st.columns([1.3, 1.3, 1])


# =====================================================================
# Neurology Note / MRI / CT ASPECT Panel
# =====================================================================

with col1:
    st.markdown(
        card_style +
        step_badge("Source Document 1") +
        "<h3>📝 Neurology Note</h3>" +
        neurology_notes[selected] +
        "</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        card_style +
        step_badge("Source Document 2") +
        "<h3>📄 Radiology Report (MRI)</h3>" +
        radiology_reports[selected] +
        "</div>",
        unsafe_allow_html=True
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
# STEP 1: Extraction Output
# =====================================================================

with st.expander("STEP 1 — Extraction Output (Mock, with intentional hallucinations)"):
    extracted = extraction_results[selected]
    st.json(extracted)


# =====================================================================
# STEP 2: Multi-Tier Validation
# =====================================================================

with st.expander("STEP 2 — Multi-Tiered Validation (Rule → RAG → Cosine → HITL)"):

    validation = validate_data(
        selected,
        extracted,
        neurology_notes[selected],
        radiology_reports[selected]
    )

    # ---- Rule-based ----
    st.subheader("1) 🔎 Rule-Based Verification")
    for msg in validation["Rule"]:
        if "❗" in msg:
            st.markdown(highlight_red(msg), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(msg), unsafe_allow_html=True)

    # ---- RAG ----
    st.markdown("---")
    st.subheader("2) 📚 RAG Verification (Semantic vs Original Note)")
    for msg in validation["RAG"]:
        if "❗" in msg:
            st.markdown(highlight_red(msg), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(msg), unsafe_allow_html=True)

    # ---- Cosine Similarity ----
    st.markdown("---")
    st.subheader("3) 📈 Cosine Similarity Flagging")
    for msg in validation["Cosine"]:
        if "❗" in msg:
            st.markdown(highlight_red(msg), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(msg), unsafe_allow_html=True)

    # ---- HITL Recommendation ----
    st.markdown("---")
    st.subheader("4) 🧑‍⚕️ HITL Review Recommendation")
    if "❗" in validation["HITL"]:
        st.markdown(highlight_red(validation["HITL"]), unsafe_allow_html=True)
    else:
        st.markdown(highlight_green(validation["HITL"]), unsafe_allow_html=True)


# =====================================================================
# STEP 3: HITL-Assisted Correction
# =====================================================================

st.markdown("---")
with st.expander("STEP 3 — Corrected Structured Output (HITL-assisted)"):

    corrected, changed = hitl_correction(selected, extracted, validation)

    if changed:
        st.markdown(
            "<p style='color:#cc0000;font-weight:700;font-size:18px;'>"
            "⚠️ Issues detected — clinician-like corrections applied</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='color:#008800;font-weight:700;font-size:18px;'>"
            "✔ No corrections needed</p>",
            unsafe_allow_html=True
        )

    st.json(corrected)


# =====================================================================
# STEP 4: Prediction (Mock model)
# =====================================================================

with st.expander("STEP 4 — Prediction (Mock Model Based on Corrected ASPECTS)"):

    # Simple rule-based probability
    if corrected["ASPECTS"] <= 5:
        prob = 0.55
    elif corrected["ASPECTS"] <= 7:
        prob = 0.32
    else:
        prob = 0.10

    st.write("Prediction is based on corrected structured data.")

    # Gradient Risk Bar
    st.markdown(f"""
    <div style='height:22px;border-radius:12px;margin-top:12px;
        background:linear-gradient(90deg, #ff6666 {prob*100}%, #e0e0e0 {prob*100}%);'>
    </div>
    <p style='font-size:16px;font-weight:600;margin-top:6px;'>{prob*100:.1f}% predicted poor outcome</p>
    """, unsafe_allow_html=True)


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


