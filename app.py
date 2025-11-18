import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")
st.title("🧠 Stroke Pipeline Demo")
st.write("A multimodal extraction–validation–prediction pipeline using text, imaging, and structured variables.")

# =====================================================================
# CSS: 카드형 박스 디자인
# =====================================================================
card_style = """
<div style="
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    margin-bottom: 25px;
    color: #000000;
">
"""

# =====================================================================
# 0) Neurology Notes – ASPECT 5 / 6 / 10
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

**Imaging Impression:**  
CT: Early ischemic change in left MCA territory.

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

**Imaging Impression:**  
CT: Subtle hypoattenuation in right MCA territory (early ischemic change).
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

**Imaging Impression:**  
CT: No acute finding. Normal brain.
"""
}

# =====================================================================
# Radiology Reports – ASPECT 문구 제거 (CT/MRI 소견)
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

# =====================================================================
# ASPECT Images (ground truth 표현 없음)
# =====================================================================

aspect_images = {
    "Example Case 1": "images/aspects1.png",
    "Example Case 2": "images/aspects2.png",
    "Example Case 3": "images/aspects3.png"
}

# =====================================================================
# Extraction Results – 논문 기반 구조화 변수
# =====================================================================

extraction_results = {
    "Example Case 1": {
        # ❌ 오류 포함 (의도된 LLM hallucination)
        "Chief_Complaint": "Right-sided weakness, dysarthria",
        "Onset_Time": "2018-08-25 21:40",
        "NIHSS": 9,
        "Hypertension": "yes",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",

        # ❌ 오류 intentionally 넣음
        "ASPECTS": 7,                # 실제는 5 → RAG에서 잡힘
        "tPA_Administered": "no",    # 실제는 yes → Rule-based에서 잡힘
        "Weakness_Side": "bilateral",# 실제는 right → similarity flag에서 잡힘

        "SBP": 178
    },

    "Example Case 2": {
        # ❌ 오류 포함 (의도된 LLM hallucination)
        "Chief_Complaint": "Aphasia, left arm heaviness",
        "Onset_Time": "2018-09-03 19:10",
        "NIHSS": 5,

        # ❌ 오류 intentionally 넣음
        "Hypertension": "no",       # 실제는 yes → rule-based에서 잡힘
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 9,               # 실제는 6 → RAG + similarity에서 잡힘

        "tPA_Administered": "no",
        "Weakness_Side": "left",
        "SBP": 162
    },

    "Example Case 3": {
        # ✔ 정상 추출됨 (검증 PASS)
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
# Validation Logic
# =====================================================================

# =====================================================================
# Validation Logic – 논문 2.3.2 Multi-tiered Validation Framework
# =====================================================================

def validate_data(selected, extracted, note_text, radiology_text):
    val = {}

    # ===============================================================
    # 1. RULE-BASED VERIFICATION (format / range only)
    # ===============================================================
    rule_msgs = []
    
    # binary check
    binary_fields = ["Hypertension", "Diabetes", "Atrial_Fibrillation", "tPA_Administered"]
    for field in binary_fields:
        if extracted[field] not in ["yes", "no", "unknown"]:
            rule_msgs.append(f"❗ {field} contains invalid value '{extracted[field]}'. Expected yes/no/unknown.")
    
    # NIHSS range
    if not (0 <= extracted["NIHSS"] <= 42):
        rule_msgs.append("❗ NIHSS is outside valid range (0–42).")
        
    # ASPECTS range
    if not (0 <= extracted["ASPECTS"] <= 10):
        rule_msgs.append("❗ ASPECTS is outside valid range (0–10).")
    
    # SBP physiological plausibility
    if extracted["SBP"] < 40 or extracted["SBP"] > 300:
        rule_msgs.append("❗ SBP value is physiologically implausible.")
    
    if not rule_msgs:
        rule_msgs.append("✔ No format/range issues detected.")
    
    val["Rule-Based"] = rule_msgs


    # ===============================================================
    # 2. RAG VERIFICATION (semantic match vs original note)
    # ===============================================================
    rag_msgs = []
    full_text = (note_text + " " + radiology_text).lower()

    # CASE 1 tPA mismatch
    if selected == "Example Case 1":
        if "tpa" in full_text and extracted["tPA_Administered"] != "yes":
            rag_msgs.append("❗ RAG: Original note indicates tPA was administered, but extraction says otherwise.")

        if "right" in full_text and extracted["Weakness_Side"] != "right":
            rag_msgs.append("❗ RAG: Weakness side mismatch vs original note.")

        if "acute infarction" in full_text and extracted["ASPECTS"] > 7:
            rag_msgs.append("❗ RAG: ASPECTS too high relative to MRI findings.")

    # CASE 2 mismatch
    if selected == "Example Case 2":
        if "early" in full_text and extracted["ASPECTS"] >= 8:
            rag_msgs.append("❗ RAG: ASPECTS inconsistent with 'early ischemic change' MRI description.")

        if "bp" in full_text and extracted["Hypertension"] == "no" and extracted["SBP"] >= 160:
            rag_msgs.append("❗ RAG: Hypertension mismatch relative to note context.")

    if not rag_msgs:
        rag_msgs.append("✔ RAG did not detect semantic inconsistencies.")

    val["RAG"] = rag_msgs


    # ===============================================================
    # 3. COSINE SIMILARITY FLAGGING (population-level anomaly)
    # ===============================================================
    cos_msgs = []

    # Mock similarity scores
    if selected == "Example Case 1":
        sim = 0.71  # anomaly
    elif selected == "Example Case 2":
        sim = 0.78  # anomaly
    else:
        sim = 0.92  # normal

    if sim < 0.82:
        cos_msgs.append(
            f"❗ Cosine similarity = {sim:.2f} (<0.82). Pattern outlier detected relative to 200 validated historical records."
        )
        cos_msgs.append(
            "   (Note: This does NOT perform clinical reasoning — it detects rare or atypical variable combinations.)"
        )
    else:
        cos_msgs.append(f"✔ Cosine similarity = {sim:.2f} (within normal validated pattern).")

    val["Cosine"] = cos_msgs


    # ===============================================================
    # 4. HITL REVIEW
    # ===============================================================
    flagged = any("❗" in msg for stage in val.values() for msg in stage)

    if flagged:
        val["HITL"] = (
            "🔎 Record requires clinician review. "
            "Automated steps detected issues via RAG or population-level inconsistency."
        )
    else:
        val["HITL"] = (
            "✔ No issues detected. Record eligible for automated acceptance "
            "(HITL reviewer will still audit a random 10% of non-flagged cases)."
        )

    return val


# =====================================================================
# UI 시작
# =====================================================================

selected = st.selectbox("Select Example Case", list(neurology_notes.keys()))

col1, col2, col3 = st.columns([1.3, 1.3, 1])

# Neurology Note
with col1:
    st.markdown(card_style + "<h3>📝 Neurology Note</h3>" +
                neurology_notes[selected] + "</div>", unsafe_allow_html=True)

# Radiology Report
with col2:
    st.markdown(card_style + "<h3>📄 Radiology Report</h3>" +
                radiology_reports[selected] + "</div>", unsafe_allow_html=True)

# ASPECT Image
with col3:
    st.markdown(card_style + "<h3>🖼️ ASPECT CT Image</h3>", unsafe_allow_html=True)
    st.image(aspect_images[selected], use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Extraction
# ============================================================

with st.expander("1. Extraction Output (Mock)"):
    extracted = extraction_results[selected]
    st.json(extracted)

# ============================================================
# Validation
# ============================================================

with st.expander("2. Validation Steps (Based on Study Framework)"):

    results = validate_data(
        selected,
        extracted,
        neurology_notes[selected],
        radiology_reports[selected]
    )

    st.subheader("1) 🔎 Rule-Based Verification (Format / Range)")
    for m in results["Rule-Based"]:
        st.write(m)

    st.markdown("---")
    st.subheader("2) 📚 RAG Verification (Semantic Check vs Original Note)")
    for m in results["RAG"]:
        st.write(m)

    st.markdown("---")
    st.subheader("3) 📈 Cosine Similarity Flagging (Population-Level Pattern Check)")
    for m in results["Cosine"]:
        st.write(m)

    st.markdown("---")
    st.subheader("4) 🧑‍⚕️ Human-in-the-Loop Review")
    st.write(results["HITL"])


# ============================================================
# Prediction
# ============================================================

with st.expander("3. Prediction (Mock)"):

    if extracted["ASPECTS"] <= 5:
        prob = 0.55
    elif extracted["ASPECTS"] <= 7:
        prob = 0.32
    else:
        prob = 0.10

    st.metric("Predicted Poor Outcome Probability", f"{prob:.2f}")

    st.markdown("Prediction displayed numerically (gauge disabled).")

# ============================================================
# CSV Export
# ============================================================

final_df = pd.DataFrame([{
    **extracted,
    "Predicted_Poor_Outcome_Probability": prob
}])

st.download_button(
    label="⬇️ Download Final Structured Data (CSV)",
    data=final_df.to_csv(index=False),
    mime="text/csv",
    file_name=f"{selected}_structured_output.csv"
)


