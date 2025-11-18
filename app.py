import streamlit as st
import pandas as pd

st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")
st.title("Stroke Pipeline Demo (Enhanced Mock Version)")
st.write("Multimodal extraction–validation–prediction pipeline using text and imaging sources (Mock).")

# ============================================================
# 0) Neurology Notes (ASPECT 5 / 6 / 10 완전 재작성)
# ============================================================

neurology_notes = {
    "Example Case 1":  # ASPECTS = 5, tPA administered
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
The patient developed sudden right arm and leg weakness with dysarthria while at home.  
Symptoms were abrupt and persistent.  
Required assistance for ambulation.  
No LOC or seizure activity.

**Vital Signs on Arrival:** BP 178/92, HR 84, RR 18, Temp 36.8°C  
**Initial NIHSS:** 9  

**Neurological Examination:**  
- Mental Status: Alert, mildly dysarthric  
- CN: Right facial droop, pupils normal  
- Motor: RUE 3/5, RLE 3/5; LUE/LLE 5/5  
- Sensory: Decreased light touch on right side  
- Cerebellum: No ataxia on left  
- Reflexes: Normal, no Babinski  

**Imaging Impression:**  
Non-contrast CT: Early ischemic changes in left MCA territory.  
**ASPECTS ≈ 5**

**Treatment:**  
IV tPA administered at 22:35 (0.9 mg/kg).
""",

    "Example Case 2":  # ASPECTS = 6
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
- Occasional alcohol use  

**Present Illness:**  
Patient developed difficulty expressing words and mild left arm heaviness.  
Symptoms fluctuated initially but became more noticeable.  
No headache or seizure-like activity.

**Vital Signs on Arrival:** BP 162/88, HR 76, RR 18, Temp 37.0°C  
**Initial NIHSS:** 5  

**Neurological Examination:**  
- Mental Status: Alert, mildly aphasic  
- CN: No gaze deviation  
- Motor: LUE 4+/5, LLE 4/5; Right side 5/5  
- Sensory: Intact bilaterally  
- Cerebellum: Intact  
- Reflexes: Normal  

**Imaging Impression:**  
CT: Subtle decreased attenuation in right MCA territory.  
**ASPECTS ≈ 6**
""",

    "Example Case 3":  # ASPECTS = 10 (normal brain)
    """
**Chief Complaint:**  
Presyncope  

**Onset:**  
August 24, 2018 at 23:30

**Past Medical History:**  
- Hypertension  
- Diabetes mellitus  
- Treated pulmonary tuberculosis  
- Chronic hepatitis B  
- No regular medications  

**Social History:**  
- Smoking 0.5 pack/day × 10 yrs  
- Alcohol 1–2 drinks/day × 20 yrs  

**Present Illness:**  
While playing billiards, the patient experienced dizziness, chills, and transient bilateral leg weakness.  
No clear unilateral symptoms.  
ADL baseline intact.

**Vital Signs on Arrival:** BP 211/90, HR 73, RR 20, Temp 36.7°C  
**Initial NIHSS:** 0  

**Neurological Examination:**  
- Mental Status: Alert  
- CN: Normal  
- Motor: UE 5/5; LE 4+/5 bilaterally  
- Sensory: Intact  
- Cerebellum: FTN/HTS intact  
- Reflexes: No pathologic signs  

**Imaging Impression:**  
CT brain: No acute lesion, no hemorrhage.  
**ASPECTS = 10**
"""
}

# ============================================================
# Radiology Reports (이미 제공된 내용 유지, 케이스별 차등)
# ============================================================

radiology_reports = {
    "Example Case 1":  # ASPECTS 5
    """
CT/MRI REPORT:

Findings:
Acute ischemic changes involving the left MCA territory.
Loss of gray–white differentiation and mild sulcal effacement noted.
No intracranial hemorrhage.

Conclusion:
Left MCA territory acute infarction (ASPECTS ~5).
""",

    "Example Case 2":  # ASPECTS 6
    """
CT/MRI REPORT:

Findings:
Subtle decreased attenuation in the right MCA territory.
No hemorrhage. No significant mass effect.

Conclusion:
Findings compatible with early ischemic change (ASPECTS ~6).
""",

    "Example Case 3":  # ASPECTS 10
    """
CT/MRI REPORT:

Findings:
No intra- or extra-axial mass lesion.
Ventricles and sulci within normal limits.
No abnormal diffusion or gradient signal.
TOF normal.

Conclusion:
Normal MRI brain (ASPECTS 10).
"""
}

# ============================================================
# Extraction Results (변수 정비: 논문 기반 변수)
# ============================================================

extraction_results = {
    "Example Case 1": {
        "Chief_Complaint": "Right-sided weakness, dysarthria",
        "Onset_Time": "2018-08-25 21:40",
        "NIHSS": 9,
        "Hypertension": "yes",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 5,
        "tPA_Administered": "yes",
        "Weakness_Side": "right",
        "SBP": 178
    },

    "Example Case 2": {
        "Chief_Complaint": "Aphasia, left arm heaviness",
        "Onset_Time": "2018-09-03 19:10",
        "NIHSS": 5,
        "Hypertension": "yes",
        "Diabetes": "yes",
        "Atrial_Fibrillation": "no",
        "ASPECTS": 6,
        "tPA_Administered": "no",
        "Weakness_Side": "left",
        "SBP": 162
    },

    "Example Case 3": {
        "Chief_Complaint": "Presyncope with bilateral leg weakness",
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

# ============================================================
# Validation Logic (Ground truth 비교 제거)
# ============================================================

def validate_data(selected, extracted):
    val = {}

    # Rule-based
    rule_msgs = []
    if extracted["NIHSS"] < 0 or extracted["NIHSS"] > 42:
        rule_msgs.append("❗ NIHSS out of valid range.")
    if extracted["ASPECTS"] < 0 or extracted["ASPECTS"] > 10:
        rule_msgs.append("❗ ASPECTS out of valid clinical range.")
    if not rule_msgs:
        rule_msgs.append("✔ No rule-based issues detected.")
    val["Rule-based"] = rule_msgs

    # RAG Verification (Mock)
    if extracted["ASPECTS"] <= 6:
        val["RAG"] = ["Retrieved context suggests ischemic change is present."]
    elif extracted["ASPECTS"] == 10:
        val["RAG"] = ["Retrieved context supports normal imaging findings."]
    else:
        val["RAG"] = ["No conflicting context detected."]

    # Cosine Similarity Flagging (Mock)
    if extracted["ASPECTS"] <= 5:
        val["Flag"] = "❗ FLAGGED: severe change detected"
    else:
        val["Flag"] = "✔ Not flagged"

    # HITL (Mock)
    val["HITL"] = "Values reviewed manually; no correction required."

    return val


# ============================================================
# UI 시작
# ============================================================

selected = st.selectbox("Select Example Case", list(neurology_notes.keys()))

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Neurology Note")
    st.markdown(neurology_notes[selected])

with col2:
    st.subheader("📄 Radiology Report")
    st.markdown(radiology_reports[selected])

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
    st.markdown("### 📌 Vector Similarity")
    st.write(results["Flag"])

    st.markdown("---")
    st.markdown("### 🧑‍⚕️ Human-in-the-loop Review")
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

