import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")
st.title("🧠 Stroke Pipeline Demo (Enhanced Mock Version)")
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
Non-contrast CT demonstrates acute ischemic changes involving the left MCA territory.  
Loss of gray–white differentiation and mild sulcal effacement noted.  
No intracranial hemorrhage.

Conclusion:  
Findings consistent with acute ischemic infarction in the left MCA territory.
""",

    "Example Case 2":
    """
CT brain shows subtle hypoattenuation involving the right MCA territory.  
No hemorrhage or mass effect is seen.

Conclusion:  
Findings compatible with early ischemic change.
""",

    "Example Case 3":
    """
CT/MRI Brain:  
No mass, no hemorrhage, ventricles normal.  
No restricted diffusion.  
TOF vascular imaging normal.

Conclusion:  
Normal brain MRI/CT appearance.
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

def validate_data(selected, extracted):
    val = {}

    # Rule-based
    rule_msgs = []
    if extracted["NIHSS"] < 0 or extracted["NIHSS"] > 42:
        rule_msgs.append("❗ NIHSS out of valid range.")
    if extracted["ASPECTS"] < 0 or extracted["ASPECTS"] > 10:
        rule_msgs.append("❗ ASPECTS out of valid range.")
    if not rule_msgs:
        rule_msgs.append("✔ No rule-based issues detected.")
    val["Rule-based"] = rule_msgs

    # RAG verification
    if extracted["ASPECTS"] <= 6:
        val["RAG"] = ["Context suggests ischemic change is present."]
    else:
        val["RAG"] = ["Context suggests no acute ischemic lesion."]

    # Vector similarity
    if extracted["ASPECTS"] <= 5:
        val["Flag"] = "❗ FLAGGED: significant ischemic burden"
    else:
        val["Flag"] = "✔ Not flagged"

    val["HITL"] = "Reviewed by human expert; no correction required."
    return val

# =====================================================================
# Gauge chart (반원 게이지)
# =====================================================================

def semicircular_gauge(score):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)

    theta = np.linspace(0, np.pi, 200)
    ax.plot(theta, [1]*200, color='#E0E0E0', linewidth=18)

    score_angle = np.pi * score
    ax.plot([score_angle, score_angle], [0, 1], color='#d62728', linewidth=4)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.grid(False)

    return fig

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

with st.expander("2. Validation Steps"):
    results = validate_data(selected, extracted)

    st.subheader("🔎 Rule-based Validation")
    for m in results["Rule-based"]:
        st.write(m)

    st.markdown("---")
    st.subheader("📚 RAG Verification")
    for m in results["RAG"]:
        st.write("- " + m)

    st.markdown("---")
    st.subheader("📌 Vector Similarity Flagging")
    st.write(results["Flag"])

    st.markdown("---")
    st.subheader("🧑‍⚕️ Human-in-the-loop Review")
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


