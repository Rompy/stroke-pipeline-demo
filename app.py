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
A 68-year-old male with a history of poorly controlled hypertension and diabetes mellitus, but without atrial fibrillation, prior stroke, dyslipidemia, cardiovascular disease, malignancy, or ESRD, presented with sudden right-sided arm and leg weakness accompanied by slurred speech. The symptoms began at approximately 21:40 on August 25, 2018 (LKW 21:30) while he was at home, and the deficits persisted, requiring assistance for ambulation. He has a social history notable for smoking half a pack per day for 10 years and consuming approximately two alcoholic drinks daily for 15 years.
On arrival, his vital signs were BP 178/92, HR 84, RR 18, and temperature 36.8°C. Neurologic exam showed mild dysarthria, right facial droop, 3/5 strength in the right upper and lower extremities, intact strength on the left, decreased light touch sensation on the right side, and no cerebellar ataxia. His initial NIHSS score was 9. There were no signs of seizure, head trauma, or altered mental status.
Given the clear onset time and absence of contraindications, IV tPA was administered at 22:35 at a dose of 0.9 mg/kg. No mechanical thrombectomy or other intra-arterial procedures were performed.
""",

    "Example Case 2":
    """
A 72-year-old female with a medical history of hypertension and diabetes mellitus, and without atrial fibrillation, dyslipidemia, cardiovascular disease, prior stroke, ESRD, or malignancy, presented with expressive aphasia and a sensation of heaviness in the left upper extremity. The symptoms began on September 3, 2018 at approximately 19:10. She denied smoking but reported occasional alcohol use.
Her symptoms initially fluctuated but eventually persisted. On examination in the emergency department, her vital signs were BP 162/88, HR 76, RR 18, and temperature 37.0°C. Neurologic exam revealed mild aphasia, 4+/5 strength in the left upper extremity, 4/5 in the left lower extremity, intact sensation, and no cranial nerve or cerebellar abnormalities. Her initial NIHSS was calculated as 5.
No IV tPA or intra-arterial intervention was performed due to clinical judgment and imaging findings. There was no loss of consciousness, seizure activity, or head trauma reported.
"""
,

    "Example Case 3":
    """
A 63-year-old male with hypertension, diabetes mellitus, a remote history of treated pulmonary tuberculosis, and chronic hepatitis B, but without atrial fibrillation, dyslipidemia, ESRD, malignancy, cardiovascular disease, or previous stroke, presented after experiencing dizziness, chills, and transient bilateral leg weakness while playing billiards. The onset occurred at around 23:30 on August 24, 2018. His social history includes smoking half a pack per day for approximately 10 years and drinking one to two alcoholic beverages daily for about 20 years.
Upon evaluation, his vital signs were notable for significantly elevated blood pressure at 211/90, with HR 73, RR 20, and temperature 36.7°C. Neurologic assessment demonstrated full strength (5/5) in both upper extremities and slightly reduced strength (4+/5) in both lower extremities, without cranial nerve deficits, cerebellar signs, or sensory impairment. His initial NIHSS score was 0.
He did not receive IV tPA or undergo any intra-arterial intervention, given the absence of focal deficits consistent with acute large-vessel ischemia and imaging findings.
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

            # --------------------
    # 📌 Feedback Loop Visual Indicator
flagged = any("❗" in msg for stage in validation.values() for msg in stage)

if flagged:
    st.markdown("""
    <div style='margin:15px 0;padding:12px 16px;
        border-left:6px solid #d9534f;background:#fdecec;border-radius:8px;'>
        <b style='color:#8B0000;font-size:15px;'>❗ Validation flagged inconsistencies</b><br>
        <span style='color:#b30000;font-size:15px;'>
        ↺ LLM Feedback Loop Triggered → Proceeding to Correction Step
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='margin:15px 0;padding:12px 16px;
        border-left:6px solid #28a745;background:#e8f8f0;border-radius:8px;'>
        <b style='color:#006400;font-size:15px;'>✔ All checks stable</b><br>
        <span style='color:#1d7d46;font-size:15px;'>
        No feedback loop triggered — auto-accept path active
        </span>
    </div>
    """, unsafe_allow_html=True)





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


