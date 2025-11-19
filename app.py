import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stroke Pipeline Demo", layout="wide")

# ===============================================================
# 1) PIPELINE FLOW DIAGRAM (TOP) - IMPROVED WITH GROUPING
# ===============================================================
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
    <h1 style='color: white; text-align: center; margin: 0;'>
        🧠 Stroke Outcome Pipeline Demo
    </h1>
    <p style='color: white; text-align: center; font-size: 18px; margin-top: 10px;'>
        Multimodal Extraction → Validation → Correction → Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# Pipeline Flow Diagram - WITH PIPELINE GROUPING
st.markdown("### 📊 Pipeline Architecture")

fig_flow = go.Figure()

# Define all stages
input_stage = "Clinical\nData"
pipeline_stages = ["LLM\nExtraction", "Rule-Based\nValidation", "RAG\nVerification", 
                   "Cosine\nSimilarity", "HITL\nReview", "Corrected\nData"]
output_stages = ["Prediction\nModel", "Risk\nScore"]
management_stage = "Patient Info\nManagement"

# Positioning
input_x = 0
pipeline_x_start = 1.6
pipeline_x_spacing = 1.1
output_x_start = pipeline_x_start + len(pipeline_stages) * pipeline_x_spacing + 0.8
management_x = output_x_start + 2.5

y_pos = 0

# Colors
input_color = '#667eea'
pipeline_colors = ['#667eea', '#ffc107', '#ffc107', '#ffc107', '#ffc107', '#28a745']
output_colors = ['#dc3545', '#dc3545']
management_color = '#6c757d'

# Add background rectangle for pipeline group
pipeline_x_positions = [pipeline_x_start + i * pipeline_x_spacing for i in range(len(pipeline_stages))]
fig_flow.add_shape(
    type="rect",
    x0=min(pipeline_x_positions) - 0.8,
    x1=max(pipeline_x_positions) + 0.8,
    y0=-0.65,
    y1=0.65,
    line=dict(color="#9370DB", width=3, dash="dash"),
    fillcolor="rgba(147, 112, 219, 0.1)",
    layer="below"
)

# Add pipeline label
fig_flow.add_annotation(
    x=(min(pipeline_x_positions) + max(pipeline_x_positions)) / 2,
    y=0.75,
    text="<b>The Pipeline</b>",
    showarrow=False,
    font=dict(size=14, color="#9370DB", family="Arial Black"),
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#9370DB",
    borderwidth=2,
    borderpad=4
)

# 1) Input Stage (Clinical Notes)
fig_flow.add_trace(go.Scatter(
    x=[input_x], y=[y_pos],
    mode='markers+text',
    marker=dict(size=110, color=input_color, line=dict(width=3, color='white')),
    text=input_stage.replace('\n', '<br>'),
    textposition='middle center',
    textfont=dict(color='white', size=13, family='Arial Black'),
    hoverinfo='text',
    hovertext="Input: Clinical Notes",
    showlegend=False
))

# Arrow: Input → Pipeline
fig_flow.add_annotation(
    x=pipeline_x_positions[0] - 0.6,
    y=y_pos,
    ax=input_x + 0.6,
    ay=y_pos,
    xref='x', yref='y', axref='x', ayref='y',
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=3,
    arrowcolor='#333'
)

# 2) Pipeline Stages
for i, stage in enumerate(pipeline_stages):
    x = pipeline_x_positions[i]
    fig_flow.add_trace(go.Scatter(
        x=[x], y=[y_pos],
        mode='markers+text',
        marker=dict(size=100, color=pipeline_colors[i], line=dict(width=3, color='white')),
        text=stage.replace('\n', '<br>'),
        textposition='middle center',
        textfont=dict(color='white', size=11, family='Arial Black'),
        hoverinfo='text',
        hovertext=f"Pipeline Stage {i+1}: {stage}",
        showlegend=False
    ))

# Arrow: Pipeline → Output
fig_flow.add_annotation(
    x=output_x_start - 0.6,
    y=y_pos,
    ax=max(pipeline_x_positions) + 0.6,
    ay=y_pos,
    xref='x', yref='y', axref='x', ayref='y',
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=3,
    arrowcolor='#333'
)

# 3) Output Stages
for i, stage in enumerate(output_stages):
    x = output_x_start + i * 1.3
    fig_flow.add_trace(go.Scatter(
        x=[x], y=[y_pos],
        mode='markers+text',
        marker=dict(size=110, color=output_colors[i], line=dict(width=3, color='white')),
        text=stage.replace('\n', '<br>'),
        textposition='middle center',
        textfont=dict(color='white', size=13, family='Arial Black'),
        hoverinfo='text',
        hovertext=f"Output: {stage}",
        showlegend=False
    ))

# Arrow: Output → Management
output_last_x = output_x_start + (len(output_stages) - 1) * 1.3
fig_flow.add_annotation(
    x=management_x - 0.6,
    y=y_pos,
    ax=output_last_x + 0.6,
    ay=y_pos,
    xref='x', yref='y', axref='x', ayref='y',
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=3,
    arrowcolor='#333'
)

# 4) Management Stage
fig_flow.add_trace(go.Scatter(
    x=[management_x], y=[y_pos],
    mode='markers+text',
    marker=dict(size=110, color=management_color, line=dict(width=3, color='white')),
    text=management_stage.replace('\n', '<br>'),
    textposition='middle center',
    textfont=dict(color='white', size=12, family='Arial Black'),
    hoverinfo='text',
    hovertext="Outcome: Patient Information Management",
    showlegend=False
))

fig_flow.update_layout(
    height=280,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1, 1]),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig_flow, use_container_width=True)

# Add explanatory text
st.info("""
**Pipeline Flow:**
1. 📝 **Clinical Notes** → Input unstructured clinical data
2. 🔄 **Multi-Tiered Validation Pipeline** → Extract, validate, and correct data through 6 layers
3. 🎯 **Prediction & Risk Assessment** → Generate outcome predictions
4. 💾 **Patient Information Management** → Store structured data for clinical decision support and research
""")

# ===============================================================
# 2) PERFORMANCE METRICS DASHBOARD (SIDEBAR)
# ===============================================================
with st.sidebar:
    st.markdown("### 📈 Model Performance")
    
    # Gauge charts for key metrics
    st.metric("Extraction Accuracy", "97.0%", "After HITL")
    st.metric("TabPFN AUROC", "0.816", "95% CI: 0.784-0.847")
    st.metric("Grounding Accuracy", "93.2%", "RAG Stage")
    st.metric("Inference Time", "8.3s", "per patient")
    st.metric("Training Cohort", "1,166", "patients")
    
    st.markdown("---")

# ===============================================================
# 10) MOCK VS REAL CLARIFICATION
# ===============================================================
st.warning("""
⚠️ **IMPORTANT: This is a CONCEPTUAL DEMONSTRATION**

**What this demo shows:**
- ✓ Pipeline workflow and logic
- ✓ Validation framework structure
- ✓ Multi-tiered correction process

**What this demo does NOT use:**
- ✗ Actual Llama 3 8B model (requires 16GB RAM)
- ✗ Real FAISS-based RAG system
- ✗ Actual TabPFN prediction model

""")

st.title("🧠 Case Demonstration")
st.write("Explore how the pipeline processes clinical data through multiple validation stages.")

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

# Badge for simplified demo
simplified_badge = "⚠️ [SIMPLIFIED DEMO]"

# =====================================================================
# 0) Neurology Notes - CORRECTED TO MATCH EXTRACTION
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
# Radiology Reports
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
# Extraction Results - CORRECTED TO MATCH NOTES
# ===============================================================

extraction_results = {
    "Example Case 1": {
        "Age": 68,
        "Sex": "male",

        "Hypertension": "yes",
        "Diabetes": "yes",
        "Dyslipidemia": "no",
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

        "tPA_Administered": "no",  # Intentional error for demo
        "IA_Thrombectomy": "no",

        "Weakness_Side": "bilateral",  # Intentional error for demo
        "SBP": 178
    },

    "Example Case 2": {
        "Age": 72,
        "Sex": "female",

        "Hypertension": "no",  # Intentional error for demo
        "Diabetes": "yes",
        "Dyslipidemia": "no",
        "Cardiovascular_Disease": "no",
        "Atrial_Fibrillation": "no",
        "Old_CVA": "no",
        "Malignancy": "no",
        "ESRD": "no",

        "MRI_Acute_Infarct": "yes",
        "MRI_No_Lesion": "no",
        "MRI_Other_Lesion": "no",

        "NIHSS": 5,
        "ASPECTS": 9,  # Intentional error for demo

        "tPA_Administered": "no",
        "IA_Thrombectomy": "no",

        "Weakness_Side": "left",
        "SBP": 162
    },

    "Example Case 3": {
        "Age": 63,
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
            rag.append("❗ Weakness side mismatch: note indicates right-sided weakness.")
        if extracted["MRI_Acute_Infarct"] == "yes" and extracted["ASPECTS"] > 7:
            rag.append("❗ ASPECT too high for acute MCA infarction.")

    if selected == "Example Case 2":
        if "hypertension" in full_text and extracted["Hypertension"] == "no":
            rag.append("❗ Hypertension mismatch: note indicates hypertension history.")
        if extracted["MRI_Acute_Infarct"] == "yes" and extracted["ASPECTS"] >= 8:
            rag.append("❗ ASPECT inconsistent with early ischemia severity.")

    if not rag:
        rag.append("✔ No semantic mismatch.")

    val["RAG"] = rag

    # ---- Cosine similarity mock ----
    cos = []
    sim = 0.71 if selected=="Example Case 1" else (0.78 if selected=="Example Case 2" else 0.92)

    if sim < 0.82:
        cos.append(f"❗ Cosine similarity {sim:.2f} → atypical pattern")
    else:
        cos.append(f"✔ Cosine similarity {sim:.2f} → typical pattern")

    val["Cosine"] = cos
    val["CosineSimilarity"] = sim

    flagged = any("❗" in msg for key in ["Rule", "RAG", "Cosine"] for msg in val[key])
    val["HITL"] = "🔎 Needs manual review." if flagged else "✔ Auto-acceptable."

    return val

# =====================================================================
# HITL ASSISTED CORRECTION MODULE
# =====================================================================

def hitl_correction(selected, extracted, validation):

    corrected = extracted.copy()

    if "❗" not in str(validation):
        return corrected, False, {}

    changes = {}

    if selected == "Example Case 1":
        if extracted["tPA_Administered"] != "yes":
            corrected["tPA_Administered"] = "yes"
            changes["tPA_Administered"] = {"from": "no", "to": "yes"}
        if extracted["Weakness_Side"] != "right":
            corrected["Weakness_Side"] = "right"
            changes["Weakness_Side"] = {"from": "bilateral", "to": "right"}
        if extracted["ASPECTS"] != 5:
            corrected["ASPECTS"] = 5
            changes["ASPECTS"] = {"from": 7, "to": 5}

    if selected == "Example Case 2":
        if extracted["Hypertension"] != "yes":
            corrected["Hypertension"] = "yes"
            changes["Hypertension"] = {"from": "no", "to": "yes"}
        if extracted["ASPECTS"] != 6:
            corrected["ASPECTS"] = 6
            changes["ASPECTS"] = {"from": 9, "to": 6}

    return corrected, len(changes) > 0, changes


# =====================================================================
# UI START
# =====================================================================

selected = st.selectbox("Select Example Case", list(neurology_notes.keys()))

# ===============================================================
# 11) TABLE 1 STATISTICS (Study Cohort Overview)
# ===============================================================
with st.expander("📊 Study Cohort Statistics (Table 1 from Paper)"):
    st.markdown("### Patient Demographics and Clinical Characteristics (n=1,166)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        demo_data = pd.DataFrame({
            'Variable': ['Age (mean ± SD)', 'Male sex', 'Hypertension', 'Diabetes mellitus', 'Atrial fibrillation'],
            'Value': ['65.68 ± 15.90', '56.2%', '57.4%', '24.4%', '14.9%']
        })
        st.dataframe(demo_data, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Clinical Scores")
        scores_data = pd.DataFrame({
            'Variable': ['NIHSS (median, IQR)', 'ASPECT (median, IQR)', 'MRI infarction', 'IV t-PA', 'IA intervention'],
            'Value': ['3 (1-7)', '9 (8-10)', '59.4%', '9.0%', '7.5%']
        })
        st.dataframe(scores_data, hide_index=True, use_container_width=True)
    
    with col3:
        st.markdown("#### Outcomes")
        outcome_data = pd.DataFrame({
            'Variable': ['Poor outcome (mRS 3-6)', 'Good outcome (mRS 0-2)', 'Follow-up rate', '3-month assessment'],
            'Value': ['28.4%', '71.6%', '65.8%', '767 patients']
        })
        st.dataframe(outcome_data, hide_index=True, use_container_width=True)
    
    # Distribution charts
    st.markdown("#### NIHSS Score Distribution")
    nihss_dist = pd.DataFrame({
        'NIHSS Range': ['0', '1-4', '5-15', '16-20', '21-42'],
        'Percentage': [25.4, 40.3, 26.3, 5.7, 2.3]
    })
    fig_nihss = px.bar(nihss_dist, x='NIHSS Range', y='Percentage', 
                       color='Percentage', color_continuous_scale='Blues',
                       title='')
    fig_nihss.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig_nihss, use_container_width=True)

col1, col2, col3 = st.columns([1.3, 1.3, 1])

# =====================================================================
# Source Documents
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
    st.image(aspect_images[selected], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# STEP 1: Extraction Output
# =====================================================================

with st.expander(f"STEP 1 — LLM Extraction Output {simplified_badge}", expanded=False):
    
    with st.container():
        st.info("""
        ℹ️ **What is this step?**
        
        This stage uses a Large Language Model (Llama 3 8B) with few-shot prompting to extract structured variables from unstructured clinical text.
        
        **Real Implementation (Paper Section 2.3.1):**
        - Model: Llama 3 8B with LoRA fine-tuning
        - Prompting: 3-shot examples
        - Parameters: Temperature 0.1, max tokens 512
        
        **This Demo:**
        - Uses pre-generated mock extractions with intentional errors for demonstration
        """)
    
    extracted = extraction_results[selected]
    st.json(extracted)
    
    st.caption("⚠️ Note: Intentional errors included to demonstrate validation pipeline")


# =====================================================================
# STEP 2: Multi-Tier Validation
# =====================================================================

with st.expander(f"STEP 2 — Multi-Tiered Validation {simplified_badge}", expanded=True):
    
    with st.container():
        st.info("""
        ℹ️ **What is this step?**
        
        Multi-layered validation using defense-in-depth approach to catch errors and hallucinations.
        
        **Real Implementation (Paper Section 2.3.2):**
        - Layer 1: Rule-based checks (Python scripts)
        - Layer 2: RAG with FAISS vector database
        - Layer 3: Cosine similarity vs 200 validated records
        - Layer 4: Human-in-the-loop review (κ=0.89 agreement)
        
        **This Demo:**
        - Simplified rule checking
        - Mock RAG logic (no actual FAISS)
        - Simulated cosine scores
        """)
    
    validation = validate_data(
        selected,
        extracted,
        neurology_notes[selected],
        radiology_reports[selected]
    )

    # Validation Progress Bar
    stages = ["Rule-Based", "RAG", "Cosine", "HITL"]
    stage_status = []
    
    for stage in stages:
        if stage == "HITL":
            passed = "✔" in validation.get("HITL", "")
        else:
            passed = all("❗" not in msg for msg in validation.get(stage, []))
        stage_status.append(1 if passed else 0)
    
    # Progress visualization
    fig_progress = go.Figure()
    colors = ['#28a745' if s == 1 else '#dc3545' for s in stage_status]
    
    fig_progress.add_trace(go.Bar(
        x=stages,
        y=[1, 1, 1, 1],
        marker=dict(color=colors),
        text=['✓ Pass' if s == 1 else '✗ Flag' for s in stage_status],
        textposition='inside',
        textfont=dict(color='white', size=14)
    ))
    
    fig_progress.update_layout(
        title="Validation Stage Results",
        height=200,
        showlegend=False,
        yaxis=dict(showticklabels=False, range=[0, 1.2]),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_progress, use_container_width=True)

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
    
    # Visualization of cosine similarity
    sim_score = validation["CosineSimilarity"]
    fig_cosine = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Similarity Score"},
        delta={'reference': 0.82, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.82], 'color': "lightgray"},
                {'range': [0.82, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.82
            }
        }
    ))
    fig_cosine.update_layout(height=250)
    st.plotly_chart(fig_cosine, use_container_width=True)
    
    for msg in validation["Cosine"]:
        if "❗" in msg:
            st.markdown(highlight_red(msg), unsafe_allow_html=True)
        else:
            st.markdown(highlight_green(msg), unsafe_allow_html=True)

    # Feedback Loop Indicator
    flagged = any("❗" in msg for key in ["Rule", "RAG", "Cosine"] for msg in validation[key])

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
with st.expander(f"STEP 3 — Corrected Output (HITL-Assisted) {simplified_badge}", expanded=True):

    with st.container():
        st.info("""
        ℹ️ **What is this step?**
        
        Human-in-the-loop correction of flagged inconsistencies.
        
        **Real Implementation (Paper Section 2.3.2):**
        - Two independent clinicians review flagged cases
        - Inter-rater agreement: κ = 0.89
        - All flagged + 10% random sample reviewed
        - Corrections fed back to LoRA adapters
        
        **This Demo:**
        - Mock automatic corrections based on validation flags
        """)

    corrected, changed, changes = hitl_correction(selected, extracted, validation)

    if changed:
        st.markdown(
            "<p style='color:#cc0000;font-weight:700;font-size:18px;'>"
            "⚠️ Issues detected — corrections applied</p>",
            unsafe_allow_html=True
        )
        
        # Show what changed
        if changes:
            st.markdown("### 🔄 Changes Made:")
            for field, change in changes.items():
                st.markdown(f"**{field}:** `{change['from']}` → `{change['to']}`")
    
    else:
        st.markdown(
            "<p style='color:#008800;font-weight:700;font-size:18px;'>"
            "✔ No corrections needed</p>",
            unsafe_allow_html=True
        )

    # Before/After Comparison Table
    st.markdown("### 📊 Before/After Comparison")
    
    comparison_data = []
    for key in extracted.keys():
        orig = extracted[key]
        corr = corrected[key]
        changed_flag = "✓ Changed" if orig != corr else ""
        comparison_data.append({
            "Field": key,
            "Original": orig,
            "Corrected": corr,
            "Status": changed_flag
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Color code changed rows
    def highlight_changes(row):
        if row['Status'] == "✓ Changed":
            return ['background-color: #ffffcc; color: #000000'] * len(row)
        return ['background-color: #ffffff; color: #000000'] * len(row)
    
    st.dataframe(
        df_comparison.style.apply(highlight_changes, axis=1),
        hide_index=True,
        use_container_width=True
    )


# =====================================================================
# STEP 4: Prediction
# =====================================================================

with st.expander(f"STEP 4 — Outcome Prediction {simplified_badge}", expanded=True):

    with st.container():
        st.info("""
        ℹ️ **What is this step?**
        
        Predicts 3-month stroke outcome (mRS 3-6: poor outcome) using machine learning.
        
        **Real Implementation (Paper Section 2.4):**
        - Model: TabPFN (best: AUROC 0.816)
        - Features: 12 variables from extracted data
        - Training: 767 patients with outcome data
        - Validation: Good calibration (Hosmer-Lemeshow p>0.05)
        
        **This Demo:**
        - Simplified rule-based probability
        - Based only on ASPECT score
        """)

    # Simple rule-based probability
    if corrected["ASPECTS"] <= 5:
        prob = 0.55
    elif corrected["ASPECTS"] <= 7:
        prob = 0.32
    else:
        prob = 0.10

    st.write("**Input Features:**")
    feature_df = pd.DataFrame({
        'Feature': ['Age', 'NIHSS', 'ASPECTS', 'Hypertension', 'Atrial Fibrillation', 'tPA Given'],
        'Value': [corrected['Age'], corrected['NIHSS'], corrected['ASPECTS'], 
                  corrected['Hypertension'], corrected['Atrial_Fibrillation'], corrected['tPA_Administered']]
    })
    st.dataframe(feature_df, hide_index=True, use_container_width=True)

    # SHAP-style Feature Importance
    st.markdown("### 📊 Feature Importance (Mock SHAP Values)")
    
    shap_data = pd.DataFrame({
        'Feature': ['NIHSS', 'Age', 'ASPECTS', 'Atrial Fibrillation', 'tPA Given', 'Hypertension'],
        'Impact': [0.35, 0.25, -0.28, 0.15, -0.12, 0.08]
    })
    
    fig_shap = px.bar(
        shap_data, 
        x='Impact', 
        y='Feature', 
        orientation='h',
        color='Impact',
        color_continuous_scale=['#dc3545', '#ffc107', '#28a745'],
        title='Feature Impact on Poor Outcome Prediction'
    )
    fig_shap.update_layout(height=300)
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.caption("🔴 Red: Increases risk | 🟢 Green: Decreases risk")

    # Gradient Risk Bar
    st.markdown(f"""
    <div style='height:22px;border-radius:12px;margin-top:12px;
        background:linear-gradient(90deg, #ff6666 {prob*100}%, #e0e0e0 {prob*100}%);'>
    </div>
    <p style='font-size:16px;font-weight:600;margin-top:6px;'>{prob*100:.1f}% predicted poor outcome (mRS 3-6)</p>
    """, unsafe_allow_html=True)


# ===============================================================
# 5) ROC CURVE & 6) CALIBRATION PLOT - USING ACTUAL IMAGES
# ===============================================================
with st.expander("📈 Model Performance Visualization (From Paper)"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ROC Curves Comparison")
        st.image("images/roc.png", use_container_width=True)
        st.caption("**Figure**: ROC curves comparing Logistic Regression (AUROC=0.700), CatBoost (AUROC=0.789), and TabPFN-Specialized (AUROC=0.816)")
    
    with col2:
        st.markdown("### Precision-Recall Comparison")
        st.image("images/prc.png", use_container_width=True)
        st.caption("**Figure**: Precision-Recall curves showing similar AUPRC (~0.315) across all models due to class imbalance")
    
    # Add summary insights
    st.success("✓ All models demonstrated good calibration (Hosmer-Lemeshow test p > 0.05)")
    st.info("""
    **Key Insights from Paper:**
    - TabPFN achieved best discrimination (AUROC = 0.816, 95% CI: 0.784-0.847)
    - All models show similar AUPRC (~0.315) due to class imbalance (28.4% poor outcomes)
    - Despite different AUROC values, all models maintained good calibration
    - AUPRC more informative than AUROC for imbalanced datasets
    - Results validate that automatically extracted data enables reliable outcome prediction
    """)


# =====================================================================
# CSV Export
# =====================================================================

final_df = pd.DataFrame([{**corrected, "Predicted_Poor_Outcome_Probability": prob}])

st.download_button(
    label="⬇️ Download Final Structured Output (CSV)",
    data=final_df.to_csv(index=False),
    mime="text/csv",
    file_name=f"{selected}_corrected_output.csv"
)

# ===============================================================
# Footer
# ===============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p style='font-size: 12px; margin-top: 10px;'>⚠️ This demo uses simplified logic for visualization. For actual implementation with reported performance, see the complete repository.</p>
</div>
""", unsafe_allow_html=True)
