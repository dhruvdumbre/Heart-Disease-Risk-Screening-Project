import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Risk Screening", page_icon="🫀", layout="wide")

# --------------------------------------------------------------------------
# Styling -- clean clinical palette (deep teal / soft off-white), not the
# generic AI-cream-and-terracotta look.
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #F5F8F8; }
    h1, h2, h3 { color: #123C4D; font-family: 'Helvetica Neue', Arial, sans-serif; }
    .risk-card {
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .risk-low { background-color: #E3F2E9; border-left: 6px solid #2E7D32; }
    .risk-moderate { background-color: #FFF6E0; border-left: 6px solid #E8A317; }
    .risk-high { background-color: #FCE8E6; border-left: 6px solid #C62828; }
    .risk-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.2rem; }
    .risk-prob { font-size: 2.4rem; font-weight: 800; color: #123C4D; }
    .disclaimer {
        background-color: #EEF3F5; border: 1px solid #C9D6DA;
        border-radius: 8px; padding: 0.9rem 1.2rem; font-size: 0.88rem;
        color: #3D5560; margin-top: 1.5rem;
    }
    .factor-pill {
        display: inline-block; background-color: #E8EEF0; color: #123C4D;
        padding: 0.3rem 0.8rem; border-radius: 20px; margin: 0.2rem;
        font-size: 0.85rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Load model artifacts
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("model_random_forest.joblib")
    log_reg_model = joblib.load("model_logistic_regression.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    nominal_cols = joblib.load("nominal_cols.joblib")
    return rf_model, log_reg_model, scaler, feature_columns, nominal_cols

try:
    rf_model, log_reg_model, scaler, feature_columns, NOMINAL_COLS = load_artifacts()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# Human-readable names for the model-explanation section
FEATURE_LABELS = {
    "age": "Age", "sex": "Sex", "trestbps": "Resting blood pressure",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina", "oldpeak": "ST depression (oldpeak)", "ca": "Blocked vessels (ca)",
    "cp_1": "Chest pain type: atypical angina", "cp_2": "Chest pain type: non-anginal",
    "cp_3": "Chest pain type: asymptomatic", "restecg_1": "Resting ECG: ST-T abnormality",
    "restecg_2": "Resting ECG: left ventricular hypertrophy", "slope_1": "ST slope: flat",
    "slope_2": "ST slope: downsloping", "thal_1": "Thalassemia: fixed defect",
    "thal_2": "Thalassemia: reversible defect",
}

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🫀 Heart Disease Risk Screening")
st.markdown(
    "Enter a patient's clinical values below to estimate their heart disease risk. "
    "This tool combines a Logistic Regression model and a Random Forest model "
    "trained on clinical data."
)

if not models_loaded:
    st.error(
        "Model files not found in this folder. Run **03_train_evaluate.py** here first "
        "-- it saves the .joblib files this app needs."
    )
    st.stop()

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 20, 90, 54)
        sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)

        st.subheader("Vitals & Labs")
        trestbps = st.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
        chol = st.slider("Cholesterol (mg/dl)", 100, 450, 220)
        fbs = st.radio("Fasting blood sugar > 120 mg/dl?", options=[1, 0],
                        format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    with col2:
        st.subheader("Cardiac Test Results I")
        cp = st.selectbox("Chest pain type", options=[0, 1, 2, 3], format_func=lambda x: {
            0: "Typical angina", 1: "Atypical angina",
            2: "Non-anginal pain", 3: "Asymptomatic"}[x])
        restecg = st.selectbox("Resting ECG result", options=[0, 1, 2], format_func=lambda x: {
            0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}[x])
        thalach = st.slider("Max heart rate achieved", 60, 210, 150)
        exang = st.radio("Exercise-induced angina?", options=[1, 0],
                          format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

    with col3:
        st.subheader("Cardiac Test Results II")
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = st.selectbox("ST segment slope", options=[0, 1, 2], format_func=lambda x: {
            0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
        ca = st.selectbox("Number of major vessels blocked (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia test result", options=[0, 1, 2], format_func=lambda x: {
            0: "Normal", 1: "Fixed defect", 2: "Reversible defect"}[x])

    submitted = st.form_submit_button("Calculate Risk", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if submitted:
    patient = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    row = pd.DataFrame([patient])
    row_encoded = pd.get_dummies(row, columns=NOMINAL_COLS, drop_first=True)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)

    rf_proba = rf_model.predict_proba(row_encoded)[0, 1]
    row_scaled = scaler.transform(row_encoded)
    lr_proba = log_reg_model.predict_proba(row_scaled)[0, 1]
    combined_proba = (rf_proba + lr_proba) / 2

    if combined_proba < 0.25:
        css_class, band = "risk-low", "Low risk"
    elif combined_proba < 0.5:
        css_class, band = "risk-moderate", "Moderate risk -- consider further screening"
    else:
        css_class, band = "risk-high", "High risk -- recommend clinical follow-up"

    st.markdown(f"""
    <div class="risk-card {css_class}">
        <div class="risk-label">{band}</div>
        <div class="risk-prob">{combined_proba:.0%} estimated risk</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Random Forest estimate", f"{rf_proba:.0%}")
    with col_b:
        st.metric("Logistic Regression estimate", f"{lr_proba:.0%}")

    if abs(rf_proba - lr_proba) > 0.25:
        st.caption("⚠️ The two models disagree notably here -- treat this case as less certain.")

    importances = pd.Series(rf_model.feature_importances_, index=feature_columns)
    top_drivers = importances.sort_values(ascending=False).head(4).index.tolist()
    readable = [FEATURE_LABELS.get(f, f) for f in top_drivers]

    st.markdown("**Top contributing factors (model-wide, not patient-specific):**")
    st.markdown("".join(f'<span class="factor-pill">{f}</span>' for f in readable), unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    <strong>This is a screening/decision-support demo, not a diagnostic device.</strong>
    It was trained on synthetic data for learning purposes. A real clinical product
    would require validated patient data, clinical trials, and regulatory clearance
    (e.g. FDA Software-as-a-Medical-Device) before being used to inform care.
    </div>
    """, unsafe_allow_html=True)
