"""
Heart Disease Prediction System
-------------------------------
Two-page Streamlit application:

1. Patient Information page
2. Prediction Result page

Clicking Predict moves the user from the input page to a dedicated
result page. "Predict Again" returns to the input page.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f9fd 0%, #eef5fb 100%);
    }

    .block-container {
        padding-top: 2.3rem;
        padding-bottom: 1.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1450px;
    }

    .top-header {
        text-align: center;
        margin-bottom: 1.1rem;
    }

    .top-title {
        color: #1769aa;
        font-size: 2.25rem;
        font-weight: 850;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .top-subtitle {
        color: #64748b;
        margin-top: 0.35rem;
        font-size: 0.96rem;
    }

    .form-title {
        text-align: center;
        color: #1769aa;
        font-weight: 850;
        font-size: 2.05rem;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stWidgetLabel"] p {
        font-size: 0.93rem !important;
        font-weight: 750 !important;
        color: #1f2937 !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        min-height: 46px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 12px !important;
        min-height: 44px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
    }

    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        min-height: 50px !important;
        width: 100%;
        border-radius: 12px !important;
        border: none !important;
        background: linear-gradient(135deg, #176aa8, #267fbd) !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 18px rgba(23, 106, 168, 0.22);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(0.97);
        transform: translateY(-1px);
    }

    .result-card {
        width: 100%;
        max-width: 670px;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 22px;
        margin-top: 0.15rem;
        padding: 2rem 2.4rem 2rem 2.4rem;
        box-shadow: 0 16px 40px rgba(27, 74, 108, 0.12);
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .result-heart {
        font-size: 4.3rem;
        line-height: 1;
        margin-bottom: 0.55rem;
    }

    .result-title {
        color: #1769aa;
        font-size: 1.9rem;
        font-weight: 850;
        margin-bottom: 0.8rem;
    }

    .prediction-box {
        padding: 1.35rem 1rem;
        border-radius: 16px;
        margin: 0.5rem auto 1.15rem auto;
        font-size: 1.5rem;
        font-weight: 850;
        max-width: 570px;
    }

    .prediction-yes {
        background: #fff1f2;
        color: #b42318;
        border: 1px solid #fecdd3;
    }

    .prediction-no {
        background: #ecfdf3;
        color: #087443;
        border: 1px solid #bbf7d0;
    }

    .result-description {
        color: #334155;
        font-size: 1rem;
        margin: 0.6rem 0 1.1rem 0;
    }

    .probability-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        margin: 0 auto 1.15rem auto;
        max-width: 570px;
        text-align: left;
    }

    .probability-label {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 0.15rem;
    }

    .probability-value {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 850;
    }

    .medical-note {
        color: #8a8f98;
        font-size: 0.81rem;
        margin-top: 1.35rem;
        line-height: 1.45;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "heart_disease_model.pkl"

FEATURE_ORDER = [
    "Age", "Sex", "RestingBP", "Cholesterol", "FastingBS", "MaxHR",
    "ExerciseAngina", "Oldpeak", "ChestPainType_ATA", "ChestPainType_NAP",
    "ChestPainType_TA", "RestingECG_Normal", "RestingECG_ST",
    "ST_Slope_Flat", "ST_Slope_Up",
]

SEX_OPTIONS = ["Male", "Female"]
CHEST_PAIN_OPTIONS = ["ASY", "ATA", "NAP", "TA"]
FASTING_BS_OPTIONS = ["0 - Normal", "1 - High"]
RESTING_ECG_OPTIONS = ["Normal", "ST", "LVH"]
EXERCISE_ANGINA_OPTIONS = ["Yes", "No"]
ST_SLOPE_OPTIONS = ["Up", "Flat", "Down"]

if "page" not in st.session_state:
    st.session_state.page = "form"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "risk_probability" not in st.session_state:
    st.session_state.risk_probability = None
if "patient_data" not in st.session_state:
    st.session_state.patient_data = None
if "encoded_input" not in st.session_state:
    st.session_state.encoded_input = None


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = {
        "Age": inputs["age"],
        "Sex": 1 if inputs["sex"] == "Male" else 0,
        "RestingBP": inputs["resting_bp"],
        "Cholesterol": inputs["cholesterol"],
        "FastingBS": 1 if inputs["fasting_bs"] == "1 - High" else 0,
        "MaxHR": inputs["max_hr"],
        "ExerciseAngina": 1 if inputs["exercise_angina"] == "Yes" else 0,
        "Oldpeak": inputs["oldpeak"],
        "ChestPainType_ATA": 0,
        "ChestPainType_NAP": 0,
        "ChestPainType_TA": 0,
        "RestingECG_Normal": 0,
        "RestingECG_ST": 0,
        "ST_Slope_Flat": 0,
        "ST_Slope_Up": 0,
    }

    cp = inputs["chest_pain"]
    if cp in ("ATA", "NAP", "TA"):
        row[f"ChestPainType_{cp}"] = 1

    ecg = inputs["resting_ecg"]
    if ecg in ("Normal", "ST"):
        row[f"RestingECG_{ecg}"] = 1

    slope = inputs["st_slope"]
    if slope in ("Flat", "Up"):
        row[f"ST_Slope_{slope}"] = 1

    return pd.DataFrame([row], columns=FEATURE_ORDER)


try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model file: {exc}")
    st.stop()


# ======================================================================
# PAGE 1: PATIENT INFORMATION
# ======================================================================
if st.session_state.page == "form":
    st.markdown(
        """
        <div class="top-header">
            <div class="top-title">Heart Disease Prediction System</div>
            <div class="top-subtitle">
                Enter the patient's clinical information to generate a prediction.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_space, center_col, right_space = st.columns([0.55, 4.9, 0.55])

    with center_col:
        st.markdown('<div class="form-title">Patient Information</div>', unsafe_allow_html=True)

        with st.form("patient_form"):
            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                age = st.number_input(
                    "Age", min_value=1, max_value=120, value=None, step=1,
                    placeholder="Enter age",
                )
                chest_pain = st.selectbox(
                    "Chest Pain Type", CHEST_PAIN_OPTIONS,
                    index=None, placeholder="Select",
                )
                cholesterol = st.number_input(
                    "Cholesterol", min_value=0, max_value=700, value=None,
                    step=1, placeholder="Enter cholesterol",
                )
                resting_ecg = st.selectbox(
                    "Resting ECG", RESTING_ECG_OPTIONS,
                    index=None, placeholder="Select",
                )
                exercise_angina = st.selectbox(
                    "Exercise Angina", EXERCISE_ANGINA_OPTIONS,
                    index=None, placeholder="Select",
                )
                st_slope = st.selectbox(
                    "ST Slope", ST_SLOPE_OPTIONS,
                    index=None, placeholder="Select",
                )

            with col_right:
                sex = st.selectbox(
                    "Sex", SEX_OPTIONS,
                    index=None, placeholder="Select",
                )
                resting_bp = st.number_input(
                    "Resting Blood Pressure", min_value=0, max_value=300,
                    value=None, step=1, placeholder="Enter resting BP",
                )
                fasting_bs = st.selectbox(
                    "Fasting Blood Sugar", FASTING_BS_OPTIONS,
                    index=None, placeholder="Select",
                )
                max_hr = st.number_input(
                    "Maximum Heart Rate", min_value=40, max_value=250,
                    value=None, step=1, placeholder="Enter MaxHR",
                )
                oldpeak = st.number_input(
                    "Oldpeak", min_value=-3.0, max_value=7.0, value=None,
                    step=0.1, format="%.1f", placeholder="Enter Oldpeak",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Predict")


    if submitted:
        fields = {
            "Age": age,
            "Sex": sex,
            "Chest Pain Type": chest_pain,
            "Resting Blood Pressure": resting_bp,
            "Cholesterol": cholesterol,
            "Fasting Blood Sugar": fasting_bs,
            "Resting ECG": resting_ecg,
            "Maximum Heart Rate": max_hr,
            "Exercise Angina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST Slope": st_slope,
        }

        missing = [name for name, value in fields.items() if value is None]

        if missing:
            st.warning("Please fill in all fields: " + ", ".join(missing))
        else:
            with st.spinner("Generating prediction..."):
                X = build_feature_row(
                    {
                        "age": age,
                        "sex": sex,
                        "resting_bp": resting_bp,
                        "cholesterol": cholesterol,
                        "fasting_bs": fasting_bs,
                        "max_hr": max_hr,
                        "exercise_angina": exercise_angina,
                        "oldpeak": oldpeak,
                        "chest_pain": chest_pain,
                        "resting_ecg": resting_ecg,
                        "st_slope": st_slope,
                    }
                )
                prediction = int(model.predict(X)[0])
                risk_prob = (
                    float(model.predict_proba(X)[0][1])
                    if hasattr(model, "predict_proba")
                    else None
                )

            st.session_state.prediction = prediction
            st.session_state.risk_probability = risk_prob
            st.session_state.patient_data = fields
            st.session_state.encoded_input = X
            st.session_state.page = "result"
            st.rerun()


# ======================================================================
# PAGE 2: RESULT
# ======================================================================
else:
    if st.session_state.prediction is None:
        st.session_state.page = "form"
        st.rerun()

    prediction = st.session_state.prediction
    risk_prob = st.session_state.risk_probability
    patient_data = st.session_state.patient_data
    encoded_input = st.session_state.encoded_input

    st.markdown(
        """
        <div class="top-header">
            <div class="top-title">Heart Disease Prediction System</div>
            <div class="top-subtitle">Prediction result</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_space, result_col, right_space = st.columns([1, 2.2, 1])

    with result_col:
        st.markdown('<div class="result-heart">❤️</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            result_text = "YES - Heart Disease Detected"
            result_class = "prediction-yes"
        else:
            result_text = "NO - Heart Disease Not Detected"
            result_class = "prediction-no"

        st.markdown(
            f'<div class="prediction-box {result_class}">{result_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="result-description">'
            'Prediction generated using the trained Gradient Boosting model.'
            '</div>',
            unsafe_allow_html=True,
        )

        if risk_prob is not None:
            st.markdown(
                f'<div class="probability-card">'
                f'<div class="probability-label">Estimated probability of heart disease</div>'
                f'<div class="probability-value">{risk_prob:.0%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        action_left, action_right = st.columns(2, gap="small")

        with action_left:
            if st.button("Dashboard", use_container_width=True):
                st.session_state.page = "form"
                st.session_state.prediction = None
                st.session_state.risk_probability = None
                st.session_state.patient_data = None
                st.session_state.encoded_input = None
                st.rerun()

        with action_right:
            if st.button("Predict Again", type="primary", use_container_width=True):
                st.session_state.page = "form"
                st.rerun()

        # Keep details available without cluttering the main result screen.
        with st.expander("View patient information"):
            summary_df = pd.DataFrame(
                {
                    "Field": list(patient_data.keys()),
                    "Value": [str(v) for v in patient_data.values()],
                }
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

        with st.expander("View exact inputs sent to the model"):
            st.dataframe(
                encoded_input.T.rename(columns={0: "value"}),
                use_container_width=True,
            )

        if patient_data.get("Cholesterol") == 0:
            st.caption(
                "Note: cholesterol was entered as 0, which the source dataset "
                "uses to mean 'not measured' — this may affect the prediction."
            )

        st.markdown(
            '<div class="medical-note">'
            'This prediction is for educational purposes only and is not a medical '
            'diagnosis or a substitute for professional medical advice.'
            '</div>',
            unsafe_allow_html=True,
        )

