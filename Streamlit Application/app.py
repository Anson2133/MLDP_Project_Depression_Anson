import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Student Depression Risk Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS Styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
    box-shadow: 4px 0 15px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stRadio > label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: white !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.1) !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    margin: 6px 0 !important;
    transition: all 0.3s ease !important;
    border: 2px solid transparent !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
    transform: translateX(4px) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    background-color: white !important;
    border-color: white !important;
}

/* Main Content */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #1e293b !important;
}

.main-title {
    font-size: 42px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px !important;
    text-align: center;
}

.subtitle {
    font-size: 18px !important;
    color: #64748b !important;
    margin-bottom: 30px !important;
    text-align: center;
    font-weight: 500;
}

/* Info Banner */
.info-banner {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-left: 5px solid #3b82f6;
    padding: 24px;
    border-radius: 16px;
    margin: 30px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.info-banner-title {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #1e40af !important;
    margin-bottom: 12px !important;
}

.info-banner-text {
    color: #1e3a8a !important;
    line-height: 1.8;
    margin: 0;
    font-size: 15px;
}

/* Section Cards */
.section-card {
    background: white;
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 28px;
    border: 1px solid #e2e8f0;
    transition: all 0.3s ease;
}

.section-card:hover {
    box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}

.section-title {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1e3a8a !important;
    margin-bottom: 24px !important;
    display: flex;
    align-items: center;
}

.section-title::before {
    content: '';
    width: 4px;
    height: 24px;
    background: linear-gradient(180deg, #3b82f6 0%, #60a5fa 100%);
    margin-right: 12px;
    border-radius: 2px;
}

/* Form Inputs */
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background-color: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #1e293b !important;
    padding: 12px 16px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    background-color: white !important;
}

label {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 8px !important;
}

/* Slider Styling */
.stSlider > div > div > div {
    background-color: #cbd5e1 !important;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%) !important;
}

.stSlider [role="slider"] {
    background-color: #1e3a8a !important;
    border: 3px solid white !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}

/* Submit Button */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 8px 20px rgba(30, 58, 138, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 28px rgba(30, 58, 138, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Alert Boxes */
.stAlert {
    border-radius: 16px !important;
    border: none !important;
    padding: 20px !important;
    font-size: 16px !important;
}

/* Error Messages */
.error-message {
    color: #dc2626;
    font-size: 13px;
    margin-top: 6px;
    font-weight: 600;
    animation: shake 0.3s ease;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-5px); }
    75% { transform: translateX(5px); }
}

/* Dataframe Styling */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
    border: 2px solid #e2e8f0;
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
    font-weight: 600;
    margin-top: 8px;
}

/* Chart Container */
.chart-container {
    background: white;
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin: 20px 0;
}

/* Risk Badge */
.risk-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    margin: 10px 0;
}

.risk-low {
    background: #dcfce7;
    color: #166534;
}

.risk-moderate {
    background: #fef9c3;
    color: #854d0e;
}

.risk-high {
    background: #fee2e2;
    color: #991b1b;
}

/* Sidebar Tips Box */
.sidebar-tip {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    padding: 18px;
    border-radius: 14px;
    margin: 16px 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.sidebar-tip h3 {
    font-size: 16px !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
}

.sidebar-tip p {
    font-size: 13px !important;
    line-height: 1.6 !important;
    opacity: 0.95;
}

/* Feature Importance Bar */
.feature-bar {
    height: 8px;
    background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
    border-radius: 4px;
    margin: 8px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# Load Model Functions
@st.cache_resource
def load_model():
    try:
        return joblib.load("logistic_regression_tuned.pkl")
    except Exception as e:
        st.error(f"⚠️ Model file not found: {e}")
        st.stop()


@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"⚠️ Scaler file not found: {e}")
        st.stop()


@st.cache_resource
def load_selected_features():
    try:
        return joblib.load("selected_features.pkl")
    except Exception as e:
        st.error(f"⚠️ Selected features file not found: {e}")
        st.stop()


# Initialize
model = load_model()
scaler = load_scaler()
selected_features = load_selected_features()

if "errors" not in st.session_state:
    st.session_state.errors = {}

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 30px 0;">
            <h1 style="color:white; margin:0; font-size: 28px;">🧠 MindCheck</h1>
            <p style="color:rgba(255,255,255,0.8); margin-top: 8px; font-size: 13px;">Student Mental Health Assessment</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "📍 Navigation Menu",
        ["🏠 Home", "📊 Predict Depression Risk", "📈 Insights & Analytics", "ℹ️ About"],
        label_visibility="visible",
    )

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.3); margin:24px 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-tip">
            <h3>💡 Quick Tips</h3>
            <p>
                Answer all questions honestly and accurately for the most reliable prediction. This assessment is confidential and designed to help identify potential mental health concerns early.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-tip">
            <h3>🆘 Need Help?</h3>
            <p>
                If you're feeling overwhelmed or experiencing mental health difficulties, please reach out to a counselor or mental health professional immediately.
            </p>
            <p style="margin-top: 12px; font-weight: 700;">
                📞 National Crisis Helpline<br>
                <span style="font-size: 15px;">1-800-273-8255</span>
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='border:1px solid rgba(255,255,255,0.3); margin:24px 0;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="text-align: center; padding: 12px; opacity: 0.7;">
            <p style="font-size: 11px; margin: 0;">
                © 2026 MindCheck<br>
                Last Updated: {datetime.now().strftime('%B %Y')}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


# HOME PAGE
if page == "🏠 Home":
    st.markdown(
        "<h1 class='main-title'>Welcome to MindCheck</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Your companion for mental health awareness and early detection</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-banner">
        <div class="info-banner-title">🎯 Our Mission</div>
        <p class="info-banner-text">
            MindCheck uses advanced machine learning to help identify students who may be at risk of depression. 
            Early detection can make a significant difference in getting the support you need. This tool provides 
            a preliminary assessment and is not a substitute for professional medical advice.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">84.4%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">94.1%</div>
                <div class="metric-label">Recall Rate</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">100%</div>
                <div class="metric-label">Confidential</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-title">🔍 How It Works</h3>
                <p style="line-height: 1.8; color: #475569;">
                    <strong>1. Complete Assessment:</strong> Answer questions about your academic life, lifestyle, and mental health.<br><br>
                    <strong>2. ML Analysis:</strong> Our trained logistic regression model analyzes your responses.<br><br>
                    <strong>3. Get Results:</strong> Receive a risk assessment with personalized recommendations.<br><br>
                    <strong>4. Take Action:</strong> Use insights to seek appropriate support if needed.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-title">⚡ Key Features</h3>
                <p style="line-height: 1.8; color: #475569;">
                    <strong>✓ Evidence-Based:</strong> Built on real student data and validated metrics.<br><br>
                    <strong>✓ Fast & Easy:</strong> Complete assessment in under 5 minutes.<br><br>
                    <strong>✓ Confidential:</strong> Your responses are private and secure.<br><br>
                    <strong>✓ Actionable Insights:</strong> Get detailed visualizations and recommendations.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-left: 5px solid #f59e0b;">
            <h3 style="color: #92400e !important; margin-bottom: 12px;">⚠️ Important Disclaimer</h3>
            <p style="color: #78350f !important; line-height: 1.8; margin: 0;">
                This tool is designed for screening purposes only and does not provide a clinical diagnosis. 
                If you're experiencing symptoms of depression or mental health concerns, please consult 
                a qualified mental health professional. In case of emergency, please contact your local 
                crisis helpline immediately.
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


# PREDICTION PAGE
elif page == "📊 Predict Depression Risk":

    st.markdown(
        "<h1 class='main-title'>Depression Risk Assessment</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Complete the form below for your personalized mental health screening</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-banner">
        <div class="info-banner-title">🔒 Confidential Assessment</div>
        <p class="info-banner-text">
            Your privacy is our priority. All information you provide is confidential and used solely 
            for generating your risk assessment. We do not store or share your personal data.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.form("assessment_form"):

        # Personal Information
        st.markdown(
            "<div class='section-card'><h2 class='section-title'>👤 Personal Information</h2>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(
                "Age", min_value=0, max_value=30, value=20, help="Enter your current age"
            )
            if "age" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['age']}</p>",
                    unsafe_allow_html=True,
                )
        with col2:
            gender = st.selectbox(
                "Gender",
                ["Select gender", "Female", "Male"],
                index=0,
                help="Select your gender",
            )
            if "gender" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['gender']}</p>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Academic Factors
        st.markdown(
            "<div class='section-card'><h2 class='section-title'>📚 Academic Factors</h2>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            cgpa = st.number_input(
                "CGPA (0.0 - 4.0)",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01,
                help="Enter your current GPA",
            )
            if "cgpa" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['cgpa']}</p>",
                    unsafe_allow_html=True,
                )
            academic_pressure = st.slider(
                "Academic Pressure",
                1,
                5,
                3,
                help="1 = Very Low, 5 = Very High",
            )
            study_satisfaction = st.slider(
                "Study Satisfaction",
                1,
                5,
                3,
                help="1 = Very Dissatisfied, 5 = Very Satisfied",
            )
        with col2:
            work_hours = st.number_input(
                "Work/Study Hours per Day",
                min_value=0,
                max_value=12,
                value=7,
                help="Average daily hours spent on work/study",
            )
            if "work_hours" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['work_hours']}</p>",
                    unsafe_allow_html=True,
                )
            degree_group = st.selectbox(
                "Degree Level",
                [
                    "Select degree level",
                    "Pre-University",
                    "Undergraduate",
                    "Postgraduate",
                ],
                index=0,
                help="Select your current education level",
            )
            if "degree_group" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['degree_group']}</p>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Lifestyle & Health
        st.markdown(
            "<div class='section-card'><h2 class='section-title'>💊 Lifestyle & Health</h2>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.slider(
                "Sleep Duration",
                0,
                3,
                1,
                format="%d",
                help="0 = <5hrs, 1 = 5-6hrs, 2 = 7-8hrs, 3 = >8hrs",
            )
            st.caption(
                ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"][
                    sleep_duration
                ]
            )
            dietary_habits = st.selectbox(
                "Dietary Habits",
                ["Select dietary habits", "Healthy", "Moderate", "Unhealthy"],
                index=0,
                help="Rate your overall eating habits",
            )
            if "dietary_habits" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['dietary_habits']}</p>",
                    unsafe_allow_html=True,
                )
        with col2:
            financial_stress = st.slider(
                "Financial Stress",
                1,
                5,
                3,
                help="1 = Very Low, 5 = Very High",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Mental Health History
        st.markdown(
            "<div class='section-card'><h2 class='section-title'>🧠 Mental Health History</h2>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            suicidal_thoughts = st.selectbox(
                "Have you ever had suicidal thoughts?",
                ["Select", "No", "Yes"],
                index=0,
                help="Your honest answer helps us provide better support",
            )
            if "suicidal_thoughts" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['suicidal_thoughts']}</p>",
                    unsafe_allow_html=True,
                )
        with col2:
            family_history = st.selectbox(
                "Family History of Mental Illness",
                ["Select", "No", "Yes"],
                index=0,
                help="Do any family members have a history of mental illness?",
            )
            if "family_history" in st.session_state.errors:
                st.markdown(
                    f"<p class='error-message'>⚠️ {st.session_state.errors['family_history']}</p>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Analyze My Mental Health Risk")

    if submitted:
        st.session_state.errors = {}

        # Validation
        if not (0 <= age <= 30):
            st.session_state.errors["age"] = "Age must be between 0 and 30"

        if gender == "Select gender":
            st.session_state.errors["gender"] = "Please select your gender"

        if not (0.0 <= cgpa <= 4.0):
            st.session_state.errors["cgpa"] = "CGPA must be between 0.0 and 4.0"

        if not (0 <= work_hours <= 12):
            st.session_state.errors["work_hours"] = (
                "Work hours must be between 0 and 12"
            )

        if degree_group == "Select degree level":
            st.session_state.errors["degree_group"] = "Please select your degree level"

        if dietary_habits == "Select dietary habits":
            st.session_state.errors["dietary_habits"] = (
                "Please select your dietary habits"
            )

        if suicidal_thoughts == "Select":
            st.session_state.errors["suicidal_thoughts"] = "Please answer this question"

        if family_history == "Select":
            st.session_state.errors["family_history"] = "Please answer this question"

        if st.session_state.errors:
            st.rerun()
        else:
            try:
                # Prepare input data
                input_data = pd.DataFrame(
                    [
                        [
                            age,
                            academic_pressure,
                            cgpa,
                            study_satisfaction,
                            sleep_duration,
                            work_hours,
                            financial_stress,
                            1 if gender == "Male" else 0,
                            1 if dietary_habits == "Moderate" else 0,
                            1 if dietary_habits == "Unhealthy" else 0,
                            1 if suicidal_thoughts == "Yes" else 0,
                            1 if family_history == "Yes" else 0,
                            1 if degree_group == "Postgraduate" else 0,
                            1 if degree_group == "Pre-University" else 0,
                            1 if degree_group == "Undergraduate" else 0,
                        ]
                    ],
                    columns=[
                        "Age",
                        "Academic Pressure",
                        "CGPA",
                        "Study Satisfaction",
                        "Sleep Duration",
                        "Work/Study Hours",
                        "Financial Stress",
                        "Gender_Male",
                        "Dietary Habits_Moderate",
                        "Dietary Habits_Unhealthy",
                        "Have you ever had suicidal thoughts ?_Yes",
                        "Family History of Mental Illness_Yes",
                        "Degree_Group_Postgraduate",
                        "Degree_Group_Pre-University",
                        "Degree_Group_Undergraduate",
                    ],
                )

                # Make prediction
                input_selected = input_data[selected_features]
                input_scaled = scaler.transform(input_selected)
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]

                # Store results
                st.session_state.last_probability = probability
                st.session_state.last_input_scaled = input_scaled
                st.session_state.last_prediction = prediction
                st.session_state.last_input_data = input_data

                st.markdown("---")

                # Display results
                col1, col2 = st.columns([2, 1])

                with col1:
                    if prediction == 1:
                        risk_level = (
                            "High" if probability > 0.66 else "Moderate-High"
                        )
                        st.error(
                            f"""
                        ### 🚨 Elevated Depression Risk Detected
                        
                        **Risk Level:** {risk_level}  
                        **Risk Probability:** {probability*100:.1f}%
                        
                        Based on your responses, our model indicates an elevated risk of depression. 
                        We strongly recommend speaking with a mental health professional who can provide 
                        personalized support and guidance.
                        """
                        )
                    else:
                        st.success(
                            f"""
                        ### ✅ Lower Depression Risk
                        
                        **Risk Probability:** {probability*100:.1f}%
                        
                        Your responses suggest a lower likelihood of depression. However, if you're 
                        experiencing any mental health concerns, please don't hesitate to reach out 
                        to a counselor or mental health professional.
                        """
                        )

                with col2:
                    # Risk gauge visualization
                    fig, ax = plt.subplots(figsize=(3, 3))
                    colors = ["#22c55e", "#facc15", "#ef4444"]

                    if probability < 0.33:
                        color_idx = 0
                        risk_text = "Low Risk"
                    elif probability < 0.66:
                        color_idx = 1
                        risk_text = "Moderate Risk"
                    else:
                        color_idx = 2
                        risk_text = "High Risk"

                    ax.pie(
                        [probability, 1 - probability],
                        colors=[colors[color_idx], "#e5e7eb"],
                        startangle=90,
                        counterclock=False,
                    )
                    circle = plt.Circle((0, 0), 0.70, fc="white")
                    ax.add_artist(circle)
                    ax.text(
                        0,
                        0,
                        f"{probability*100:.1f}%\n{risk_text}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        fontweight="bold",
                    )
                    st.pyplot(fig)
                    plt.close()

                st.info(
                    """
                **Important Note:** This assessment is a screening tool and does not constitute a 
                clinical diagnosis. For accurate diagnosis and treatment, please consult with a 
                qualified mental health professional.
                """
                )

                # Show input summary
                with st.expander("📋 View Your Responses"):
                    st.dataframe(input_data, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")


# INSIGHTS PAGE
elif page == "📈 Insights & Analytics":

    st.markdown(
        "<h1 class='main-title'>Insights & Analytics</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Understanding your assessment results and model predictions</p>",
        unsafe_allow_html=True,
    )

    if "last_probability" not in st.session_state:
        st.warning(
            """
        ### 📊 No Assessment Data Available
        
        Please complete the depression risk assessment first to view your personalized insights.
        Navigate to **📊 Predict Depression Risk** to get started.
        """
        )
    else:
        prob = st.session_state.last_probability
        prediction = st.session_state.last_prediction

        # Risk Summary Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div class="metric-label">Risk Probability</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            risk_category = (
                "Low Risk" if prob < 0.33 else "Moderate" if prob < 0.66 else "High Risk"
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 24px;">{risk_category}</div>
                    <div class="metric-label">Risk Category</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            status = "⚠️ At Risk" if prediction == 1 else "✅ Not At Risk"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 22px;">{status}</div>
                    <div class="metric-label">Prediction</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk Distribution
        st.markdown(
            """
            <div class="chart-container">
                <h3 style="margin-bottom: 20px;">🎯 Your Risk Level</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )

        risk_labels = ["Low\nRisk", "Moderate\nRisk", "High\nRisk"]
        risk_thresholds = [0.33, 0.66, 1.0]

        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ["#22c55e", "#facc15", "#ef4444"]

        for i, (label, threshold) in enumerate(zip(risk_labels, risk_thresholds)):
            start = 0 if i == 0 else risk_thresholds[i - 1]
            width = threshold - start
            alpha = 1.0 if start <= prob < threshold else 0.3
            ax.barh(0, width, left=start, color=colors[i], alpha=alpha, height=0.5)

        ax.axvline(prob, color="#1e3a8a", linewidth=3, linestyle="--", label="Your Score")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([0, 0.33, 0.66, 1.0])
        ax.set_xticklabels(["0%", "33%", "66%", "100%"])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper right")
        ax.set_xlabel("Risk Probability", fontsize=11, fontweight="bold")

        st.pyplot(fig)
        plt.close()

        # Feature Contributions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="chart-container">
                <h3 style="margin-bottom: 20px;">📊 Key Factors Influencing Your Result</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )

        coef = model.coef_[0]
        contributions = coef * st.session_state.last_input_scaled[0]

        contrib_df = pd.DataFrame(
            {"Feature": selected_features, "Contribution": contributions}
        ).sort_values("Contribution", key=abs, ascending=False)[:8]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = ["#ef4444" if c > 0 else "#22c55e" for c in contrib_df["Contribution"]]
        ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors_bar)
        ax.set_xlabel("Contribution to Depression Risk", fontsize=11, fontweight="bold")
        ax.set_title(
            "Top Factors Contributing to Your Risk Score",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

        st.pyplot(fig)
        plt.close()

        st.caption(
            "🔴 Red bars = Increases risk  |  🟢 Green bars = Decreases risk"
        )

        # Recommendations
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-title">💡 Personalized Recommendations</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )

        recommendations = []

        if prob >= 0.5:
            recommendations.append(
                "🆘 **Seek Professional Help:** Consider scheduling an appointment with a mental health counselor or therapist."
            )

        # Check specific risk factors from input data
        input_data = st.session_state.last_input_data

        if input_data["Sleep Duration"].values[0] <= 1:
            recommendations.append(
                "😴 **Improve Sleep:** Aim for 7-8 hours of quality sleep each night. Establish a consistent bedtime routine."
            )

        if input_data["Academic Pressure"].values[0] >= 4:
            recommendations.append(
                "📚 **Manage Academic Stress:** Break tasks into smaller steps, use time management techniques, and don't hesitate to ask for help."
            )

        if input_data["Financial Stress"].values[0] >= 4:
            recommendations.append(
                "💰 **Address Financial Concerns:** Explore financial aid options, student budgeting resources, or speak with a financial counselor."
            )

        if input_data["Study Satisfaction"].values[0] <= 2:
            recommendations.append(
                "🎯 **Increase Study Satisfaction:** Consider joining study groups, exploring different learning methods, or speaking with an academic advisor."
            )

        recommendations.append(
            "🏃 **Physical Activity:** Regular exercise can significantly improve mental health. Aim for 30 minutes of activity daily."
        )
        recommendations.append(
            "🤝 **Social Connection:** Maintain relationships with friends and family. Social support is crucial for mental wellbeing."
        )

        for rec in recommendations:
            st.markdown(f"- {rec}")


# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown(
        "<h1 class='main-title'>About MindCheck</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Learn more about our depression screening tool and methodology</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="section-card">
        <h3 class="section-title">🎯 Project Overview</h3>
        <p style="line-height: 1.8; color: #475569;">
            MindCheck is a machine learning-based depression risk assessment tool designed specifically 
            for students. Built using a Logistic Regression model trained on real student data, this 
            application aims to provide early detection of depression risk factors to help students 
            seek timely support and intervention.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-title">📈 Model Performance</h3>
                <p style="color: #475569; line-height: 1.8;">
                    <strong>Algorithm:</strong> Logistic Regression (Tuned)<br>
                    <strong>Accuracy:</strong> 84.4%<br>
                    <strong>Recall:</strong> 94.1%<br>
                    <strong>Precision:</strong> 84.0%<br>
                    <strong>F1-Score:</strong> 87.2%<br><br>
                    Our model prioritizes recall to minimize false negatives, ensuring 
                    we identify as many at-risk students as possible.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-title">🔬 Features Used</h3>
                <p style="color: #475569; line-height: 1.8;">
                    The model analyzes multiple factors:<br><br>
                    • Academic Pressure & Performance<br>
                    • Sleep Duration & Quality<br>
                    • Financial Stress Levels<br>
                    • Study Satisfaction<br>
                    • Lifestyle Habits<br>
                    • Mental Health History<br>
                    • Family Background
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div class="section-card">
        <h3 class="section-title">⚙️ Development Process</h3>
        <p style="color: #475569; line-height: 1.8;">
            <strong>1. Data Collection:</strong> Analyzed student depression dataset with 27,929 records<br>
            <strong>2. Data Preprocessing:</strong> Cleaned data, handled missing values, and engineered features<br>
            <strong>3. Model Training:</strong> Trained multiple models and selected Logistic Regression based on performance<br>
            <strong>4. Hyperparameter Tuning:</strong> Optimized model parameters using RandomizedSearchCV<br>
            <strong>5. Validation:</strong> Evaluated using stratified k-fold cross-validation<br>
            <strong>6. Deployment:</strong> Built this interactive Streamlit application for easy access
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="section-card" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);">
        <h3 style="color: #1e40af !important; margin-bottom: 12px;">👨‍💻 Developer Information</h3>
        <p style="color: #1e3a8a !important; line-height: 1.8; margin: 0;">
            <strong>Student Name:</strong> Foo Jing Heng Anson (2401482A)<br>
            <strong>Tutorial Group:</strong> T07<br>
            <strong>Module:</strong> Machine Learning for Developers (CAI2C08)<br>
            <strong>Institution:</strong> Temasek Polytechnic - School of Informatics & IT<br>
            <strong>Academic Year:</strong> 2025/2026 October Semester
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="section-card" style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);">
        <h3 style="color: #92400e !important; margin-bottom: 12px;">⚖️ Ethical Considerations</h3>
        <p style="color: #78350f !important; line-height: 1.8; margin: 0;">
            This tool is designed to assist, not replace, professional mental health assessment. 
            We acknowledge the sensitive nature of mental health data and have implemented this tool 
            with privacy and ethical considerations in mind. All assessments are confidential and 
            no personal data is stored. Users are encouraged to seek professional help for accurate 
            diagnosis and treatment.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )