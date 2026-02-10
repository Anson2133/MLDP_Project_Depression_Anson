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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #ffffff;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a 0%, #2563eb 100%);
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stRadio > label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: white !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.1) !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    margin: 4px 0 !important;
    transition: all 0.2s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Main Content */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #1e293b !important;
}

.main-title {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 8px !important;
    text-align: center;
}

.subtitle {
    font-size: 16px !important;
    color: #64748b !important;
    margin-bottom: 30px !important;
    text-align: center;
    font-weight: 400;
}

/* Info Banner */
.info-banner {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 20px;
    border-radius: 8px;
    margin: 25px 0;
    border: 1px solid #e2e8f0;
}

.info-banner-title {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    margin-bottom: 10px !important;
}

.info-banner-text {
    color: #475569 !important;
    line-height: 1.6;
    margin: 0;
    font-size: 14px;
}

/* Section Cards */
.section-card {
    background: white;
    padding: 28px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 24px;
    border: 1px solid #e2e8f0;
}

.section-title {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    margin-bottom: 20px !important;
    padding-bottom: 12px;
    border-bottom: 2px solid #e2e8f0;
}

/* Form Inputs */
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #1e293b !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #334155 !important;
    margin-bottom: 6px !important;
}

/* Slider Styling */
.stSlider > div > div > div {
    background-color: #e2e8f0 !important;
}

.stSlider > div > div > div > div {
    background-color: #2563eb !important;
}

.stSlider [role="slider"] {
    background-color: #1e3a8a !important;
    border: 2px solid white !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}

/* Submit Button */
.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #1e40af !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}

/* Alert Boxes */
.stAlert {
    border-radius: 6px !important;
    border: none !important;
    padding: 16px !important;
    font-size: 14px !important;
}

/* Error Messages */
.error-message {
    color: #dc2626;
    font-size: 13px;
    margin-top: 4px;
    font-weight: 500;
}

/* Dataframe Styling */
.stDataFrame {
    border-radius: 6px !important;
    overflow: hidden !important;
}

/* Metric Cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    text-align: center;
    border: 1px solid #e2e8f0;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #2563eb;
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
    margin-top: 8px;
}

/* Chart Container */
.chart-container {
    background: white;
    padding: 24px;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 20px 0;
    border: 1px solid #e2e8f0;
}

/* Sidebar Tips Box */
.sidebar-tip {
    background: rgba(255, 255, 255, 0.1);
    padding: 14px;
    border-radius: 6px;
    margin: 14px 0;
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.sidebar-tip h3 {
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
}

.sidebar-tip p {
    font-size: 12px !important;
    line-height: 1.5 !important;
    opacity: 0.9;
}

</style>
""",
    unsafe_allow_html=True,
)


import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(__file__)

# Load Model Functions
@st.cache_resource
def load_model():
    try:
        return joblib.load(
            os.path.join(BASE_DIR, "logistic_regression_tuned.pkl")
        )
    except Exception as e:
        st.error(f"Model file not found: {e}")
        st.stop()


@st.cache_resource
def load_scaler():
    try:
        return joblib.load(
            os.path.join(BASE_DIR, "scaler.pkl")
        )
    except Exception as e:
        st.error(f"Scaler file not found: {e}")
        st.stop()


@st.cache_resource
def load_selected_features():
    try:
        return joblib.load(
            os.path.join(BASE_DIR, "selected_features.pkl")
        )
    except Exception as e:
        st.error(f"Selected features file not found: {e}")
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
        <div style="text-align:center; padding: 16px 0 24px 0;">
            <h2 style="color:white; margin:0; font-size: 22px; font-weight: 600;">Depression Risk Assessment</h2>
            <p style="color:rgba(255,255,255,0.8); margin-top: 6px; font-size: 12px;">ML-Based Screening Tool</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
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
        <div style="text-align: center; padding: 10px; opacity: 0.6;">
            <p style="font-size: 10px; margin: 0;">
                v1.0 | {datetime.now().strftime('%Y')}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


# HOME PAGE
if page == "🏠 Home":
    st.markdown(
        "<h1 class='main-title'>Student Depression Risk Assessment</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Evidence-based machine learning screening tool for early depression detection</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="info-banner">
        <div class="info-banner-title">About This Tool</div>
        <p class="info-banner-text">
            This application uses a trained logistic regression model to assess depression risk factors among students. 
            It analyzes multiple aspects including academic pressure, lifestyle habits, and mental health history to provide 
            a preliminary risk assessment. This is a screening tool only and does not replace professional diagnosis.
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
                <div class="metric-value">87.2%</div>
                <div class="metric-label">F1 Score</div>
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
                <h3 class="section-title">Assessment Process</h3>
                <p style="line-height: 1.7; color: #475569; font-size: 14px;">
                    <strong>Step 1:</strong> Complete the questionnaire with information about your academic life, 
                    lifestyle habits, and mental health history.<br><br>
                    <strong>Step 2:</strong> The model analyzes your responses using trained coefficients to calculate 
                    a risk probability.<br><br>
                    <strong>Step 3:</strong> Review your results and personalized recommendations based on the 
                    contributing factors.<br><br>
                    <strong>Step 4:</strong> If indicated, seek professional support from a qualified mental health provider.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3 class="section-title">Key Features Analyzed</h3>
                <p style="line-height: 1.7; color: #475569; font-size: 14px;">
                    • Academic Pressure and Performance (CGPA)<br>
                    • Work/Study Hours and Time Management<br>
                    • Sleep Duration and Quality<br>
                    • Financial Stress Levels<br>
                    • Study Satisfaction and Engagement<br>
                    • Dietary Habits and Lifestyle<br>
                    • Personal and Family Mental Health History<br>
                    • Demographic Factors
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 18px; border-radius: 6px; margin: 24px 0; border: 1px solid #fcd34d;">
            <h4 style="color: #92400e !important; margin: 0 0 10px 0; font-size: 15px; font-weight: 600;">Important Disclaimer</h4>
            <p style="color: #78350f !important; line-height: 1.6; margin: 0; font-size: 13px;">
                This assessment tool is designed for screening purposes only and does not constitute a clinical diagnosis. 
                If you are experiencing symptoms of depression or mental health concerns, please consult with a qualified 
                mental health professional. In case of emergency or crisis, please contact your local emergency services 
                or crisis helpline immediately.
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
                "Academic Pressure (1 = Lowest, 5 = Highest)",
                1,
                5,
                3,
            )
            study_satisfaction = st.slider(
                "Study Satisfaction (1 = Least Satisfied, 5 = Most Satisfied)",
                1,
                5,
                3,
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
                "Sleep Duration (0 = <5hrs, 1 = 5-6hrs, 2 = 7-8hrs, 3 = >8hrs)",
                0,
                3,
                1,
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
                "Financial Stress (1 = Lowest, 5 = Highest)",
                1,
                5,
                3,
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

                # Display results in a professional manner
                if prediction == 1:
                    risk_level = "High" if probability > 0.66 else "Moderate-High"
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
        "<h1 class='main-title'>Model Insights & Analysis</h1>", unsafe_allow_html=True
    )
    st.markdown(
        "<p class='subtitle'>Understanding prediction factors and model performance</p>",
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

        # Professional Risk Summary
        st.markdown("### Assessment Summary")
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Risk Probability", f"{prob*100:.1f}%")

        with col2:
            risk_category = (
                "Low" if prob < 0.33 else "Moderate" if prob < 0.66 else "High"
            )
            st.metric("Risk Category", risk_category)

        with col3:
            status = "At Risk" if prediction == 1 else "Not At Risk"
            st.metric("Classification", status)
            
        with col4:
            confidence = max(prob, 1-prob) * 100
            st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("---")

        # Feature Contribution Analysis
        st.markdown("### Feature Contribution Analysis")
        st.markdown("This chart shows which factors had the strongest influence on your risk assessment.")

        coef = model.coef_[0]
        contributions = coef * st.session_state.last_input_scaled[0]

        contrib_df = pd.DataFrame(
            {"Feature": selected_features, "Contribution": contributions}
        ).sort_values("Contribution", key=abs, ascending=False)[:10]

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_bar = ["#dc2626" if c > 0 else "#16a34a" for c in contrib_df["Contribution"]]
        
        bars = ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors_bar, alpha=0.8)
        ax.set_xlabel("Contribution Score", fontsize=11, fontweight='500')
        ax.set_ylabel("Feature", fontsize=11, fontweight='500')
        ax.set_title("Top 10 Contributing Factors", fontsize=13, fontweight='600', pad=15, loc='left')
        ax.axvline(0, color='#64748b', linewidth=1, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2, linestyle='--')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', 
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=9, color='#475569', fontweight='500')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.caption("🔴 Red = Increases depression risk  |  🟢 Green = Decreases depression risk")

        st.markdown("---")

   
        st.markdown("### Model Feature Importance")
        st.markdown("Understanding which features the model considers most important overall.")
        
    
        all_coefs = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)[:10]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_coef = ['#dc2626' if c > 0 else '#16a34a' for c in all_coefs['Coefficient']]
        
        bars = ax.barh(all_coefs['Feature'], all_coefs['Coefficient'], color=colors_coef, alpha=0.8)
        ax.set_xlabel('Model Coefficient', fontsize=11, fontweight='500')
        ax.set_ylabel('Feature', fontsize=11, fontweight='500')
        ax.set_title('Top 10 Model Coefficients', fontsize=13, fontweight='600', pad=15, loc='left')
        ax.axvline(0, color='#64748b', linewidth=1, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2, linestyle='--')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', 
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=9, color='#475569', fontweight='500')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.caption("These coefficients show the general influence each feature has on the model's predictions.")

        st.markdown("---")

        # Personalized Recommendations
        st.markdown("### Personalized Recommendations")
        
        input_data = st.session_state.last_input_data

        recommendations = []

        if prob >= 0.5:
            recommendations.append({
                "priority": "High",
                "category": "Professional Support",
                "recommendation": "Consider scheduling an appointment with a mental health counselor or therapist.",
                "icon": "🆘"
            })

        if input_data["Sleep Duration"].values[0] <= 1:
            recommendations.append({
                "priority": "High",
                "category": "Sleep Hygiene",
                "recommendation": "Aim for 7-8 hours of quality sleep each night. Establish a consistent bedtime routine.",
                "icon": "😴"
            })

        if input_data["Academic Pressure"].values[0] >= 4:
            recommendations.append({
                "priority": "High",
                "category": "Stress Management",
                "recommendation": "Break academic tasks into smaller steps, use time management techniques, and seek help when needed.",
                "icon": "📚"
            })

        if input_data["Financial Stress"].values[0] >= 4:
            recommendations.append({
                "priority": "Medium",
                "category": "Financial Wellness",
                "recommendation": "Explore financial aid options, student budgeting resources, or speak with a financial counselor.",
                "icon": "💰"
            })

        if input_data["Study Satisfaction"].values[0] <= 2:
            recommendations.append({
                "priority": "Medium",
                "category": "Academic Engagement",
                "recommendation": "Join study groups, explore different learning methods, or speak with an academic advisor.",
                "icon": "🎯"
            })

        # Always add these
        recommendations.append({
            "priority": "Medium",
            "category": "Physical Activity",
            "recommendation": "Regular exercise can significantly improve mental health. Aim for 30 minutes of activity daily.",
            "icon": "🏃"
        })
        
        recommendations.append({
            "priority": "Medium",
            "category": "Social Connection",
            "recommendation": "Maintain relationships with friends and family. Social support is crucial for mental wellbeing.",
            "icon": "🤝"
        })

        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(
                rec_df[['priority', 'category', 'recommendation', 'icon']],
                column_config={
                    "priority": "Priority",
                    "category": "Category",
                    "recommendation": "Recommendation",
                    "icon": "📋"
                },
                hide_index=True,
                use_container_width=True
            )


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
