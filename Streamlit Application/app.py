import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="Student Depression Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #1e293b !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #0f172a !important;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #64748b !important;
    margin-bottom: 25px;
}

.info-banner {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #3b82f6;
    padding: 20px;
    border-radius: 12px;
    margin: 25px 0;
}

.info-banner-title {
    font-size: 18px;
    font-weight: 600;
    color: #1e40af !important;
    margin-bottom: 10px;
}

.info-banner-text {
    color: #1e40af !important;
    line-height: 1.6;
    margin: 0;
}

.section-card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 25px;
    border: 1px solid #e2e8f0;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #0f172a !important;
    margin-bottom: 25px;
}

.stNumberInput > div > div > input {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
}

.stSelectbox > div > div > select {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: white !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > select:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #334155 !important;
    margin-bottom: 6px !important;
}

.stSlider > div > div > div {
    background-color: #e2e8f0 !important;
}

.stSlider > div > div > div > div {
    background-color: #667eea !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #475569 !important;
}

.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

.error-message {
    color: #dc2626;
    font-size: 13px;
    margin-top: 4px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load("logistic_regression_tuned.pkl")
    except:
        st.error("Model file not found")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except:
        st.error("Scaler file not found")
        st.stop()

@st.cache_resource
def load_selected_features():
    try:
        return joblib.load("selected_features.pkl")
    except:
        st.error("Selected features file not found")
        st.stop()

selected_features = load_selected_features()

model = load_model()
scaler = load_scaler()

if 'errors' not in st.session_state:
    st.session_state.errors = {}

with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 12px 0 20px 0;">
            <h2 style="color:white; margin:0;">Navigation</h2>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["Predict Depression Risk"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.35); margin:18px 0;'>", unsafe_allow_html=True)

   
    st.markdown("""
        <div style="
            background:#4f5dbb;
            padding:16px;
            border-radius:12px;
            margin-bottom:16px;
        ">
            <h3 style="color:white; margin:0 0 8px 0;">Quick Tips</h3>
            <p style="
                color:white;
                margin:0;
                font-size:14px;
                line-height:1.5;
            ">
                Fill in all fields honestly and accurately to ensure the most reliable prediction results.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.35); margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="
            background:#4f5dbb;
            padding:16px;
            border-radius:12px;
        ">
            <h3 style="color:white; margin:0 0 8px 0;">Need Help?</h3>
            <p style="
                color:white;
                margin:0;
                font-size:14px;
                line-height:1.5;
            ">
                If you are feeling overwhelmed, please reach out to a counselor or mental health professional.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Student Depression Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Evidence-based mental health screening tool</p>", unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
    <div class="info-banner-title">Confidential Mental Health Assessment</div>
    <p class="info-banner-text">
        This assessment uses machine learning to evaluate depression risk factors among students. Your responses are confidential and will help determine if you might benefit from professional support. This tool is not a diagnosis—please consult a mental health professional for personalized advice.
    </p>
</div>
""", unsafe_allow_html=True)

with st.form("assessment_form"):
    
    st.markdown("<div class='section-card'><h2 class='section-title'>Personal Information</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=59, value=25)
        if 'age' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['age']}</p>", unsafe_allow_html=True)
    with col2:
        gender = st.selectbox("Gender", ["Select gender", "Female", "Male"], index=0)
        if 'gender' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['gender']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-card'><h2 class='section-title'>Academic Factors</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.number_input("CGPA", min_value=5.03, max_value=10.0, value=7.66, step=0.01)
        if 'cgpa' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['cgpa']}</p>", unsafe_allow_html=True)
        academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
    with col2:
        work_hours = st.number_input("Work/Study Hours per Day", min_value=0, max_value=12, value=7)
        if 'work_hours' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['work_hours']}</p>", unsafe_allow_html=True)
        degree_group = st.selectbox("Degree Level", ["Select degree level", "Pre-University", "Undergraduate", "Postgraduate"], index=0)
        if 'degree_group' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['degree_group']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-card'><h2 class='section-title'>Lifestyle & Health</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sleep_duration = st.slider("Sleep Duration Category", 0, 3, 1)
        dietary_habits = st.selectbox("Dietary Habits", ["Select dietary habits", "Healthy", "Moderate", "Unhealthy"], index=0)
        if 'dietary_habits' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['dietary_habits']}</p>", unsafe_allow_html=True)
    with col2:
        financial_stress = st.slider("Financial Stress Level", 1, 5, 3)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-card'><h2 class='section-title'>Mental Health History</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Select", "No", "Yes"], index=0)
        if 'suicidal_thoughts' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['suicidal_thoughts']}</p>", unsafe_allow_html=True)
    with col2:
        family_history = st.selectbox("Family History of Mental Illness", ["Select", "No", "Yes"], index=0)
        if 'family_history' in st.session_state.errors:
            st.markdown(f"<p class='error-message'>{st.session_state.errors['family_history']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    submitted = st.form_submit_button("Submit Assessment")

if submitted:
    st.session_state.errors = {}
    
    if not (18 <= age <= 59):
        st.session_state.errors['age'] = "Age must be between 18 and 59"
    
    if gender == "Select gender":
        st.session_state.errors['gender'] = "Please select your gender"
    
    if not (5.03 <= cgpa <= 10.0):
        st.session_state.errors['cgpa'] = "CGPA must be between 5.03 and 10.0"
    
    if not (0 <= work_hours <= 12):
        st.session_state.errors['work_hours'] = "Work hours must be between 0 and 12"
    
    if degree_group == "Select degree level":
        st.session_state.errors['degree_group'] = "Please select your degree level"
    
    if dietary_habits == "Select dietary habits":
        st.session_state.errors['dietary_habits'] = "Please select your dietary habits"
    
    if suicidal_thoughts == "Select":
        st.session_state.errors['suicidal_thoughts'] = "Please answer this question"
    
    if family_history == "Select":
        st.session_state.errors['family_history'] = "Please answer this question"
    
    if st.session_state.errors:
        st.rerun()
    else:
        try:
            input_data = pd.DataFrame([[
            age, academic_pressure, cgpa, study_satisfaction, sleep_duration, work_hours, financial_stress,
            1 if gender == "Male" else 0,
            1 if dietary_habits == "Moderate" else 0,
            1 if dietary_habits == "Unhealthy" else 0,
            1 if suicidal_thoughts == "Yes" else 0,
            1 if family_history == "Yes" else 0,
            1 if degree_group == "Postgraduate" else 0,
            1 if degree_group == "Pre-University" else 0,
            1 if degree_group == "Undergraduate" else 0,
    ]], columns=['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
             'Work/Study Hours', 'Financial Stress', 'Gender_Male', 'Dietary Habits_Moderate',
             'Dietary Habits_Unhealthy', 'Have you ever had suicidal thoughts ?_Yes',
             'Family History of Mental Illness_Yes', 'Degree_Group_Postgraduate',
             'Degree_Group_Pre-University', 'Degree_Group_Undergraduate'])
            
            input_selected = input_data[selected_features]
            input_scaled = scaler.transform(input_selected)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            st.markdown("---")
            
            if prediction == 1:
                st.error(f"High Risk of Depression Detected\n\nRisk Probability: {probability*100:.1f}%\n\nWe strongly recommend speaking with a mental health professional.")
            else:
                st.success(f"Low Risk of Depression\n\nRisk Probability: {probability*100:.1f}%\n\nYour responses suggest a lower likelihood of depression.")
            
            st.info("This is a screening tool only and does not replace professional medical advice.")
            
        except Exception as e:
          st.error(f"Error during prediction: {e}")