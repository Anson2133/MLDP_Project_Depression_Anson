import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Depression Risk Predictor",
    layout="centered"
)

st.title("Depression Risk Prediction System")
st.write("Fill in the details below and click **Predict**.")


@st.cache_resource
def load_model():
    model_path = os.path.join("..", "logistic_regression_tuned.pkl")
    return joblib.load(model_path)

model = load_model()

st.header("Personal & Academic Information")

age = st.number_input("Age", min_value=10, max_value=100, value=20)
academic_pressure = st.slider("Academic Pressure", 0, 5, 5)
cgpa = st.slider("CGPA", 0.0, 4.0, 3.0)
study_satisfaction = st.slider("Study Satisfaction", 0, 5, 5)
sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 7)
work_hours = st.slider("Work / Study Hours per day", 0, 16, 6)
financial_stress = st.slider("Financial Stress", 0, 10, 5)

st.header("Lifestyle & Background")

gender = st.selectbox("Gender", ["Female", "Male"])
diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
suicidal = st.selectbox(
    "Have you ever had suicidal thoughts?",
    ["No", "Yes"]
)
family_history = st.selectbox(
    "Family History of Mental Illness",
    ["No", "Yes"]
)
degree = st.selectbox(
    "Degree Level",
    ["Other", "Pre-University", "Undergraduate", "Postgraduate"]
)


input_data = np.array([
    age,
    academic_pressure,
    cgpa,
    study_satisfaction,
    sleep_duration,
    work_hours,
    financial_stress,
    1 if gender == "Male" else 0,
    1 if diet == "Moderate" else 0,
    1 if diet == "Unhealthy" else 0,
    1 if suicidal == "Yes" else 0,
    1 if family_history == "Yes" else 0,
    1 if degree == "Postgraduate" else 0,
    1 if degree == "Pre-University" else 0,
    1 if degree == "Undergraduate" else 0
]).reshape(1, -1)


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"High Risk of Depression Detected\n\n"
            f"Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"Low Risk of Depression Detected\n\n"
            f"Probability: {probability:.2%}"
        )