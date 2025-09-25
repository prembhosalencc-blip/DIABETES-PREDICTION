import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

# Page config
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("🩺 Diabetes Prediction System")

# Load model and scaler
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Tabs for Dashboard and Prediction
tab1, tab2 = st.tabs(["📊 Dashboard", "🔍 Prediction"])

# Tab 1: Dashboard
with tab1:
    st.header("Dataset Overview")
    data = pd.read_csv("data/pima-indians-diabetes.csv")
    
    # Show dataset metrics
    total_patients = data.shape[0]
    diabetic_count = data[data['Outcome'] == 1].shape[0]
    non_diabetic_count = data[data['Outcome'] == 0].shape[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", total_patients)
    col2.metric("Diabetic", diabetic_count)
    col3.metric("Non-Diabetic", non_diabetic_count)
    
    # Pie chart for diabetic distribution
    pie_df = pd.DataFrame({
        'Category': ['Non-Diabetic', 'Diabetic'],
        'Count': [non_diabetic_count, diabetic_count]
    })
    pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
        theta="Count:Q",
        color="Category:N",
        tooltip=["Category", "Count"]
    )
    st.altair_chart(pie_chart, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Feature Correlations")
    corr = data.corr()
    st.dataframe(corr)

# Tab 2: Prediction
with tab2:
    st.header("Enter Patient Details for Prediction")
    pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 900, 79)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 0, 120, 30)
    
    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            st.error(f"🚨 High Risk of Diabetes ({probability:.2f}% probability)")
        else:
            st.success(f"✅ Low Risk of Diabetes ({100-probability:.2f}% probability)")
