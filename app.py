import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import time

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Advanced Diabetes Prediction System with Telehealth Features')

# --------- IoT Wearable Data Simulation ---------
st.header("Simulated IoT Wearable Device Data")
iot_placeholder = st.empty()

# Simulate streaming data for demonstration (non-blocking approach would be ideal)
for i in range(10):  # shorter loop for demo
    glucose = np.random.normal(120, 10)
    bp_systolic = np.random.normal(120, 5)
    heart_rate = np.random.normal(70, 5)
    alert_msg = ""
    if glucose < 70:
        alert_msg = "⚠️ Low glucose alert!"
    elif glucose > 180:
        alert_msg = "⚠️ High glucose alert!"
    data_str = f"Glucose: {glucose:.1f} mg/dL | Systolic BP: {bp_systolic:.1f} mmHg | Heart Rate: {heart_rate:.0f} bpm {alert_msg}"
    iot_placeholder.write(data_str)
    time.sleep(1)

# --------- User Input for Prediction ---------
st.header('Patient Data Input for Prediction')

def user_input_features():
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose_input = st.number_input('Glucose', min_value=40, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=20, max_value=130, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=5, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=15, max_value=900, value=79)
    bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Age', min_value=10, max_value=100, value=33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose_input,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction Results')
st.write('Diabetes Positive' if prediction[0] == 1 else 'Diabetes Negative')
st.write(f'Positive Probability: {prediction_proba[0][1]:.2f}, Negative Probability: {prediction_proba[0][0]:.2f}')

if prediction_proba[0][1] > 0.5:
    st.error("Alert: High risk of diabetes detected! Please consult a healthcare professional.")
else:
    st.success("Low risk of diabetes based on input data.")

# --------- AI Chatbot for Symptom Assessment ---------
st.header("Symptom Assessment Chatbot")
symptoms = st.text_area("Describe any symptoms you are experiencing:")

if st.button("Assess Symptom Risk"):
    risk_keywords = ['fatigue', 'blurred vision', 'frequent urination', 'thirst']
    risk_score = sum(word in symptoms.lower() for word in risk_keywords)
    if risk_score > 2:
        st.warning("High symptom risk indicators detected. Consider undergoing a full assessment.")
    elif risk_score > 0:
        st.info("Some symptom risk indicators noted.")
    else:
        st.success("Symptoms currently do not indicate high risk.")

# --------- Interactive SHAP Visualization ---------
st.header("Interactive Risk Factor Exploration")

# Interactive sliders for key features
glucose_slider = st.slider('Glucose', min_value=40, max_value=200, value=int(input_df['Glucose']))
bmi_slider = st.slider('BMI', min_value=10.0, max_value=70.0, value=float(input_df['BMI']))
age_slider = st.slider('Age', min_value=10, max_value=100, value=int(input_df['Age']))

# Create dataframe from sliders to get SHAP for adjusted values
interactive_data = input_df.copy()
interactive_data['Glucose'] = glucose_slider
interactive_data['BMI'] = bmi_slider
interactive_data['Age'] = age_slider

interactive_scaled = scaler.transform(interactive_data)
interactive_pred = model.predict(interactive_scaled)
interactive_shap_exp = shap.TreeExplainer(model)
shap_values = interactive_shap_exp.shap_values(interactive_scaled)

# For binary classifier, select predicted class shap values
if isinstance(shap_values, list):
    shap_vals = shap_values[interactive_pred[0]][0]
else:
    shap_vals = shap_values[0]

feature_names = interactive_data.columns.tolist()

shap_vals = np.array(shap_vals).flatten()
min_len = min(len(feature_names), len(shap_vals))
shap_df = pd.DataFrame({
    'Feature': feature_names[:min_len],
    'SHAP Value': shap_vals[:min_len]
}).sort_values(by='SHAP Value', key=abs, ascending=False)

fig, ax = plt.subplots()
colors = ['red' if val < 0 else 'green' for val in shap_df['SHAP Value']]
ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
ax.set_xlabel("SHAP Value (Impact)")
ax.invert_yaxis()
st.pyplot(fig)
