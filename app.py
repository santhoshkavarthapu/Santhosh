import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def user_input_features():
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    glucose = st.number_input('Glucose', min_value=40, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=20, max_value=130, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=5, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=15, max_value=900, value=79)
    bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Age', min_value=10, max_value=100, value=33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

st.title('Diabetes Prediction System with Risk Visualization')

input_df = user_input_features()

# Scale user input
input_scaled = scaler.transform(input_df)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction')
st.write('Diabetes Positive' if prediction[0] == 1 else 'Diabetes Negative')

st.subheader('Prediction Probability')
st.write(f'Positive: {prediction_proba[0][1]:.2f}, Negative: {prediction_proba[0][0]:.2f}')

# SHAP explanation and manual bar plot
st.subheader('Feature Impact on Prediction')

explainer = shap.TreeExplainer(model)
shap_values_all = explainer.shap_values(input_scaled)

if isinstance(shap_values_all, list):
    sample_shap_values = shap_values_all[prediction[0]][0]
else:
    sample_shap_values = shap_values_all[0]

sample_shap_values = np.array(sample_shap_values).flatten()
feature_names = input_df.columns.tolist()

st.write("Feature names length:", len(feature_names))
st.write("SHAP values length:", len(sample_shap_values))

min_length = min(len(feature_names), len(sample_shap_values))
shap_df = pd.DataFrame({
    'Feature': feature_names[:min_length],
    'SHAP Value': sample_shap_values[:min_length]
}).sort_values(by='SHAP Value', key=abs, ascending=False)

fig, ax = plt.subplots()
colors = ['red' if val < 0 else 'green' for val in shap_df['SHAP Value']]
ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
ax.set_xlabel('SHAP Value (Impact)')
ax.invert_yaxis()
st.pyplot(fig)

threshold = 0.5
if prediction_proba[0][1] > threshold:
    st.error("Alert: High risk of diabetes detected! Please consult a healthcare professional.")
else:
    st.success("Low risk of diabetes based on input data.")
