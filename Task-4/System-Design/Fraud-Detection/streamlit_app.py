import streamlit as st
import numpy as np
import joblib

# Load model artifacts
model = joblib.load("model/classifier.pkl")
scaler = joblib.load("model/feature_scaler.pkl")

st.set_page_config(page_title="Banknote Fraud Detector", layout="centered")

st.title("💵 Banknote Authentication System")
st.write("Check whether a banknote is **Genuine or Fake** using ML.")

st.divider()

# User Inputs
variance = st.number_input("Variance", value=0.0)
skewness = st.number_input("Skewness", value=0.0)
kurtosis = st.number_input("Kurtosis", value=0.0)
entropy = st.number_input("Entropy", value=0.0)

if st.button("Analyze Banknote"):
    input_data = np.array([[variance, skewness, kurtosis, entropy]])
    scaled_data = scaler.transform(input_data)

    probability = model.predict_proba(scaled_data)[0][1]

    if probability > 0.4:
        st.error(f"🚨 Fake Banknote Detected (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Genuine Banknote (Confidence: {1-probability:.2f})")

st.caption("Model: Logistic Regression | Threshold Tuned")
