import streamlit as st
import numpy as np
import joblib
import os

# Safe base directory (local + cloud dono ke liye)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "classifier.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "feature_scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.set_page_config(page_title="Banknote Fraud Detector", layout="centered")

st.title("💵 Banknote Authentication System")
st.write("Detect whether a banknote is **Genuine or Fake**")

variance = st.number_input("Variance", value=0.0)
skewness = st.number_input("Skewness", value=0.0)
kurtosis = st.number_input("Kurtosis", value=0.0)
entropy = st.number_input("Entropy", value=0.0)

if st.button("Analyze"):
    X = np.array([[variance, skewness, kurtosis, entropy]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]

    if prob > 0.4:
        st.error(f"🚨 Fake Banknote Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Genuine Banknote (Confidence: {1-prob:.2f})")
