import streamlit as st
import numpy as np
import joblib
import cv2
import os
from scipy.stats import skew, kurtosis

# ==============================
# SAFE PATH HANDLING (IMPORTANT)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "classifier.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "feature_scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Banknote Scanner", layout="centered")

st.title("💵 Banknote Authentication Scanner")
st.write("Upload a banknote image to check whether it is **Genuine or Fake**.")

uploaded_file = st.file_uploader("Upload Banknote Image", type=["jpg", "jpeg", "png"])

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(float)

    variance = gray.var()
    skewness = skew(gray.flatten())
    kurt = kurtosis(gray.flatten())
    entropy = -np.sum((gray/255) * np.log2((gray/255) + 1e-10))

    return np.array([[variance, skewness, kurt, entropy]])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Banknote", use_column_width=True)

    features = extract_features(image)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    st.subheader("Scan Result")

    if prediction == 1:
        st.error("❌ Fake Banknote Detected")
    else:
        st.success("✅ Genuine Banknote")
