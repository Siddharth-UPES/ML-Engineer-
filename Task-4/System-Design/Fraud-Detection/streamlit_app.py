import streamlit as st
import numpy as np
import cv2
import joblib
from scipy.stats import skew, kurtosis

# Load model & scaler
model = joblib.load("model/classifier.pkl")
scaler = joblib.load("model/feature_scaler.pkl")

st.set_page_config(page_title="Banknote Scanner", layout="centered")

st.title("💵 Banknote Authentication Scanner")
st.write("Upload a banknote image to detect whether it is **Genuine** or **Fake**")

uploaded_file = st.file_uploader("📤 Upload Banknote Image", type=["jpg", "png", "jpeg"])

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)

    variance = gray.var()
    skewness = skew(gray.flatten())
    kurt = kurtosis(gray.flatten())
    entropy = -np.sum(
        (gray/255) * np.log2((gray/255) + 1e-10)
    )

    return np.array([[variance, skewness, kurt, entropy]])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Banknote", use_column_width=True)

    features = extract_features(image)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    st.subheader("🔍 Scan Result")

    if prediction == 0:
        st.success("✅ Genuine Banknote")
    else:
        st.error("❌ Fake Banknote")
