# Banknote Authentication Scanner – ML System Design

## Overview
This project implements a **banknote authentication scanner** that detects whether a currency note is **Genuine or Fake** using machine learning.  
The system simulates a real-world banknote scanner by accepting an image input, extracting statistical features, and performing real-time inference.

---

## System Architecture

<p align="center">
  <img src="diagram.png" width="700">
</p>

---

## Data Ingestion
- Input is a **banknote image** uploaded by the user
- Images are processed in real time using OpenCV

---

## Feature Engineering
Statistical features are extracted from the grayscale image:
- **Variance** – texture variation
- **Skewness** – distribution asymmetry
- **Kurtosis** – sharpness of peaks
- **Entropy** – randomness in pixel intensity

These features are commonly used in banknote authentication datasets.

---

## Model Training
- Dataset: Banknote Authentication dataset
- Feature scaling using **StandardScaler**
- Model: **Logistic Regression**
- Threshold tuning applied to improve detection performance
- Trained model and scaler stored as `.pkl` files

---

## Inference Flow
1. User uploads a banknote image
2. Features are automatically extracted from the image
3. Features are scaled using the trained scaler
4. ML model predicts fraud probability
5. Threshold-based decision is displayed:
   - Genuine
   - Fake

---

## Deployment
- Application built using **Streamlit**
- Deployed on **Streamlit Cloud**
- Supports real-time image-based inference

---

## Monitoring & Retraining Strategy
- Monitor feature distributions from uploaded images
- Track prediction confidence
- Retrain the model periodically or when data drift is detected

---

## Technologies Used
- Python
- Streamlit
- OpenCV
- Scikit-learn
- NumPy
- SciPy
- Joblib

---

## Conclusion
This project demonstrates a complete **production-style ML system**, integrating image processing, feature engineering, model inference, and deployment.  
It closely resembles real-world fraud detection scanners used in financial systems.
