# Task 1: End-to-End ML Pipeline – Banknote Authentication

## Objective
The goal of Task 1 is to build an end-to-end machine learning pipeline to classify banknotes as **genuine or fake** using the Banknote Authentication dataset.

---

## Dataset
**Banknote Authentication Dataset**

**Features:**
- Variance
- Skewness
- Kurtosis
- Entropy

**Target:**
- `class`  
  - 0 → Genuine  
  - 1 → Fake  

---

## Approach

### 1. Data Loading
The dataset is loaded from a CSV file and column names are assigned manually, as the raw dataset does not include headers.

### 2. Data Preprocessing
- Features and target variable are separated
- Feature scaling is applied using **StandardScaler**
- The trained scaler is saved as a `.pkl` file to ensure consistent preprocessing during inference

### 3. Train-Test Split
The data is split into training and testing sets using a fixed `random_state` to ensure reproducibility.

### 4. Model Training
- A **Logistic Regression** model is trained on the scaled training data
- Logistic Regression is chosen for its simplicity, speed, and interpretability

### 5. Model Persistence
- The trained model is saved as `banknote_model.pkl`
- The scaler is saved as `scaler.pkl`
- These files are later used for deployment and inference

### 6. Evaluation
The model is evaluated using standard classification metrics:
- Precision
- Recall
- F1-score

---

## Output Artifacts
- `banknote_model.pkl` – trained classification model
- `scaler.pkl` – feature scaler used during training

---

## Tools & Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Conclusion
This task demonstrates a complete and reproducible machine learning workflow, covering data preprocessing, model training, evaluation, and model persistence, forming the foundation for further debugging, optimization, and deployment tasks.
