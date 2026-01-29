import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# Load dataset
df = pd.read_csv("data/banknote_authentication.csv", header=None)
df.columns = ["variance", "skewness", "kurtosis", "entropy", "class"]

X = df.drop("class", axis=1)
y = df["class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Default prediction (threshold = 0.5)
y_pred_default = model.predict(X_test_scaled)
default_f1 = f1_score(y_test, y_pred_default)

# Threshold tuning
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred_tuned = (y_prob > 0.4).astype(int)
tuned_f1 = f1_score(y_test, y_pred_tuned)

print("F1-score (Default Threshold 0.5):", default_f1)
print("F1-score (Tuned Threshold 0.4):", tuned_f1)

print("\nClassification Report after Threshold Tuning:\n")
print(classification_report(y_test, y_pred_tuned))

# Save improved model
joblib.dump(model, "models/banknote_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
