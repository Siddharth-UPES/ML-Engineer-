import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("data/banknote_authentication.csv", header=None)
df.columns = ["variance", "skewness", "kurtosis", "entropy", "class"]

X = df.drop("class", axis=1)
y = df["class"]

# 2. FIX RANDOMNESS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. PREVENT DATA LEAKAGE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 6. Save stable artifacts
joblib.dump(model, "models/banknote_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
