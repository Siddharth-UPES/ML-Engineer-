import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_and_preprocess(path):
    df = pd.read_csv(path)

    X = df.drop("class", axis=1)
    y = df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")
    return X_scaled, y
