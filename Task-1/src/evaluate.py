from sklearn.metrics import classification_report
import joblib
from data_preprocessing import load_and_preprocess
from sklearn.model_selection import train_test_split

X, y = load_and_preprocess("data/banknote_authentication.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/banknote_model.pkl")
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
