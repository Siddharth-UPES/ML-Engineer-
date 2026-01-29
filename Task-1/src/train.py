from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from data_preprocessing import load_and_preprocess

X, y = load_and_preprocess("data/banknote_authentication.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "models/banknote_model.pkl")
