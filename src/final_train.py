import joblib
from sklearn.linear_model import LogisticRegression
from src.data_loader import get_train_test_data

X_train, X_test, y_train, y_test = (
    get_train_test_data()
)

joblib.dump(
    X_train.columns.tolist(),
    "models/features.pkl"
)

model = LogisticRegression(
    C=0.1,
    class_weight="balanced",
    max_iter=1000,
    penalty="l2",
    solver="liblinear"
)

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/churn_model.pkl"
)

print("Model saved successfully!")