import joblib

from src.preprocess import preprocess_data

model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")


def predict_customer(df):

    df = preprocess_data(df)

    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    df = df.reindex(
        columns=features,
        fill_value=0
    )

    prediction = model.predict(df)

    probability = model.predict_proba(df)[:, 1]

    return prediction, probability