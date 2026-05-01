from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle
import pandas as pd

# =========================
# Load artifacts (once)
# =========================
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model = pickle.load(open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "rb"))
transformer = pickle.load(open(os.path.join(ARTIFACTS_DIR, "transformer.pkl"), "rb"))
columns = pickle.load(open(os.path.join(ARTIFACTS_DIR, "columns.pkl"), "rb"))

THRESHOLD = 0.3

# =========================
# App init
# =========================
app = FastAPI(title="Fraud Detection API")


# =========================
# Input Schema
# =========================
class Transaction(BaseModel):
    data: dict


# =========================
# Preprocess
# =========================
def preprocess_input(data):

    df = pd.DataFrame([data])

    # Ensure correct column order
    df = df.reindex(columns=columns, fill_value=0)

    # Scale Amount
    if 'Amount' in df.columns:
        df['Amount'] = scaler.transform(df[['Amount']])

    # Power transform
    df = transformer.transform(df)

    return df


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict_fraud")
def predict_fraud(transaction: Transaction):

    processed = preprocess_input(transaction.data)

    prob = model.predict_proba(processed)[0][1]
    prediction = 1 if prob > THRESHOLD else 0

    return {
        "prediction": int(prediction),
        "fraud_probability": float(prob)
    }