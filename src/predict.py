import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model = pickle.load(open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "rb"))
transformer = pickle.load(open(os.path.join(ARTIFACTS_DIR, "transformer.pkl"), "rb"))
columns = pickle.load(open(os.path.join(ARTIFACTS_DIR, "columns.pkl"), "rb"))

THRESHOLD = 0.3


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


def predict(data):

    processed = preprocess_input(data)

    prob = model.predict_proba(processed)[0][1]
    prediction = 1 if prob > THRESHOLD else 0

    return {
        "prediction": int(prediction),
        "fraud_probability": float(prob)
    }


if __name__ == "__main__":

    sample = {
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536346,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }

    print(predict(sample))