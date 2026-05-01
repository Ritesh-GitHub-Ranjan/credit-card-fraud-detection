import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier


# =========================
# 1. Load Data
# =========================
def load_data(path):
    df = pd.read_csv(path)

    df = df.drop(columns=['Time'], errors='ignore')
    df = df.dropna()

    return df


# =========================
# 2. Split + Preprocess
# =========================
def prepare_data(df):

    y = df['Class']
    X = df.drop('Class', axis=1)

    # Split FIRST (important)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale Amount
    scaler = StandardScaler()
    if 'Amount' in X_train.columns:
        X_train['Amount'] = scaler.fit_transform(X_train[['Amount']])
        X_test['Amount'] = scaler.transform(X_test[['Amount']])

    # Power Transform
    pt = PowerTransformer()
    X_train = pt.fit_transform(X_train)
    X_test = pt.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, pt


# =========================
# 3. Train Model
# =========================
def train_model(X_train, y_train):

    ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    model = XGBClassifier(
        learning_rate=0.2,
        max_depth=4,
        min_child_weight=4,
        n_estimators=130,
        gamma=0.1,
        subsample=1,
        colsample_bytree=1,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_res, y_train_res)

    return model


# =========================
# 4. Save Artifacts
# =========================
ARTIFACTS_DIR = "project/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def save_artifacts(model, scaler, transformer, columns):

    pickle.dump(model, open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb"))
    pickle.dump(transformer, open(os.path.join(ARTIFACTS_DIR, "transformer.pkl"), "wb"))
    pickle.dump(columns, open(os.path.join(ARTIFACTS_DIR, "columns.pkl"), "wb"))

    print("Model saved successfully!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data("project/data/creditcard.csv")

    X_train, X_test, y_train, y_test, scaler, pt = prepare_data(df)

    model = train_model(X_train, y_train)

    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC: {roc:.4f}")

    save_artifacts(model, scaler, pt, df.drop('Class', axis=1).columns.tolist())