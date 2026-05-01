# Credit Card Fraud Detection System

## Overview

This repository contains an end-to-end machine learning system for detecting fraudulent credit card transactions. The solution is designed with a production-oriented mindset, focusing on model performance, system reliability, and business impact.

The system handles extreme class imbalance, captures non-linear transaction behavior, and exposes predictions through a REST API for real-time usage.

---

## Problem Context

Credit card fraud detection is a high-stakes classification problem characterized by:

* Severe class imbalance (~0.17% fraud cases)
* Asymmetric cost of errors
* Dynamic fraud patterns
* Need for real-time decisioning

A missed fraudulent transaction results in direct financial loss, whereas excessive false positives degrade customer experience. The system must balance these competing objectives.

---

## Business Objective

The model is optimized for practical deployment in financial systems with the following priorities:

* Maximize fraud detection (high recall) to reduce monetary loss
* Maintain controlled false positives (precision) to avoid blocking genuine users
* Enable real-time inference through API integration
* Ensure reproducibility and scalability

This reflects the trade-off between **risk prevention** and **customer experience**, which is central to fraud systems.

---

## Dataset

* Total transactions: 284,807
* Fraudulent transactions: 492 (~0.172%)
* Features:

  * V1–V28: PCA-transformed anonymized variables
  * Amount: Transaction value
  * Class: Target label (1 = fraud, 0 = non-fraud)

The dataset is highly anonymized, requiring the model to learn implicit patterns rather than relying on domain-specific features.

---

## Methodology

### Data Preparation

* Removed non-informative features (e.g., Time)
* Applied StandardScaler on transaction amount
* Used PowerTransformer to stabilize feature distributions
* Ensured transformations were fit only on training data to avoid leakage

### Handling Class Imbalance

* Applied Random Oversampling on training data only
* Avoided synthetic sampling during inference to maintain real-world consistency

### Model Selection

Multiple algorithms were evaluated:

* Logistic Regression (baseline, interpretable)
* Random Forest (ensemble, non-linear)
* XGBoost (boosting-based, high performance)

### Final Model Choice

**XGBoost** was selected due to:

* Strong performance on imbalanced data
* Ability to capture complex non-linear relationships
* Robust generalization

---

## Model Performance

* ROC-AUC: ~0.98+
* High recall ensures most fraudulent transactions are detected
* Precision controlled to reduce unnecessary intervention

### Metric Selection Rationale

Accuracy is not used as a primary metric due to class imbalance. Instead:

* ROC-AUC → overall separability
* Recall → fraud detection capability
* Precision → customer experience impact

---

## Threshold Optimization

The classification threshold is explicitly tuned instead of using the default 0.5.

* Lower threshold → higher recall (detect more fraud)
* Higher threshold → higher precision (reduce false alarms)

A threshold of 0.3 is used to balance fraud detection and false positives, aligning with business priorities.

---

## System Design

The project follows a modular and production-aware architecture:

* Training pipeline separated from inference pipeline
* Preprocessing artifacts persisted and reused
* Consistent feature ordering enforced during prediction
* API layer abstracts model complexity

---

## Architecture Overview

```id="arch001"
Client Request → FastAPI Endpoint → Preprocessing (Scaler + Transformer)
              → Model Inference (XGBoost)
              → Threshold Decision → Response (Prediction + Probability)
```

---

## Project Structure

```id="struct001"
credit-card-fraud-detection/

├── app.py                     # FastAPI application (serving layer)
├── requirements.txt          # Dependencies
├── README.md
├── .gitignore

├── artifacts/                # Serialized objects
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── transformer.pkl
│   └── columns.pkl

├── src/
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Inference logic

└── notebooks/
    └── fraud_detection_analysis.ipynb   # EDA and experimentation
```

---

## API Deployment

### Run Locally

```id="run001"
git clone https://github.com/Ritesh-GitHub-Ranjan/credit-card-fraud-detection.git
cd credit-card-fraud-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload
```

### Endpoint

POST `/predict_fraud`

### Request Example

```id="req001"
{
  "data": {
    "V1": -1.359807,
    "V2": -0.072781,
    "...": "...",
    "V28": -0.021053,
    "Amount": 149.62
  }
}
```

### Response Example

```id="res001"
{
  "prediction": 0,
  "fraud_probability": 0.02
}
```

---

## Key Design Decisions

### Training–Serving Consistency

All preprocessing steps (scaling, transformation) are saved and reused during inference to prevent training-serving skew.

### Artifact Management

Model and preprocessing objects are stored in a dedicated artifacts directory, ensuring portability across environments.

### Feature Alignment

Input data is reindexed to match training feature order, preventing silent errors during prediction.

### Modular Code Structure

Separation of concerns between training, inference, and API layers improves maintainability.

---

## Trade-offs and Considerations

* Oversampling improves recall but may introduce noise
* Lower threshold increases detection but may raise false positives
* PCA features improve privacy but reduce interpretability

---

## Limitations

* No real-time streaming pipeline
* No concept drift handling
* Limited explainability due to PCA transformation
* No monitoring or alerting system

---

## Future Work

* Real-time inference using streaming frameworks
* Model monitoring and drift detection
* Cost-sensitive learning using financial impact
* Explainability using SHAP or LIME
* Containerization and cloud deployment

---

## How to Reproduce

```id="rep001"
python src/train.py
python src/predict.py
```

---

## Author

Ritesh Ranjan
Data Science and Machine Learning Practitioner

---
