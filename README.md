# **Credit Card Fraud Detection**

This project applies **Machine Learning** to detect fraudulent credit card transactions.
It includes **EDA, preprocessing, imbalance handling, model building, evaluation**, and final insights.

---

## ** Project Overview**

Credit card fraud costs financial institutions millions every year.
The goal of this project is to:

* **Analyze** anonymized credit card transaction data
* **Handle data imbalance** (fraud cases are <1%)
* **Train multiple ML models** to classify fraud vs. non-fraud
* **Evaluate models** using business-relevant metrics
* **Select the best model** based on ROC-AUC, precision, recall & F1-score

The notebook follows a full end-to-end Data Science workflow.

---

## ** Project Structure**

```
│── Credit_Card_Fraud_Detection_CapStone_Project_(1).ipynb
│── README.md
└── data/
    └── creditcard.csv (Not included - Kaggle dataset)
```

---

## ** Dataset**

* **Source**: Kaggle Credit Card Fraud Detection Dataset
* **Rows**: 284,807
* **Fraud cases**: 492 (~0.17%)
* **Features**:

  * PCA-transformed features: V1–V28
  * `Amount` – transaction amount
  * `Time` – seconds elapsed
  * `Class` – **1 = fraud**, **0 = genuine**

---

## ** Steps Performed**

### **1️. Importing Libraries**

NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, etc.

---

### **2️. Exploratory Data Analysis (EDA)**

* Verified **missing values** → none
* Checked **class imbalance** → extremely imbalanced
* Visualized:

  * Fraud vs Non-fraud distribution
  * Time-based fraud patterns
  * Amount distribution
  * Correlation heatmaps

---

### **3️. Data Preprocessing**

Performed:

* Standard scaling for numerical columns
* **PowerTransformer** to fix skewness
* Splitting: **80% train, 20% test**

---

### **4️. Imbalanced Data Handling**

Tested multiple techniques:

* **Random Under-Sampling**
* **Random Over-Sampling**
* **SMOTE**
* Class-weight adjustments (for models like Logistic Regression)

---

### **5️. Model Building**

Models trained:

| Model                   | Key Points                               |
| ----------------------- | ---------------------------------------- |
| **Logistic Regression** | With L1/L2 penalty, class weights        |
| **Random Forest**       | Tuned with hyperparameters               |
| **XGBoost**             | Handling imbalance with scale_pos_weight |
| (Optional) SVM, KNN     | Based on dataset behavior                |

Each model had a **dedicated function** for training & evaluation.

---

### **6️. Evaluation Metrics**

Evaluated on both Train & Test sets:

* **Accuracy**
* **Precision (fraud-focused)**
* **Recall (fraud detection rate)**
* **F1-Score**
* **ROC–AUC**
* **Confusion Matrix**
* **Custom Probability Threshold Optimization**

---

## **7. Final Results (Example)**

*(Replace with your actual results if needed)*

| Metric        | Value  |
| ------------- | ------ |
| **Accuracy**  | 0.9595 |
| **Precision** | 0.9554 |
| **Recall**    | 0.9641 |
| **F1-Score**  | 0.9597 |
| **ROC-AUC**   | 0.9949 |

**Confusion Matrix:**

```
[[1146   54]
 [  43 1157]]
```

**Interpretation:**

* Model catches most frauds (high recall)
* Low false positives
* Excellent separation between classes (AUC ≈ 0.995)

---

## ** Key Insights**

* Fraudulent transactions show **distinct patterns** in certain PCA-components.
* Fraud amounts vary widely, unlike non-fraud amounts which cluster more.
* Imbalance handling drastically improves **recall**.
* Random Forest & Logistic Regression perform exceptionally well.

---

## ** Installation & Requirements**

### **1. Clone Repository**

```bash
git clone https://github.com/your-username/fraud-detection
cd fraud-detection
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run Notebook**

Open Jupyter Notebook:

```bash
jupyter notebook
```

---

## ** Future Improvements**

* Deploy model via FastAPI / Flask
* Real-time streaming detection using Kafka
* Model monitoring & drift detection
* AutoML-based hyperparameter tuning
* Feature engineering on Time and Amount

---

## ** License**

This project is licensed under the MIT License.

---

## ** Author**

**Ritesh Ranjan**
Data Science & Machine Learning Practitioner

---
