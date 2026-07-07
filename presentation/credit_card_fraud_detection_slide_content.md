# Credit Card Fraud Detection Capstone — Slide Content

## Slide 1 — Title

**Credit Card Fraud Detection Using Machine Learning**

Parviz Ali and Maryam Shabani 
Capstone Project

---

## Slide 2 — Project Objective

Build a reproducible machine learning pipeline to detect fraudulent credit card transactions.

Project goals:

- Analyze a highly imbalanced fraud dataset
- Build preprocessing and model training pipelines
- Compare supervised and anomaly detection models
- Select the best final model
- Prepare an inference and deployment plan

---

## Slide 3 — Business Problem

Credit card fraud creates:

- Financial loss
- Customer trust concerns
- Operational review workload
- Risk management challenges

Main challenge:

Fraud is rare, so accuracy alone can be misleading.

---

## Slide 4 — Dataset Overview

Dataset: Kaggle/ULB Credit Card Fraud Detection

- Total transactions: 284,807
- Fraud transactions: 492
- Fraud rate: about 0.17%
- Features: Time, Amount, V1 through V28
- Target: Class

Class labels:

- 0 = Non-Fraud
- 1 = Fraud

---

## Slide 5 — Class Imbalance Challenge

The dataset is extremely imbalanced.

Why this matters:

- A model can have high accuracy while missing fraud
- Recall is important because missed fraud is costly
- Precision is important because false alerts create review burden
- PR-AUC is important for rare positive-class problems

---

## Slide 6 — Preprocessing Pipeline

File:

`src/preprocess.py`

Pipeline steps:

1. Load raw dataset
2. Validate columns
3. Check missing values
4. Create stratified train/test split
5. Use random_state=42
6. Fit StandardScaler only on training data
7. Save official processed split

Important control:

All models used the same official train/test split.

---

## Slide 7 — Models Built

Three models were compared:

1. Logistic Regression Baseline
2. XGBoost Supervised Classifier
3. Autoencoder Anomaly Detection Model

Shared evaluation:

`src/evaluation.py`

Shared metrics output:

`reports/metrics_summary.csv`

---

## Slide 8 — Model Results

| Model | Precision | Recall | F1-score | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.7190 |
| XGBoost | 0.5478 | 0.8776 | 0.6745 | 0.8547 |
| Autoencoder | 0.1781 | 0.6633 | 0.2808 | 0.3800 |

Key finding:

XGBoost had the best overall balance.

---

## Slide 9 — Final Model Recommendation

Recommended model:

**XGBoost supervised classification model**

Why XGBoost:

- Best F1-score
- Best PR-AUC
- Strong ROC-AUC
- High fraud recall
- Better precision than Logistic Regression
- Fewer false positives than Autoencoder

Final model artifact:

`model_artifacts/xgboost_fraud_model.joblib`

---

## Slide 10 — Inference and Deployment Plan

One-sample inference script:

`src/predict.py`

Deployment plan:

`deployment/cloudrun_deploy.md`

Future deployment approach:

1. Build FastAPI app
2. Load saved XGBoost model
3. Accept transaction JSON
4. Return fraud probability
5. Deploy to Google Cloud Run

---

## Slide 11 — Limitations

Project limitations:

- Dataset features are anonymized
- Business interpretation of V1-V28 is limited
- Fraud behavior changes over time
- Model should be validated on newer data
- Threshold should be optimized by business cost
- Live API deployment is planned but not completed

---

## Slide 12 — Future Work and Closing

Future work:

- Add SHAP explainability
- Optimize threshold using business cost
- Build FastAPI endpoint
- Deploy to Google Cloud Run
- Monitor model drift
- Add dashboard reporting

Conclusion:

XGBoost is the recommended final model for this capstone project.