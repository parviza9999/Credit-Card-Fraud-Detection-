# Credit Card Fraud Detection Capstone — Zoom Presentation Outline
## Project Team

- Parviz Ali
- Maryam Shabani
## Slide 1 — Title

**Credit Card Fraud Detection Using Machine Learning**

Presenter: Parviz Ali and Maryam Shabani 
Capstone Project

### Speaker Notes

Today we will present our capstone project on credit card fraud detection using machine learning. The goal was to build a reproducible pipeline, compare multiple models, and recommend the best model for detecting fraudulent transactions.

---

## Slide 2 — Business Problem

Credit card fraud creates financial loss, customer trust issues, and operational review burden.

The challenge is that fraud is rare, so the dataset is highly imbalanced.

### Speaker Notes

The business problem is that fraudulent transactions are very costly, but they are also very rare compared with normal transactions. This makes the project challenging because a model can appear accurate by predicting almost everything as non-fraud. For this reason, I focused on recall, precision, F1-score, ROC-AUC, and PR-AUC instead of accuracy alone.

---

## Slide 3 — Dataset Overview

Dataset: Kaggle/ULB Credit Card Fraud Detection

- Total transactions: 284,807
- Fraud transactions: 492
- Fraud rate: approximately 0.17%
- Features: Time, Amount, V1 through V28
- Target: Class

### Speaker Notes

The dataset contains 284,807 credit card transactions, but only 492 are fraudulent. That is about 0.17 percent of the data. The features V1 through V28 are anonymized PCA-transformed variables, so detailed business interpretation of each feature is limited.

---

## Slide 4 — Class Imbalance

Fraud transactions are a very small percentage of the dataset.

Why this matters:

- Accuracy alone is misleading
- False negatives are important because missed fraud is costly
- False positives are also important because they create review workload

### Speaker Notes

The most important issue in this project is class imbalance. If a model predicts every transaction as non-fraud, it would still get very high accuracy, but it would be useless. Therefore, I evaluated models using recall, precision, F1-score, ROC-AUC, PR-AUC, and confusion matrix.

---

## Slide 5 — Preprocessing Pipeline

Pipeline file:

`src/preprocess.py`

Steps:

1. Load raw dataset
2. Validate columns
3. Check missing values
4. Create stratified train/test split
5. Use random_state=42
6. Fit scaler only on training data
7. Save official processed files

### Speaker Notes

I created a preprocessing pipeline to make the project reproducible. The most important point is that all models use the same official train/test split. The scaler is fitted only on the training data and then applied to the test data to avoid data leakage.

---

## Slide 6 — Models Built

Three models were compared:

1. Logistic Regression Baseline
2. XGBoost Supervised Classifier
3. Autoencoder Anomaly Detection Model

### Speaker Notes

I compared a simple supervised baseline, a stronger supervised model, and an anomaly detection model. Logistic Regression provides a simple benchmark. XGBoost is strong for tabular data. The Autoencoder was trained only on normal transactions and detects fraud using reconstruction error.

---

## Slide 7 — Model Evaluation Metrics

Metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

Shared evaluation file:

`src/evaluation.py`

### Speaker Notes

To keep the evaluation consistent, I used a shared evaluation script. This allowed all model results to be saved into the same metrics summary file. For this fraud detection problem, recall and PR-AUC are especially important because fraud is the rare positive class.

---

## Slide 8 — Model Results

XGBoost results:

- Accuracy: 0.9985
- Precision: 0.5478
- Recall: 0.8776
- F1-score: 0.6745
- ROC-AUC: 0.9809
- PR-AUC: 0.8547

Autoencoder results:

- Accuracy: 0.9942
- Precision: 0.1781
- Recall: 0.6633
- F1-score: 0.2808
- ROC-AUC: 0.9367
- PR-AUC: 0.3800

### Speaker Notes

XGBoost was the strongest model. It detected 86 out of 98 fraud cases in the test set and missed only 12 fraud cases. The Autoencoder detected 65 fraud cases but had more false positives and lower precision. The baseline model provides a simple comparison point.

---

## Slide 9 — Final Model Recommendation

Recommended model:

`XGBoost supervised classification model`

Reason:

- Highest recall
- Best F1-score
- Strongest PR-AUC
- Strong ROC-AUC
- Better performance than Autoencoder benchmark

### Speaker Notes

I recommend XGBoost as the final model because it had the strongest overall performance. In fraud detection, we want to catch as much fraud as possible while controlling false positives. XGBoost gave the best balance among the models tested.

---

## Slide 10 — Inference and Deployment Plan

Created one-sample inference script:

`src/predict.py`

Deployment plan:

`deployment/cloudrun_deploy.md`

Proposed future deployment:

1. Build FastAPI app
2. Load XGBoost model
3. Accept transaction JSON
4. Return fraud probability
5. Deploy to Google Cloud Run

### Speaker Notes

I also created a one-sample prediction script to show how the trained model can be used for inference. The future deployment plan is to wrap the model in a FastAPI application and deploy it to Google Cloud Run.

---

## Slide 11 — Limitations

Limitations:

- Dataset is anonymized
- Feature interpretation is limited
- Fraud patterns change over time
- Model should be validated on newer data
- Threshold should be optimized using business cost

### Speaker Notes

There are some limitations. Since the features are anonymized, business interpretation is limited. Also, fraud patterns change over time, so this model would need monitoring and retraining before production use.

---

## Slide 12 — Future Work and Closing

Future work:

- Add SHAP explainability
- Optimize threshold by business cost
- Build FastAPI endpoint
- Deploy to Google Cloud Run
- Monitor data drift
- Add dashboard reporting

### Speaker Notes

Future work would include explainability, threshold optimization, deployment, and monitoring. Overall, this project demonstrates a complete machine learning lifecycle from data preparation through model comparison and deployment planning.