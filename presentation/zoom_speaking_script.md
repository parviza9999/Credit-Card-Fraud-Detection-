# Zoom Speaking Script — Credit Card Fraud Detection Capstone

## Opening

Good morning/afternoon everyone. My name is Parviz Ali. Maryam Shabani and I worked on this capstone project.

Today I will present our credit card fraud detection project. The objective was to build a reproducible machine learning pipeline, compare multiple fraud detection models, and recommend the best model for identifying fraudulent transactions.

---

## Slide 1 — Title

This project is titled **Credit Card Fraud Detection Using Machine Learning**.

The focus of the project was not just to train a model, but to build a complete machine learning workflow. That includes data preparation, preprocessing, model training, standardized evaluation, model comparison, inference, and deployment planning.

---

## Slide 2 — Project Objective

The main objective was to detect fraudulent credit card transactions using machine learning.

The project goals were to analyze a highly imbalanced fraud dataset, create a reproducible preprocessing pipeline, train multiple models, compare their performance fairly, and select a final recommended model.

A key requirement was that all models had to use the same processed train/test split so that the comparison would be fair and consistent.

---

## Slide 3 — Business Problem

Credit card fraud is a significant business problem because it can create financial losses, customer trust issues, and operational review burden.

The main challenge is that fraud is rare. This means a model can have very high accuracy and still fail at the actual business goal if it misses fraudulent transactions.

Because of that, I did not rely on accuracy alone. I focused more on recall, precision, F1-score, ROC-AUC, PR-AUC, and the confusion matrix.

---

## Slide 4 — Dataset Overview

The dataset used was the Kaggle/ULB Credit Card Fraud Detection dataset.

It contains 284,807 total transactions and only 492 fraud transactions. That means fraud represents about 0.17 percent of the dataset.

The dataset has the columns Time, Amount, V1 through V28, and Class. The Class column is the target variable, where 0 means non-fraud and 1 means fraud.

The V1 through V28 features are anonymized PCA-transformed features, so feature-level business interpretation is limited.

---

## Slide 5 — Class Imbalance Challenge

The biggest technical challenge in this project was class imbalance.

Since fraud is only about 0.17 percent of the dataset, a model could predict almost everything as non-fraud and still show very high accuracy. That would not be useful in a real fraud detection setting.

For this reason, recall and PR-AUC were very important. Recall tells us how much fraud we caught. Precision tells us how many flagged transactions were actually fraud. PR-AUC is especially useful when the positive class is rare.

---

## Slide 6 — Preprocessing Pipeline

A reproducible preprocessing pipeline was created in `src/preprocess.py`.

The pipeline loads the raw dataset, validates the required columns, checks missing values, reports duplicate rows, and creates a stratified train/test split.

It also uses `random_state=42` for reproducibility.

For scaling, the StandardScaler was fitted only on the training data and then applied to the test data. This prevents data leakage.

The most important control is that all models used the same official processed split.

---

## Slide 7 — Models Built

Three models were built and compared.

The first was a Logistic Regression baseline. This gave us a simple supervised benchmark.

The second was XGBoost, which is a stronger supervised model and is well suited for structured tabular data.

The third was an Autoencoder anomaly detection model. The Autoencoder was trained only on normal transactions and used reconstruction error to identify possible fraud.

All models used the shared evaluation file `src/evaluation.py`, and results were saved into `reports/metrics_summary.csv`.

---

## Slide 8 — Model Results

The three models showed different tradeoffs.

Logistic Regression had the highest recall at about 91.84 percent, but its precision was only about 6.10 percent. This means it caught many fraud cases but also created many false positives.

XGBoost had recall of about 87.76 percent and precision of about 54.78 percent. It also had the best F1-score and PR-AUC.

The Autoencoder had recall of about 66.33 percent and precision of about 17.81 percent. It was useful as an anomaly detection benchmark, but it did not outperform the supervised models.

Overall, XGBoost had the best balance.

---

## Slide 9 — Final Model Recommendation

The final recommended model is the XGBoost supervised classification model.

Even though Logistic Regression had slightly higher recall, the precision was too low, meaning it would generate too many false fraud alerts.

XGBoost provided a much better balance between catching fraud and controlling false positives. It also had the strongest F1-score and PR-AUC.

For this project, XGBoost is the best final model because it provides the strongest overall fraud detection performance.

---

## Slide 10 — Inference and Deployment Plan

A one-sample inference script was created in `src/predict.py`.

This script loads the saved XGBoost model from `model_artifacts/xgboost_fraud_model.joblib`, scores one transaction, and returns the fraud probability and prediction label.

A deployment plan was also created in `deployment/cloudrun_deploy.md`.

The future deployment approach is to build a FastAPI application, load the saved XGBoost model, accept transaction data as JSON, return a fraud probability, and deploy the API to Google Cloud Run.

---

## Slide 11 — Limitations

There are several limitations.

First, the dataset is anonymized, so detailed business interpretation of individual features is limited.

Second, fraud behavior changes over time, so the model should be validated with newer data before production use.

Third, the threshold should be optimized based on business cost. For example, the cost of missing fraud may be much higher than the cost of reviewing a false positive.

Finally, this project includes a deployment plan, but not a live production API.

---

## Slide 12 — Future Work and Closing

Future work would include adding SHAP explainability for XGBoost, optimizing the classification threshold using business cost, building a FastAPI endpoint, deploying to Google Cloud Run, monitoring data drift, and creating dashboard reporting.

In conclusion, this project built a complete fraud detection workflow from data preparation through model comparison and deployment planning.

The final recommended model is XGBoost because it provides the best overall balance of precision, recall, F1-score, ROC-AUC, and PR-AUC.

Thank you. I’m happy to answer any questions.