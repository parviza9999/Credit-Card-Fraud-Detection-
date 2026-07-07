# Credit Card Fraud Detection Capstone Final Report

## 1. Executive Summary

This capstone project developed and compared machine learning models for credit card fraud detection using the Kaggle/ULB credit card fraud dataset. The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.17% of all records.

The project focused on building a reproducible fraud detection pipeline, comparing supervised and anomaly detection approaches, and selecting a final model for future deployment.

Three models were evaluated using the same official processed train/test split:

1. Logistic Regression Baseline
2. XGBoost Supervised Classification Model
3. Autoencoder Anomaly Detection Model

Based on the evaluation results, the XGBoost model is recommended as the final model because it delivered the strongest overall fraud detection performance, including the highest fraud recall, F1-score, ROC-AUC, and PR-AUC.

---

## 2. Business Problem

Credit card fraud detection is a high-value business problem because fraudulent transactions can cause financial loss, customer trust issues, and operational review costs.

The main challenge is that fraud cases are rare compared with normal transactions. Because of this imbalance, a model can achieve very high accuracy by predicting nearly all transactions as non-fraud. Therefore, accuracy alone is not a sufficient success metric.

The project emphasized metrics that are more meaningful for fraud detection:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrix

---

## 3. Dataset Overview

The project used the Kaggle/ULB Credit Card Fraud Detection dataset.

Dataset characteristics:

* Total transactions: 284,807
* Fraud transactions: 492
* Fraud rate: approximately 0.17%
* Features: 30 input features
* Target column: `Class`

  * `0` = Non-Fraud
  * `1` = Fraud

The dataset includes anonymized PCA-transformed features `V1` through `V28`, along with `Time`, `Amount`, and `Class`.

Because the features are anonymized, detailed business interpretation of individual variables is limited.

---

## 4. Exploratory Data Analysis

Exploratory data analysis confirmed that the dataset is extremely imbalanced. Fraudulent transactions make up a very small percentage of total transactions.

Key EDA findings:

* The dataset contains a severe class imbalance.
* Fraud cases are rare but important.
* Transaction `Amount` is right-skewed.
* Accuracy is not a reliable model selection metric by itself.
* Precision, recall, F1-score, ROC-AUC, and PR-AUC are more meaningful for this problem.

The imbalance influenced the modeling strategy. Supervised models used class imbalance handling, while the Autoencoder was trained as an anomaly detection benchmark.

---

## 5. Preprocessing Pipeline

A reproducible preprocessing pipeline was created in:

```text
src/preprocess.py
```

The preprocessing pipeline performs the following steps:

1. Loads the raw dataset from `data/raw/creditcard.csv`
2. Validates required columns
3. Checks missing values
4. Reports duplicate rows
5. Creates one official stratified train/test split
6. Uses `random_state=42` for reproducibility
7. Fits `StandardScaler` only on the training data
8. Applies the fitted scaler to the test data
9. Saves processed outputs under `data/processed`

The official processed files are:

```text
X_train.csv
X_test.csv
y_train.csv
y_test.csv
X_train_scaled.csv
X_test_scaled.csv
scaler.joblib
preprocessing_metadata.json
```

All models used this same official split. No model created its own train/test split. This ensures fair comparison across models.

---

## 6. Models Built

### 6.1 Logistic Regression Baseline

File:

```text
src/train_logistic_baseline.py
```

The Logistic Regression model was used as a simple supervised baseline. It used scaled features and `class_weight="balanced"` to address class imbalance.

Purpose:

* Establish a simple benchmark
* Provide an interpretable baseline
* Compare against more advanced models

---

### 6.2 XGBoost Supervised Classification Model

File:

```text
src/train_xgboost.py
```

The XGBoost model used the unscaled official train/test split. XGBoost was selected because it is a strong tree-based supervised learning algorithm and performs well on structured/tabular data.

The model used `scale_pos_weight` to account for the severe class imbalance.

The calculated class imbalance weight was approximately:

```text
scale_pos_weight = 577.2868
```

---

### 6.3 Autoencoder Anomaly Detection Model

File:

```text
src/train_autoencoder.py
```

The Autoencoder was used as an anomaly detection benchmark. It used the scaled official split and was trained only on non-fraud training transactions.

Fraud detection was based on reconstruction error. Transactions with reconstruction error above the selected threshold were classified as fraud.

The threshold strategy used:

```text
training_normal_reconstruction_error_percentile = 99.5
```

---

## 7. Standardized Evaluation

A shared evaluation utility was used to standardize reporting across models.

File:

```text
src/evaluation.py
```

The shared evaluation script calculates:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

The model comparison summary is saved in:

```text
reports/metrics_summary.csv
```

This provides a consistent comparison across all models.

---

## 8. Model Evaluation Results

The project compared model performance using the same official test set.

### XGBoost Results

At threshold `0.50`, XGBoost produced:

* Accuracy: 0.9985
* Precision: 0.5478
* Recall: 0.8776
* F1-score: 0.6745
* ROC-AUC: 0.9809
* PR-AUC: 0.8547

Confusion matrix:

* True negatives: 56,793
* False positives: 71
* False negatives: 12
* True positives: 86

Interpretation:

The XGBoost model detected 86 out of 98 fraud cases in the test set. It missed 12 fraud cases and incorrectly flagged 71 non-fraud transactions as fraud.

---

### Autoencoder Results

At the 99.5 percentile reconstruction error threshold, the Autoencoder produced:

* Accuracy: 0.9942
* Precision: 0.1781
* Recall: 0.6633
* F1-score: 0.2808
* ROC-AUC: 0.9367
* PR-AUC: 0.3800

Confusion matrix:

* True negatives: 56,564
* False positives: 300
* False negatives: 33
* True positives: 65

Interpretation:

The Autoencoder detected 65 out of 98 fraud cases. It missed 33 fraud cases and produced 300 false positives. It is useful as an anomaly detection benchmark, but it did not outperform XGBoost.

---

### Logistic Regression Baseline

The Logistic Regression baseline was included to provide a simple supervised benchmark. Its results are stored in:

```text
reports/logistic_baseline_results.json
reports/metrics_summary.csv
```

The baseline helps demonstrate whether more advanced models provide meaningful improvement.

---

## 9. Final Model Recommendation

The recommended final model is:

```text
XGBoost supervised classification model
```

Model artifact:

```text
model_artifacts/xgboost_fraud_model.joblib
```

XGBoost is recommended because it provided the strongest overall performance across the most important fraud detection metrics.

Key reasons for selecting XGBoost:

* Highest fraud recall
* Best F1-score
* Strongest PR-AUC
* Strong ROC-AUC
* Lower false positives than Autoencoder
* Better fraud detection than the anomaly detection benchmark

For fraud detection, recall and PR-AUC are especially important because the positive class is rare and costly to miss.

---

## 10. One-Sample Inference

A one-sample prediction script was created in:

```text
src/predict.py
```

The script loads the trained XGBoost model and scores one transaction from the processed test set.

Example command:

```powershell
python src\predict.py --row-index 0 --threshold 0.50
```

The output includes:

* Fraud probability
* Predicted class
* Prediction label
* Threshold used

This demonstrates how the final model can be used for future inference.

---

## 11. Deployment Plan

A deployment plan was created in:

```text
deployment/cloudrun_deploy.md
```

The proposed deployment approach is:

1. Save the trained XGBoost model as a `.joblib` artifact
2. Build a FastAPI application
3. Load the model when the API starts
4. Accept one transaction as JSON input
5. Return fraud probability and prediction label
6. Package the application using Docker
7. Deploy the container to Google Cloud Run
8. Monitor predictions and model performance over time

This approach supports future real-time fraud scoring.

---

## 12. Limitations

The project has several limitations:

* The dataset is anonymized, limiting feature-level business interpretation.
* Fraud behavior changes over time, so model retraining may be required.
* The model was trained on a historical dataset and should be validated on newer data before production use.
* False positives may create operational review burden.
* The threshold should be selected based on business cost of false positives and false negatives.
* The project does not yet include a live deployed API.

---

## 13. Future Work

Recommended future improvements:

1. Perform formal threshold optimization based on business cost.
2. Add SHAP explainability for XGBoost.
3. Build a FastAPI inference endpoint.
4. Deploy the model to Google Cloud Run.
5. Add monitoring for model drift.
6. Add batch scoring for larger transaction sets.
7. Evaluate model performance on newer transaction data.
8. Create a dashboard for fraud prediction monitoring.

---

## 14. Conclusion

This project successfully created a reproducible fraud detection machine learning pipeline, trained multiple models, standardized evaluation reporting, and selected a final recommended model.

The XGBoost model is the best-performing model in this project and is recommended for future deployment planning.

The project demonstrates the full machine learning lifecycle:

* Data loading
* EDA
* Preprocessing
* Model training
* Model evaluation
* Model comparison
* Inference
* Deployment planning
