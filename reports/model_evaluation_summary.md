# Model Evaluation Summary

## Objective

The objective of this project was to detect fraudulent credit card transactions using a highly imbalanced dataset.

## Models Compared

The following models were evaluated using the same official train/test split:

1. Logistic Regression Baseline
2. XGBoost Supervised Classifier
3. Autoencoder Anomaly Detection Model

## Evaluation Metrics

Because the dataset is highly imbalanced, accuracy alone is not sufficient. The main metrics used were:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix

## Model Comparison

Refer to `reports/metrics_summary.csv` for the standardized comparison table.

## Key Findings

XGBoost produced the strongest overall performance. It had the best balance of fraud recall, precision, F1-score, ROC-AUC, and PR-AUC.

The Autoencoder was useful as an anomaly-detection benchmark, but it produced more false positives and lower precision.

The Logistic Regression baseline provided a simple, interpretable benchmark for comparison.

## Final Recommendation

XGBoost is recommended as the final model because it provides the strongest fraud detection performance using the official processed train/test split.

## Limitations

- The dataset is anonymized, so business interpretation of features is limited.
- Fraud cases are extremely rare.
- Model performance should be validated on newer transaction data before production use.
- False positives may create operational review burden.

## Future Work

- Perform threshold optimization based on business cost of false positives and false negatives.
- Add SHAP explainability for XGBoost.
- Build FastAPI inference endpoint.
- Deploy the model using Google Cloud Run.
- Monitor model drift and fraud pattern changes over time.