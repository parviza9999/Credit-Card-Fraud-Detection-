# Deployment Plan: Credit Card Fraud Detection Model

## Objective

Deploy the final XGBoost fraud detection model as an API so a user or system can submit one transaction and receive:

- Fraud probability
- Predicted class
- Prediction label

## Final Model

The recommended model is:

```text
XGBoost supervised classification model
model_artifacts/xgboost_fraud_model.joblib