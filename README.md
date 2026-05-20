# Credit Card Fraud Detection Capstone

## Project Overview

This capstone project applies machine learning to credit card fraud detection using the Kaggle/ULB Credit Card Fraud dataset. The dataset contains anonymized European cardholder transactions and is highly imbalanced: only about 0.17% of transactions are fraudulent. The primary objective is to build a robust fraud detection pipeline with repeatable engineering, strong validation, and a clear comparison between simple and advanced models.

## Goals

- Demonstrate data engineering and automated preprocessing for fraud detection.
- Compare baseline and advanced models to show real performance gains.
- Maintain reproducibility with fixed random seeds and environment dependencies.
- Provide clear analysis of class imbalance, bias mitigation, and model failure modes.
- Deliver an end-to-end workflow that supports a one-sample inference quick start.

## Dataset

- Source: Kaggle Credit Card Fraud Detection dataset
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Data characteristics:
  - Transactions from European cardholders
  - 284,807 total transactions
  - 492 fraudulent cases (0.17% of transactions)
  - 30 features: `Time`, `V1`–`V28`, `Amount`, and `Class`

## Project Structure

- `data/` - raw and external dataset storage
- `notebooks/` - exploratory data analysis, modeling experiments, and reporting notebooks
- `reports/` - metrics, diagrams, and project documentation outputs
- `src/` - modular Python code for loading data, preprocessing, training, evaluation, and inference
- `requirements.txt` - Python dependencies for reproducible setup

## Project Phases

Phase 1: Project Initiation & Setup – Finalize project proposal, dataset, GitHub setup, and project roadmap.
Phase 2: Data Engineering & EDA – Implement data loading, EDA, and the preprocessing pipeline.
Phase 3: Baseline Regression & Anomaly Detection – Implement a multi-parameter linear regression baseline
* using linear regression as a baseline usually means one of two setups:
    * Target Reconstruction: Using the regression model to predict a continuous feature based on the others, using the prediction error (residual) as an anomaly score.
    * Proxy Classification: Running a standard linear probability model where the targets are 0 and 1 to establish a quick linear decision boundary before jumping into the complex reconstructions of the Autoencoder.
Phase 4: Supervised Modeling & Evaluation – Implement XGBoost and evaluate performance against the baseline.
Phase 5: Validation & Project Closure – Unsupervised Modeling - Autoencoder-based anomaly detection.
Phase 6: Validate the final model, compile the report, and present.

## Quick Start

### 1. Install environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare data
```powershell
python src/data_loader.py --download --output data/raw
```
### 3. Train models
TODO