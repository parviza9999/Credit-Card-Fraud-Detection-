"""
Shared model evaluation utilities for the Credit Card Fraud Detection Capstone.

This file standardizes model evaluation across baseline, XGBoost, and
Autoencoder models.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model_performance(y_true, y_pred, y_pred_proba):
    """
    Compute standard classification metrics for highly imbalanced fraud detection.

    Args:
        y_true: True labels.
        y_pred: Predicted class labels.
        y_pred_proba: Fraud probability or anomaly score where higher means more likely fraud.

    Returns:
        Dictionary of classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
    }

    return metrics


def save_metrics_to_report(
    metrics_dict,
    file_path="reports/metrics_summary.csv",
    model_name="Baseline",
):
    """
    Append or update model performance in a shared metrics summary file.

    Args:
        metrics_dict: Dictionary of model metrics.
        file_path: Output CSV path.
        model_name: Name of the model being reported.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([metrics_dict])
    df_new.insert(0, "model", model_name)

    try:
        df_existing = pd.read_csv(file_path)
        df_existing = df_existing[df_existing["model"] != model_name]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_final = df_new

    df_final.to_csv(file_path, index=False)
    print(f"Metrics saved/updated successfully in {file_path}")