import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model_performance(y_true, y_pred, y_pred_proba):
    """
    Computes standard classification metrics for highly imbalanced fraud detection datasets.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba)
    }
    return metrics

def save_metrics_to_report(metrics_dict, file_path="../reports/metrics_summary.csv", model_name="Baseline"):
    """
    Appends or creates a dataframe tracking model performance across experiments.
    """
    df_new = pd.DataFrame([metrics_dict])
    df_new.insert(0, 'model', model_name)
    
    try:
        df_existing = pd.read_csv(file_path)
        # Filter out previous runs of the same model name to avoid clutter
        df_existing = df_existing[df_existing['model'] != model_name]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_final = df_new
        
    df_final.to_csv(file_path, index=False)
    print(f"Metrics saved/updated successfully in {file_path}")