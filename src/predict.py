"""
One-sample inference script for the final XGBoost fraud detection model.

This script loads:
- XGBoost model from model_artifacts/xgboost_fraud_model.joblib
- One transaction row from data/processed/X_test.csv by row index

It returns:
- Fraud probability
- Predicted class: 0 = Non-Fraud, 1 = Fraud

Run example:

python src/predict.py --row-index 0 --threshold 0.50
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: Path):
    """
    Load trained model artifact.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    return model


def load_sample(input_path: Path, row_index: int) -> pd.DataFrame:
    """
    Load one transaction sample from the processed test set.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index must be between 0 and {len(df) - 1}")

    sample = df.iloc[[row_index]]

    return sample


def predict_transaction(model, sample: pd.DataFrame, threshold: float) -> dict:
    """
    Predict fraud probability and class for one transaction.
    """
    fraud_probability = float(model.predict_proba(sample)[:, 1][0])
    predicted_class = int(fraud_probability >= threshold)

    result = {
        "threshold": threshold,
        "fraud_probability": fraud_probability,
        "predicted_class": predicted_class,
        "prediction_label": "Fraud" if predicted_class == 1 else "Non-Fraud",
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-sample fraud prediction.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="model_artifacts/xgboost_fraud_model.joblib",
        help="Path to trained XGBoost model artifact.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/X_test.csv",
        help="Path to processed test features.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index from X_test.csv to score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Fraud classification threshold.",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input)

    model = load_model(model_path)
    sample = load_sample(input_path, args.row_index)
    result = predict_transaction(model, sample, args.threshold)

    print("\nOne-Sample Fraud Prediction")
    print("---------------------------")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()