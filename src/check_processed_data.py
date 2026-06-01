"""
Validation script for processed data files.

This script confirms that the official processed train/test split exists,
loads correctly, and preserves the expected fraud class distribution.

Run from project root:

python src/check_processed_data.py --input data/processed
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


REQUIRED_FILES = [
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",
    "X_train_scaled.csv",
    "X_test_scaled.csv",
    "scaler.joblib",
    "preprocessing_metadata.json",
]


def check_required_files(input_dir: Path) -> None:
    """Confirm all required processed files exist."""
    missing_files = []

    for filename in REQUIRED_FILES:
        file_path = input_dir / filename
        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(f"Missing processed files: {missing_files}")

    print("All required processed files found.")


def load_processed_data(input_dir: Path):
    """Load processed train/test files."""
    X_train = pd.read_csv(input_dir / "X_train.csv")
    X_test = pd.read_csv(input_dir / "X_test.csv")
    y_train = pd.read_csv(input_dir / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(input_dir / "y_test.csv").squeeze("columns")

    X_train_scaled = pd.read_csv(input_dir / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(input_dir / "X_test_scaled.csv")

    scaler = joblib.load(input_dir / "scaler.joblib")

    with open(input_dir / "preprocessing_metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, metadata


def validate_shapes(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled) -> None:
    """Validate that feature and target shapes match."""
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test row counts do not match.")

    if X_train.shape != X_train_scaled.shape:
        raise ValueError("X_train and X_train_scaled shapes do not match.")

    if X_test.shape != X_test_scaled.shape:
        raise ValueError("X_test and X_test_scaled shapes do not match.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test columns do not match.")

    if list(X_train.columns) != list(X_train_scaled.columns):
        raise ValueError("Scaled and unscaled training columns do not match.")

    print("Shape validation passed.")


def print_summary(X_train, X_test, y_train, y_test, metadata) -> None:
    """Print summary of processed data."""
    print("\nProcessed Data Summary")
    print("----------------------")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nClass Distribution")
    print("------------------")
    print("Training set:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True) * 100)

    print("\nTesting set:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True) * 100)

    print("\nMetadata")
    print("--------")
    print(f"random_state: {metadata.get('random_state')}")
    print(f"test_size: {metadata.get('test_size')}")
    print(f"train_fraud_count: {metadata.get('train_fraud_count')}")
    print(f"test_fraud_count: {metadata.get('test_fraud_count')}")
    print(f"leakage_prevention: {metadata.get('leakage_prevention')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed data files.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed",
        help="Directory containing processed data files.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)

    check_required_files(input_dir)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
        metadata,
    ) = load_processed_data(input_dir)

    validate_shapes(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    print_summary(X_train, X_test, y_train, y_test, metadata)

    print("\nProcessed data validation completed successfully.")


if __name__ == "__main__":
    main()