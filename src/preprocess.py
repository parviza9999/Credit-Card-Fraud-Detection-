"""
Preprocessing pipeline for the Credit Card Fraud Detection Capstone.

This script creates the official cleaned dataset and train/test split used by
all models in the project.

Key controls:
- Uses one stratified train/test split.
- Uses fixed random_state for reproducibility.
- Fits StandardScaler only on training data to prevent data leakage.
- Saves both unscaled and scaled split features.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Class"


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate expected dataset structure.

    Args:
        df: Raw input DataFrame.

    Raises:
        ValueError: If required columns or target values are invalid.
    """
    required_columns = ["Time", "Amount", TARGET_COLUMN] + [f"V{i}" for i in range(1, 29)]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError("Target column contains missing values.")

    valid_targets = set(df[TARGET_COLUMN].unique())

    if not valid_targets.issubset({0, 1}):
        raise ValueError(f"Unexpected target values found: {valid_targets}")

    print("Dataset validation passed.")


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    """
    Clean the dataset.

    For this fraud dataset, duplicates are not removed by default because repeated
    transaction-like patterns may contain useful fraud information. Missing values
    are checked and reported.

    Args:
        df: Raw input DataFrame.
        drop_duplicates: Whether to remove duplicate rows.

    Returns:
        Cleaned DataFrame.
    """
    df_clean = df.copy()

    missing_total = int(df_clean.isnull().sum().sum())
    duplicate_count = int(df_clean.duplicated().sum())

    print(f"Total missing values: {missing_total}")
    print(f"Duplicate rows detected: {duplicate_count}")

    if missing_total > 0:
        raise ValueError("Missing values found. Add an imputation strategy before modeling.")

    if drop_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after = len(df_clean)
        print(f"Dropped duplicate rows: {before - after}")
    else:
        print("Duplicate rows retained by design.")

    return df_clean


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """
    Create official stratified train/test split.

    Args:
        df: Cleaned DataFrame.
        test_size: Test set size.
        random_state: Random seed.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("Train/test split complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train fraud rate: {y_train.mean() * 100:.4f}%")
    print(f"y_test fraud rate: {y_test.mean() * 100:.4f}%")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scale features using StandardScaler.

    The scaler is fitted only on X_train and then applied to X_test to prevent
    data leakage.

    Args:
        X_train: Training features.
        X_test: Testing features.

    Returns:
        X_train_scaled_df, X_test_scaled_df, fitted scaler
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    print("Scaling complete. Scaler was fitted only on training data.")

    return X_train_scaled_df, X_test_scaled_df, scaler


def save_outputs(
    output_dir: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    scaler: StandardScaler,
    metadata: dict,
) -> None:
    """
    Save processed datasets and preprocessing artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    X_train_scaled.to_csv(output_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test_scaled.csv", index=False)

    joblib.dump(scaler, output_dir / "scaler.joblib")

    with open(output_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Processed files saved to: {output_dir}")


def run_preprocessing(
    input_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    drop_duplicates: bool = False,
) -> None:
    """
    Run full preprocessing pipeline.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print(f"Loading raw dataset from: {input_path}")

    df = pd.read_csv(input_path)

    validate_dataset(df)
    df_clean = clean_dataset(df, drop_duplicates=drop_duplicates)

    X_train, X_test, y_train, y_test = create_train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    metadata = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "random_state": random_state,
        "test_size": test_size,
        "target_column": TARGET_COLUMN,
        "drop_duplicates": drop_duplicates,
        "raw_rows": int(df.shape[0]),
        "raw_columns": int(df.shape[1]),
        "clean_rows": int(df_clean.shape[0]),
        "clean_columns": int(df_clean.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_fraud_count": int(y_train.sum()),
        "test_fraud_count": int(y_test.sum()),
        "train_fraud_percent": float(y_train.mean() * 100),
        "test_fraud_percent": float(y_test.mean() * 100),
        "feature_columns": list(X_train.columns),
        "saved_outputs": [
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
            "X_train_scaled.csv",
            "X_test_scaled.csv",
            "scaler.joblib",
            "preprocessing_metadata.json",
        ],
        "leakage_prevention": "StandardScaler fitted only on X_train and applied to X_test.",
    }

    save_outputs(
        output_dir=output_dir,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        scaler=scaler,
        metadata=metadata,
    )

    print("Preprocessing pipeline completed successfully.")
    print(json.dumps(metadata, indent=4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create official train/test split and scaled datasets.")
    parser.add_argument("--input", type=str, default="data/raw/creditcard.csv", help="Path to raw creditcard.csv.")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for processed files.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set proportion.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--drop-duplicates", action="store_true", help="Drop duplicate rows if enabled.")

    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_duplicates=args.drop_duplicates,
    )


if __name__ == "__main__":
    main()