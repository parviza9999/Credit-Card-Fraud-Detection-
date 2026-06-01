"""
Data loading utility for the Credit Card Fraud Detection Capstone.

This module loads the Kaggle/ULB creditcard.csv dataset from either:
1. A folder path containing creditcard.csv
2. A full path to creditcard.csv
"""

import argparse
from pathlib import Path
import shutil

import pandas as pd


DEFAULT_FILENAME = "creditcard.csv"


class DataLoader:
    def __init__(self, path: str, filename: str = DEFAULT_FILENAME):
        """
        Initialize the DataLoader.

        Args:
            path: Folder containing creditcard.csv or full path to creditcard.csv.
            filename: Dataset filename if path is a folder.
        """
        path = Path(path)

        if path.is_dir():
            self.file_path = path / filename
        else:
            self.file_path = path

        print(f"Dataset path: {self.file_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset.

        Returns:
            pandas DataFrame containing the dataset.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file. Got: {self.file_path}")

        data = pd.read_csv(self.file_path)
        print(f"Data loaded successfully from {self.file_path}")
        print(f"Dataset shape: {data.shape}")

        return data


def copy_dataset(input_path: str, output_dir: str = "data/raw") -> Path:
    """
    Copy an existing dataset file into data/raw.

    Args:
        input_path: Existing path to creditcard.csv.
        output_dir: Target raw data directory.

    Returns:
        Destination file path.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    destination = output_dir / input_path.name

    if input_path.resolve() != destination.resolve():
        shutil.copy2(input_path, destination)

    print(f"Copied dataset from {input_path} to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or copy the credit card fraud dataset.")
    parser.add_argument("--input", type=str, help="Path to an existing creditcard.csv file.")
    parser.add_argument("--output", type=str, default="data/raw", help="Target raw data directory.")
    parser.add_argument("--load-only", action="store_true", help="Load dataset from --input without copying.")

    args = parser.parse_args()

    if not args.input:
        raise ValueError("Please provide --input pointing to creditcard.csv or a folder containing it.")

    if args.load_only:
        loader = DataLoader(args.input)
        df = loader.load_data()
    else:
        copied_path = copy_dataset(args.input, args.output)
        loader = DataLoader(copied_path)
        df = loader.load_data()

    print(df.head())


if __name__ == "__main__":
    main()