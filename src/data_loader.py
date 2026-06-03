import argparse
from pathlib import Path

import pandas as pd
class DataLoader:
    def __init__(self, file_path: str, filename: str = 'creditcard.csv'):
        """Initialize the DataLoader with the file path."""
        self.file_path = Path(file_path)
        # Combine file_path with filename
        self.file_path = self.file_path / filename

        print(f'{self.file_path=}')

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from the file path."""
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file is empty: {self.file_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {exc}") from exc


def download_dataset(output_dir: Path, dataset: str = 'mlg-ulb/creditcardfraud') -> Path:
    """Download the Kaggle dataset and place the CSV in the output directory."""
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud", output_dir=output_dir)

    print("Path to dataset files:", path)

    return path
def main() -> None:
    parser = argparse.ArgumentParser(description="Load or download the credit card fraud dataset.")
    parser.add_argument("--download", action="store_true", help="Download the dataset from Kaggle.")
    parser.add_argument("--input", type=str, help="Path to an existing CSV file to copy into raw data.")
    parser.add_argument("--output", type=str, default="data/raw", help="Target directory for raw dataset files.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        csv_path = download_dataset(output_dir)
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        destination = output_dir / input_path.name
        if input_path.resolve() != destination.resolve():
            destination.write_bytes(input_path.read_bytes())
        print(f"Copied dataset from {input_path} to {destination}")
        csv_path = destination
    else:
        raise ValueError("Either --download or --input must be provided.")

    loader = DataLoader(csv_path)
    df = loader.load_data()
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    main()