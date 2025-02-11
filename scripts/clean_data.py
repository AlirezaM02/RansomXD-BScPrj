# data_cleaning.py
import pandas as pd
import argparse
from pathlib import Path


def load_and_clean_files(benign_path, ransom_path):
    """
    Demonstrates cleaning raw data files without any feature engineering
    Returns cleaned DataFrames for benign and ransomware samples separately
    """
    # Persian digit conversion mapping
    persian_map = {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }

    def clean_single_file(df):
        """Cleaning operations for individual files"""
        # Convert Persian numbers in critical columns
        numeric_cols = ["ID", "CUCKOO_ID", "MALICIOUS", "DOWNLOADED"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str).str.strip('"').replace(persian_map, regex=True)
                )
                # Handle empty strings before conversion
                df[col] = df[col].replace("", "0")
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clean text columns
        text_cols = ["NAME", "LOCATION", "CATEGORY"]
        for col in text_cols:
            if col in df.columns:
                # Remove quotes and extract filename from path
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip('"')
                    .str.split("\\")
                    .str[-1]  # Extract filename from Windows path
                    .str.split("/")
                    .str[-1]  # Extract filename from Unix path
                )

        # Ensure MALICIOUS column exists
        if "MALICIOUS" not in df.columns:
            df["MALICIOUS"] = 0 if "benign" in str(benign_path).lower() else 1

        return df

    # Load and clean both datasets separately
    benign_df = clean_single_file(pd.read_csv(benign_path))
    ransom_df = clean_single_file(pd.read_csv(ransom_path))

    return benign_df, ransom_df


def save_cleaned_data(benign_df, ransom_df, output_dir):
    """Save cleaned data to CSV for inspection"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benign_df.to_csv(output_dir / "cleaned_benign.csv", index=False)
    ransom_df.to_csv(output_dir / "cleaned_ransom.csv", index=False)


def cleaning_demo(input_dir, output_dir):
    """Demonstration of cleaning functionality with sample data"""
    input_path = Path(input_dir)
    print("\n=== Raw Data Demonstration ===")

    # Show before cleaning
    raw_benign = pd.read_csv(input_path / "benign.csv").head(3)
    raw_ransom = pd.read_csv(input_path / "ransom.csv").head(3)

    print("\nSample Before Cleaning (Benign):")
    print(raw_benign[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]])

    print("\nSample Before Cleaning (Ransom):")
    print(raw_ransom[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]])

    # Clean data
    clean_benign, clean_ransom = load_and_clean_files(
        input_path / "benign.csv", input_path / "ransom.csv"
    )

    # Show after cleaning
    print("\nSample After Cleaning (Benign):")
    print(clean_benign[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]].head(3))

    print("\nSample After Cleaning (Ransom):")
    print(clean_ransom[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]].head(3))

    # Save cleaned data
    save_cleaned_data(clean_benign, clean_ransom, output_dir)
    print(f"\nCleaned data saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data cleaning demonstration")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory containing raw CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory for cleaned CSV files",
    )

    args = parser.parse_args()

    cleaning_demo(args.input, args.output)
