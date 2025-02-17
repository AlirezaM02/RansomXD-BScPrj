# data_cleaning.py
import pandas as pd
import argparse
from pathlib import Path
import pickle
import numpy as np


def clean_single_file(df, is_benign=True):
    """Cleans an individual file (benign or ransomware)"""
    # Convert Persian numbers in critical columns
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
    numeric_cols = ["ID", "CUCKOO_ID", "MALICIOUS", "DOWNLOADED"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip('"').replace(persian_map, regex=True)
            )

            df[col] = pd.to_numeric(df[col].replace("", "0"), errors="coerce")

    # Clean text columns
    text_cols = ["NAME", "LOCATION", "CATEGORY"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip('"')
                .str.split("\\")
                .str[-1]  # Windows path
                .str.split("/")
                .str[-1]
            )  # Unix path

    # Set MALICIOUS flag
    df["MALICIOUS"] = 0 if is_benign else 1
    return df


def load_and_clean_files(benign_path, ransom_path):
    """
    Loads, cleans, and combines the benign and ransomware datasets.
    Returns a single cleaned and shuffled DataFrame.
    """
    # Load and clean both datasets
    benign_df = clean_single_file(pd.read_csv(benign_path), is_benign=True)
    ransom_df = clean_single_file(pd.read_csv(ransom_path), is_benign=False)

    # Combine and shuffle
    combined_df = pd.concat([benign_df, ransom_df], ignore_index=True)
    return combined_df.sample(frac=1, random_state=42).reset_index(drop=True)


def save_cleaned_data(combined_df, output_dir):
    """
    Saves the cleaned and combined data to both CSV and pickle formats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    csv_path = output_dir / "cleaned_combined_data.csv"
    combined_df.to_csv(csv_path, index=False)

    # Save to pickle
    pickle_path = output_dir / "cleaned_combined_data.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(combined_df, f)

    print(f"\nCleaned data saved to:")
    print(f"- CSV: {csv_path}")
    print(f"- Pickle: {pickle_path}")


def cleaning_demo(input_dir, output_dir):
    """Demonstrates the cleaning and combining process"""
    input_path = Path(input_dir)

    # Show before cleaning
    raw_benign = pd.read_csv(input_path / "benign.csv").head(3)
    raw_ransom = pd.read_csv(input_path / "ransom.csv").head(3)

    print("\n=== Raw Data Demonstration ===")
    print("\nSample Before Cleaning (Benign):")
    print(raw_benign[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]])

    print("\nSample Before Cleaning (Ransom):")
    print(raw_ransom[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]])

    # Clean and combine data
    combined_df = load_and_clean_files(
        input_path / "benign.csv", input_path / "ransom.csv"
    )

    # Show after cleaning
    print("\nSample After Cleaning (Combined):")
    print(combined_df[["ID", "NAME", "CUCKOO_ID", "MALICIOUS"]].head(5))

    # Save cleaned data
    save_cleaned_data(combined_df, output_dir)


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
        default="data/cleaned",
        help="Output directory for cleaned CSV and pickle files",
    )

    args = parser.parse_args()
    cleaning_demo(args.input, args.output)
