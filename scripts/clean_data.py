# %% [markdown]
# # Ransomware detection using Intelligent Methods - Isfahan University of Technology
# ### Alireza Mirzaei - BSC Project
# ### Winter of 1403 - Spring of 2025

# %%
# clean_data.py
import pandas as pd
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
def clean_single_file(df, is_benign=True):
    """Cleans an individual file (benign or ransomware)"""
    try:
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

        text_cols = ["NAME", "LOCATION", "CATEGORY"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip('"')
                df[col] = df[col].apply(lambda x: x.split("\\")[-1].split("/")[-1])

        df["MALICIOUS"] = 0 if is_benign else 1
        return df

    except Exception as e:
        logger.error(f"Error in clean_single_file: {str(e)}")
        raise


# %%
def save_data(df, output_dir, filename_prefix):
    """Save DataFrame to both CSV and pickle formats"""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename_prefix}.csv"
    pickle_path = output_dir / f"{filename_prefix}.pkl"

    df.to_csv(csv_path, index=False)
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)

    print("\nCleaned data saved to:")
    print(f"- CSV: {csv_path}")
    print(f"- Pickle: {pickle_path}")


# %%
def main():
    try:
        base_dir = Path(__file__).resolve().parent.parent
        print(base_dir)
        # Load and clean data
        input_dir = base_dir / "data/raw"
        output_dir = base_dir / "data/processed/dataset"

        benign_df = clean_single_file(
            pd.read_csv(input_dir / "benign.csv"), is_benign=True
        )
        ransom_df = clean_single_file(
            pd.read_csv(input_dir / "ransom.csv"), is_benign=False
        )

        # Save individual cleaned files
        save_data(benign_df, output_dir, "cleaned_benign")
        save_data(ransom_df, output_dir, "cleaned_ransom")

        # Combine and shuffle
        combined_df = pd.concat([benign_df, ransom_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=71).reset_index(drop=True)

        # Save combined data
        save_data(combined_df, output_dir, "cleaned_combined")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


# %%
if __name__ == "__main__":
    main()

# %%
