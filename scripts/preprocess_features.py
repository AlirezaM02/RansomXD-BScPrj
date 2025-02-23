# %% [markdown]
# # Ransomware detection using Intelligent Methods - Isfahan University of Technology
# ### Alireza Mirzaei - BSC Project
# ### Winter of 1403 - Spring of 2025

# %%
# preprocess_features.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import logging
import json

# %%
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


# %%
def load_and_clean_features(path):
    """
    Load and clean feature data from the RDPM dataset
    Handles Persian digits and creates proper binary labels
    """
    try:
        df = pd.read_csv(BASE_DIR / path)
        logger.info(f"Loaded {len(df)} samples from {path}")

        # Persian digit mapping (if needed)
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

        # Permission columns that need cleaning
        perm_cols = ["r", "rw", "rx", "rwc", "rwx", "rwxc"]

        # Clean and convert permission columns
        for col in perm_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(persian_map, regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Create binary labels and category features
        df["MALICIOUS"] = (df["label"] == "M").astype(int)

        # Handle NaN values and convert to string
        df["category"] = df["category"].fillna("unknown").astype(str)

        # Extract clean category names
        df["clean_category"] = df["category"].apply(
            lambda x: x.split(".")[0] if "." in x else x
        )

        # Log summary statistics
        logger.info(f"Class distribution: {df['MALICIOUS'].value_counts().to_dict()}")
        logger.info(f"Categories found: {df['clean_category'].nunique()}")

        return df

    except Exception as e:
        logger.error(f"Error in load_and_clean_features: {str(e)}")
        raise


# %%
def engineer_features(df):
    """
    Calculate additional features based on permission patterns
    """
    try:
        df = df.copy()

        # Avoid division by zero and log(0)
        eps = 1e-6

        # Feature engineering based on permission patterns
        df["perm_ratio"] = (df["rwx"] + eps) / (df["r"] + eps)
        df["complexity_score"] = np.log1p(df["rwxc"] + eps) * np.sqrt(df["rwx"] + eps)
        df["write_ratio"] = (df["rw"] + eps) / (df["r"] + eps)
        df["execute_ratio"] = (df["rx"] + eps) / (df["r"] + eps)

        # Weighted permission complexity
        df["weighted_perm"] = (
            0.3 * df["r"]
            + 0.2 * df["rw"]
            + 0.2 * df["rx"]
            + 0.1 * df["rwc"]
            + 0.1 * df["rwx"]
            + 0.1 * df["rwxc"]
        )

        logger.info(
            "Added engineered features: perm_ratio, complexity_score, write_ratio, execute_ratio, weighted_perm"
        )

        return df

    except Exception as e:
        logger.error(f"Error in engineer_features: {str(e)}")
        raise


# %%
def prepare_training_data(df):
    """
    Prepare final feature set for training
    """
    # Base permission features
    base_features = ["r", "rw", "rx", "rwc", "rwx", "rwxc"]

    # Engineered features
    engineered_features = [
        "perm_ratio",
        "complexity_score",
        "write_ratio",
        "execute_ratio",
        "weighted_perm",
    ]

    # Combine features
    feature_cols = base_features + engineered_features

    # Create feature matrix and target
    X = df[feature_cols]
    y = df["MALICIOUS"]

    return X, y, feature_cols


# %%
def preprocess_data(df, test_size=0.2):
    """
    Complete preprocessing pipeline including train/test split and scaling
    """
    try:
        # Prepare features
        X, y, feature_cols = prepare_training_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_scaled, y_train
        )

        # Log preprocessing results
        logger.info(f"Training set shape after SMOTE: {X_train_resampled.shape}")
        logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

        return {
            "X_train": X_train_resampled,
            "X_test": X_test_scaled,
            "y_train": y_train_resampled,
            "y_test": y_test,
            "feature_names": feature_cols,
            "scaler": scaler,
        }

    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise


# %%
def save_preprocessed_data(processed_data, output_dir="data/processed"):
    """
    Save all preprocessed data and artifacts
    """
    try:
        output_dir = BASE_DIR / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save features as pkl
        dataset_path = output_dir / "features/preprocessed_dataset.pkl"
        with open(dataset_path, "wb") as f:
            pickle.dump(processed_data, f)

        # Save feature names separately for easy access
        feature_names_path = output_dir / "features/feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump(processed_data["feature_names"], f, indent=2)

        # Save scaler
        scaler_path = BASE_DIR / Path("models") / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(processed_data["scaler"], f)

        logger.info(f"Saved preprocessed data to {output_dir}")
        logger.info(f"Saved scaler to {scaler_path}")

    except Exception as e:
        logger.error(f"Error in save_preprocessed_data: {str(e)}")
        raise


# %%
def main():
    try:
        # Load and clean features
        df = load_and_clean_features("data/raw/features.csv")

        # Engineer additional features
        df = engineer_features(df)

        # Preprocess data
        processed_data = preprocess_data(df)

        # Save all artifacts
        save_preprocessed_data(processed_data)

        logger.info("Feature preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


# %%
if __name__ == "__main__":
    main()

# %%
