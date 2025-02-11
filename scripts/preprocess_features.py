import pandas as pd
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def load_features(path):
    """Load and validate feature data with Persian-to-English conversion"""
    df = pd.read_csv(path)

    # Persian digit mapping
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

    # Clean numeric columns
    num_cols = ["r", "rw", "rx", "rwc", "rwx", "rwxc"]
    for col in num_cols:
        df[col] = df[col].astype(str).replace(persian_map, regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create binary labels
    df["MALICIOUS"] = df["label"].map({"B": 0, "M": 1}).astype(int)

    return df.drop(columns=["ID", "label", "sample", "category"])


def engineer_features(df):
    """Feature calculations"""
    # Replace zeros to avoid log(0)
    df["rwxc_safe"] = df["rwxc"].replace(0, 1e-6)

    # Revised features
    df["perm_complexity"] = 0.4 * df["rwx"] + 0.3 * df["rwxc_safe"] + 0.3 * df["rwc"]
    df["risk_score"] = (
        np.log1p(df["rwxc_safe"]) * np.sqrt(df["rwx"] + 1e-6)  # Avoid sqrt(0)
    )

    return df.drop(columns=["rwxc_safe"])


def preprocess_pipeline(df):
    """Preprocessing Workflow with label handling"""
    # Get feature names before dropping target
    feature_names = df.drop("MALICIOUS", axis=1).columns.tolist()

    # Split FIRST to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("MALICIOUS", axis=1),
        df["MALICIOUS"],
        test_size=0.2,
        stratify=df["MALICIOUS"],  # Stratify on original labels
        random_state=random.randint(10, 100),
    )

    # Print class distribution before SMOTE
    print("Class counts before SMOTE:", np.bincount(y_train))

    # Apply SMOTE ONLY to training data
    smote = SMOTE(sampling_strategy="auto")
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("Class counts after SMOTE:", np.bincount(y_train_res))

    # Normalize AFTER resampling
    scaler = MinMaxScaler()
    X_train_final = scaler.fit_transform(X_train_res)
    X_test_final = scaler.transform(X_test)

    # Convert labels to numpy arrays to ensure consistent types
    y_train_res = np.array(y_train_res)
    y_test = np.array(y_test)

    # Verify shapes and types
    print("\nShape verification:")
    print(f"X_train: {X_train_final.shape}")
    print(f"X_test: {X_test_final.shape}")
    print(f"y_train: {y_train_res.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train_final, X_test_final, y_train_res, y_test, scaler, feature_names


def inspect_data(X_train, X_test, y_train, y_test, feature_names):
    """Print a sample of preprocessed data for inspection"""
    # Convert arrays back to DataFrames for readability
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["MALICIOUS"] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["MALICIOUS"] = y_test

    print("\n=== Preprocessed Data Sample ===")
    print("\nTraining Data (First 5 Rows):")
    print(train_df.head())

    print("\nTest Data (First 5 Rows):")
    print(test_df.head())

    print("\nData Summary:")
    print(test_df.describe())

    print("\nTraining Data Class Distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nTest Data Class Distribution:")
    print(pd.Series(y_test).value_counts())

    print("\nFeature Statistics (Training):")
    print(train_df.drop("MALICIOUS", axis=1).describe())


def save_artifacts(X_train, X_test, y_train, y_test, scaler, feature_names):
    """Persist processed data"""
    artifacts = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
    }

    # Save dataset as a dictionary for better organization
    with open("data/processed/dataset.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    # Save scaler separately
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    # Load and clean
    df = load_features("data/raw/features.csv")

    # Feature engineering
    df = engineer_features(df)

    # Process data
    X_train_final, X_test_final, y_train_res, y_test, scaler, feature_names = (
        preprocess_pipeline(df)
    )

    # Inspect data before saving
    inspect_data(X_train_final, X_test_final, y_train_res, y_test, feature_names)

    # Save artifacts
    save_artifacts(
        X_train_final, X_test_final, y_train_res, y_test, scaler, feature_names
    )
