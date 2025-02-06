import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import pickle


def load_features(path):
    """Load and validate feature data with Persian-to-Arabic conversion"""
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
    """Create enhanced permission-based features"""
    # Ratios
    df["rw_ratio"] = df["rw"] / (df["r"] + 1e-6)
    df["rx_ratio"] = df["rx"] / (df["r"] + 1e-6)

    # Composite scores
    df["perm_complexity"] = 0.4 * df["rwx"] + 0.3 * df["rwxc"] + 0.2 * df["rwc"]
    df["risk_score"] = np.log1p(df["rwxc"]) * np.sqrt(df["rwx"])

    # Binary flags
    df["high_risk"] = (df["rwxc"] > df["rwxc"].quantile(0.9)).astype(int)

    return df


def preprocess_pipeline(df):
    """End-to-end preprocessing workflow"""
    # Handle missing values
    df = df.fillna(df.median())

    # Split data
    X = df.drop("MALICIOUS", axis=1)
    y = df["MALICIOUS"]

    # Balance classes
    X_res, y_res = SMOTE().fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
    )

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def save_artifacts(X_train, X_test, y_train, y_test, scaler):
    """Persist processed data"""
    with open("data/processed/dataset.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    # Load and clean
    df = load_features("data/raw/features.csv")

    # Feature engineering
    df = engineer_features(df)

    # Process data
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)

    # Save outputs
    save_artifacts(X_train, X_test, y_train, y_test, scaler)

    print("Processed Data Shapes:")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class Distribution (Train): {np.unique(y_train, return_counts=True)}")
