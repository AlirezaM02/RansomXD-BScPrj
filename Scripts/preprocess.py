import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np


def load_data(feature_path, label_path):
    # Load and merge datasets
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    df = pd.merge(features, labels, on="ID")
    return df


def clean_data(df):
    # Handle Persian numbers and quotations
    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
    df = df.replace(
        {
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
        },
        regex=True,
    )

    # Convert numerical columns
    num_cols = ["r", "rw", "rx", "rwc", "rwx", "rwxc"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Drop irrelevant columns
    df = df.drop(
        columns=["ID", "NAME", "DATEADDED", "MD5_HASH", "SHA256_HASH"], errors="ignore"
    )

    return df


def preprocess_main():
    df = load_data("data/raw/benign.csv", "data/raw/ransom.csv")
    df = clean_data(df)

    # Handle class imbalance
    malicious = df[df.MALICIOUS == 1]
    benign = df[df.MALICIOUS == 0]
    benign_downsampled = resample(
        benign, replace=False, n_samples=len(malicious), random_state=42
    )
    df = pd.concat([malicious, benign_downsampled])

    # Split data
    X = df.drop("MALICIOUS", axis=1)
    y = df["MALICIOUS"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save processed data
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)


if __name__ == "__main__":
    preprocess_main()
