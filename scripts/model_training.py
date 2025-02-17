# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from .preprocess_features import train_test_split
import json
import os


def load_preprocessed_data(data_path):
    """Load preprocessed data from pickle file"""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
        data["feature_names"],
    )


def train_random_forest(X_train, y_train):
    """Train and return Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train and return XGBoost model"""
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


# Neural Network Architecture
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(X_train, y_train, X_val, y_val):
    """Train PyTorch MLP model"""
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with early stopping
    best_val_recall = 0
    patience_counter = 0
    patience = 5

    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_recall = recall_score(y_val, val_preds.numpy())

            # Early stopping
            if val_recall > best_val_recall:
                best_val_recall = val_recall
                patience_counter = 0
                best_weights = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best weights
    model.load_state_dict(best_weights)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    if isinstance(model, MLP):  # PyTorch model
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            y_pred_proba = model(X_test_tensor).numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
    else:  # Traditional models
        y_pred = model.predict(X_test)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }


def save_model(model, model_name):
    """Save trained model to disk"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    if isinstance(model, MLP):
        torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}.pth"))
    else:
        with open(os.path.join(models_dir, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f)


def plot_roc_curves(models, X_test, y_test):
    """Plot ROC curves for each model"""
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if isinstance(model, MLP):
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred_proba = model(X_test_tensor).numpy()
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")


def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test, feature_names = load_preprocessed_data(
        "data/processed/dataset.pkl"
    )

    # Split validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Train models
    models = {
        "RandomForest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    # Train MLP
    mlp_model, mlp_history = train_mlp(X_train, y_train, X_val, y_val)
    models["MLP"] = mlp_model

    # Evaluate models
    metrics = []
    for name, model in models.items():
        model_metrics = evaluate_model(model, X_test, y_test, name)
        metrics.append(model_metrics)
        save_model(model, name)

        # Save and visualize results
        save_results(metrics)
        plot_roc_curves(models, X_test, y_test)

        # Print summary
        print("\n=== Model Performance Summary ===")

        plt.plot([0, 1], [0, 1], "k--", label="Baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.savefig("reports/roc_curves.png")
        plt.show()
    print(
        pd.DataFrame(metrics).drop(
            ["confusion_matrix", "classification_report"], axis=1
        )
    )


if __name__ == "__main__":
    main()
