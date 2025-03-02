# %% [markdown]
# # Ransomware detection using Intelligent Methods - Isfahan University of Technology
# ### Alireza Mirzaei - BSC Project
# ### Winter of 1403 - Spring of 2025

# %%
# model_training.py
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
import logging
import seaborn as sns
from datetime import datetime

# %%
# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


# %%
class RansomwareDetector(nn.Module):
    """Neural network for ransomware detection"""

    def __init__(self, input_size, hidden_sizes=[128, 64]):
        super(RansomwareDetector, self).__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.3),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# %%
class ModelTrainer:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.models_dir = BASE_DIR / Path("models")
        self.results_dir = BASE_DIR / Path("results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load preprocessed data"""
        try:
            with open(
                BASE_DIR / "data/processed/features/preprocessed_dataset.pkl", "rb"
            ) as f:
                data = pickle.load(f)
            logger.info("Successfully loaded preprocessed data")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=np.sum(y_train == 0) / np.sum(y_train == 1),
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def train_neural_net(self, X_train, y_train, X_val, y_val, use_cv=False, n_folds=5):
        """Train Neural Network model with optional cross-validation"""
        logger.info("Training Neural Network model...")
        input_size = X_train.shape[1]

        if not use_cv:
            # Regular training without cross-validation
            model = RansomwareDetector(input_size).to(self.device)

            # Create data loaders
            train_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_train).to(self.device),
                    torch.FloatTensor(y_train).reshape(-1, 1).to(self.device),
                ),
                batch_size=64,
                shuffle=True,
            )

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=5
            )

            best_val_auc = 0
            patience = 10
            patience_counter = 0

            for epoch in range(100):
                # Training code remains the same...
                model.train()
                total_loss = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(torch.FloatTensor(X_val).to(self.device))
                    val_auc = roc_auc_score(y_val, val_outputs.cpu().numpy())

                    scheduler.step(val_auc)

                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Val AUC: {val_auc:.4f}"
                    )

            model.load_state_dict(best_state)
            return model

        else:
            # Cross-validation training
            logger.info(f"Using {n_folds}-fold cross-validation")
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_results = []

            # Combine training and validation data for CV
            X_combined = np.vstack((X_train, X_val))
            y_combined = np.concatenate((y_train, y_val))

            fold_models = []
            fold_val_aucs = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_combined)):
                logger.info(f"Training fold {fold + 1}/{n_folds}")

                # Split data for this fold
                X_fold_train, X_fold_val = X_combined[train_idx], X_combined[val_idx]
                y_fold_train, y_fold_val = y_combined[train_idx], y_combined[val_idx]

                # Initialize model for this fold
                fold_model = RansomwareDetector(input_size).to(self.device)

                # Create data loader for this fold
                fold_train_loader = DataLoader(
                    TensorDataset(
                        torch.FloatTensor(X_fold_train).to(self.device),
                        torch.FloatTensor(y_fold_train).reshape(-1, 1).to(self.device),
                    ),
                    batch_size=64,
                    shuffle=True,
                )

                criterion = nn.BCELoss()
                optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", patience=5
                )

                best_fold_val_auc = 0
                patience = 10
                patience_counter = 0

                for epoch in range(100):
                    fold_model.train()
                    total_loss = 0

                    for batch_X, batch_y in fold_train_loader:
                        optimizer.zero_grad()
                        outputs = fold_model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                    # Validation
                    fold_model.eval()
                    with torch.no_grad():
                        fold_val_outputs = fold_model(
                            torch.FloatTensor(X_fold_val).to(self.device)
                        )
                        fold_val_auc = roc_auc_score(
                            y_fold_val, fold_val_outputs.cpu().numpy()
                        )

                        scheduler.step(fold_val_auc)

                        if fold_val_auc > best_fold_val_auc:
                            best_fold_val_auc = fold_val_auc
                            best_fold_state = fold_model.state_dict()
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            logger.info(
                                f"Fold {fold + 1} early stopping at epoch {epoch}"
                            )
                            break

                    if (epoch + 1) % 10 == 0:
                        logger.info(
                            f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {total_loss / len(fold_train_loader):.4f}, Val AUC: {fold_val_auc:.4f}"
                        )

                # Save best model for this fold
                fold_model.load_state_dict(best_fold_state)
                fold_models.append(fold_model)
                fold_val_aucs.append(best_fold_val_auc)

                # Evaluate on validation set
                fold_model.eval()
                with torch.no_grad():
                    fold_outputs = fold_model(
                        torch.FloatTensor(X_fold_val).to(self.device)
                    )
                    fold_preds = (fold_outputs.cpu().numpy() > 0.5).astype(int)
                    fold_report = classification_report(
                        y_fold_val, fold_preds, output_dict=True
                    )
                    cv_results.append(
                        {
                            "fold": fold + 1,
                            "auc": best_fold_val_auc,
                            "accuracy": fold_report["accuracy"],
                            "precision": fold_report["weighted avg"]["precision"],
                            "recall": fold_report["weighted avg"]["recall"],
                            "f1": fold_report["weighted avg"]["f1-score"],
                        }
                    )

            # Log CV results
            cv_df = pd.DataFrame(cv_results)
            logger.info(f"\nCross-validation results:\n{cv_df}")
            logger.info(
                f"Mean AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}"
            )

            # Select best model
            best_fold_idx = np.argmax(fold_val_aucs)
            best_model = fold_models[best_fold_idx]
            logger.info(
                f"Selected model from fold {best_fold_idx + 1} with validation AUC: {fold_val_aucs[best_fold_idx]:.4f}"
            )

            # Save CV results
            cv_results_file = (
                self.results_dir / f"{self.model_name}_{self.timestamp}_cv_results.csv"
            )
            cv_df.to_csv(cv_results_file, index=False)
            logger.info(f"CV results saved to {cv_results_file}")

            # Plot CV results
            plt.figure(figsize=(10, 6))
            metrics = ["auc", "accuracy", "precision", "recall", "f1"]
            for metric in metrics:
                plt.plot(cv_df["fold"], cv_df[metric], "o-", label=metric)
            plt.xlabel("Fold")
            plt.ylabel("Score")
            plt.title("Cross-validation Results")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                self.results_dir / f"{self.model_name}_{self.timestamp}_cv_results.png",
                dpi=300,
            )
            plt.close()

            return best_model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                y_pred_proba = (
                    model(torch.FloatTensor(X_test).to(self.device)).cpu().numpy()
                )
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        y_test = np.asarray(y_test).ravel()  # flatten the y_test

        # Calculate metrics
        results = {
            "accuracy": float(np.mean(y_pred == y_test)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        # Generate plots
        self._generate_plots(y_test, y_pred, y_pred_proba)

        return results

    # Replace the _generate_plots method with this enhanced version

    def _generate_plots(self, y_true, y_pred, y_pred_proba):
        """Generate and save evaluation plots"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        axes[0, 0].plot([0, 1], [0, 1], "k--")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend(loc="lower right")

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        axes[0, 1].plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision-Recall Curve")
        axes[0, 1].legend(loc="lower left")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("True")
        axes[1, 0].set_title("Confusion Matrix")

        # Classification threshold analysis
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
            rec = recall_score(y_true, y_pred_thresh, zero_division=0)

            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)

        axes[1, 1].plot(thresholds, f1_scores, label="F1 Score")
        axes[1, 1].plot(thresholds, precisions, label="Precision")
        axes[1, 1].plot(thresholds, recalls, label="Recall")
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Metrics vs. Classification Threshold")
        axes[1, 1].legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(
            self.results_dir / f"{self.model_name}_{self.timestamp}_plots.png", dpi=300
        )
        plt.close()

    def save_results(self, results):
        """Save evaluation results"""
        results_file = (
            self.results_dir / f"{self.model_name}_{self.timestamp}_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {results_file}")

    def save_model(self, model):
        """Save trained model"""
        n = 1
        model_file = self.models_dir / f"batch{n}/{self.model_name}_model"

        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), str(model_file) + ".pth")
        else:
            with open(str(model_file) + ".pkl", "wb") as f:
                pickle.dump(model, f)

        logger.info(f"Model saved to {model_file}")


# %%
# Add to main function:
def main():
    try:
        # Set device
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Initialize trainers
        trainers = {
            "random_forest": ModelTrainer("random_forest", device),
            "xgboost": ModelTrainer("xgboost", device),
            "neural_net": ModelTrainer("neural_net", device),
            "neural_net_cv": ModelTrainer("neural_net_cv", device),
        }

        # Train and evaluate all models
        for name, trainer in trainers.items():
            logger.info(f"\nTraining {name} model...")

            # Load data
            data = trainer.load_data()
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            # Train appropriate model
            if name == "random_forest":
                model = trainer.train_random_forest(X_train, y_train)

            elif name == "xgboost":
                model = trainer.train_xgboost(X_train, y_train)

            elif name == "neural_net":
                # Create validation split for neural net
                X_train_nn, X_val, y_train_nn, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train
                )
                model = trainer.train_neural_net(
                    X_train_nn, y_train_nn, X_val, y_val, use_cv=False
                )

            else:  # neural_net with cv
                # Create validation split for neural net
                X_train_nn, X_val, y_train_nn, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train
                )
                model = trainer.train_neural_net(
                    X_train_nn, y_train_nn, X_val, y_val, use_cv=True
                )

            # Evaluate and save results
            results = trainer.evaluate_model(model, X_test, y_test)
            trainer.save_results(results)
            trainer.save_model(model)

            # Log summary metrics
            logger.info(f"\n{name.upper()} Results:")
            logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
            logger.info(f"Average Precision: {results['avg_precision']:.4f}")
            logger.info(f"Accuracy: {results['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


# %%
if __name__ == "__main__":
    main()

# %%
