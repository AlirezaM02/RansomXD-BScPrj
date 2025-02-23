# %%
# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
import logging
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split

# %%
# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        self.models_dir = Path("models")
        self.results_dir = Path("results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load preprocessed data"""
        try:
            base_dir = (
                Path(__file__).resolve().parent
            )  # Move up two levels from current file
            data_path = base_dir / "data/processed/preprocessed_dataset.pkl"
            with open(data_path, "rb") as f:
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

    def train_neural_net(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        logger.info("Training Neural Network model...")
        input_size = X_train.shape[1]
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

    def _generate_plots(self, y_true, y_pred, y_pred_proba):
        """Generate and save evaluation plots"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ax1.plot(fpr, tpr)
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title("Confusion Matrix")

        # Save plot
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{self.model_name}_{self.timestamp}_plots.png")
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
        model_file = self.models_dir / f"{self.model_name}_{self.timestamp}_model"

        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), str(model_file) + ".pth")
        else:
            with open(str(model_file) + ".pkl", "wb") as f:
                pickle.dump(model, f)

        logger.info(f"Model saved to {model_file}")


# %%
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
            else:  # neural_net
                # Create validation split for neural net
                X_train_nn, X_val, y_train_nn, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, stratify=y_train
                )
                model = trainer.train_neural_net(X_train_nn, y_train_nn, X_val, y_val)

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
