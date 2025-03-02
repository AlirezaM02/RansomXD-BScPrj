# %% [markdown]
# # Ransomware detection using Intelligent Methods - Isfahan University of Technology
# ### Alireza Mirzaei - BSC Project
# ### Winter of 1403 - Spring of 2025

# %%
# visualization_analysis.py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
from typing import Dict, Tuple, List
import shap
import json
import logging
import gc

# %% Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES = []
with open(BASE_DIR / "data/processed/features/feature_names.json") as file:
    features = json.load(file)
    FEATURES = features.copy()


# %% Ransomware Visualizer
class RansomwareVisualizer:
    """Unified class for ransomware detection visualization and analysis"""

    def __init__(self, config: Dict[str, Path]):
        """
        Initialize visualizer with configuration

        Args:
            config: Dictionary containing path configurations
        """
        self.data_path = config["data_path"]
        self.results_dir = config["results_dir"]
        self.models_dir = config["models_dir"]
        self.feature_names = config["feature_names"]

        # Create directories if they don't exist
        for dir_path in [self.results_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        self.colors = sns.color_palette("husl", 8)

        # Data cache
        self._data_cache = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and combine training and test data with caching"""
        if self._data_cache is not None:
            return self._data_cache

        try:
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
            X = np.concatenate([data["X_train"], data["X_test"]])
            y = np.concatenate([data["y_train"], data["y_test"]])
            logger.info(f"Data loaded successfully. Shape: {X.shape}")

            # Cache the data to avoid repeated loading
            self._data_cache = (X, y)
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def plot_class_distribution(self, y: np.ndarray):
        """Plot class distribution with percentages"""
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=y)
        total = len(y)

        # Add percentage labels
        for p in ax.patches:
            percentage = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(
                percentage,
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
            )

        plt.title("Class Distribution in Dataset")
        plt.xlabel("Class (0: Benign, 1: Ransomware)")
        plt.ylabel("Count")
        self._save_plot("class_distribution.png")
        plt.close()  # Explicitly close the figure

    def plot_feature_correlations(self, X: np.ndarray):
        """Plot enhanced correlation matrix with annotations"""
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X

        # Calculate correlation matrix
        corr = X_df.corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
        )
        plt.title("Feature Correlation Matrix")
        self._save_plot("feature_correlations.png")
        plt.close()

        # Help garbage collection
        del corr, X_df
        gc.collect()

    def plot_model_comparison(self, results_dict: Dict[str, Dict]):
        """Plot comparison of multiple models' performance metrics"""
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        model_names = list(results_dict.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        for i, (model_name, results) in enumerate(results_dict.items()):
            performance = [results.get(metric, 0) for metric in metrics]
            ax.bar(x + i * width, performance, width, label=model_name)

        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        plt.tight_layout()
        self._save_plot("model_comparison.png")
        plt.close()

    def plot_feature_importance(
        self, model, X: np.ndarray, y: np.ndarray, model_name: str
    ):
        """Memory-optimized feature importance visualization"""
        plt.figure(figsize=(12, 6))

        # Use built-in feature importance if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Only top 15 features for clarity

            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.title(f"{model_name} - Top Feature Importance")
            plt.xlabel("Importance")
            self._save_plot(f"{model_name}_feature_importance.png")
            plt.close()

        # Permutation importance (more memory-efficient than SHAP)
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        indices = np.argsort(r.importances_mean)[-15:]  # Only top 15

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(indices)), r.importances_mean[indices], align="center")
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.title(f"{model_name} - Permutation Importance")
        plt.xlabel("Mean decrease in accuracy")
        self._save_plot(f"{model_name}_permutation_importance.png")
        plt.close()

    def plot_reduced_feature_distributions(self, X: np.ndarray, y: np.ndarray):
        """Plot distributions for only the most important features"""
        # Select only top features to plot - reduces memory usage
        top_features_idx = np.argsort(np.var(X, axis=0))[
            -6:
        ]  # Select 6 most variable features
        top_features = [self.feature_names[i] for i in top_features_idx]

        # Create DataFrame with only selected features
        df = pd.DataFrame(
            {feature: X[:, i] for i, feature in zip(top_features_idx, top_features)}
        )
        df["target"] = y

        # Create figure with fewer subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, feature in enumerate(top_features):
            sns.histplot(data=df, x=feature, hue="target", kde=True, ax=axes[i])
            axes[i].set_title(f"{feature} Distribution by Class")

        plt.tight_layout()
        self._save_plot("top_feature_distributions.png")
        plt.close()

        # Clean up
        del df
        gc.collect()

    def _save_plot(self, filename: str):
        """Helper method to save plots with consistent settings"""
        try:
            plt.savefig(self.results_dir / filename, dpi=200, bbox_inches="tight")
            logger.info(f"Plot saved successfully: {filename}")
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {e}")


class FeatureAnalyzer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        plt.style.use("seaborn-v0_8")

    def analyze_correlations(self, X: np.ndarray) -> Dict:
        """
        Analyze feature correlations using multiple methods
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(X, columns=self.feature_names)

        # Pearson correlation
        pearson_corr = df.corr()

        # Create correlation plot with hierarchical clustering
        plt.figure(figsize=(12, 10))
        clustergrid = sns.clustermap(
            pearson_corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            figsize=(12, 12),
        )
        plt.title("Hierarchically Clustered Feature Correlations")

        # Find highly correlated features
        high_corr = np.where(np.abs(pearson_corr) > 0.7)
        high_corr_pairs = [
            (self.feature_names[i], self.feature_names[j], pearson_corr.iloc[i, j])
            for i, j in zip(*high_corr)
            if i < j  # Avoid duplicates and self-correlations
        ]

        return {
            "correlation_matrix": pearson_corr,
            "high_correlations": high_corr_pairs,
            "clustergrid": clustergrid,
        }

    def analyze_feature_importance(self, model, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Comprehensive feature importance analysis using multiple methods
        """
        results = {}

        # Built-in feature importance (for tree-based models)
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in model.estimators_], axis=0
            )

            results["mdi_importance"] = {"importance": importance, "std": std}

            # Plot MDI importance
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importance)[::-1]
            plt.bar(
                range(len(importance)),
                importance[indices],
                yerr=std[indices],
                align="center",
            )
            plt.xticks(
                range(len(indices)),
                [self.feature_names[i] for i in indices],
                rotation=45,
            )
            plt.title("Feature Importance (MDI)")
            results["mdi_plot"] = plt.gcf()
            plt.close()

        # Permutation importance
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42)

        results["permutation_importance"] = {
            "mean": r.importances_mean,
            "std": r.importances_std,
        }

        # Plot permutation importance
        plt.figure(figsize=(10, 6))
        indices = np.argsort(r.importances_mean)[::-1]
        plt.bar(
            range(len(r.importances_mean)),
            r.importances_mean[indices],
            yerr=r.importances_std[indices],
            align="center",
        )
        plt.xticks(
            range(len(indices)), [self.feature_names[i] for i in indices], rotation=45
        )
        plt.title("Feature Importance (Permutation)")
        results["permutation_plot"] = plt.gcf()
        plt.close()

        # SHAP values for detailed feature impact
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        results["shap_plot"] = plt.gcf()
        plt.close()

        return results

    def feature_distributions(self, X: np.ndarray, y: np.ndarray):
        """
        Analyze and plot feature distributions by class
        """
        df = pd.DataFrame(X, columns=self.feature_names)
        df["target"] = y

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.ravel()

        for idx, feature in enumerate(self.feature_names):
            if idx < len(axes):
                sns.boxplot(x="target", y=feature, data=df, ax=axes[idx])
                axes[idx].set_title(f"{feature} Distribution by Class")

        plt.tight_layout()
        return fig


# %% Main function
def main():
    """Main execution function with enhanced visualizations"""
    # Configuration
    config = {
        "data_path": BASE_DIR / "data/processed/features/preprocessed_dataset.pkl",
        "results_dir": BASE_DIR / "results",
        "models_dir": BASE_DIR / "models/batch1",
        "feature_names": FEATURES,
    }

    # Initialize visualizer
    visualizer = RansomwareVisualizer(config)

    try:
        # Load data (only once)
        X, y = visualizer.load_data()

        # Generate basic visualizations
        visualizer.plot_class_distribution(y)
        visualizer.plot_feature_correlations(X)
        visualizer.plot_reduced_feature_distributions(X, y)

        # Process one model at a time to reduce memory usage
        model_paths = {
            "random_forest": config["models_dir"] / "random_forest_model.pkl",
            "xgboost": config["models_dir"] / "xgboost_model.pkl",
        }

        for model_name, path in model_paths.items():
            if not path.exists():
                logger.warning(f"Model not found: {path}")
                continue

            logger.info(f"Processing {model_name} model")

            # Load model
            with open(path, "rb") as f:
                model = pickle.load(f)

            # Analyze model
            visualizer.plot_feature_importance(model, X, y, model_name)

            # Clean up to free memory
            del model
            gc.collect()

        logger.info("All visualizations completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.exception("Detailed error information:")


# %% Main Execution
if __name__ == "__main__":
    main()
# %%
