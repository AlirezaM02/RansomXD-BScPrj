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
import logging

# %% Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FEATURES = [
    "r",
    "rw",
    "rx",
    "rwc",
    "rwx",
    "rwxc",
    "perm_ratio",
    "complexity_score",
    "write_ratio",
    "execute_ratio",
    "weighted_perm",
]
BASE_DIR = Path(__file__).resolve().parent.parent


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

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and combine training and test data"""
        try:
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
            X = np.concatenate([data["X_train"], data["X_test"]])
            y = np.concatenate([data["y_train"], data["y_test"]])
            logger.info(f"Data loaded successfully. Shape: {X.shape}")
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

    def plot_feature_correlations(self, X: np.ndarray):
        """Plot enhanced correlation matrix with annotations"""
        corr = pd.DataFrame(
            np.corrcoef(X.T), columns=self.feature_names, index=self.feature_names
        )

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

    def plot_feature_importance_analysis(
        self, model, X: np.ndarray, y: np.ndarray, model_name: str
    ):
        """Enhanced feature importance analysis with statistical testing"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Built-in importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in model.estimators_], axis=0
            )
            indices = np.argsort(importances)[::-1]

            ax1.bar(
                range(len(importances)),
                importances[indices],
                yerr=std[indices],
                color=self.colors[0],
                align="center",
            )
            ax1.set_title(f"{model_name} Feature Importance (MDI)")
            ax1.set_xticks(range(len(importances)))
            ax1.set_xticklabels(
                [self.feature_names[i] for i in indices], rotation=45, ha="right"
            )

        # Permutation importance
        r = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        sorted_idx = r.importances_mean.argsort()

        ax2.boxplot(
            [r.importances[sorted_idx[i]] for i in range(X.shape[1])],
            vert=False,
            labels=[self.feature_names[i] for i in sorted_idx],
        )
        ax2.set_title("Permutation Importance")

        plt.tight_layout()
        self._save_plot(f"{model_name}_feature_importance.png")

    def _save_plot(self, filename: str):
        """Helper method to save plots with consistent settings"""
        try:
            plt.savefig(self.results_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
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
    """Main execution function"""
    # Configuration
    config = {
        "data_path": BASE_DIR / "data/processed/features/preprocessed_dataset.pkl",
        "results_dir": BASE_DIR / "results",
        "models_dir": BASE_DIR / "models/batch1",
        "feature_names": FEATURES,
    }

    # Initialize visualizer
    visualizer = RansomwareVisualizer(config)
    feature_analyzer = FeatureAnalyzer(FEATURES)

    try:
        # Load data
        X, y = visualizer.load_data()

        # Generate visualizations
        visualizer.plot_class_distribution(y)
        visualizer.plot_feature_correlations(X)

        # Load and analyze model
        with open(config["models_dir"] / "random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)

        # Analyze features
        logger.info(
            f"Feature Correlation Analysis {feature_analyzer.analyze_correlations(X)}\n"
        )
        logger.info(
            f"Feature Importance Analysis {
                feature_analyzer.analyze_feature_importance(rf_model, X, y)
            }\n"
        )

        logger.info(
            f"Feature Importance Analysis {
                feature_analyzer.feature_distributions(X, y)
            }\n"
        )

        visualizer.plot_feature_importance_analysis(rf_model, X, y, "RandomForest")

        logger.info("All visualizations completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")


# %% Main Execution
if __name__ == "__main__":
    main()
# %%
