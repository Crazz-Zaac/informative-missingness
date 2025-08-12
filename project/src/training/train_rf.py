from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import pandas as pd
import yaml

from src.config.schemas import ExperimentConfig, ModelTypeEnum
from src.data.dataset import TabularDataset
from src.models.random_forest import RandomForestModel
from loguru import logger


class RandomForestTrainer:
    """Handles the complete training process"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        logger.info(f"Initializing RandomForestTrainer ")
        # since the hyperparameters is a class, we need to access it dynamically
        model_hyperparams = self.config.model.hyperparameters.RandomForest
        if model_hyperparams is None:
            raise ValueError(
                f"No hyperparameters provided for model: {ModelTypeEnum.RF}"
            )

        self.rf_fixed_params = model_hyperparams.fixed_params
        self.rf_grid_search_params = model_hyperparams.grid_search_params

        self.dataset = TabularDataset(
            window_size=self.config.data.tabular.window_size, config=self.config
        )
        self.random_state = self.rf_fixed_params.get("random_state")
        self.model = RandomForestModel(config=self.rf_fixed_params)

    def save_and_plot(
        self, mean_importance: pd.DataFrame, recalls: list, f1s: list, aucs: list
    ):

        plot_dir = self.config.plots_dir
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        mean_importance.plot(
            kind="barh", figsize=(10, 6), title="Permutation Feature Importance"
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/feature_importance_{timestamp}.png")

        metrics_df = pd.DataFrame(
            {"fold": list(range(1, 6)), "recall": recalls, "f1": f1s, "roc_auc": aucs}
        )
        metrics_df.to_csv(f"{plot_dir}/fold_metrics_{timestamp}.csv", index=False)

    def run_training(self):
        recalls, f1s, aucs, pr_aucs = [], [], [], []
        best_estimators = []

        logger.info("Loading and preparing the data...")
        X, y, groups = self.dataset.load_and_split_data()

        sgkf = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )
        logger.info("Fixed model hyperparameters:")
        for key, value in self.rf_fixed_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Grid search parameters to explore:")
        for key, value in self.rf_grid_search_params.items():
            logger.info(f"  {key}: {value}")

        try:
            for fold, (train_idx, test_idx) in enumerate(
                sgkf.split(X, y, groups=groups)
            ):
                logger.info(f"\n=== Running Fold {fold + 1} ===")

                X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # sampling data
                logger.info("Balancing the training set using RandomOverSampler")
                oversample = RandomOverSampler(
                    sampling_strategy="minority", random_state=self.random_state
                )
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                # Ensure consistent column names
                X_train.columns = X_train.columns.astype(str)
                X_test.columns = X_test.columns.astype(str)

                # Grid Search with fixed and search parameters
                base_model = self.model._initialize_model()
                
                logger.info("Running GridSearchCV...")
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=self.rf_grid_search_params,
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                best_estimators.append(best_model)

                logger.info(
                    f"Best hyperparameters from GridSearchCV (Fold {fold + 1}):"
                )
                for key, val in grid_search.best_params_.items():
                    logger.info(f"  {key}: {val}")

                # Evaluation
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]

                recalls.append(recall_score(y_test, y_pred, pos_label=1))
                f1s.append(f1_score(y_test, y_pred, pos_label=1))
                aucs.append(roc_auc_score(y_test, y_prob))
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_aucs.append(auc(recall, precision))

                logger.info(
                    f"Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}, ROC-AUC: {aucs[-1]:.4f}, PR-AUC: {pr_aucs[-1]:.4f}"
                )

            # Log model parameters
            logger.info("Logging fixed model hyperparameters:")
            for key, value in self.rf_fixed_params.items():
                logger.info(f"  {key}: {value}")
            logger.info("Logging grid search parameters:")
            for key, value in self.rf_grid_search_params.items():
                logger.info(f"  {key}: {value}")

        
            # Summary
            logger.info("\n=== Training Summary ===")
            logger.info(
                f"Mean Recall: {pd.Series(recalls).mean():.4f} ± {pd.Series(recalls).std():.4f}"
            )
            logger.info(
                f"Mean F1:     {pd.Series(f1s).mean():.4f} ± {pd.Series(f1s).std():.4f}"
            )
            logger.info(
                f"Mean ROC-AUC: {pd.Series(aucs).mean():.4f} ± {pd.Series(aucs).std():.4f}"
            )
            logger.info(
                f"Mean PR-AUC: {pd.Series(pr_aucs).mean():.4f} ± {pd.Series(pr_aucs).std():.4f}"
            )

            # Save best model from last fold (or optionally from best overall)
            self.model.model = best_estimators[-1]

            return self.model

        except Exception as e:
            logger.exception(f"Experiment failed due to an unexpected error {e}.")
        finally:
            logger.success("Experiment finished.")

    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = ExperimentConfig(**yaml.safe_load(f))
        return cls(config)
