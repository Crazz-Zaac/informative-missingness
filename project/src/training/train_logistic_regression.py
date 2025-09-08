from pathlib import Path
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    recall_score,
    f1_score,
    roc_auc_score,
)

import pandas as pd
import yaml
import numpy as np

from src.data.dataset import TabularDataset
from src.models.logistic_regression import LogisticRegressionModel
from src.config.schemas import ExperimentConfig, ModelTypeEnum


class LogisticRegressionTrainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        logger.info("Initializing LogisticRegressionTrainer")

        model_hyperparams = self.config.model.hyperparameters.LogisticRegression
        if model_hyperparams is None:
            raise ValueError(
                f"No hyperparameters provided for model: {ModelTypeEnum.LR}"
            )

        self.fixed_params = model_hyperparams.fixed_params
        self.grid_search_params = model_hyperparams.grid_search_params

        self.dataset = TabularDataset(
            window_size=self.config.data.tabular.window_size, config=self.config
        )
        self.model = LogisticRegressionModel(config=self.fixed_params)

    def run_training(self):
        recalls, f1s, aucs, pr_aucs = [], [], [], []
        best_estimators = []

        logger.info("Loading and preparing the data...")
        X, y, groups = self.dataset.load_and_split_data()
        if np.isnan(X.values).any():
            logger.info("NaN values detected in X — applying imputation...")
            imputer = SimpleImputer(strategy="mean")  # or median, most_frequent, constant
            X = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

        sgkf = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=42
        )

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
                    sampling_strategy="minority", random_state=42
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
                    param_grid=self.grid_search_params,
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
            for key, value in self.fixed_params.items():
                logger.info(f"  {key}: {value}")
            logger.info("Logging grid search parameters:")
            for key, value in self.rf_grid_search_params.items():
                logger.info(f"  {key}: {value}")

            # Summary
            logger.info("\n=== Training Summary ===")
            logger.success(
                f"Mean Recall: {pd.Series(recalls).mean():.4f} ± {pd.Series(recalls).std():.4f}"
            )
            logger.success(
                f"Mean F1:     {pd.Series(f1s).mean():.4f} ± {pd.Series(f1s).std():.4f}"
            )
            logger.success(
                f"Mean ROC-AUC: {pd.Series(aucs).mean():.4f} ± {pd.Series(aucs).std():.4f}"
            )
            logger.success(
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
