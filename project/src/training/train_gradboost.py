from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import pandas as pd
from wandb import config
import yaml
from loguru import logger

from project.src.models.gradient_boosting import GradientBoostingModel
from src.config.schemas import ExperimentConfig, ModelTypeEnum
from src.data.dataset import TabularDataset


class GradientBoostingTrainer:

    def __init__(self, config: ExperimentConfig):
        logger.info(f"Initializing GradientBoostingTrainer")
        model_hyperparams = config.model.hyperparameters.GradientBoosting

        if model_hyperparams is None:
            raise ValueError(
                f"No hyperparameters provided for model: {ModelTypeEnum.GRADBOOST}"
            )

        self.gradboost_fixed_params = model_hyperparams.fixed_params
        self.gradboost_grid_search_params = model_hyperparams.grid_search_params
        self.dataset = TabularDataset(
            window_size=config.data.tabular.window_size, config=config
        )
        self.random_state = self.gradboost_fixed_params.get("random_state")
        self.model = GradientBoostingModel(config=self.gradboost_fixed_params)

    def run_training(self):
        recalls, f1s, aucs, pr_aucs = [], [], [], []
        best_estimators = []

        logger.info("Loading and preparing the data...")
        X, y, groups = self.dataset.load_and_split_data()

        logger.info("Starting training process...")
        skf = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        logger.info("Fixed model parameters:")
        for key, value in self.gradboost_fixed_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Grid search parameters:")
        for key, value in self.gradboost_grid_search_params.items():
            logger.info(f"  {key}: {value}")

        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups=groups)):
                logger.info(f"Training fold {fold + 1}...")
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Handle class imbalance
                logger.info("Handling class imbalance using RandomOverSampler...")
                ros = RandomOverSampler(random_state=self.random_state)
                X_train_resampled, y_train_resampled = ros.fit_resample(
                    X_train, y_train
                )

                X_train.columns = X_train.columns.astype(str)
                X_val.columns = X_val.columns.astype(str)

                # Grid search with fixed and search parameters
                base_model = self.model.fit(X_train_resampled, y_train_resampled)

                logger.info("Running GridSearchCV...")
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=self.gradboost_grid_search_params,
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=1,
                )
                grid_search.fit(X_train_resampled, y_train_resampled)

                best_model = grid_search.best_estimator_
                best_estimators.append(best_model)

                logger.info(f"Best hyperparameters for fold {fold + 1}:")
                for key, val in grid_search.best_params_.items():
                    logger.info(f"  {key}: {val}")

                # Model evaluation
                y_pred = grid_search.predict(X_val)
                y_proba = grid_search.predict_proba(X_val)[:, 1]

                recalls.append(recall_score(y_val, y_pred, pos_label=1))
                f1s.append(f1_score(y_val, y_pred, pos_label=1))
                aucs.append(roc_auc_score(y_val, y_proba))
                precision, recall, _ = precision_recall_curve(y_val, y_proba)
                pr_aucs.append(auc(recall, precision))

                logger.info(
                    f"Recall: {recalls[-1]:.4f}, F1: {f1s[-1]:.4f}, ROC-AUC: {aucs[-1]:.4f}, PR-AUC: {pr_aucs[-1]:.4f}"
                )

            logger.info("Logging fixed model hyperparameters:")
            for key, value in self.gradboost_fixed_params.items():
                logger.info(f"  {key}: {value}")
            logger.info("Logging grid search parameters:")
            for key, value in self.gradboost_grid_search_params.items():
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
            logger.error(f"Error occurred during training: {e}")
        finally:
            logger.success("Experiment finished.")

    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = ExperimentConfig(**yaml.safe_load(f))
        return cls(config)
