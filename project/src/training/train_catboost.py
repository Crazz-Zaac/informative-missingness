import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
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
from loguru import logger
from pathlib import Path

from src.models.catboost import CatBoostModel
from src.data.dataset import TabularDataset
from src.config.schemas import ExperimentConfig, ModelTypeEnum


class CatBoostTrainer:
    """Trainer for CatBoost model"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        logger.info("Initializing CatBoostTrainer")

        model_hyperparams = self.config.model.hyperparameters.CatBoost
        if model_hyperparams is None:
            raise ValueError(
                f"No hyperparameters provided for model: {ModelTypeEnum.CatBoost}"
            )

        self.fixed_params = model_hyperparams.fixed_params

        self.grid_search_params = model_hyperparams.grid_search_params

        self.dataset = TabularDataset(
            window_size=self.config.data.tabular.window_size, config=self.config
        )
        self.model = CatBoostModel(config=self.fixed_params)

    def run_training(self):
        recalls, f1_scores, aucs, pr_aucs = [], [], [], []
        best_estimators = []
        best_params = []

        logger.info("Loading and preparing the data...")
        X, y, groups = self.dataset.load_and_split_data()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        try:
            for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
                logger.info(f"\n=== Running Fold {fold + 1} ===")

                X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # sampling the data
                logger.info("Applying Random Over Sampling...")
                oversample = RandomOverSampler(
                    sampling_strategy="minority", random_state=42
                )
                X_train, y_train = oversample.fit_resample(X_train, y_train)

                X_train.columns = X_train.columns.astype(str)
                X_test.columns = X_test.columns.astype(str)

                train_pool = catboost.Pool(data=X_train, label=y_train)
                test_pool = catboost.Pool(data=X_test, label=y_test)

                logger.info("Running Grid Search...")
                base_model = self.model._initialize_model()
                grid_search_results = base_model.grid_search(
                    self.grid_search_params,
                    train_pool,
                    cv=3,
                    shuffle=False,
                    verbose=3,
                    plot=True,
                )
                best_params_fold = grid_search_results["params"]
                best_estimators.append(best_params_fold)

                logger.info(
                    f"Best hyperparameters from GridSearchCV (Fold {fold + 1}):"
                )
                for key, val in best_params_fold.items():
                    logger.info(f"  {key}: {val}")

                final_model = CatBoostClassifier(
                    **self.fixed_params,
                    **best_params_fold,
                    task_type="GPU",
                    eval_metric="AUC",
                    custom_metric=["Precision", "Recall", "F1", "PRAUC"],
                )

                final_model.fit(
                    train_pool,
                    eval_set=test_pool,
                    verbose=100,
                    plot=True,
                    metric_period=100,
                )

                # Evaluation
                y_pred_proba = final_model.predict_proba(test_pool)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)

                # metrics calculation
                fold_recall = recall_score(y_test, y_pred)
                fold_f1 = f1_score(y_test, y_pred)
                fold_auc = roc_auc_score(y_test, y_pred_proba)
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                fold_pr_auc = auc(recall, precision)

                recalls.append(fold_recall)
                f1_scores.append(fold_f1)
                aucs.append(fold_auc)
                pr_aucs.append(fold_pr_auc)

            logger.info("\n=== Training Summary ===")
            logger.success(
                f"Mean Recall: {pd.Series(recalls).mean():.4f} ± {pd.Series(recalls).std():.4f}"
            )
            logger.success(
                f"Mean F1:     {pd.Series(f1_scores).mean():.4f} ± {pd.Series(f1_scores).std():.4f}"
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
