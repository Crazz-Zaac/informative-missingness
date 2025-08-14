from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    precision_recall_curve,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
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

        self.catboost_fixed_params = model_hyperparams.fixed_params

        self.catboost_search_params = model_hyperparams.grid_search_params

        self.dataset = TabularDataset(
            window_size=self.config.data.tabular.window_size, config=self.config
        )
        self.model = CatBoostModel(config=self.catboost_fixed_params)

    def run_training(self):
        recalls, f1_scores, aucs, pr_aucs = [], [], [], []
        best_estimators = []

        logger.info("Loading and preparing the data...")
        X, y, groups = self.dataset.load_and_split_data()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
        logger.info("Fixed model hyperparameters:")
        for key, value in self.catboost_fixed_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("Grid search parameters:")
        for key, value in self.catboost_search_params.items():
            logger.info(f"  {key}: {value}")
        
        try:
            for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
                logger.info(f"\n=== Running Fold {fold + 1} ===")

                X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # sampling the data
                logger.info("Applying Random Over Sampling...")
                oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                
                X_train.columns = X_train.columns.astype(str)
                X_test.columns = X_test.columns.astype(str)

                logger.info("Running Grid Search...")
                base_model = self.model._initialize_model()
                grid_search = 
