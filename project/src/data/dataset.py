import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import KNNImputer
from loguru import logger

from .tabular_preprocessing import TabularPreprocessingConfig
from src.config.schemas import ExperimentConfig



class TabularDataset:
    def __init__(self, window_size: int, config: ExperimentConfig):
        self.window_size = window_size
        self.input_filename = f"lab_events_{window_size}_days_prior.parquet"
        self.config = config
        self.feature_type = self.config.data.tabular.feature_type
        self.aggregation_window_size = self.config.data.tabular.aggregation_window_size
        self.training_feature = self.config.data.tabular.training_feature
        self.age_threshold = self.config.data.tabular.age_threshold
        self.insurance_type = self.config.data.tabular.insurance_type
        self.random_state = 42

    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Preprocessing data from {self.input_filename} with window size {self.window_size} days.")

        config_obj = TabularPreprocessingConfig.from_defaults(
            window_size=self.window_size,
            feature_type=self.feature_type,
            aggregation_window_size=self.aggregation_window_size,
            training_feature=self.training_feature,
            age_threshold=self.age_threshold,
            insurance_type=self.insurance_type,
        )

        data = config_obj.preprocess_and_save(self.input_filename)
        data = data.reset_index()
        # dump the data
        if data.empty:
            raise ValueError("Loaded data is empty. Please check the dataset.")

        X = data.drop(columns=[self.training_feature])
        y = data[self.training_feature]
        
        sgkf = StratifiedGroupKFold(
            n_splits=5, 
            shuffle=True,
            random_state=self.random_state
        )
        X = data.drop(columns=[self.training_feature])
        y = data[self.training_feature]
        groups = data['subject_id']
        train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        logger.info("Imputing data using KNN Imputer")
        imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        return X_train_imputed, X_test_imputed, y_train, y_test

