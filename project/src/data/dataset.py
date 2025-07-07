import pandas as pd
from typing import Tuple
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
        

    def load_and_split_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:  # pd.Series, pd.Series
        logger.info(
            f"Preprocessing data from {self.input_filename} with window size {self.window_size} days."
        )

        config_obj = TabularPreprocessingConfig.from_defaults(
            window_size=self.window_size,
            feature_type=self.feature_type,
            aggregation_window_size=self.aggregation_window_size,
            training_feature=self.training_feature,
            age_threshold=self.age_threshold,
            insurance_type=self.insurance_type,
        )

        X, y, groups = config_obj.preprocess_and_save(self.input_filename)
        
        return X, y, groups

