import pandas as pd
from typing import Tuple
from loguru import logger

from .data_preprocessing import TabularPreprocessingConfig
from src.data.tabular_data_processor import TabularDataPreprocessor
from src.config.schemas import ExperimentConfig


class TabularDataset:
    def __init__(self, window_size: int, config: ExperimentConfig):
        self.window_size = window_size
        self.input_filename = f"{config.data.tabular.training_data}_lab_events_data_{window_size}_days_prior.parquet"
        self.config = config
        self.feature_combinations = self.config.data.tabular.feature_combinations
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
            training_data=self.config.data.tabular.training_data,
            window_size=self.window_size,
            feature_type=self.feature_type,
            feature_combinations=self.feature_combinations,
            aggregation_window_size=self.aggregation_window_size,
            training_feature=self.training_feature,
            age_threshold=self.age_threshold,
            insurance_type=self.insurance_type,
        )

        data_processor = TabularDataPreprocessor(config=config_obj)
        logger.info("Starting data preprocessing...")
        X, y, groups = data_processor.preprocess_and_save(self.input_filename)

        return X, y, groups

