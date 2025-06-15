import pandas as pd
from typing import Tuple

from .tabular_preprocessing import TabularPreprocessingConfig
from src.config.schemas import ExperimentConfig
from loguru import logger


class TabularDataset:
    def __init__(self, window_size: int, config: ExperimentConfig):
        self.window_size = window_size
        self.input_filename = f"lab_events_{window_size}_days_prior.parquet"
        self.config = config
        self.feature_type = self.config.data.tabular.feature_type
        self.aggregation_window_size = self.config.data.tabular.aggregation_window_size
        self.training_feature = self.config.data.tabular.training_feature
        self.age_threshold = self.config.data.tabular.age_threshold

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and concatenate multiple Parquet files."""
        logger.info(
            f"Preprocessing data from {self.input_filename} with window size {self.window_size} days."
        )
        config_obj = TabularPreprocessingConfig.from_defaults(
            window_size=self.window_size,
            feature_type=self.feature_type,
            aggregation_window_size=self.aggregation_window_size,
            training_feature=self.training_feature,
            age_threshold=self.age_threshold,
        )
        data = config_obj.preprocess_and_save(self.input_filename)
        X_trainable = data.drop(columns=self.training_feature)
        y_target = data[self.training_feature]
        return X_trainable, y_target
        # config_obj.process_window_file_only()
        # if self.feature_type == "numeric":
        #     numeric_data = config_obj.preprocess_and_save(self.input_filename)
        #     X_trainable = numeric_data.drop(columns=self.training_feature)
        #     y_target = numeric_data[self.training_feature]
        #     return X_trainable, y_target

        # elif self.feature_type == "categorical":
        #     categorical_data = config_obj.preprocess_and_save(self.input_filename)
        #     X_trainable = categorical_data.drop(columns=self.training_feature)
        #     y_target = categorical_data[self.training_feature]
        #     return X_trainable, y_target
        # if self.feature_type in ["numeric", "categorical"]:
        #     data = config_obj.preprocess_and_save(self.input_filename)
        #     X_trainable = data.drop(columns=self.training_feature)
        #     y_target = data[self.training_feature]
        #     return X_trainable, y_target

        # else:
        #     raise ValueError(
        #         f"Unsupported feature type: {self.feature_type}. Choose 'numeric' or 'categorical'."
        #     )
