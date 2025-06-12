import pandas as pd
from typing import Dict

from .tabular_preprocessing import TabularPreprocessingConfig
from src.config.schemas import ExperimentConfig


class TabularDataset:
    def __init__(self, window_size: int, config: ExperimentConfig):
        self.window_size = window_size
        self.input_filename = f"lab_events_{window_size}_days_prior.parquet"
        self.config = config
        self.feature_type = self.config.data.tabular.feature_type
        self.aggregation_window_size = self.config.data.tabular.aggregation_window_size

    def load_data(self) -> pd.DataFrame:
        """Load and concatenate multiple Parquet files."""
        config_obj = TabularPreprocessingConfig.from_defaults(
            window_size=self.window_size,
            feature_type=self.feature_type,
            aggregation_window_size=self.aggregation_window_size,
        )
        config_obj.process_window_file_only()

        if self.feature_type == "numeric":
            numeric_data = config_obj.preprocess_and_save(self.input_filename)
            # numeric_path = config_obj.preprocessed_data_dir / numeric_data_path
            # numeric_data = pd.read_parquet(numeric_path)
            return numeric_data
        elif self.feature_type == "categorical":
            categorical_data = config_obj.preprocess_and_save(self.input_filename)
            # categorical_path = config_obj.preprocessed_data_dir / categorical_data_path
            # categorical_data = pd.read_parquet(categorical_path)
            return categorical_data
        else:
            raise ValueError(
                f"Unsupported feature type: {self.feature_type}. Choose 'numeric' or 'categorical'."
            )
