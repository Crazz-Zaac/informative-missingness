import pandas as pd
from typing import Dict

from .tabular_preprocessing import TabularPreprocessingConfig
from src.config.schemas import ExperimentConfig


class TabularDataset:
    def __init__(self, window_size: int, config: ExperimentConfig):
        self.window_size = window_size
        self.input_filename = f"lab_events_{window_size}_days_prior.parquet"
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load and concatenate multiple Parquet files."""
        config_obj = TabularPreprocessingConfig.from_defaults(
            window_size=self.window_size
        )
        config_obj.process_window_file_only()
        numeric_data_path, categorical_data_path = config_obj.preprocess_and_save(
            self.input_filename
        )
        feature_type = self.config.data.tabular.feature_type
        if feature_type == "numeric":
            numeric_path = config_obj.preprocessed_data_dir / numeric_data_path
            numeric_data = pd.read_parquet(numeric_path)
            return numeric_data
        elif feature_type == "categorical":
            categorical_path = config_obj.preprocessed_data_dir / categorical_data_path
            categorical_data = pd.read_parquet(categorical_path)
            return categorical_data
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}. Choose 'numeric' or 'categorical'.")
