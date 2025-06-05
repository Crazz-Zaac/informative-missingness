import pandas as pd
from typing import Dict

from .tabular_preprocessing import TabularPreprocessingConfig


class TabularDataset:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.input_filename = f"lab_events_{window_size}_days_prior.parquet"
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and concatenate multiple Parquet files."""
        config_obj = TabularPreprocessingConfig.from_defaults(window_size=self.window_size)
        config_obj.process_all_files()
        numeric_data_path, categorical_data_path = config_obj.preprocess_and_save(
            self.input_filename
        )

        numeric_path = config_obj.preprocessed_data_dir / numeric_data_path
        categorical_path = config_obj.preprocessed_data_dir / categorical_data_path

        print(f"[DEBUG] Reading numeric data from: {numeric_path}")
        print(f"[DEBUG] Reading categorical data from: {categorical_path}")

        return {
            "numeric_data": pd.read_parquet(numeric_path),
            "categorical_data": pd.read_parquet(categorical_path),
        }

