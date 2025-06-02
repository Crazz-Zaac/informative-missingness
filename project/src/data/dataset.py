import pandas as pd
from typing import Dict

from .tabular_preprocessing import TabularPreprocessingConfig


class TabularDataset:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.input_filename = f"lab_events_{window_size}_days_prior.parquet"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and concatenate multiple Parquet files."""
        config_obj = TabularPreprocessingConfig(window_size=self.window_size)
        config_obj.process_all_files()
        numeric_data, categorical_data = config_obj.preprocess_and_save(
            self.input_filename
        )
        return {
            "numeric_data": numeric_data,
            "categorical_data": categorical_data,
        }
