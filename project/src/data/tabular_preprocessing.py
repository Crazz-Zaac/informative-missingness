from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


class TabularPreprocessingConfig(BaseModel):
    # Directory paths for raw and preprocessed data
    parent_dir: Path = Path(__file__).parent.parent.parent
    raw_data_dir: Path = Field(
        default=parent_dir / "dataset" / "raw",
        description="Directory containing raw Parquet files",
    )
    preprocessed_data_dir: Path = Field(
        default=parent_dir / "dataset" / "preprocessed_tabular",   
        description="Directory to save processed Parquet files",
    )
    window_size: int = Field(
        default=7,
        description="Size of the sliding window for time series data (e.g., 7 or 14 days)",
    )        

    def extract_window_size(self, filename: str) -> int:
        """Extract window_size (e.g., 7 or 14) from filenames like '*_7_days.parquet'."""
        match = re.search(r"_(\d+)_days_prior", filename)
        if not match:
            raise ValueError(
                f"Filename '{filename}' must follow pattern '*_X_days_prior.parquet'"
            )
        return int(match.group(1))

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load patient data from a specific Parquet file."""
        patients_data = pd.read_parquet(self.raw_data_dir / filename)
        patients_data["charttime"] = pd.to_datetime(patients_data["charttime"])
        patients_data["dischtime"] = pd.to_datetime(patients_data["dischtime"])
        return patients_data

    def prepare_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Same as your existing method"""
        numeric_data = data.pivot_table(
            index="hadm_id",
            columns="feature_id",
            values="valuenum",
            aggfunc="mean",
            fill_value=np.nan,
        )
        numeric_data = numeric_data.sort_index(axis=1)
        numeric_data = numeric_data.ffill(axis=1).bfill(axis=1)
        return numeric_data

    def prepare_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Same as your existing method"""
        data["has_measurement"] = 1
        categorical_data = data.pivot_table(
            index="hadm_id",
            columns="feature_id",
            values="has_measurement",
            aggfunc="max",
            fill_value=0,
        )
        categorical_data = categorical_data.sort_index(axis=1)
        categorical_data.columns = pd.Index([col + "_measured" for col in categorical_data.columns])
        return categorical_data

    def preprocess_and_save(self, input_filename: str):
        """Process a single file and save results"""
        # Load data
        patients_data = self.load_data(input_filename)

        # Calculate days before discharge
        patients_data["days_before_discharge"] = (
            patients_data["dischtime"] - patients_data["charttime"]
        ).dt.days

        # Filter data based on window_size
        patients_data = patients_data[
            (patients_data["days_before_discharge"] >= 0)
            & (patients_data["days_before_discharge"] < self.window_size)
        ]

        # Create feature identifiers
        patients_data["feature_id"] = (
            "itemid_"
            + patients_data["itemid"].astype(str)
            + "_day_"
            + patients_data["days_before_discharge"].astype(str)
        )

        # Prepare data
        numeric_data = self.prepare_numeric_data(patients_data)
        categorical_data = self.prepare_categorical_data(patients_data)

        # Handle targets
        targets = (
            patients_data[["hadm_id", "target"]].drop_duplicates().set_index("hadm_id")
        )

        # Merge and save
        processed_numeric_data = numeric_data.join(targets).reset_index()
        processed_categorical_data = categorical_data.join(targets).reset_index()

        # Generate output filenames
        base_name = os.path.splitext(input_filename)[0]  # removes .parquet
        numeric_output = f"{base_name}_numeric.parquet"
        categorical_output = f"{base_name}_categorical.parquet"

        # Save files
        processed_numeric_data.to_parquet(
            os.path.join(self.preprocessed_data_dir, numeric_output), index=False
        )
        processed_categorical_data.to_parquet(
            os.path.join(self.preprocessed_data_dir, categorical_output), index=False
        )

        print(
            f"Processed {input_filename} -> {numeric_output} and {categorical_output}"
        )
        return numeric_output, categorical_output

    def process_all_files(self):
        """Process all Parquet files in the raw directory"""
        for file in self.raw_data_dir.glob("*.parquet"):
            self.preprocess_and_save(file.name)


# if __name__ == "__main__":
#     # Process both 7-day and 14-day files
#     config = TabularPreprocessingConfig()
#     config.process_all_files()

    # If you need different window sizes for different files:
    # config_7 = TabularPreprocessingConfig(window_size=7)
    # config_7.preprocess_and_save("your_7_days_file.parquet")

    # config_14 = TabularPreprocessingConfig(window_size=14)
    # config_14.preprocess_and_save("your_14_days_file.parquet")
