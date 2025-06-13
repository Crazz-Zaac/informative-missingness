from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


class TabularPreprocessingConfig(BaseModel):
    raw_data_dir: Path
    preprocessed_data_dir: Path
    window_size: int
    aggregation_window_size: int  # aggregation by hours, e.g., 2 hours
    feature_type: str

    @classmethod
    # Create a configuration instance with default paths and specified window size.
    def from_defaults(
        cls,
        window_size: int,
        aggregation_window_size: int,
        feature_type: str,
    ) -> "TabularPreprocessingConfig":
        """Create a configuration instance with default paths and specified window size."""
        parent_dir = Path(__file__).parent.parent.parent
        return cls(
            raw_data_dir=parent_dir / "dataset" / "raw",
            preprocessed_data_dir=parent_dir / "dataset" / "preprocessed_tabular",
            window_size=window_size,
            aggregation_window_size=aggregation_window_size,  # Default aggregation window size (e.g., 2 hours)
            feature_type=feature_type,  # e.g., "numeric" or "categorical"
        )

    def assign_time_bin(self, hours_before_discharge, window_hours: int) -> np.ndarray:
        """Assign records to fixed time bins (e.g., 0-6h, 6-12h)."""
        return np.floor(hours_before_discharge / window_hours) * window_hours

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
        patients_data.loc[:, "charttime"] = pd.to_datetime(patients_data["charttime"])
        patients_data.loc[:, "dischtime"] = pd.to_datetime(patients_data["dischtime"])
        return patients_data

    def prepare_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare numeric data by pivoting and filling missing values."""
        data = data.copy()
        numeric_data = data.pivot_table(
            index="hadm_id",
            columns="feature_id",
            values="valuenum",
            aggfunc="mean",
        )
        numeric_data = numeric_data.fillna(-999)
        # numeric_data = numeric_data.sort_index(axis=1)
        # numeric_data = numeric_data.ffill(axis=1).bfill(axis=1)
        return numeric_data

    def prepare_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare categorical data by pivoting and creating binary features."""
        data = data.copy()
        data.loc[:, "has_measurement"] = 1
        categorical_data = data.pivot_table(
            index="hadm_id",
            columns="feature_id",
            values="has_measurement",
            aggfunc="max",
            fill_value=0,
        )
        categorical_data = categorical_data.sort_index(axis=1)
        categorical_data.columns = pd.Index(
            [col + "_measured" for col in categorical_data.columns]
        )
        return categorical_data

    def preprocess_and_save(self, input_filename: str):
        """Process a single file and save results"""
        # Load data
        patients_data = self.load_data(input_filename)

        # Calculate hours before discharge
        patients_data.loc[:, "days_before_discharge"] = (
            patients_data["dischtime"] - patients_data["charttime"]
        ).dt.total_seconds() / 3600  # convert to hours

        # aggregating all lab events per admission into a single row with many columns
        filtering_hours = self.window_size * 24  # convert days to hours
        patients_data = patients_data[
            (patients_data["days_before_discharge"] >= 0)
            & (patients_data["days_before_discharge"] < filtering_hours)
        ].copy()

        # Assign time bins
        patients_data.loc[:, "time_bin"] = self.assign_time_bin(
            patients_data["days_before_discharge"], self.aggregation_window_size
        )
        # Convert time_bin to string for feature_id
        patients_data.loc[:, "feature_id"] = (
            "itemid_"
            + patients_data["itemid"].astype(str)
            + "_last_"
            + patients_data["time_bin"].astype(str)
            + "_hours_prior"
        )

        # Handle targets
        targets = (
            patients_data[["hadm_id", "target"]].drop_duplicates().set_index("hadm_id")
        )
        # Generate output filenames
        base_name = os.path.splitext(input_filename)[0]  # removes .parquet

        if self.feature_type == "numeric":
            # Prepare data
            numeric_data = self.prepare_numeric_data(patients_data)
            # Merge and save
            numeric_output = f"{base_name}_numeric.parquet"
            # Join targets with numeric data
            processed_numeric_data = numeric_data.join(targets).reset_index()
            # Save files
            processed_numeric_data.to_parquet(
                os.path.join(self.preprocessed_data_dir, numeric_output), index=False
            )
            # Return the output filenames
            return processed_numeric_data

        if self.feature_type == "categorical":
            categorical_data = self.prepare_categorical_data(patients_data)
            processed_categorical_data = categorical_data.join(targets).reset_index()
            categorical_output = f"{base_name}_categorical.parquet"
            processed_categorical_data.to_parquet(
                os.path.join(self.preprocessed_data_dir, categorical_output),
                index=False,
            )
            return processed_categorical_data

    def process_window_file_only(self):
        """Process the file that matches the specified window size"""
        pattern = re.compile(rf".*_{self.window_size}_days_prior\.parquet")

        for file in self.raw_data_dir.glob("*.parquet"):
            if pattern.match(file.name):
                print(f"Processing: {file.name}")
                self.preprocess_and_save(file.name)
                return  # process only the first match

        raise FileNotFoundError(
            f"No file found in {self.raw_data_dir} for window size {self.window_size}"
        )


# if __name__ == "__main__":
#     # Process both 7-day and 14-day files
#     config = TabularPreprocessingConfig()
#     config.process_all_files()

# If you need different window sizes for different files:
# config_7 = TabularPreprocessingConfig(window_size=7)
# config_7.preprocess_and_save("your_7_days_file.parquet")

# config_14 = TabularPreprocessingConfig(window_size=14)
# config_14.preprocess_and_save("your_14_days_file.parquet")
