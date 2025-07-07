from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from sklearn.impute import KNNImputer
from typing import List
from sklearn.preprocessing import LabelEncoder
from loguru import logger


class TabularPreprocessingConfig(BaseModel):
    raw_data_dir: Path
    preprocessed_data_dir: Path
    window_size: int
    aggregation_window_size: int  # aggregation by hours, e.g., 2 hours
    feature_type: str
    training_feature: str
    age_threshold: int
    insurance_type: str

    @classmethod
    # Create a configuration instance with default paths and specified window size.
    def from_defaults(
        cls,
        window_size: int,
        aggregation_window_size: int,
        feature_type: str,
        training_feature: str,
        age_threshold: int,  # Default age threshold for filtering patients
        insurance_type: str,
    ) -> "TabularPreprocessingConfig":
        """Create a configuration instance with default paths and specified window size."""
        parent_dir = Path(__file__).parent.parent.parent
        return cls(
            raw_data_dir=parent_dir / "dataset" / "raw",
            preprocessed_data_dir=parent_dir / "dataset" / "preprocessed_tabular",
            window_size=window_size,
            aggregation_window_size=aggregation_window_size,  # Default aggregation window size (e.g., 2 hours)
            feature_type=feature_type,  # e.g., "numeric" or "categorical"
            training_feature=training_feature,  # e.g., "target"
            age_threshold=age_threshold,  # Default age threshold for filtering patients
            insurance_type=insurance_type,  #
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
        logger.info(f"Loading data from {filename} and removing any duplicate rows.")
        patients_data = pd.read_parquet(self.raw_data_dir / filename)
        patients_data = patients_data.drop_duplicates(
            subset=["subject_id", "hadm_id", "charttime", "itemid"]
        )
        patients_data.loc[:, "charttime"] = pd.to_datetime(patients_data["charttime"])
        patients_data.loc[:, "dischtime"] = pd.to_datetime(patients_data["dischtime"])
        return patients_data

    def map_race(self, race):
        if pd.isna(race):
            return "Unknown or Not Reported"

        race = race.upper()

        if "HISPANIC" in race or "LATINO" in race or "SOUTH AMERICAN" in race:
            return "Hispanic or Latino"
        elif "WHITE" in race:
            return "White"
        elif "BLACK" in race or "AFRICAN" in race:
            return "Black or African American"
        elif "ASIAN" in race:
            return "Asian"
        elif "PACIFIC ISLANDER" in race or "NATIVE HAWAIIAN" in race:
            return "Native Hawaiian or Other Pacific Islander"
        elif "AMERICAN INDIAN" in race or "ALASKA NATIVE" in race:
            return "American Indian or Alaska Native"
        elif "DECLINED" in race or "UNABLE" in race or "UNKNOWN" in race:
            return "Unknown or Not Reported"
        else:
            return "Other"

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
        non_trainable_features = ["gender", "anchor_age", "race", "target"]
        patients_data = self.load_data(input_filename)

        # drop features that are not needed for training
        columns_to_drop = [
            f for f in non_trainable_features if f != self.training_feature
        ]
        patients_data = patients_data.drop(columns=columns_to_drop, errors="ignore")
        patients_data = patients_data.dropna(subset=["charttime", "dischtime"])

        if self.training_feature == "race":
            # map race to numerical values
            logger.info("Mapping race to numerical values")
            race_encoder = LabelEncoder()
            patients_data["race"] = patients_data["race"].apply(self.map_race)
            patients_data["race"] = race_encoder.fit_transform(patients_data["race"])

        if self.training_feature == "gender":
            # map M and F to 1 and 0
            logger.info("Mapping Male to 1 and Female to 0")
            patients_data["gender"] = patients_data["gender"].map({"M": 1, "F": 0})

        if self.training_feature == "anchor_age":
            logger.info(
                f"Setting anchor_age to 0 if age < {self.age_threshold} otherwise 1"
            )
            patients_data["anchor_age"] = (
                patients_data["anchor_age"] >= self.age_threshold
            ).astype(int)

        if "target" in patients_data.columns:
            cohort_data = patients_data[
                ["hadm_id", "subject_id", "target"]
            ].drop_duplicates()
        else:
            raise ValueError("Missing target column in data; cannot proceed.")

        patients_data["hours_before_discharge"] = (
            patients_data["dischtime"] - patients_data["charttime"]
        ).dt.total_seconds() / 3600
        patients_data["bin"] = (
            patients_data["hours_before_discharge"] // self.aggregation_window_size
        )
        patients_data = patients_data[patients_data["bin"].notna()]
        patients_data.loc[:, "bin"] = patients_data["bin"].astype(int)
        patients_data = patients_data[patients_data["bin"] >= 0]

        patients_data["itemid_bin"] = (
            patients_data["itemid"].astype(int).astype(str)
            + "_"
            + patients_data["bin"].astype(str)
        )

        patients_data = (
            patients_data.groupby(["hadm_id", "itemid", "bin"])["valuenum"]
            .mean()
            .unstack(level=-1)
            .interpolate(method="linear", axis=1, limit_area="inside")
            .ffill(axis=1)
            .bfill(axis=1)
            .reset_index()
        )

        patients_data = patients_data.set_index(["hadm_id", "itemid"])
        wide_df = patients_data.unstack(level=-1)
        # Only swap levels if columns is a MultiIndex
        if isinstance(wide_df.columns, pd.MultiIndex):
            wide_df.columns = wide_df.columns.swaplevel(0, 1)
        wide_df = wide_df.sort_index(axis=1)
        wide_df.columns = [
            "_".join(map(str, col)) if isinstance(col, tuple) else str(col)
            for col in wide_df.columns
        ]
        patients_data = wide_df.copy()
        
        target_data = cohort_data.set_index("hadm_id")["target"].reindex(
            patients_data.index
        )

        groups = (
            cohort_data.set_index("hadm_id")
            .reindex(patients_data.index)["subject_id"]
            .values
        )

        # Generate output filenames
        base_name = os.path.splitext(input_filename)[0]  # removes .parquet
        numeric_output = f"{base_name}_numeric.parquet"
        patients_data.to_parquet(
            os.path.join(self.preprocessed_data_dir, numeric_output), index=False
        )
        logger.info(f"Numeric data shape: {patients_data.shape}")

        # Return the output filenames
        return patients_data, target_data, groups

        # if self.feature_type == "numeric":
        #     logger.info("Processing numeric data")
        #     processed_numeric_data = self.prepare_numeric_data(patients_data)
        #     # Merge and save
        #     numeric_output = f"{base_name}_numeric.parquet"
        #     # Save files
        #     processed_numeric_data.to_parquet(
        #         os.path.join(self.preprocessed_data_dir, numeric_output), index=False
        #     )
        #     # log statistics of the numeric data
        #     logger.info(f"Numeric data shape: {processed_numeric_data.shape}")

        #     # Return the output filenames
        #     return processed_numeric_data

        # if self.feature_type == "categorical":
        #     logger.info("Processing categorical data")
        #     categorical_data = self.prepare_categorical_data(patients_data)
        #     processed_categorical_data = categorical_data.join(targets).reset_index()
        #     categorical_output = f"{base_name}_categorical.parquet"
        #     processed_categorical_data.to_parquet(
        #         os.path.join(self.preprocessed_data_dir, categorical_output),
        #         index=False,
        #     )
        #     logger.info(f"Categorical data shape: {processed_categorical_data.shape}")
        #     return processed_categorical_data

    def process_window_file_only(self):
        """Process the file that matches the specified window size"""
        pattern = re.compile(rf".*_{self.window_size}_days_prior\.parquet")

        for file in self.raw_data_dir.glob("*.parquet"):
            if pattern.match(file.name):
                logger.info(f"Processing file: {file.name}")
                self.preprocess_and_save(file.name)
                return  # process only the first match

        raise FileNotFoundError(
            f"No file found in {self.raw_data_dir} for window size {self.window_size}"
        )
