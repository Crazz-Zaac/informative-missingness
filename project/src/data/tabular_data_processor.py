import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from loguru import logger
from .data_preprocessing import TabularPreprocessingConfig


class TabularDataPreprocessor:

    def __init__(self, config: TabularPreprocessingConfig):
        self.config = config
        self.raw_data_dir = config.raw_data_dir
        self.training_data = config.training_data
        self.preprocessed_data_dir = config.preprocessed_data_dir
        self.window_size = config.window_size
        self.aggregation_window_size = config.aggregation_window_size
        self.feature_combinations = config.feature_combinations
        self.feature_type = config.feature_type
        self.training_feature = config.training_feature
        self.age_threshold = config.age_threshold
        self.insurance_type = config.insurance_type
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TabularDataPreprocessor with config: {self.config}")

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

    def _compute_x_m_delta(self, patients_data):
        patients_data = patients_data.copy()
        patients_data["itemid"] = patients_data["itemid"].astype(int)

        all_admissions = patients_data["hadm_id"].unique()
        all_items = patients_data["itemid"].unique()

        # Average values within each bin: x
        x_values = (
            patients_data.groupby(["hadm_id", "itemid", "bin"])["valuenum"]
            .mean()
            .unstack(level=["bin"])
            .rename(columns=lambda c: f"x_(bin{c})")
            .interpolate(method="linear", axis=1, limit_area="inside")
            .ffill(axis=1)
            .bfill(axis=1)
        )

        x_values = x_values.fillna(0)

        # Reindex to include all admissions & items
        x_values = x_values.reindex(
            pd.MultiIndex.from_product([all_admissions, all_items], names=["hadm_id", "itemid"])
        )

        # unstack to match m_df and delta_df structure
        x_df = x_values.unstack(level="itemid")


        # 1 if data is present, else 0: m
        m_values = (
            patients_data.groupby(["hadm_id", "itemid", "bin"])["valuenum"]
            .count()
            .unstack(level=["bin"])
            .rename(columns=lambda c: f"m_(bin{c})")
        )
        # Reindexing to ensure all admission-item combinations are present
        m_values = m_values.reindex(
            pd.MultiIndex.from_product(
                [all_admissions, all_items], names=["hadm_id", "itemid"]
            )
        )
        m_values = m_values.fillna(0).astype(int)
        # unstack to get hadm_id as index and itemid_bin as columns
        m_df = m_values.unstack(level="itemid")

        # Calculate delta for each admission: delta
        delta = np.zeros_like(m_values.values, dtype=float)
        delta[:, 0] = 1 - m_values.values[:, 0]

        for t in range(1, m_values.shape[1]):
            prev = delta[:, t - 1]
            obs = m_values.values[:, t]
            delta[:, t] = np.where(
                obs == 1,  # observed
                0,  # Reset delta if observed
                1 + prev,  # Increment delta if not observed
            )
        # normalize by number of features
        delta = delta / m_values.shape[1]

        # Create delta DataFrame with proper column names
        delta_df = pd.DataFrame(
            delta,
            index=m_values.index,
            columns=[f"delta_bin{col}" for col in m_values.columns],
        )
        # unstack to get hadm_id as index and itemid_bin as columns
        delta_df = delta_df.unstack(level="itemid")

        # Return the requested feature combination
        if self.feature_combinations == "x":
            return x_df
        elif self.feature_combinations == "m":
            return m_df
        elif self.feature_combinations == "delta":
            return delta_df
        elif self.feature_combinations == "x_m":
            return pd.concat([x_df, m_df], axis=1)
        elif self.feature_combinations == "x_delta":
            return pd.concat([x_df, delta_df], axis=1)
        elif self.feature_combinations == "m_delta":
            return pd.concat([m_df, delta_df], axis=1)
        elif self.feature_combinations == "x_m_delta":
            return pd.concat([x_df, m_df, delta_df], axis=1)
        else:
            raise ValueError(
                f"Invalid feature_combinations: {self.feature_combinations}. "
                "Must be one of: 'x', 'm', 'delta', 'x_m', 'x_delta', 'm_delta', 'x_m_delta'"
            )

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

        # If training with race, ensure race column exists
        if self.training_feature == "race":
            if "race" in patients_data.columns:
                logger.info("Mapping race to numerical values")
                # using OneHotEncoder for multiclass
                race_encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore"
                )
                patients_data["race"] = patients_data["race"].apply(self.map_race)
                patients_data["race"] = race_encoder.fit_transform(
                    patients_data[["race"]]
                )
                cohort_data = patients_data[
                    ["hadm_id", "subject_id", "race"]
                ].drop_duplicates()
            else:
                raise ValueError("Missing race column in data; cannot proceed.")

        elif self.training_feature == "gender":
            if "gender" in patients_data.columns:
                logger.info("Mapping Male to 1 and Female to 0")
                patients_data["gender"] = patients_data["gender"].map({"M": 1, "F": 0})
                cohort_data = patients_data[
                    ["hadm_id", "subject_id", "gender"]
                ].drop_duplicates()
            else:
                raise ValueError("Missing gender column in data; cannot proceed.")

        elif self.training_feature == "anchor_age":
            if "anchor_age" in patients_data.columns:
                logger.info(
                    f"Setting anchor_age to 0 if age < {self.age_threshold} otherwise 1"
                )
                patients_data["anchor_age"] = (
                    patients_data["anchor_age"] >= self.age_threshold
                ).astype(int)
                cohort_data = patients_data[
                    ["hadm_id", "subject_id", "anchor_age"]
                ].drop_duplicates()
            else:
                raise ValueError("Missing anchor_age column in data; cannot proceed.")

        elif self.training_feature == "target":
            if "target" in patients_data.columns:
                cohort_data = patients_data[
                    ["hadm_id", "subject_id", "target"]
                ].drop_duplicates()
            else:
                raise ValueError("Missing target column in data; cannot proceed.")

        else:
            raise ValueError(f"Unknown training feature: {self.training_feature}")

        # TODO: descritize the time based on noon and midnight

        # descritize the time into bins based on aggregation_window_size
        # Convert charttime and dischtime to datetime if not already and calculate hours before discharge
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

        # prepare dataframe based on feature combinations
        logger.info(f"Computing the feature combinations: {self.feature_combinations}")
        df_ts = self._compute_x_m_delta(patients_data=patients_data)
        logger.info(f"Computed features with shape: {df_ts.shape}")

        # flattening the column and removing the multiIndex columns
        df_ts.columns = [
            "_".join(map(str, col)) if isinstance(col, tuple) else col
            for col in df_ts.columns
        ]

        # setting hadm_id as index and reindexing training feature data
        target_data = cohort_data.set_index("hadm_id")[self.training_feature].reindex(
            df_ts.index
        )

        groups = (
            cohort_data.set_index("hadm_id").reindex(df_ts.index)["subject_id"].values
        )

        # Generate output filenames
        base_name = os.path.splitext(input_filename)[0]  # removes .parquet
        file_saved_to = f"{base_name}_{self.training_feature}.parquet"
        df_ts.to_parquet(
            os.path.join(self.preprocessed_data_dir, file_saved_to), index=False
        )
        logger.info(
            f"Training Data: {self.feature_combinations} - Data shape: {df_ts.shape}"
        )
        logger.info(
            f"Data prepared for {self.training_feature} and saved to {file_saved_to}"
        )

        # Return the output filenames
        return df_ts, target_data, groups

    def process_training_data_file(self):
        pattern = (
            f"{self.training_data}_lab_events_{self.window_size}_days_prior.parquet"
        )
        for file in self.raw_data_dir.glob("*.parquet"):
            if file.name == pattern:
                logger.info(f"Processing file: {file.name}")
                self.preprocess_and_save(file.name)
                return
        raise FileNotFoundError(f"No file matching '{pattern}' in {self.raw_data_dir}")
