import pandas as pd

from pathlib import Path
from psycopg2.extras import execute_values
from sqlalchemy import create_engine


class DemographicsDataFetcher:
    """
    This script fetches demographic data from a specified source and saves it to a local file.
    Merges target data with the cohort data
    """

    def __init__(
        self, db_url, logger, raw_output_dir, cohort_dir, cohort, cohort_target
    ):
        self.logger = logger
        self.raw_output_dir = raw_output_dir
        self.cohort_dir = cohort_dir
        self.cohort = cohort
        self.cohort_target = cohort_target

        # initialize file name
        self.filename = f"{cohort}_with_demographics_data.parquet"

        # define the cohort's target file name
        # this should match the correct file name inside the cohort directory
        # self.cohort_target = f"mimic_iv_labs_{self.cohort}_tr_{self.days}_days.csv.gz"
        self.db_url = db_url
        self.engine = self._get_engine()
        self.logger.info("Initializing DemographicsDataFetcher...")

    def _get_engine(self):
        return create_engine(self.db_url)

        

    def _create_temp_table(self, cursor):
        """Create a temporary table with target data."""
        target_data = pd.read_csv(self.cohort_dir / self.cohort_target)
        self.logger.info("Creating temporary target table...")
        cursor.execute(
            """
            CREATE TEMP TABLE temp_target(
                subject_id INT,
                hadm_id INT,
                admittime TIMESTAMP,
                dischtime TIMESTAMP,
                target INT
            )
            """
        )

        target_data["admittime"] = pd.to_datetime(target_data["admittime"])
        target_data["dischtime"] = pd.to_datetime(target_data["dischtime"])

        values = list(target_data.itertuples(index=False, name=None))
        execute_values(
            cursor,
            "INSERT INTO temp_target (subject_id, hadm_id, admittime, dischtime, target) VALUES %s",
            values,
        )
        self.logger.success("Temporary target table created and populated.")

    def fetch_and_save(self) -> str:
        """Fetch demographics data and save it to a parquet file."""
        self.logger.info("Fetching demographics data...")
        with self.engine.connect() as conn:
            cursor = conn.connection.cursor()

            # Create temporary target table
            self._create_temp_table(cursor)

            # Fetch demographics data
            cursor.execute(
                """
                SELECT DISTINCT
                    c.subject_id,
                    c.hadm_id,
                    c.admittime,
                    c.dischtime,
                    c.target,
                    p.gender,
                    p.anchor_age,
                    a.race
                FROM temp_target c
                JOIN mimiciv_hosp.admissions a ON c.hadm_id = a.hadm_id
                JOIN mimiciv_hosp.patients p ON c.subject_id = p.subject_id
                """
            )
            demographics_data = cursor.fetchall()

        self.logger.success("Demographics data fetched successfully.")
        # Prepare DataFrame
        columns = [
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "target",
            "gender",
            "anchor_age",
            "race",
        ]
        demographics_df = pd.DataFrame(demographics_data, columns=columns)

        # Data type conversions
        demographics_df["admittime"] = pd.to_datetime(demographics_df["admittime"])
        demographics_df["dischtime"] = pd.to_datetime(demographics_df["dischtime"])
        demographics_df["anchor_age"] = pd.to_numeric(
            demographics_df["anchor_age"], errors="coerce"
        )
        demographics_df["target"] = pd.to_numeric(
            demographics_df["target"], errors="coerce"
        )

        # Save to parquet
        output_path = self.raw_output_dir / self.filename
        demographics_df.to_parquet(output_path, index=False)
        rel_path = Path(output_path).relative_to(self.raw_output_dir.parent)
        self.logger.info(f"Demographics data fetched and saved: {rel_path}\n")

        return output_path
