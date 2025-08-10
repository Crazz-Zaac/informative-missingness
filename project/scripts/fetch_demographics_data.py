from loguru import logger
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path
from psycopg2.extras import execute_values


class DemographicsDataFetcher:
    """
    This script fetches demographic data from a specified source and saves it to a local file.
    Merges target data with the cohort data
    """
    def __init__(self):
        logger.info("Initializing DemographicsDataFetcher...")

        # Load environment variables
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path)

        # DB connection parameters
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_user = os.getenv("DB_USER")
        self.db_pass = os.getenv("DB_PASS")
        self.db_name = os.getenv("DB_NAME")

        # Paths
        self.dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
        self.output_dir = self.dataset_dir / "raw"
        self.cohort_dir = self.dataset_dir / "MIMIC-IV-data"
        self.filename = "aplasia_with_demographics_data.parquet"
        self.cohort_target = "mimic_iv_target_apl_tr_14_days.csv.gz"

        self.engine = self._get_engine()

    def _get_engine(self):
        """Create and return SQLAlchemy engine."""
        url = f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        logger.info(
            f"Connecting to database {self.db_name} at {self.db_host}:{self.db_port}..."
        )
        return create_engine(url)

    def _create_temp_table(self, cursor):
        """Create a temporary table with target data."""
        target_data = pd.read_csv(
            self.cohort_dir / self.cohort_target
        )
        logger.info("Creating temporary target table...")
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
        logger.success("Temporary target table created and populated.")

    def fetch_and_save(self):
        """Fetch demographics data and save it to a parquet file."""
        logger.info("Fetching demographics data...")
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
        
        # Close the cursor and connection
        logger.success("Demographics data fetched successfully.")
        # conn.connection.commit()
        # cursor.close()
        # conn.close()

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
        output_path = self.output_dir / self.filename
        demographics_df.to_parquet(output_path, index=False)
        logger.info(f"Demographics data saved to {output_path}")


if __name__ == "__main__":
    fetcher = DemographicsDataFetcher()
    fetcher.fetch_and_save()
