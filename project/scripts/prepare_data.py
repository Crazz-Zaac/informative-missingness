import os
import sys
import argparse

from dotenv import load_dotenv
from pathlib import Path
from loguru import logger

from fetch_demographics_data import DemographicsDataFetcher
from fetch_labevents_data import LabEventsExtractor


class PrepareData:
    def __init__(self):
        # Define paths
        self.dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
        self.output_dir = self.dataset_dir / "raw"
        self.cohort_dir = self.dataset_dir / "MIMIC-IV-data"
        self.temp_dir = self.dataset_dir / "temp"
        self.raw_output_dir = self.dataset_dir / "raw"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.raw_output_dir.mkdir(parents=True, exist_ok=True)

        # Load env and DB connection
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path)
        self.db_host = os.getenv("DB_HOST")
        self.db_port = os.getenv("DB_PORT")
        self.db_user = os.getenv("DB_USER")
        self.db_pass = os.getenv("DB_PASS")
        self.db_name = os.getenv("DB_NAME")

        self.db_url = self._get_engine()

        # Create logs directory
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.remove()
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            level="INFO",
        )

    def _get_engine(self):
        """Create and return SQLAlchemy engine."""
        url = f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        logger.info(
            f"Connecting to database {self.db_name} at {self.db_host}:{self.db_port}..."
        )
        return url


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Extracting data from the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        python scripts/prepare_data.py --cohort <cohort_name (apl or nf)> --days <days (7 or 14)>
        """,
    )
    parser.add_argument(
        "--cohort", choices=["apl", "nf"], required=True, help="Cohort name"
    )

    parser.add_argument(
        "--cohort_target", required=True, help="Cohort's respective target file"
    )

    parser.add_argument("--days", required=True, type=int, help="Number of days")
    args = parser.parse_args()

    try:
        prep = PrepareData()
        # Initialize DemographicsDataFetcher
        demographics_fetcher = DemographicsDataFetcher(
            db_url=prep.db_url,
            logger=logger,
            raw_output_dir=prep.raw_output_dir,
            cohort_dir=prep.cohort_dir,
            cohort=args.cohort,
            cohort_target=args.cohort_target,
        )
        demographics_path = demographics_fetcher.fetch_and_save()

        batch_size = 1000
        # Initialize LabEventsExtractor
        lab_events_extractor = LabEventsExtractor(
            db_url=prep.db_url,
            logger=logger,
            temp_dir=prep.temp_dir,
            raw_output_dir=prep.raw_output_dir,
            demographics_file_path=demographics_path,
            cohort=args.cohort,
            days=args.days,
        )
        lab_events_extractor.run(batch_size=batch_size)
        logger.success(f"Lab events data fetched and saved")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
