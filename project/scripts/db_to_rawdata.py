"""
MIMIC Data Extractor

A unified script to extract lab events data with demographics for medical cohorts.
Always extracts demographics first, then lab events, and combines them into a single parquet file.

Usage:
    python db_to_rawdata.py --cohort aplasia --days 7
    python db_to_rawdata.py --cohort aplasia --days 14
    python db_to_rawdata.py --cohort heart_failure --days 7
"""

import argparse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from pathlib import Path
import multiprocessing as mp
import sys
from loguru import logger
from psycopg2.extras import execute_values


class MimicDataExtractor:
    """Extractor for demographics and lab events data."""
    
    def __init__(self, cohort_name: str):
        self.cohort_name = cohort_name.lower()
        self._setup_logging()
        self._setup_paths()
        self._load_environment()
        self._setup_database()
        
    def _setup_logging(self):
        """Configure logging for both console and file output."""
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Console logging
        logger.add(
            sys.stderr,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO"
        )
        
        # File logging
        logger.add(
            log_dir / f"{self.cohort_name}_extraction_{{time:YYYY-MM-DD_HH-mm-ss}}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days"
        )
        
        # Special logging for patients with no lab events
        self.no_labs_logger = logger.bind(category="no_labs")
        logger.add(
            log_dir / f"{self.cohort_name}_patients_no_lab_events_{{time:YYYY-MM-DD_HH-mm-ss}}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: record["extra"].get("category") == "no_labs",
            level="INFO"
        )
        
        logger.info(f"Logging setup complete - logs will be saved to: {log_dir}")
    
    def _setup_paths(self):
        """Setup directory paths and file names."""
        base_dir = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parents[1].exists() else Path.cwd()
        
        self.dataset_dir = base_dir / "dataset"
        self.output_dir = self.dataset_dir / "raw"
        self.temp_dir = self.dataset_dir / "temp"
        self.cohort_dir = self.dataset_dir / "MIMIC-IV-data"
        
        # Create directories
        for dir_path in [self.output_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # File naming conventions
        self.target_file = f"mimic_iv_target_{self.cohort_name}_tr_14_days.csv.gz"
        self.demographics_file = f"{self.cohort_name}_with_demographics_data.parquet"
        
    def _load_environment(self):
        """Load database configuration from environment."""
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"
        if not dotenv_path.exists():
            dotenv_path = Path.cwd() / ".env"
        
        load_dotenv(dotenv_path)
        
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASS"),
            'database': os.getenv("DB_NAME")
        }
        
        # Validate configuration
        missing_configs = [k for k, v in self.db_config.items() if not v]
        if missing_configs:
            raise ValueError(f"Missing database configuration: {missing_configs}")
    
    def _setup_database(self):
        """Create database engine."""
        url = (f"postgresql+psycopg2://{self.db_config['user']}:{self.db_config['password']}"
               f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        self.engine = create_engine(url)
        logger.info(f"Connected to database: {self.db_config['database']}")
    
    def _extract_demographics(self) -> pd.DataFrame:
        """Extract demographics data and return as DataFrame."""
        logger.info("=" * 60)
        logger.info(f"EXTRACTING DEMOGRAPHICS FOR COHORT: {self.cohort_name.upper()}")
        logger.info("=" * 60)
        
        # Load target data
        target_path = self.cohort_dir / self.target_file
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target_path}")
        
        target_data = pd.read_csv(target_path)
        logger.info(f"Loaded target data: {len(target_data)} records")
        
        with self.engine.connect() as conn:
            cursor = conn.connection.cursor()
            
            # Create temporary table
            self._create_temp_table(cursor, target_data)
            
            # Fetch demographics data
            cursor.execute("""
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
            """)
            
            demographics_data = cursor.fetchall()
            logger.success(f"Demographics data fetched: {len(demographics_data)} records")
        
        # Create DataFrame
        columns = ["subject_id", "hadm_id", "admittime", "dischtime", "target", 
                  "gender", "anchor_age", "race"]
        demographics_df = pd.DataFrame(demographics_data, columns=columns)
        
        # Data type conversions
        demographics_df["admittime"] = pd.to_datetime(demographics_df["admittime"])
        demographics_df["dischtime"] = pd.to_datetime(demographics_df["dischtime"])
        demographics_df["anchor_age"] = pd.to_numeric(demographics_df["anchor_age"], errors="coerce")
        demographics_df["target"] = pd.to_numeric(demographics_df["target"], errors="coerce")
        
        # Save demographics for reference
        demographics_path = self.output_dir / self.demographics_file
        demographics_df.to_parquet(demographics_path, index=False)
        logger.success(f"Demographics data saved to: {demographics_path}")
        
        return demographics_df
    
    def _create_temp_table(self, cursor, target_data):
        """Create temporary table with target data."""
        logger.info("Creating temporary target table...")
        cursor.execute("""
            CREATE TEMP TABLE temp_target(
                subject_id INT,
                hadm_id INT,
                admittime TIMESTAMP,
                dischtime TIMESTAMP,
                target INT
            )
        """)
        
        target_data["admittime"] = pd.to_datetime(target_data["admittime"])
        target_data["dischtime"] = pd.to_datetime(target_data["dischtime"])
        
        values = list(target_data.itertuples(index=False, name=None))
        execute_values(
            cursor,
            "INSERT INTO temp_target (subject_id, hadm_id, admittime, dischtime, target) VALUES %s",
            values,
        )
        logger.success("Temporary target table created and populated.")
    
    def extract_lab_events_with_demographics(self, days_prior: int) -> str:
        """
        Extract lab events data with demographics for specified days prior to discharge.
        This is the main method that should be called - it handles the full workflow.
        """
        logger.info("=" * 80)
        logger.info(f"STARTING COMPLETE DATA EXTRACTION FOR COHORT: {self.cohort_name.upper()}")
        logger.info(f"TIME WINDOW: {days_prior} days prior to discharge")
        logger.info("=" * 80)
        
        # Step 1: Extract demographics (always required)
        demographics_df = self._extract_demographics()
        
        # Step 2: Extract lab events with demographics
        logger.info("=" * 60)
        logger.info("EXTRACTING LAB EVENTS WITH DEMOGRAPHICS")
        logger.info("=" * 60)
        
        # Process in batches
        batch_size = 1000
        batches = [
            (demographics_df.iloc[i:i + batch_size], i // batch_size)
            for i in range(0, len(demographics_df), batch_size)
        ]
        
        # Parallel processing
        with mp.Pool(mp.cpu_count() - 1) as pool:
            logger.info(f"Processing {len(batches)} batches of size {batch_size}...")
            batch_args = [(batch_df, batch_idx, days_prior, self.db_config, self.temp_dir) 
                         for batch_df, batch_idx in batches]
            parquet_files = pool.map(self._process_lab_batch, batch_args)
        
        # Concatenate results
        valid_files = [f for f in parquet_files if f]
        logger.info(f"Valid batch files: {len(valid_files)}")
        
        if not valid_files:
            raise RuntimeError("No valid lab events data extracted!")
        
        # Combine all batches
        final_df = self._combine_batch_files(valid_files)
        
        # Generate summary
        self._log_extraction_summary(final_df, len(demographics_df))
        
        # Save final result (this is the file for ML training)
        output_filename = f"{self.cohort_name}_lab_events_data_{days_prior}_days_prior.parquet"
        output_path = self.output_dir / output_filename
        final_df.to_parquet(output_path, index=False)
        
        logger.success("=" * 80)
        logger.success(f"✅ EXTRACTION COMPLETE!")
        logger.success(f"📁 Raw data for further preprocessing: {output_path}")
        logger.success(f"📁 Demographics reference: {self.output_dir / self.demographics_file}")
        logger.success(f"📁 Logs saved to: {Path.cwd() / 'logs'}")
        logger.success("=" * 80)
        
        return str(output_path)
    
    def _process_lab_batch(self, args) -> str:
        """Process a batch of lab events data."""
        batch_df, batch_idx, days_prior, db_config, temp_dir = args
        
        # Create engine for this process
        url = (f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
               f"@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        engine = create_engine(url)
        
        try:
            batch_ids = batch_df[["subject_id", "hadm_id"]].drop_duplicates()
            
            if batch_ids.empty:
                return None
            
            # Prepare ID tuples for SQL query
            id_tuples = []
            for row in batch_ids.itertuples(index=False):
                if pd.isna(row.subject_id) or pd.isna(row.hadm_id):
                    continue
                sid = int(float(row.subject_id))
                hid = int(float(row.hadm_id))
                id_tuples.append((sid, hid))
            
            if not id_tuples:
                logger.warning(f"Batch {batch_idx} has no valid IDs. Skipping.")
                return None
            
            # SQL query for lab events
            id_tuple_str = ",".join([f"({sid},{hid})" for sid, hid in id_tuples])
            query = f"""
                SELECT DISTINCT
                    le.subject_id, 
                    le.hadm_id, 
                    le.itemid, 
                    le.charttime, 
                    le.valuenum
                FROM mimiciv_hosp.labevents le
                WHERE (le.subject_id, le.hadm_id) IN ({id_tuple_str})
            """
            
            lab_df = pd.read_sql(text(query), engine)
            logger.info(f"Batch {batch_idx} fetched: {len(lab_df)} lab event records")
            
            # Merge with demographics (this ensures demographics are always included)
            merged_df = batch_df[
                ["subject_id", "hadm_id", "dischtime", "target", "gender", "anchor_age", "race"]
            ].merge(lab_df, on=["subject_id", "hadm_id"], how="left")
            
            # Apply time filtering
            merged_df = self._apply_time_filter(merged_df, days_prior, batch_idx)
            
            if merged_df.empty:
                logger.warning(f"Batch {batch_idx} is empty after filtering.")
                return None
            
            # Save batch file
            file_path = temp_dir / f"lab_batch_{batch_idx}.parquet"
            merged_df.to_parquet(file_path, index=False)
            logger.info(f"Batch {batch_idx} saved: {len(merged_df)} records")
            
            return str(file_path)
            
        finally:
            engine.dispose()
    
    def _apply_time_filter(self, df: pd.DataFrame, days_prior: int, batch_idx: int) -> pd.DataFrame:
        """Apply time filtering to lab events data."""
        if df.empty:
            return df
            
        df["charttime"] = pd.to_datetime(df["charttime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        
        # Log patients with no lab events
        patients_no_labs = df[df["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
        if not patients_no_labs.empty:
            logger.info(f"Batch {batch_idx}: {len(patients_no_labs)} patients with NO lab events")
            for _, row in patients_no_labs.iterrows():
                self.no_labs_logger.info(
                    f"BATCH {batch_idx} - Subject ID: {row['subject_id']}, "
                    f"Hospital Admission ID: {row['hadm_id']}"
                )
        
        # Filter by time window (keep rows without lab events OR within time window)
        time_filtered = df[
            df["charttime"].isna() |
            ((df["charttime"] >= df["dischtime"] - pd.Timedelta(days=days_prior)) &
             (df["charttime"] <= df["dischtime"]))
        ]
        
        logger.info(f"Batch {batch_idx} after time filtering: {len(time_filtered)} records")
        
        return time_filtered.drop_duplicates()
    
    def _combine_batch_files(self, batch_files: list) -> pd.DataFrame:
        """Combine all batch files into a single DataFrame."""
        logger.info(f"Combining {len(batch_files)} batch files...")
        
        dfs_to_concat = []
        total_rows = 0
        
        for pq_file in batch_files:
            df_temp = pd.read_parquet(pq_file)
            total_rows += len(df_temp)
            dfs_to_concat.append(df_temp)
        
        logger.info(f"Total rows before concatenation: {total_rows}")
        final_df = pd.concat(dfs_to_concat, ignore_index=True)
        logger.info(f"Total rows after concatenation: {len(final_df)}")
        
        return final_df
    
    def _log_extraction_summary(self, final_df: pd.DataFrame, original_cohort_size: int):
        """Log comprehensive extraction summary."""
        unique_patients = final_df[['subject_id', 'hadm_id']].drop_duplicates()
        patients_with_no_labs = final_df[final_df["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
        patients_with_labs = final_df[~final_df["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
        
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY:")
        logger.info(f"Original cohort size: {original_cohort_size}")
        logger.info(f"Total unique patients in final dataset: {len(unique_patients)}")
        logger.info(f"Patients WITH lab events: {len(patients_with_labs)}")
        logger.info(f"Patients WITHOUT lab events: {len(patients_with_no_labs)}")
        logger.info(f"Missing patients: {original_cohort_size - len(unique_patients)}")
        logger.info(f"Total records in final dataset: {len(final_df)}")
        logger.info("=" * 60)
        
        # Log to no-labs file
        self.no_labs_logger.info("=" * 50)
        self.no_labs_logger.info("FINAL SUMMARY - PATIENTS WITH NO LAB EVENTS:")
        self.no_labs_logger.info(f"Total patients with no lab events: {len(patients_with_no_labs)}")
        for _, row in patients_with_no_labs.iterrows():
            self.no_labs_logger.info(f"  Subject ID: {row['subject_id']}, Hospital Admission ID: {row['hadm_id']}")
        self.no_labs_logger.info("=" * 50)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract lab events data with demographics for medical cohorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --cohort aplasia --days 7
  %(prog)s --cohort aplasia --days 14  
  %(prog)s --cohort heart_failure --days 7

Note: Demographics are always extracted first as they are required for the final output.
The final output file contains lab events with demographics and is ready for ML training.
        """
    )
    
    parser.add_argument(
        '--cohort',
        required=True,
        help='Cohort name (e.g., aplasia, heart_failure)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        required=True,
        help='Days prior to discharge for lab events extraction'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = MimicDataExtractor(args.cohort)
        
        # Perform complete extraction (demographics + lab events)
        output_path = extractor.extract_lab_events_with_demographics(args.days)
        
        print(f"\n🎉 EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"📊 Final output file (ready for ML training): {output_path}")
        print(f"📁 Check logs directory for detailed extraction logs")
            
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()