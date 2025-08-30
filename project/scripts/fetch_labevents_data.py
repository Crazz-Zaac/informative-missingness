import multiprocessing as mp
import pandas as pd
from sqlalchemy import text
from sqlalchemy import create_engine
from pathlib import Path
import tempfile
import logging
from functools import partial


class LabEventsExtractor:
    def __init__(
        self,
        db_url,
        logger,
        raw_output_dir,
        temp_dir,
        demographics_file_path,
        cohort,
        days,
    ):
        self.logger = logger
        self.raw_output_dir = Path(raw_output_dir)
        self.temp_dir = Path(temp_dir)
        self.demographics_file_path = demographics_file_path
        self.cohort = cohort
        self.days = days

        # prepare file names
        self.output_file_name = f"{cohort}_lab_events_data_with_demographics.parquet"
        self.days_prior_file = f"{cohort}_lab_events_data_{days}_days_prior.parquet"

        # Load cohort with demographics file
        parquet_path = self.raw_output_dir / self.demographics_file_path
        self.cohort_df = pd.read_parquet(parquet_path)

        self.db_url = db_url
        self.engine = self._get_engine()

    def _get_engine(self):
        return create_engine(self.db_url)

    @staticmethod
    def fetch_batch(args, db_url, temp_dir):
        """Static method for multiprocessing - receives all data as arguments"""
        batch_df, batch_idx = args
        try:
            # Setup simple logging for this process
            logging.basicConfig(
                level=logging.INFO, 
                format="%(asctime)s | %(levelname)s | %(message)s"
            )

            engine = create_engine(db_url)
            batch_ids = batch_df[["subject_id", "hadm_id"]].drop_duplicates()
            if batch_ids.empty:
                logging.warning(f"Batch {batch_idx} is empty. Skipping.")
                return None

            # Build ID tuples
            id_tuples = []
            for row in batch_ids.itertuples(index=False):
                if pd.isna(row.subject_id) or pd.isna(row.hadm_id):
                    continue
                sid = int(float(row.subject_id))
                hid = int(float(row.hadm_id))
                id_tuples.append((sid, hid))

            if not id_tuples:
                logging.warning(f"Batch {batch_idx} has no valid IDs. Skipping.")
                return None

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

            df = pd.read_sql(text(query), engine)
            logging.info(f"Batch {batch_idx} fetched with {len(df)} lab event records.")

            merged_df = batch_df.merge(df, on=["subject_id", "hadm_id"], how="left")
            logging.info(
                f"Batch {batch_idx} after merge: {len(merged_df)} total records"
            )

            # Log patients with no labs
            patients_no_labs = merged_df[merged_df["itemid"].isna()][
                ["subject_id", "hadm_id"]
            ].drop_duplicates()
            if not patients_no_labs.empty:
                logging.info(
                    f"BATCH {batch_idx} - {len(patients_no_labs)} patients with no lab events"
                )

            if not merged_df.empty:
                merged_df["charttime"] = pd.to_datetime(merged_df["charttime"])
                merged_df["dischtime"] = pd.to_datetime(merged_df["dischtime"])
                # Filter for records within 7 days before discharge
                time_filtered = merged_df[
                    merged_df["charttime"].isna()
                    | (
                        (
                            merged_df["charttime"]
                            >= merged_df["dischtime"] - pd.Timedelta(days=7)
                        )
                        & (merged_df["charttime"] <= merged_df["dischtime"])
                    )
                ]
                merged_df = time_filtered

            merged_df = merged_df.drop_duplicates()
            engine.dispose()

            file_path = Path(temp_dir) / f"lab_batch_{batch_idx}.parquet"
            merged_df.to_parquet(file_path, index=False)
            logging.info(f"Batch {batch_idx} saved to {file_path}")
            return str(file_path)

        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            return None

    def run(self, batch_size: int = 1000):
        self.logger.info("=" * 60)
        self.logger.info("STARTING LAB EVENTS EXTRACTION")
        self.logger.info("=" * 60)

        # Create temporary directory for batch files
        temp_dir = Path(tempfile.mkdtemp())
        self.logger.info(f"Using temporary directory: {temp_dir}")

        # Prepare batches
        batch_args = []
        for i in range(0, len(self.cohort_df), batch_size):
            batch_df = self.cohort_df.iloc[i : i + batch_size].copy()
            batch_idx = i // batch_size
            batch_args.append((batch_df, batch_idx))

        self.logger.info(
            f"Batch size: {batch_size} - Processing {len(batch_args)} total batches"
        )

        # Create partial function with fixed arguments
        process_func = partial(
            self.fetch_batch,
            db_url=self.db_url,
            temp_dir=str(temp_dir)
        )

        # Use multiprocessing with map
        with mp.Pool(min(mp.cpu_count() - 1, len(batch_args))) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_func, batch_args)):
                if result:
                    results.append(result)
                    self.logger.info(f"Completed batch {i+1}/{len(batch_args)}")
                else:
                    self.logger.warning(f"Batch {i+1} failed or returned no data")

        parquet_files = results

        if not parquet_files:
            self.logger.error("No valid parquet files to concatenate!")
            return

        # Concatenate files in chunks to avoid memory issues
        chunks = []
        for pq_file in parquet_files:
            try:
                df_chunk = pd.read_parquet(pq_file)
                chunks.append(df_chunk)
                self.logger.info(f"Loaded {len(df_chunk)} records from {pq_file}")
            except Exception as e:
                self.logger.error(f"Error loading {pq_file}: {e}")

        if not chunks:
            self.logger.error("No data loaded from any parquet files!")
            return

        final_df = pd.concat(chunks, ignore_index=True)

        # Save the unfiltered data
        output_path = self.raw_output_dir / self.output_file_name
        final_df.to_parquet(output_path, index=False)
        rel_path = Path(output_path).relative_to(self.raw_output_dir.parent)
        self.logger.info(f"✅ Saved {len(final_df)} records to {rel_path}")

        # Save x-days prior subsets (filtered)
        if not final_df.empty:
            final_df["charttime"] = pd.to_datetime(final_df["charttime"])
            final_df["dischtime"] = pd.to_datetime(final_df["dischtime"])

            days_prior_df = final_df[
                (
                    final_df["charttime"]
                    >= final_df["dischtime"] - pd.Timedelta(days=self.days)
                )
                & (final_df["charttime"] <= final_df["dischtime"])
            ]

            days_prior_path = self.raw_output_dir / self.days_prior_file
            days_prior_df.to_parquet(days_prior_path, index=False)
            self.logger.info(
                f"✅ {self.days_prior_file} days dataset saved with {len(days_prior_df)} records."
            )
        else:
            self.logger.warning("No data to save for days prior subset.")

        # Clean up temporary files
        self._cleanup_temp_files(parquet_files, temp_dir)

        self.logger.info("Lab events extraction completed successfully.")

    def _cleanup_temp_files(self, parquet_files, temp_dir):
        """Clean up temporary files"""
        cleaned_count = 0
        for pq_file in parquet_files:
            try:
                Path(pq_file).unlink()
                cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Could not delete temp file {pq_file}: {e}")

        try:
            # Try to remove temp directory if empty
            if temp_dir.exists():
                temp_dir.rmdir()
                self.logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"Could not remove temp directory {temp_dir}: {e}")

        self.logger.info(f"Cleaned up {cleaned_count} temporary files")