import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from pathlib import Path
import multiprocessing as mp
import polars as pl
import sys
from loguru import logger

# Setup logging to both file and console
log_dir = Path.cwd() / "logs"
log_dir.mkdir(exist_ok=True)

# Remove default logger
logger.remove()

# Add console logging (INFO level and above)
logger.add(
    sys.stderr, 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
    level="INFO"
)

# Add file logging (DEBUG level and above) - creates new file each run
logger.add(
    log_dir / "lab_events_extraction_{time:YYYY-MM-DD_HH-mm-ss}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function} | {message}",
    level="DEBUG",
    rotation="10 MB",  # Rotate when file gets too large
    retention="7 days"  # Keep logs for 7 days
)

# Add separate file for patients with no lab events (for easy analysis)
no_labs_logger = logger.bind(category="no_labs")
logger.add(
    log_dir / "patients_no_lab_events_{time:YYYY-MM-DD_HH-mm-ss}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    filter=lambda record: record["extra"].get("category") == "no_labs",
    level="INFO"
)

logger.info("Logging setup complete - logs will be saved to: " + str(log_dir))

dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)
 

# Prepare output directory
temp_dir = Path(__file__).resolve().parents[1] / "dataset" / "temp"
output_dir = Path(__file__).resolve().parents[1] / "dataset" / "raw"
output_dir.mkdir(exist_ok=True)

# Load cohort
parquet_path = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "raw"
    / "cohort_with_demographic_data.parquet"
)
cohort_df = pd.read_parquet(parquet_path)


def fetch_batch(args):
    batch_df, batch_idx = args
    engine = get_engine()

    batch_ids = batch_df[["subject_id", "hadm_id"]].drop_duplicates()

    if batch_ids.empty:
        return None

    # Create tuples for the IN clause - handle potential float values
    id_tuples = []
    for row in batch_ids.itertuples(index=False):
        if pd.isna(row.subject_id) or pd.isna(row.hadm_id):
            continue  # Skip invalid rows
        sid = int(float(row.subject_id))
        hid = int(float(row.hadm_id))
        id_tuples.append((sid, hid))
    
    if not id_tuples:  # No valid IDs in this batch
        logger.warning(f"Batch {batch_idx} has no valid IDs. Skipping.")
        return None
        
    # Convert to string format for SQL
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
    logger.info(f"Batch {batch_idx} fetched with {len(df)} lab event records.")
    
    # Always merge with batch_df to preserve all patients
    merged_df = batch_df[
        ["subject_id", "hadm_id", "dischtime", "target", "gender", "anchor_age", "race"]
    ].merge(df, on=["subject_id", "hadm_id"], how="left")
    
    logger.info(f"Batch {batch_idx} after merge: {len(merged_df)} total records")
    
    # Log patients with no lab events
    patients_no_labs = merged_df[merged_df["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
    if not patients_no_labs.empty:
        logger.info(f"Batch {batch_idx}: {len(patients_no_labs)} patients with NO lab events")
        
        # Log to special no-labs file
        no_labs_logger.info(f"BATCH {batch_idx} - Patients with no lab events:")
        for _, row in patients_no_labs.iterrows():
            no_labs_logger.info(f"  Subject ID: {row['subject_id']}, Hospital Admission ID: {row['hadm_id']}")
        
        # Also log to main debug log
        logger.debug(f"Batch {batch_idx} - Patients with no lab events: {patients_no_labs.values.tolist()}")
    
    if not merged_df.empty:
        # Filter for the time window only for rows that have lab events
        merged_df["charttime"] = pd.to_datetime(merged_df["charttime"])
        merged_df["dischtime"] = pd.to_datetime(merged_df["dischtime"])
        
        # Count patients before time filtering
        patients_before_time_filter = merged_df[["subject_id", "hadm_id"]].drop_duplicates()
        
        # Keep rows without lab events (charttime is NaN) OR within time window
        time_filtered = merged_df[
            merged_df["charttime"].isna() |
            ((merged_df["charttime"] >= merged_df["dischtime"] - pd.Timedelta(days=7))
             & (merged_df["charttime"] <= merged_df["dischtime"]))
        ]
        
        # Count patients after time filtering
        patients_after_time_filter = time_filtered[["subject_id", "hadm_id"]].drop_duplicates()
        
        # Log patients filtered out by time window
        patients_filtered_by_time = len(patients_before_time_filter) - len(patients_after_time_filter)
        if patients_filtered_by_time > 0:
            logger.warning(f"Batch {batch_idx}: {patients_filtered_by_time} patients filtered out due to time window")
            
            # Find which specific patients were filtered out
            filtered_out_patients = patients_before_time_filter.merge(
                patients_after_time_filter, 
                on=["subject_id", "hadm_id"], 
                how="left", 
                indicator=True
            ).query('_merge == "left_only"')[["subject_id", "hadm_id"]]
            
            if not filtered_out_patients.empty:
                logger.debug(f"Batch {batch_idx} - Patients filtered by time: {filtered_out_patients.values.tolist()}")
        
        logger.info(f"Batch {batch_idx} after time filtering: {len(time_filtered)} records, {len(patients_after_time_filter)} unique patients")
        
        # Final check for patients with no lab events after time filtering
        final_patients_no_labs = time_filtered[time_filtered["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
        if not final_patients_no_labs.empty:
            logger.info(f"Batch {batch_idx}: {len(final_patients_no_labs)} patients with no lab events in final dataset")
        
        if time_filtered.empty:
            logger.warning(f"Batch {batch_idx} after time filtering is empty.")
            return None
            
        merged_df = time_filtered

    if merged_df.duplicated().any():
        logger.warning(f"Batch {batch_idx} contains duplicate records. Removing duplicates.")
        merged_df = merged_df.drop_duplicates()

    engine.dispose()

    # Save to parquet
    file_path = temp_dir / f"lab_batch_{batch_idx}.parquet"
    merged_df.to_parquet(file_path, index=False)
    logger.info(f"Batch {batch_idx} saved to {file_path}")
    return str(file_path)


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("STARTING LAB EVENTS EXTRACTION")
    logger.info("="*60)
    
    subject_ids = cohort_df["subject_id"].unique().tolist()
    batch_size = 1000
    batches = [
        (cohort_df.iloc[i : i + batch_size], i // batch_size)
        for i in range(0, len(cohort_df), batch_size)
    ]

    with mp.Pool(mp.cpu_count() - 1) as pool:
        logger.info(f"Starting to fetch lab events in {len(batches)} batches of size {batch_size}...")
        parquet_files = pool.map(fetch_batch, batches)

    # DEBUG: Check what we got back
    logger.info(f"Total batches processed: {len(parquet_files)}")
    none_batches = [i for i, f in enumerate(parquet_files) if f is None]
    logger.info(f"Batches that returned None: {len(none_batches)} - {none_batches}")
    
    # Concatenate results
    parquet_files = [f for f in parquet_files if f]
    logger.info(f"Valid parquet files to concatenate: {len(parquet_files)}")
    
    if parquet_files:
        dfs_to_concat = []
        total_rows = 0
        
        for pq in parquet_files:
            df_temp = pd.read_parquet(pq)
            logger.info(f"File {pq}: {len(df_temp)} rows")
            total_rows += len(df_temp)
            dfs_to_concat.append(df_temp)
        
        logger.info(f"Total rows before concatenation: {total_rows}")
        final_df = pd.concat(dfs_to_concat, ignore_index=True)
        logger.info(f"Total rows after concatenation: {len(final_df)}")
        
        # Final summary statistics
        unique_patients = final_df[['subject_id', 'hadm_id']].drop_duplicates()
        patients_with_no_labs = final_df[final_df["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
        patients_with_labs = final_df[~final_df["itemid"].isna()][["subject_id", "hadm_id"]].drop_duplicates()
        
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY:")
        logger.info(f"Total unique patients in final dataset: {len(unique_patients)}")
        logger.info(f"Patients WITH lab events: {len(patients_with_labs)}")
        logger.info(f"Patients WITHOUT lab events: {len(patients_with_no_labs)}")
        logger.info(f"Original cohort size: {len(cohort_df)}")
        logger.info(f"Missing patients: {len(cohort_df) - len(unique_patients)}")
        
        # Log summary to no-labs file as well
        no_labs_logger.info("=" * 50)
        no_labs_logger.info("FINAL SUMMARY - PATIENTS WITH NO LAB EVENTS:")
        no_labs_logger.info(f"Total patients with no lab events: {len(patients_with_no_labs)}")
        no_labs_logger.info("Complete list:")
        for _, row in patients_with_no_labs.iterrows():
            no_labs_logger.info(f"  Subject ID: {row['subject_id']}, Hospital Admission ID: {row['hadm_id']}")
        no_labs_logger.info("=" * 50)
        
        # Show some examples in main log
        if not patients_with_no_labs.empty:
            logger.info("First 5 patients with no lab events:")
            for _, row in patients_with_no_labs.head(5).iterrows():
                logger.info(f"  Subject ID: {row['subject_id']}, Hospital Admission ID: {row['hadm_id']}")
        
        logger.info("=" * 60)
        
        final_df.to_parquet(output_dir / "lab_event_data_with_demographics.parquet", index=False)
        logger.info(f"✅ Done. Saved {len(final_df)} lab events to lab_event_data_with_demographics.parquet")
        logger.info(f"📁 Logs saved to: {log_dir}")
        
    else:
        logger.error("No valid parquet files to concatenate!")