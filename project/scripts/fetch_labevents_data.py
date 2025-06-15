import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from pathlib import Path
import multiprocessing as mp
import polars as pl
import loguru

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
        loguru.logger.debug(
            f"Processing row: subject_id={row.subject_id}, hadm_id={row.hadm_id}"
        )
        sid = int(float(row.subject_id)) if pd.notna(row.subject_id) else 0
        hid = int(float(row.hadm_id)) if pd.notna(row.hadm_id) else 0
        id_tuples.append((sid, hid))

    # Convert to string format for SQL
    id_tuple_str = ",".join([f"({sid},{hid})" for sid, hid in id_tuples])

    query = f"""
        SELECT 
            le.subject_id, 
            le.hadm_id, 
            le.itemid, 
            le.charttime, 
            le.valuenum
        FROM mimiciv_hosp.labevents le
        WHERE (le.subject_id, le.hadm_id) IN ({id_tuple_str})
    """

    df = pd.read_sql(text(query), engine)
    loguru.logger.info(f"Batch {batch_idx} fetched with {len(df)} records.")
    if df.empty:
        loguru.logger.warning(f"Batch {batch_idx} is empty. Skipping.")
        return None

    if df.duplicated().any():
        loguru.logger.warning(
            f"Batch {batch_idx} contains duplicate records. Removing duplicates."
        )
        df = df.drop_duplicates()

    # Add dischtime and target by merging with batch_df
    df = df.merge(
        batch_df[
            [
                "subject_id",
                "hadm_id",
                "dischtime",
                "target",
                "gender",
                "anchor_age",
                "race",
            ]
        ],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    loguru.logger.info(
        f"Batch {batch_idx} merged with cohort data. Total records: {len(df)}"
    )
    if df.empty:
        loguru.logger.warning(f"Batch {batch_idx} after merge is empty. Skipping.")
        return None

    # Filter for the time window
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df = df[
        (df["charttime"] >= df["dischtime"] - pd.Timedelta(days=7))
        & (df["charttime"] <= df["dischtime"])
    ]

    engine.dispose()

    # Save to parquet
    file_path = temp_dir / f"lab_batch_{batch_idx}.parquet"
    df.to_parquet(file_path, index=False)
    loguru.logger.info(f"Batch {batch_idx} saved to {file_path}")
    return str(file_path)


if __name__ == "__main__":
    subject_ids = cohort_df["subject_id"].unique().tolist()
    batch_size = 1000
    batches = [
        (cohort_df.iloc[i : i + batch_size], i // batch_size)
        for i in range(0, len(cohort_df), batch_size)
    ]

    with mp.Pool(mp.cpu_count() - 1) as pool:
        loguru.logger.info(
            f"Starting to fetch lab events in {len(batches)} batches of size {batch_size}..."
        )
        parquet_files = pool.map(fetch_batch, batches)

    # Concatenate results
    parquet_files = [f for f in parquet_files if f]
    final_df = pd.concat(
        [pd.read_parquet(pq) for pq in parquet_files], ignore_index=True
    )
    final_df.to_parquet(output_dir / "final_lab_events_7_days_prior.parquet", index=False)

    print(f"✅ Done. Saved {len(final_df)} lab events to final_lab_events_7_days_prior.parquet")
