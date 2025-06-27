#!/usr/bin/env python3

import pandas as pd
from sklearn.impute import KNNImputer

# Load lab measurements (14-day window) and ensure item IDs are strings for consistency
df = pd.read_csv("mimic_iv_labs_nf_14_days.csv.gz", compression="gzip")
df["itemid"] = df["itemid"].astype("int").astype("str")

# Compute how frequently each lab test is recorded across admissions
freq_items = (
    df[["itemid", "admid"]]
    .drop_duplicates()
    .groupby("itemid")["admid"]
    .size()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"admid": "count"})
)
freq_items["freq_ts"] = freq_items["count"] / df["admid"].nunique()

# Compute the average number of measurements per admission for each lab test
ts_items = (
    df.groupby(["admid", "itemid"])
    .size()
    .unstack()
    .mean(axis=0)
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={0: "num_ts"})
)

# Merge frequency and count information
freq_ts_items = ts_items.merge(freq_items[["itemid", "freq_ts"]], on="itemid")

# Select lab tests with at least npoints and present in minfreq of admissions (customise)
npoints = 5
minfreq = 0.75
sel_vars = freq_ts_items[
    (freq_ts_items["num_ts"] >= npoints) &
    (freq_ts_items["freq_ts"] >= minfreq)
]["itemid"].tolist()

# Keep only the selected lab tests
df = df[df["itemid"].isin(sel_vars)]

# Compute lab-wise mean and standard deviation based only on observed values
lab_stats = df.groupby("itemid")["valuenum"].agg(["mean", "std"]).rename(columns={"mean": "lab_mean", "std": "lab_std"})

# Merge stats into main dataframe to allow normalisation later
df = df.merge(lab_stats, on="itemid")

# Pivot lab measurements into time series format (one row per admission-lab, one column per day)
df_ts = (
    df
    .groupby(["admid", "itemid", "lab_mean", "lab_std", "day"])["valuenum"]
    .mean()
    .unstack(level=-1)
    .interpolate(method='linear', axis=1, limit_area="inside")  # interpolate only between observed values
    .ffill(axis=1)  # fill missing values forward (after last measurement)
    .bfill(axis=1)  # fill missing values backward (before first measurement)
    .reset_index()
)

# Normalise time series using precomputed mean and std, drop stats columns
time_bins = list(range(0, 14))
# NOTE: if training with random forests you can skip normalisation (tree-based models are scale free)
df_ts[time_bins] = df_ts[time_bins].subtract(df_ts["lab_mean"], axis=0).divide(df_ts["lab_std"], axis=0) # can skip this line
df_ts = df_ts.drop(columns=["lab_mean", "lab_std"]).set_index(["admid", "itemid"])

# Unstack to get one row per admission with multiple lab*time columns
df_mx = df_ts.unstack(level=-1)
df_mx.columns = df_mx.columns.swaplevel(0, 1)
df_mx = df_mx.sort_index(axis=1)
df_mx.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in df_mx.columns]

# Impute any remaining missing values using k-nearest neighbours imputation
imputer = KNNImputer(n_neighbors=5)
df_imp = pd.DataFrame(
    imputer.fit_transform(df_mx),
    index=df_mx.index,
    columns=df_mx.columns
)

# Save final processed and imputed dataset
df_imp.to_csv("mimic_iv_labs_nf_14_days_processed.csv.gz", index=True, compression="gzip")

print("Data processing complete. Output saved to mimic_iv_labs_nf_14_days_processed.csv.gz")

