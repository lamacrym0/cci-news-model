# preprocessing_us.py
import polars as pl
import os
import torch
import numpy as np

events_dir = "data/processed/events_by_month_filtered"
cci_path = "data/cci_ocde.parquet"
output_X = "data/X_us.pt"
output_y = "data/y_us.pt"

df_cci = pl.read_parquet(cci_path)
df_cci = df_cci.filter(pl.col("REF_AREA") == "USA")

df_cci = df_cci.with_columns(
    pl.col("TIME_PERIOD").str.replace("-", "").alias("YYYYMM")
)

df_cci = df_cci.drop(["TIME_PERIOD", "REF_AREA"])

df_cci = (
    df_cci.sort(["YYYYMM"])
    .with_columns(
        (pl.col("OBS_VALUE") - pl.col("OBS_VALUE").shift(1)).alias("delta_cci")
    )
    .drop_nulls("delta_cci")
)

df_full = None
for month_file in sorted(os.listdir(events_dir)):
    if not month_file.endswith(".parquet"):
        continue

    df = pl.scan_parquet(os.path.join(events_dir, month_file))

    df = df.with_columns(
        pl.concat_list(["ActionGeo_CountryCode", "Actor1Geo_CountryCode", "Actor2Geo_CountryCode"])
        .list.drop_nulls()
        .alias("countries")
    )

    df = df.explode("countries")

    df = df.filter(pl.col("countries") == "US")

    df = df.drop("countries")

    df = df.with_columns(pl.col("EventCode").str.slice(0, 2).alias("EventRoot"))
    
    df = df.filter(pl.col("EventRoot").str.contains(r"^\d{2}$"))

    

    if df_full is None:
        df_full = df
    else:
        df_full = pl.concat([df_full, df], how="diagonal")

df_full = (
        df_full.group_by(["SQLDATE", "EventRoot"])
        .agg([
            (pl.col("AvgTone") * pl.col("NumMentions")).sum().alias("tone_weighted_sum"),
            pl.col("NumMentions").sum().alias("total_mentions"),
        ])
        .with_columns([
            (pl.col("tone_weighted_sum") / pl.col("total_mentions")).alias("mean_tone_weighted"),
            pl.col("total_mentions").alias("mean_mentions")
        ])
        .select(["EventRoot", "mean_tone_weighted", "mean_mentions", "SQLDATE"])
    )

df_full = df_full.collect()

df_wide = df_full.pivot(
    values=["mean_tone_weighted", "mean_mentions"],
    index=["SQLDATE"],
    on="EventRoot"
).fill_null(0)

df_wide.sort("SQLDATE")

df_wide = df_wide.with_columns(
    pl.col("SQLDATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date")
).sort("date")

df_wide = df_wide.with_columns(
    pl.col("date").dt.strftime("%Y%m").alias("YYYYMM")
)

df_cci_shifted = (
    df_cci
    .with_columns([
        pl.col("delta_cci").shift(1).alias("feature_delta_cci"),
        pl.col("OBS_VALUE").shift(1).alias("feature_cci_prev"),
        pl.col("delta_cci").alias("target_delta_cci")
    ])
    .drop_nulls(["feature_delta_cci", "feature_cci_prev", "target_delta_cci"])
    .select(["YYYYMM", "feature_delta_cci", "feature_cci_prev", "target_delta_cci"])
)

df_merged = df_wide.join(df_cci_shifted, on="YYYYMM", how="inner")

df_merged = df_merged.fill_null(0)

df_merged = df_merged.sort("date")

# print first date and last date
print(f"Data from {df_merged['date'].min()} to {df_merged['date'].max()}")
exit()

SEQ_LEN = 60

# Conversion en numpy
features = df_merged.drop(["date", "YYYYMM", "target_delta_cci", "feature_delta_cci"]).to_numpy()
targets = df_merged["target_delta_cci"].to_numpy()
months = df_merged["YYYYMM"].to_numpy()

X, y = [], []

# On va créer UNE séquence par mois (many-to-one)
unique_months = np.unique(months)

for month in unique_months:
    # indices des lignes appartenant à ce mois
    idx = np.where(months == month)[0]
    if len(idx) == 0:
        continue

    # fin de la séquence = dernier jour de ce mois
    idx_end = idx[-1]
    idx_start = idx_end - SEQ_LEN + 1
    if idx_start < 0:
        continue  # pas assez d'historique

    # features des 60 derniers jours (many-to-one)
    X.append(features[idx_start:idx_end + 1])
    y.append(targets[idx_end])  # ΔCCI du mois courant

X = torch.tensor(np.stack(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

torch.save(X, output_X)
torch.save(y, output_y)

print("✅ Saved:")
print(f"   X: {X.shape} (n_months, seq_len, n_features)")
print(f"   y: {y.shape} (n_months, 1)")
print(f"   Sequence length: {SEQ_LEN} days")
print(f"   Number of features: {X.shape[2]}")

