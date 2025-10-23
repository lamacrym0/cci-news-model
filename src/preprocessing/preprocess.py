# preprocess.py
import polars as pl
import os
import torch
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
events_dir = "data/processed/events_by_month_filtered"
cci_path = "data/cci_ocde.parquet"
output_X = "data/X.pt"
output_y = "data/y.pt"
output_metadata = "data/metadata.pt"

SEQUENCE_LENGTH = 6  # Number of months to look back

# ---------------------------
# Mapping OCDE ISO3 -> FIPS
# ---------------------------
ocde_to_fips = {
    "AUS": "AS", "AUT": "AU", "BEL": "BE", "BRA": "BR", "CAN": "CA", "CHE": "SZ",
    "CHL": "CI", "CHN": "CH", "COL": "CO", "CRI": "CS", "CZE": "EZ", "DEU": "GM",
    "DNK": "DA", "EST": "EN", "ESP": "SP", "FIN": "FI", "FRA": "FR", "GBR": "UK",
    "GRC": "GR", "HUN": "HU", "IDN": "ID", "IND": "IN", "IRL": "EI", "ISR": "IS",
    "ITA": "IT", "JPN": "JA", "KOR": "KS", "LTU": "LH", "LVA": "LG", "LUX": "LU",
    "MEX": "MX", "NLD": "NL", "NZL": "NZ", "POL": "PL", "PRT": "PO", "RUS": "RS",
    "SVK": "LO", "SVN": "SI", "SWE": "SW", "TUR": "TU", "USA": "US", "ZAF": "SF"
}

# ---------------------------
# Load CCI
# ---------------------------
df_cci = pl.scan_parquet(cci_path).collect()
df_cci = df_cci.with_columns(
    pl.col("REF_AREA").replace(ocde_to_fips).alias("FIPS")
).drop("REF_AREA")

# Convert TIME_PERIOD to YYYYMM for merge
df_cci = df_cci.with_columns(
    pl.col("TIME_PERIOD").str.replace("-", "").alias("YYYYMM")
)

# Compute monthly ΔCCI by country
df_cci = (
    df_cci.sort(["FIPS", "YYYYMM"])
    .with_columns(
        (pl.col("OBS_VALUE") - pl.col("OBS_VALUE").shift(1)).over("FIPS").alias("delta_cci")
    )
    .drop_nulls("delta_cci")
)

# ---------------------------
# Load and aggregate events by month/country
# ---------------------------
all_months = []
for month_file in tqdm(sorted(os.listdir(events_dir)), desc="Processing monthly event files"):
    if not month_file.endswith(".parquet"):
        continue
    month_str = month_file.replace(".parquet", "")
    df = pl.scan_parquet(os.path.join(events_dir, month_file)).collect()

    # Extract unique countries per event
    df = df.with_columns(
        pl.concat_list(["ActionGeo_CountryCode", "Actor1Geo_CountryCode", "Actor2Geo_CountryCode"])
        .list.drop_nulls()
        .alias("countries")
    )

    df = df.explode("countries")

    # Add EventRoot (first two digits of EventCode)
    df = df.with_columns(pl.col("EventCode").str.slice(0, 2).alias("EventRoot"))
    
    # Filter valid EventRoots
    df = df.filter(pl.col("EventRoot").str.contains(r"^\d{2}$"))

    # Aggregate by country and EventRoot
    agg = (
        df.group_by(["countries", "EventRoot"])
        .agg([
            (pl.col("AvgTone") * pl.col("NumMentions")).sum().alias("tone_weighted_sum"),
            pl.col("NumMentions").sum().alias("total_mentions"),
        ])
        .with_columns([
            (pl.col("tone_weighted_sum") / pl.col("total_mentions")).alias("mean_tone_weighted"),
            pl.col("total_mentions").alias("mean_mentions"),
            pl.lit(month_str).alias("YYYYMM")
        ])
        .select(["countries", "EventRoot", "mean_tone_weighted", "mean_mentions", "YYYYMM"])
    )
    all_months.append(agg)

# Combine all months
df_all = pl.concat(all_months)

# Pivot to wide format: one row per (country, month)
df_wide = df_all.pivot(
    values=["mean_tone_weighted", "mean_mentions"],
    index=["countries", "YYYYMM"],
    on="EventRoot"
).fill_null(0)

# ---------------------------
# Merge with CCI targets
# ---------------------------
df_full = df_wide.join(
    df_cci.select(["FIPS", "YYYYMM", "delta_cci"]),
    left_on=["countries", "YYYYMM"],
    right_on=["FIPS", "YYYYMM"],
    how="inner"
).sort(["countries", "YYYYMM"])

print(f"Total data points: {df_full.height}")
print(f"Countries: {df_full['countries'].n_unique()}")
print(f"Date range: {df_full['YYYYMM'].min()} to {df_full['YYYYMM'].max()}")

# ---------------------------
# Create sequences for LSTM
# ---------------------------
feature_cols = [c for c in df_full.columns if c not in ("countries", "YYYYMM", "delta_cci")]
n_features = len(feature_cols)

X_sequences = []
y_sequences = []
metadata = []  # Store (country, end_date) for each sequence

# Group by country and create sequences
for country in tqdm(df_full["countries"].unique(), desc="Creating sequences"):
    country_data = df_full.filter(pl.col("countries") == country).sort("YYYYMM")
    
    # Convert to numpy for easier slicing
    features = country_data.select(feature_cols).to_numpy()
    targets = country_data["delta_cci"].to_numpy()
    dates = country_data["YYYYMM"].to_list()
    
    # Create sequences of length SEQUENCE_LENGTH
    for i in range(len(features) - SEQUENCE_LENGTH):
        X_seq = features[i:i+SEQUENCE_LENGTH]  # Shape: (seq_len, n_features)
        y_val = targets[i+SEQUENCE_LENGTH]      # Predict next month
        
        X_sequences.append(X_seq)
        y_sequences.append(y_val)
        metadata.append({
            "country": country,
            "end_date": dates[i+SEQUENCE_LENGTH-1],
            "target_date": dates[i+SEQUENCE_LENGTH]
        })

# Convert to tensors
X = torch.tensor(X_sequences, dtype=torch.float32)  # Shape: (n_samples, seq_len, n_features)
y = torch.tensor(y_sequences, dtype=torch.float32).unsqueeze(1)  # Shape: (n_samples, 1)

# Normalize features (important for LSTM)
X_mean = X.mean(dim=(0, 1), keepdim=True)  # Mean over samples and time
X_std = X.std(dim=(0, 1), keepdim=True) + 1e-8
X = (X - X_mean) / X_std

# Normalize target
y_mean = y.mean()
y_std = y.std() + 1e-8
y_normalized = (y - y_mean) / y_std

# Save tensors and metadata
os.makedirs("data", exist_ok=True)
torch.save(X, output_X)
torch.save(y_normalized, output_y)
torch.save({
    "X_mean": X_mean,
    "X_std": X_std,
    "y_mean": y_mean,
    "y_std": y_std,
    "feature_cols": feature_cols,
    "sequence_length": SEQUENCE_LENGTH,
    "metadata": metadata
}, output_metadata)

print(f"\n✅ Saved:")
print(f"   X: {X.shape} (n_samples, seq_len, n_features)")
print(f"   y: {y_normalized.shape} (n_samples, 1)")
print(f"   Sequence length: {SEQUENCE_LENGTH} months")
print(f"   Number of features: {n_features}")
print(f"   Total sequences: {len(X_sequences)}")