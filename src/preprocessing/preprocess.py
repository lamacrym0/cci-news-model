# src/preprocessing/preprocess.py
# Preprocessing pipeline: GDELT Events + CCI → X.parquet / y.parquet
# Fully lazy, robust, no silent crashes, memory-safe
import polars as pl
from pathlib import Path
from typing import List, Dict
import sys

# === CONFIGURATION ===
CCI_PATH = Path("data/CCI_OCDE.csv")
EVENTS_DIR = Path("data/events")
OUTPUT_DIR = Path("data/rows")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FIPS mapping: Alpha-3 → FIPS (2-letter)
ALPHA3_TO_FIPS: Dict[str, str] = {
    "CHL": "CL", "CRI": "CR", "POL": "PL", "PRT": "PT", "LTU": "LT",
    "CHN": "CH", "ITA": "IT", "FIN": "FI", "LUX": "LU", "RUS": "RU",
    "BRA": "BR", "AUT": "AT", "BEL": "BE", "CHE": "SZ", "HUN": "HU",
    "DEU": "GM", "MEX": "MX", "GRC": "GR", "GBR": "UK", "COL": "CO",
    "JPN": "JA", "SWE": "SW", "IND": "IN", "KOR": "KS", "TUR": "TU",
    "ISR": "IS", "AUS": "AS", "FRA": "FR", "NLD": "NL", "LVA": "LV",
    "SVK": "LO", "CZE": "EZ", "IDN": "ID", "EST": "EN", "USA": "US",
    "DNK": "DK", "IRL": "EI", "ZAF": "SA", "ESP": "SP", "NZL": "NZ", "SVN": "SI"
}

# === 1. LOAD CCI DATA ===
print("Loading CCI data...")
if not CCI_PATH.exists():
    print(f"[ERROR] CCI file not found: {CCI_PATH}")
    sys.exit(1)

cci = (
    pl.scan_csv(str(CCI_PATH), separator=",")
    .select(["TIME_PERIOD", "REF_AREA", "OBS_VALUE"])
    .with_columns([
        pl.col("TIME_PERIOD").str.to_date("%Y-%m", strict=False),
        pl.col("OBS_VALUE").cast(pl.Float64, strict=False)
    ])
    .filter(pl.col("TIME_PERIOD").is_not_null())
)

# Cache intermediate
cci.collect().write_parquet("data/cci_ocde.parquet")

# === 2. GET UNIQUE MONTHS (as strings: "2021-11") ===
print("Extracting unique months from CCI...")
months_df = cci.select("TIME_PERIOD").unique().collect()

if months_df.is_empty():
    print("[ERROR] No valid months found in CCI data.")
    sys.exit(1)

months: List[str] = months_df["TIME_PERIOD"].dt.strftime("%Y-%m").to_list()
print(f"Found {len(months)} months to process.")

# === 3. PROCESS EACH MONTH: JOIN EVENTS + CCI ===
print("Joining GDELT events with CCI by month...")
results = []

for month_str in months:  # e.g., "2021-11"
    yyyymm = month_str.replace("-", "")  # e.g., "202111"

    # --- CCI slice for this month ---
    cci_month = (
        cci.filter(pl.col("TIME_PERIOD") == month_str)
        .with_columns(pl.lit(month_str).alias("DATE"))
        .drop("TIME_PERIOD")
        .collect()  # Small: one month → safe
    )

    if cci_month.is_empty():
        print(f"Warning: no CCI data for month {month_str}, skipping")
        continue

    # --- Find event file ---
    parquet_files = list(EVENTS_DIR.glob(f"{yyyymm}.parquet"))
    if not parquet_files:
        print(f"Warning: no event files for month {yyyymm}, skipping")
        continue

    events_path = parquet_files[0]
    try:
        dx = pl.scan_parquet(str(events_path))
    except Exception as e:
        print(f"[ERROR] Failed to read {events_path}: {e}")
        continue

    # --- Transform SQLDATE → DATE (YYYY-MM) ---
    dx = dx.with_columns(
        pl.col("SQLDATE")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, "%Y%m%d", strict=False)
        .dt.strftime("%Y-%m")
        .alias("DATE")
    ).drop("SQLDATE")

    # --- Lazy inner join on DATE ---
    joined = dx.join(
        pl.LazyFrame(cci_month),
        on="DATE",
        how="inner"
    )

    results.append(joined.collect())

# === 4. CONCAT ALL MONTHS ===
if not results:
    print("No joined data found. Creating empty output.")
    empty_df = pl.DataFrame()
    empty_df.write_parquet(str(OUTPUT_DIR / "data.parquet"))
    empty_df.write_parquet(str(OUTPUT_DIR / "X.parquet"))
    empty_df.write_parquet(str(OUTPUT_DIR / "y.parquet"))
    sys.exit(0)

print("Concatenating all monthly results...")
all_rows = pl.concat(results, how="vertical")
all_rows.write_parquet(str(OUTPUT_DIR / "data.parquet"))

# === 5. FINAL PROCESSING (LAZY) ===
print("Final aggregation and pivoting...")
data = pl.scan_parquet(str(OUTPUT_DIR / "data.parquet"))

# Map REF_AREA (alpha-3) → FIPS
data = data.with_columns(
    pl.col("REF_AREA")
    .replace_strict(ALPHA3_TO_FIPS, default=None)
    .alias("REF_AREA_FIPS")
)

# Drop original REF_AREA and ActionGeo_CountryCode (redundant)
data = data.drop(["REF_AREA", "ActionGeo_CountryCode"]).rename({"REF_AREA_FIPS": "REF_AREA"})

# Clean EventCode
df = data.with_columns(
    pl.col("EventCode").cast(pl.Utf8).str.strip_chars().alias("EventCode_clean")
).drop("EventCode")

# === 6. AGGREGATE BY (REF_AREA, DATE, EventCode, OBS_VALUE) ===
agg_df = (
    df.group_by(["REF_AREA", "DATE", "EventCode_clean", "OBS_VALUE"])
    .agg([
        pl.col("NumMentions").sum().alias("NumMentions_sum"),
        pl.col("AvgTone").mean().alias("AvgTone_mean")
    ])
)

# === 7. PIVOT: Mentions & Tone (LAZY) ===
print("Pivoting mentions and tone...")
pivot_mentions = (
    agg_df.pivot(
        values="NumMentions_sum",
        index=["REF_AREA", "DATE", "OBS_VALUE"],
        columns="EventCode_clean"
    )
    .fill_null(0)
    .rename(lambda col: f"Mentions_{col}" if col not in ["REF_AREA", "DATE", "OBS_VALUE"] else col)
)

pivot_tone = (
    agg_df.pivot(
        values="AvgTone_mean",
        index=["REF_AREA", "DATE", "OBS_VALUE"],
        columns="EventCode_clean"
    )
    .fill_null(0)
    .rename(lambda col: f"Tone_{col}" if col not in ["REF_AREA", "DATE", "OBS_VALUE"] else col)
)

# === 8. JOIN PIVOTS + ADD INDEX ===
final_df = pivot_mentions.join(pivot_tone, on=["REF_AREA", "DATE", "OBS_VALUE"])

# Stable row index
final_df = final_df.with_row_index("index")

# === 9. ENCODE REF_AREA → Integer ID ===
unique_areas = final_df.select("REF_AREA").unique().sort("REF_AREA")
area_to_id = {
    area: idx for idx, area in enumerate(unique_areas["REF_AREA"].to_list())
}

encoded_df = final_df.with_columns(
    pl.col("REF_AREA").replace(area_to_id).cast(pl.Int32).alias("REF_AREA_ID")
).drop("REF_AREA")

# Reorder columns: index first
encoded_df = encoded_df.select([
    "index", "REF_AREA_ID", "DATE", "OBS_VALUE",
    pl.exclude("index", "REF_AREA_ID", "DATE", "OBS_VALUE")
])

# === 10. WRITE FINAL X / y ===
print(f"Writing final datasets to {OUTPUT_DIR}/")
encoded_df.select(pl.exclude("OBS_VALUE", "DATE")).write_parquet(str(OUTPUT_DIR / "X.parquet"))
encoded_df.select(["index", "OBS_VALUE"]).write_parquet(str(OUTPUT_DIR / "y.parquet"))

print(f"Done! X: {OUTPUT_DIR}/X.parquet, y: {OUTPUT_DIR}/y.parquet")
print(f"   → {encoded_df.collect().height:,} rows, {len(encoded_df.columns)} columns")