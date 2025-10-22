# src/ingestion/gdelt_fetch.py
# Scalable & parallel GDELT Events ingestion (RAM-safe, 8 threads)
import polars as pl
from pathlib import Path
import zipfile
import io
import requests
import re
from tqdm import tqdm
from typing import Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
import pyarrow.parquet as pq

# === GDELT CONFIGURATION ===
COLUMNS_TO_KEEP = [
    "GLOBALEVENTID",
    "SQLDATE",
    "EventCode",
    "NumMentions",
    "AvgTone",
    "ActionGeo_CountryCode",
]

# Column indices (0-based) from GDELT 2.0 Events schema (verified on real file)
COLUMN_INDEX_MAP = {
    "GLOBALEVENTID": 0,
    "SQLDATE": 1,
    "EventCode": 26,
    "NumMentions": 31,
    "AvgTone": 34,
    "ActionGeo_CountryCode": 53,
    "Actor1Geo_CountryCode": 37,
    "Actor2Geo_CountryCode": 45
}

COLUMN_INDICES = list(COLUMN_INDEX_MAP.values())
COLUMN_NAMES = list(COLUMN_INDEX_MAP.keys())

# FIPS country codes to keep
COUNTRIES_FIPS = [
    "AS", "AU", "BE", "BR", "SZ", "CI", "CH", "CO", "CS", "EZ",
    "GM", "DA", "SP", "EN", "FI", "FR", "UK", "GR", "HU", "ID",
    "IN", "EI", "IS", "IT", "JA", "KS", "LH", "LU", "LG", "MX",
    "NL", "NZ", "PL", "PO", "RS", "LO", "SI", "SW", "TU", "SF", "US"
]

# Time range (YYYYMM as int)
START_MONTH = 201503
END_MONTH = 202509

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" / "events_by_month_filtered"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
URLS_PARQUET = DATA_DIR / "gdelt_events_urls.parquet"

# === 1. Extract month from URL (supports .translation.export.CSV.zip) ===
def url_to_month(url: str) -> int:
    match = re.search(r"(\d{8})\d{6}\.(?:translation\.)?export\.CSV\.zip$", url)
    return int(match.group(1)[:6]) if match else 0

# === 2. Generator: filtered URLs by month ===
def iter_urls_by_month(min_month: int, max_month: int) -> Iterator[Tuple[int, str]]:
    df_urls = pl.read_parquet(URLS_PARQUET)
    url_col = next((c for c in ["event_url", "url", "URL"] if c in df_urls.columns), None)
    if not url_col:
        raise ValueError(f"URL column not found. Available: {df_urls.columns}")
    print(f"{len(df_urls)} URLs loaded from {URLS_PARQUET.name} (column: '{url_col}')")

    urls = df_urls[url_col].to_list()
    filtered_count = 0
    for url in urls:
        month = url_to_month(url)
        if month == 0:
            continue  # Skip malformed URLs
        if min_month <= month <= max_month:
            yield month, url
            filtered_count += 1
    print(f"{filtered_count} URLs match the date range [{min_month} - {max_month}]")

# === 3. Download + extract + filter + append to Parquet (disk-based) ===
def download_and_append(url: str, month: int):
    raw_path = PROCESSED_DIR / f"{month}_raw.parquet"
    try:
        with requests.get(url, stream=True, timeout=30) as r: # stream is useless bcse of zipfile i guess
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pl.read_csv(
                        f,
                        separator="\t",
                        has_header=False,
                        columns=COLUMN_INDICES,
                        new_columns=COLUMN_NAMES,
                        dtypes=[
                            pl.Int64,   # GLOBALEVENTID
                            pl.Int32,   # SQLDATE
                            pl.Utf8,    # EventCode
                            pl.Int32,   # NumMentions
                            pl.Float64, # AvgTone
                            pl.Utf8,    # ActionGeo_CountryCode
                            pl.Utf8,    # Actor1Geo_CountryCode
                            pl.Utf8     # Actor2Geo_CountryCode
                        ],
                        null_values=["", "NULL"]
                    )
                    df = df.filter(
                        pl.col("ActionGeo_CountryCode").is_in(COUNTRIES_FIPS) |
                        pl.col("Actor1Geo_CountryCode").is_in(COUNTRIES_FIPS) |
                        pl.col("Actor2Geo_CountryCode").is_in(COUNTRIES_FIPS)
                    )
                    if df.is_empty():
                        return

                    table = df.to_arrow()

                    # THREAD-SAFE APPEND
                    if raw_path.exists():
                        writer = pq.ParquetWriter(raw_path, table.schema)
                        writer.write_table(table)
                        writer.close()
                    else:
                        pq.write_table(table, raw_path, compression="zstd")
    except Exception as e:
        print(f"[ERROR] Failed {url}: {e}")

# === 4. Final deduplication per month ===
def deduplicate_month(month: int):
    raw_path = PROCESSED_DIR / f"{month}_raw.parquet"
    final_path = PROCESSED_DIR / f"{month}.parquet"
    if not raw_path.exists():
        return
    try:
        df = pl.read_parquet(raw_path)
        before = len(df)
        df = df.unique(subset="GLOBALEVENTID", maintain_order=True)
        after = len(df)
        df.write_parquet(final_path, compression="zstd")
        raw_path.unlink()  # Clean up
        print(f"{month}.parquet → {after:,} unique events ({before - after:,} duplicates removed)")
    except Exception as e:
        print(f"[ERROR] Deduplication failed for {month}: {e}")

# === 5. Schema test on a real file ===
def test_schema() -> bool:
    test_url = "http://data.gdeltproject.org/gdeltv2/20241001000000.export.CSV.zip"
    print(f"\nTesting schema on: {test_url}")
    try:
        with requests.get(test_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pl.read_csv(
                        f, separator="\t",
                        has_header=False,
                        columns=COLUMN_INDICES,
                        new_columns=COLUMN_NAMES,
                        dtypes=[
                            pl.Int64,   # GLOBALEVENTID
                            pl.Int32,   # SQLDATE
                            pl.Utf8,    # EventCode
                            pl.Int32,   # NumMentions
                            pl.Float64, # AvgTone
                            pl.Utf8,    # ActionGeo_CountryCode
                            pl.Utf8,    # Actor1Geo_CountryCode
                            pl.Utf8     # Actor2Geo_CountryCode
                        ]
                    ).filter(pl.col("ActionGeo_CountryCode").is_in(COUNTRIES_FIPS))
                    print(f"OK: {len(df)} rows extracted")
                    print(f"Sample: {df.row(0, named=True)}")
                    return True
    except Exception as e:
        print(f"Schema test failed: {e}")
        return False

# === 6. Main: parallel + disk-safe ===
def main():
    print(f"Starting processing: {START_MONTH} → {END_MONTH}")
    print(f"Columns & indices: {dict(zip(COLUMN_NAMES, COLUMN_INDICES))}")
    print(f"Countries filtered: {len(COUNTRIES_FIPS)} FIPS codes\n")

    if not test_schema():
        print("Stopping: fix parsing first.")
        return

    # Generate URL list
    url_list = list(iter_urls_by_month(START_MONTH, END_MONTH))
    if not url_list:
        print("No files to process. Check date range and URL column.")
        return
    print(f"\n{len(url_list)} files to download.\n")

    # === PARALLEL DOWNLOAD (8 threads) ===
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_and_append, url, month)
            for month, url in url_list
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            pass

    # === DEDUPLICATION ===
    print("\nDeduplicating monthly files...")
    months = sorted({month for month, _ in url_list})
    for month in months:
        deduplicate_month(month)

    print(f"\nDone! {len(months)} months processed.")
    print(f"Output: {PROCESSED_DIR.resolve()}")

if __name__ == "__main__":
    main()