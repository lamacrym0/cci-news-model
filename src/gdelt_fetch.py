# src/ingestion/gdelt_fetch.py
# GDELT Events ingestion — Optimized, streaming, batched, server-proof
import polars as pl
from pathlib import Path
import zipfile
import tempfile
import requests
import re
from tqdm import tqdm
from typing import Iterator, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import shutil

# === GDELT CONFIGURATION ===
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

# === 1. Extract month from URL ===
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
            continue
        if min_month <= month <= max_month:
            yield month, url
            filtered_count += 1
    print(f"{filtered_count} URLs match date range [{min_month} - {max_month}]")

# === 3. Download + stream + parse + collect per month ===
def download_and_collect(url: str) -> Tuple[int, pl.DataFrame]:
    month = url_to_month(url)
    if month == 0:
        return month, pl.DataFrame()

    try:
        # Streaming download to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
            tmp_path = tmp.name

        # Extract and parse
        with zipfile.ZipFile(tmp_path) as z:
            with z.open(z.namelist()[0]) as f:
                df = pl.read_csv(
                    f,
                    separator="\t",
                    has_header=False,
                    columns=COLUMN_INDICES,
                    new_columns=COLUMN_NAMES,
                    dtypes=[
                        pl.Int64, pl.Int32, pl.Utf8, pl.Int32,
                        pl.Float64, pl.Utf8, pl.Utf8, pl.Utf8
                    ],
                    null_values=["", "NULL"]
                )
                df = df.filter(
                    pl.col("ActionGeo_CountryCode").is_in(COUNTRIES_FIPS) |
                    pl.col("Actor1Geo_CountryCode").is_in(COUNTRIES_FIPS) |
                    pl.col("Actor2Geo_CountryCode").is_in(COUNTRIES_FIPS)
                )
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        return month, df

    except Exception as e:
        print(f"[ERROR] Failed {url}: {e}")
        return month, pl.DataFrame()

# === 4. Process one month: collect, concat, dedup, write ===
def process_month(month: int, urls: List[str]):
    print(f"Processing month {month} ({len(urls)} files)...")
    dfs = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_and_collect, url) for url in urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Month {month}", leave=False):
            _, df = future.result()
            if not df.is_empty():
                dfs.append(df)

    if not dfs:
        print(f"{month}: No data after filtering.")
        return

    # Concat + deduplicate
    df_month = pl.concat(dfs)
    before = len(df_month)
    df_month = df_month.unique(subset="GLOBALEVENTID", maintain_order=True)
    after = len(df_month)

    # Write final Parquet
    final_path = PROCESSED_DIR / f"{month}.parquet"
    df_month.write_parquet(final_path, compression="zstd", compression_level=3)
    print(f"{month}.parquet → {after:,} events ({before - after:,} duplicates removed)")

# === 5. Schema test ===
def test_schema() -> bool:
    test_url = "http://data.gdeltproject.org/gdeltv2/20241001000000.export.CSV.zip"
    print(f"\nTesting schema on: {test_url}")
    try:
        month, df = download_and_collect(test_url)
        if df.is_empty():
            print("No rows match country filter.")
        else:
            print(f"OK: {len(df)} rows extracted")
            print(f"Sample: {df.row(0, named=True)}")
        return True
    except Exception as e:
        print(f"Schema test failed: {e}")
        return False

# === 6. Main ===
def main():
    print(f"Starting GDELT ingestion: {START_MONTH} → {END_MONTH}")
    print(f"Output → {PROCESSED_DIR.resolve()}")
    print(f"Countries filtered: {len(COUNTRIES_FIPS)} FIPS codes\n")

    if not test_schema():
        print("Stopping: fix parsing first.")
        return

    # Group URLs by month
    url_by_month = defaultdict(list)
    for month, url in iter_urls_by_month(START_MONTH, END_MONTH):
        url_by_month[month].append(url)

    if not url_by_month:
        print("No files to process.")
        return

    print(f"\n{len(url_by_month)} months to process\n")

    # Process months in parallel (one thread per month)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_month, month, urls)
            for month, urls in sorted(url_by_month.items())
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Months"):
            pass

    print(f"\nDone! Processed {len(url_by_month)} months.")
    print(f"Output: {PROCESSED_DIR.resolve()}")

if __name__ == "__main__":
    main()