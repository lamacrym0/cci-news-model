import polars as pl
from pathlib import Path

CCI_PARQUET = 'data/cci/cci_ocde.parquet'
EVENTS_DIR = 'data/events/'
DATA_DIR = 'data/rows'
DATA_PARQUET = 'data/rows/data.parquet'
X_PARQUET = 'data/rows/X.parquet'
Y_PARQUET = 'data/rows/y.parquet'

# FIPS mapping dictionary (Alpha-3 -> FIPS two-letter)
alpha3_to_fips = {
    "CHL":"CL","CRI":"CR","POL":"PL","PRT":"PT","LTU":"LT",
    "CHN":"CH","ITA":"IT","FIN":"FI","LUX":"LU","RUS":"RU",
    "BRA":"BR","AUT":"AT","BEL":"BE","CHE":"SZ","HUN":"HU",
    "DEU":"GM","MEX":"MX","GRC":"GR","GBR":"UK","COL":"CO",
    "JPN":"JA","SWE":"SW","IND":"IN","KOR":"KS","TUR":"TU",
    "ISR":"IS","AUS":"AS","FRA":"FR","NLD":"NL","LVA":"LV",
    "SVK":"LO","CZE":"EZ","IDN":"ID","EST":"EN","USA":"US",
    "DNK":"DK","IRL":"EI","ZAF":"SA","ESP":"SP","NZL":"NZ","SVN":"SI"
}

cci = pl.scan_parquet(CCI_PARQUET)

# collect distinct months as python strings
months = cci.select(pl.col("TIME_PERIOD")).unique().collect().to_numpy()

results = []  # collect eager DataFrames here
Path(EVENTS_DIR).mkdir(parents=True, exist_ok=True)
events_dir = Path(EVENTS_DIR)

for month in months:
    # filter the month from cci and prepare SQLDATE column (eager)
    cci_month = (
        cci
        .filter(pl.col("TIME_PERIOD") == month)
        .with_columns(pl.col("TIME_PERIOD").alias("DATE"))
        .drop("TIME_PERIOD")
    )
    month = month[0].replace('-','')

    # ensure path matches parquet files; use glob if directory contains many files
    
    parquet_files = list(events_dir.glob(f"{month}.parquet"))

    if not parquet_files:
        # skip this month if no parquet files found
        print(f"Warning: no event files for month {month}, skipping")
        continue
    
    events_path = str(events_dir) + f"/{month}.parquet"
    # scan_parquet returns a LazyFrame; join lazily against the small eager cci_month by turning it into a lazy frame
    dx = pl.scan_parquet(events_path)
    dx = dx.with_columns(
        pl.col("SQLDATE")
        .cast(pl.Utf8)
        .str.to_date("%Y%m%d", strict=False)
        .dt.strftime("%Y-%m")
        .alias("DATE")
    ).drop("SQLDATE").with_columns(
        pl.when(
            pl.col("ActionGeo_CountryCode")
            .is_not_null())
        .then(
            pl.col("ActionGeo_CountryCode"))
        .otherwise(
            pl.col("Actor2Geo_CountryCode")
        )
        .alias("CountryCode")
    ).drop(['ActionGeo_CountryCode','Actor1Geo_CountryCode','Actor2Geo_CountryCode'])

    joined_lazy = dx.join(cci_month, on="DATE", how="inner")

    # collect the joined partition into memory and append
    joined = joined_lazy.collect()
    results.append(joined)


# concatenate all month-level results and write once
if results:
    all_rows = pl.concat(results, how="vertical")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    all_rows.write_parquet(DATA_PARQUET)
else:
    print("No data to store.")
    exit(1)

print("Data saved")
data = pl.scan_parquet(DATA_PARQUET)

# Map alpha-3 -> fips and add as new column
data = (
    data
    .with_columns(
        pl.col("REF_AREA").replace(alpha3_to_fips, default="UNKNOWN").alias("REF_AREA")
    )
)

data = data.filter(pl.col('REF_AREA') == pl.col('CountryCode'))

df = data.with_columns([
    pl.col("EventCode").cast(str).str.strip_chars().alias("EventCode_clean")
])

# Group by REF_AREA and index (month), then pivot
agg_df = df.group_by(["REF_AREA", "DATE", "EventCode_clean",'OBS_VALUE']).agg([
    pl.col("NumMentions").sum().alias("NumMentions_sum"),
    pl.col("AvgTone").mean().alias("AvgTone_mean")
])

# Pivot so each EventCode becomes a column
pivot_mentions = agg_df.collect().pivot(
    values="NumMentions_sum",
    index=["REF_AREA", "DATE",'OBS_VALUE'],
    columns="EventCode_clean",
)

pivot_tone = agg_df.collect().pivot(
    values="AvgTone_mean",
    index=["REF_AREA", "DATE",'OBS_VALUE'],
    columns="EventCode_clean"
)
pivot_mentions = pivot_mentions.rename({
    col: f"Mentions_{col}" for col in pivot_mentions.columns if col not in ["REF_AREA", "DATE", 'OBS_VALUE']
})

pivot_tone = pivot_tone.rename({
    col: f"Tone_{col}" for col in pivot_tone.columns if col not in ["REF_AREA", "DATE", 'OBS_VALUE']
})
pivot_mentions = pivot_mentions.fill_null(0)
pivot_tone = pivot_tone.fill_null(0)


# Join both pivoted tables
final_df = pivot_mentions.join(pivot_tone, on=["REF_AREA", "DATE"]).sort(['DATE','REF_AREA'])

final_df = final_df.with_columns(
    pl.row_index().alias('index')
)

# Step 1: Get unique REF_AREA values
unique_areas = final_df.select("REF_AREA").unique().sort("REF_AREA").to_list()

# Step 2: Create a mapping dictionary
area_to_id = {
    area: idx for idx, area in enumerate(unique_areas)
}

# Step 3: Apply the mapping
encoded_df = final_df.with_columns([
    pl.col("REF_AREA").replace(area_to_id).cast(int).alias("REF_AREA")
])

print("Save final data")

encoded_df.select(pl.exclude('OBS_VALUE','DATE')).write_parquet(X_PARQUET)
encoded_df.select(pl.col(['index','OBS_VALUE'])).write_parquet(Y_PARQUET)