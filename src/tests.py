import matplotlib.pyplot as plt
import polars as pl
import os
import glob
import numpy as np

dir_path = "data/events_by_month_filtered"

df_cci = pl.read_parquet("data/cci_ocde.parquet")

# plot cci distribution
plt.figure(figsize=(10, 6))
plt.hist(df_cci["OBS_VALUE"], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of CCI OCDE")
plt.xlabel("CCI OCDE")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

fips_codes = {
    'SVK': 'LO',  # Slovaquie :contentReference[oaicite:0]{index=0}
    'TUR': 'TU',  # Turquie :contentReference[oaicite:1]{index=1}
    'FRA': 'FR',  # France :contentReference[oaicite:2]{index=2}
    'CHN': 'CH',  # Chine :contentReference[oaicite:3]{index=3}
    'LVA': 'LG',  # Lettonie :contentReference[oaicite:4]{index=4}
    'ISR': 'IS',  # Israël :contentReference[oaicite:5]{index=5}
    'ZAF': 'SF',  # Afrique du Sud :contentReference[oaicite:6]{index=6}
    'FIN': 'FI',  # Finlande :contentReference[oaicite:7]{index=7}
    'GBR': 'UK',  # Royaume-Uni :contentReference[oaicite:8]{index=8}
    'CZE': 'EZ',  # République tchèque :contentReference[oaicite:9]{index=9}
    'POL': 'PL',  # Pologne :contentReference[oaicite:10]{index=10}
    'NLD': 'NL',  # Pays-Bas :contentReference[oaicite:11]{index=11}
    'KOR': 'KS',  # Corée du Sud :contentReference[oaicite:12]{index=12}
    'SVN': 'SI',  # Slovénie :contentReference[oaicite:13]{index=13}
    'IND': 'IN',  # Inde :contentReference[oaicite:14]{index=14}
    'EST': 'EN',  # Estonie :contentReference[oaicite:15]{index=15}
    'USA': 'US',  # États-Unis :contentReference[oaicite:16]{index=16}
    'SWE': 'SW',  # Suède :contentReference[oaicite:17]{index=17}
    'ESP': 'SP',  # Espagne :contentReference[oaicite:18]{index=18}
    'AUS': 'AS',  # Australie :contentReference[oaicite:19]{index=19}
    'NZL': 'NZ',  # Nouvelle-Zélande :contentReference[oaicite:20]{index=20}
    'RUS': 'RS',  # Russie :contentReference[oaicite:21]{index=21}
    'IRL': 'EI',  # Irlande :contentReference[oaicite:22]{index=22}
    'DEU': 'GM',  # Allemagne :contentReference[oaicite:23]{index=23}
    'LTU': 'LH',  # Lituanie :contentReference[oaicite:24]{index=24}
    'MEX': 'MX',  # Mexique :contentReference[oaicite:25]{index=25}
    'COL': 'CO',  # Colombie :contentReference[oaicite:26]{index=26}
    'JPN': 'JA',  # Japon :contentReference[oaicite:27]{index=27}
    'CHE': 'SZ',  # Suisse :contentReference[oaicite:28]{index=28}
    'GRC': 'GR',  # Grèce :contentReference[oaicite:29]{index=29}
    'ITA': 'IT',  # Italie :contentReference[oaicite:30]{index=30}
    'BEL': 'BE',  # Belgique :contentReference[oaicite:31]{index=31}
    'LUX': 'LU',  # Luxembourg :contentReference[oaicite:32]{index=32}
    'CRI': 'CS',  # Costa Rica :contentReference[oaicite:33]{index=33}
    'AUT': 'AU',  # Autriche :contentReference[oaicite:34]{index=34}
    'CHL': 'CI',  # Chili :contentReference[oaicite:35]{index=35}
    'DNK': 'DA',  # Danemark :contentReference[oaicite:36]{index=36}
    'IDN': 'ID',  # Indonésie :contentReference[oaicite:37]{index=37}
    'PRT': 'PO',  # Portugal :contentReference[oaicite:38]{index=38}
    'HUN': 'HU',  # Hongrie :contentReference[oaicite:39]{index=39}
    'BRA': 'BR',  # Brésil :contentReference[oaicite:40]{index=40}
}

df_cci = df_cci.with_columns(
    pl.col("REF_AREA").replace(fips_codes).alias("FIPS")
).drop("REF_AREA")

countries = df_cci["FIPS"].unique().to_list()

# create a dictionary with countries as keys and counter value 0
country_counters = {country: 0 for country in countries}

for file_path in glob.glob(os.path.join(dir_path, "*.parquet")):
    df_event = pl.read_parquet(file_path)

    for country in countries:
        # count events for each country
        country_counters[country] += df_event.filter((pl.col("ActionGeo_CountryCode") == country)
                                                     | (pl.col("Actor1Geo_CountryCode") == country)
                                                     | (pl.col("Actor2Geo_CountryCode") == country)).height

# bar plot of country_counters
plt.figure(figsize=(12, 8))
plt.bar(country_counters.keys(), country_counters.values(), color='lightgreen', edgecolor='black')
plt.title("Number of Events by Country")
plt.xlabel("Country")
plt.ylabel("Number of Events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

vals = np.array(list(country_counters.values()))
print("Min:", vals.min(), " | Median:", np.median(vals), " | Max:", vals.max())
print("Countries with >1e6 events:", np.sum(vals > 1e6))