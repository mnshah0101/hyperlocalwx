# hyperlocalwx — Minimal README

Tiny helpers for Team 46 to work with the hourly feature parquet and stations.

## Prerequisites

- Python ≥ 3.9
- OpenTopo API key (sign up at https://opentopography.org/developers)
- Natural Earth shapefiles (for distance-to-water features)

## Install

1. Clone this repository
2. Install the package:

```bash
cd hyperlocalwx
pip install -e .
```

This will automatically install dependencies: `pandas`, `pyarrow`, `geopandas`, `shapely`

3. Download Natural Earth shapefiles (required for distance-to-water features):

```bash
cd raw/natural_earth

# Download coastline
wget https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip
unzip ne_10m_coastline.zip

# Download lakes
wget https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_lakes.zip
unzip ne_10m_lakes.zip

# Clean up zip files
rm *.zip

cd ../..
```

**Why these files are needed:** The `dist_coast_m` and `dist_lake_m` features require Natural Earth shapefiles. Without them, these features will be NaN.

---

## Workflow Overview

The complete workflow has two main phases:

### Phase 1: Data Acquisition (Required First)
1. **Fetch raw data** - Download NOAA ISD-Lite weather data and DEM terrain data
2. **Build features** - Process raw data into feature parquet files

### Phase 2: Training & Modeling
3. **Create targets & split** - Generate next-hour prediction targets and train/valid splits
4. **Train models** - Use the prepared data to train ML models

---

## Getting Started

### Step 1: Fetch Data and Build Features

First, you must download and process the raw data for your region:

```bash
# Set your OpenTopo API key (required for DEM terrain features)
export OPENTOPO_KEY=your_key_here

# Fetch data and build features for a region (recommended - all-in-one)
python examples/01_fetch_and_build_region.py --region sf_coast_inland

# Or run steps individually:
# python scripts/get_data.py --region sf_coast_inland
# python scripts/build_features.py --region sf_coast_inland
```

**Available regions:**
- `sf_coast_inland` - San Francisco coast and inland areas
- `co_front_range` - Colorado Front Range
- `wa_rainshadow` - Washington rainshadow
- `lake_michigan` - Lake Michigan area
- `phoenix_uhi` - Phoenix urban heat island
- `cape_cod` - Cape Cod region

**This creates:**
- `data/features_hourly_<region>.parquet` - Main feature file with hourly data
- `data/station_index.parquet` - Station metadata (automatically built!)
- `data/station_aliases.json` - Station lookup aliases (automatically built!)

### Step 2: Create Targets and Train/Valid Split

After building features, prepare the data for training:

```bash
python examples/02_make_targets_and_split.py --region sf_coast_inland --q 0.8
```

This creates:
- `data/train_ready_<region>.parquet` - Features with next-hour targets
- `data/train.parquet` - Training split (80% by default)
- `data/valid.parquet` - Validation split (20% by default)

### Step 3: Train Your Model

Now you can use the prepared data for modeling (see examples below).

---

## Quickstart (After Data is Ready)

```python
import hyperlocalwx as wx

# 1) Load features (pipeline output)
df = wx.load_features("sf_coast_inland")  # or a full .parquet path

# 2) Create next-hour target
df = wx.create_next_hour_target(df)  # adds 'temp_c_tplus1'

# 3) Time-based split (not random)
train, valid = wx.time_split(df, train_quantile=0.8)

# 4) (Optional) enrich with station metadata (needs station_index.parquet)
valid = wx.enrich_with_station_meta(valid)

# 5) Baselines
print(wx.persistence_baseline(valid))
print(wx.climatology_baseline(train, valid))
```

---

## What’s in the features parquet

Produced by the pipeline as:

```
data/features_hourly_<region>.parquet
```

**Core columns**

* Keys: `time` (UTC, hourly), `station` (ICAO)
* Weather: `temp_c`, `dewpoint_c`, `wind_dir_deg`, `wind_spd_ms`
  Optional: `slp_hpa`, `precip_1h_mm`, `precip_6h_mm`
* Terrain (sampled at station): `elevation_m`, `slope_deg`, `aspect_deg`, `relief_1km_m`, `tpi_1km_m`
* Hydro proximity: `dist_coast_m`, `dist_lake_m`
* Encodings: `hour_utc`, `hour_sin`, `hour_cos`

> All times are **UTC**. Temperature rows already filtered to valid values.

---

## Station Index

The station index provides a unified view of all stations with metadata and static features. **It's now automatically built when you run `build_features.py`!**

### Automatic Creation (Recommended)

The station index is created automatically during feature building:

```bash
python scripts/build_features.py --region sf_coast_inland
```

This produces:
- `data/station_index.parquet` - One row per station with metadata + terrain features
- `data/station_aliases.json` - Alias map for ICAO/USAF/WBAN lookup

### Manual Creation (If Needed)

If you need to rebuild the index separately:

```bash
python scripts/build_station_index.py --region sf_coast_inland
```

Or in Python:

```python
import hyperlocalwx as wx

# Load your metadata
meta = pd.read_parquet("data/noaa_meta_sf_coast_inland.parquet")

# Build index (with optional static features)
idx = wx.build_station_index(
    meta_df=meta,
    region_key="sf_coast_inland",
    static_features_df=static_features  # optional
)

# Save aliases for quick lookup
wx.save_station_aliases(idx)
```

**Resolve any key to canonical ID**

```python
sid = wx.resolve_station_id("KSFO")          # "724940-23234"
sid2 = wx.resolve_station_id("724940-23234") # same
```

**Convenience selectors**

```python
from hyperlocalwx import stations_in_region, stations_in_bbox

r = stations_in_region("sf_coast_inland")
b = stations_in_bbox(-123.2, 36.8, -120.5, 38.8)
```

**Enrich hourly rows with station metadata**

```python
df = wx.load_features("sf_coast_inland")
df = wx.create_next_hour_target(df)
df = wx.enrich_with_station_meta(df)  # adds lat/lon/elev/static features
```

---

## Minimal modeling starter (XGBoost)

```python
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import hyperlocalwx as wx

df = wx.load_features("sf_coast_inland")
df = wx.create_next_hour_target(df)
train, valid = wx.time_split(df, 0.8)

FEATS = [
  "temp_c","dewpoint_c","wind_spd_ms","wind_dir_deg",
  "elevation_m","slope_deg","aspect_deg","relief_1km_m","tpi_1km_m",
  "dist_coast_m","dist_lake_m","hour_sin","hour_cos"
]

X_tr, y_tr = train[FEATS].fillna(train[FEATS].median(numeric_only=True)), train["temp_c_tplus1"]
X_va, y_va = valid[FEATS].fillna(train[FEATS].median(numeric_only=True)), valid["temp_c_tplus1"]

model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, n_jobs=-1)
model.fit(X_tr, y_tr)

pred = model.predict(X_va)
rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
mae  = float(mean_absolute_error(y_va, pred))
r2   = float(r2_score(y_va, pred))
print(f"RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
```

---

## Common pitfalls

* **Timezones:** keep everything in **UTC**.
* **Splits:** split by **time**, not random.
* **Leakage:** features must reflect current/past only for next-hour prediction.
* **Missing values:** impute per split (train vs valid), not globally.

---

## Troubleshooting

### Installation Issues

**`ModuleNotFoundError: No module named 'hyperlocalwx'`**
```bash
# Make sure you're in the project directory
cd hyperlocalwx
pip install -e .
```

**Missing dependencies during feature building**
```bash
# Reinstall with all dependencies
pip install -e . --force-reinstall
```

### Data Pipeline Issues

**`❌ Features file not found: data/features_hourly_<region>.parquet`**

You need to run the feature building pipeline first:
```bash
export OPENTOPO_KEY=your_key_here
python scripts/get_data.py --region <region>
python scripts/build_features.py --region <region>
```

**`❌ Station index not found: data/station_index.parquet`**

The station index should be built automatically, but you can rebuild it:
```bash
python scripts/build_station_index.py --region <region>
```

**`❌ DEM file not found`**

You need to set your OpenTopo API key before fetching data:
```bash
export OPENTOPO_KEY=your_actual_key_here
python scripts/get_data.py --region <region>
```

Get a free API key from: https://opentopography.org/developers

**`NaN values in 'dist_coast_m' and 'dist_lake_m'`**

You need to download Natural Earth shapefiles manually:
```bash
cd raw/natural_earth

# Download coastline
wget https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip
unzip ne_10m_coastline.zip

# Download lakes
wget https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_lakes.zip
unzip ne_10m_lakes.zip

# Clean up
rm *.zip
cd ../..

# Rebuild features with the shapefiles now present
python scripts/build_features.py --region <region>
```

These shapefiles are required for calculating distances to coastlines and lakes. Download from Natural Earth: https://www.naturalearthdata.com/downloads/10m-physical-vectors/

**`❌ Missing required columns: ['station', 'time', 'temp_c']`**

Make sure you're using the correct DataFrame from the pipeline:
```python
df = wx.load_features("sf_coast_inland")  # Correct
# Not: df = pd.read_csv("some_other_file.csv")
```

### Workflow Issues

**Empty train or validation set after split**

Check your train_quantile parameter:
```python
train, valid = wx.time_split(df, train_quantile=0.8)  # 80% train, 20% valid
```

**Duplicate columns after enrichment (_x, _y suffixes)**

This is expected if your DataFrame already has those features. The enrichment will only add missing columns. Use `suffixes=("", "_from_index")` if you want to explicitly see which columns came from the index.

### Error Messages

All functions now provide helpful error messages with:
- ❌ Clear description of what went wrong
- 💡 Suggestions on how to fix it
- 📝 Exact commands to run

If you see an error, read the full message - it will tell you exactly what to do!

---

## Quick Reference: Complete Workflow

```bash
# 1. Install the package
pip install -e .

# 2. Set your OpenTopo API key (free from opentopography.org)
export OPENTOPO_KEY=your_actual_key_here

# 3. Fetch data and build features (downloads NOAA data + terrain)
python examples/01_fetch_and_build_region.py --region sf_coast_inland

# 4. Create next-hour targets and train/valid split
python examples/02_make_targets_and_split.py --region sf_coast_inland --q 0.8

# 5. Train your models using the prepared data
# Files ready: data/train.parquet and data/valid.parquet
```

## What's New in This Version

**Automatic Station Index** - No need to manually build station index, it's created during feature building

**Multi-Region Support** - Region-specific filenames (station_index_<region>.parquet) prevent conflicts when working with multiple regions

**Better Error Messages** - All functions provide clear, helpful error messages with fix suggestions

**Complete Dependencies** - All required packages are now in `pyproject.toml`

**Standalone Tools** - New `scripts/build_station_index.py` for rebuilding index separately

**Progress Indicators** - Feature building shows progress for each step

**Input Validation** - Functions validate inputs and provide specific error messages

## API Reference

### Core Functions

**Data Loading**
- `load_features(region)` - Load features parquet by region name or path
- `load_station_index(path=None, region=None)` - Load station metadata index

**Preprocessing**
- `create_next_hour_target(df, drop_last=True)` - Add next-hour temperature target
- `time_split(df, train_quantile=0.8)` - Time-based train/valid split

**Station Management**
- `build_station_index(meta_df, region_key, static_features_df=None, out_path=None)` - Build station metadata index
- `save_station_aliases(station_index, out_path=None, region=None)` - Create alias lookup JSON
- `resolve_station_id(key, alias_path=None, region=None)` - Resolve ICAO/USAF/WBAN to station_id
- `stations_in_region(region, index_path=None)` - Get all stations in a region
- `stations_in_bbox(west, south, east, north, index_path=None, region=None)` - Get stations in bounding box
- `enrich_with_station_meta(df, index_path=None, region=None)` - Add station metadata to hourly data

**Baselines**
- `persistence_baseline(df_valid, target_col="temp_c_tplus1")` - Naive baseline (next hour = current hour)
- `climatology_baseline(df_train, df_valid, target_col="temp_c_tplus1")` - Per-station-hour average baseline

All functions include comprehensive error checking and helpful messages.
