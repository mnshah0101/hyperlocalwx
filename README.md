# hyperlocalwx — Minimal README

Tiny helpers for Team 46 to work with the hourly feature parquet and stations.

## Prerequisites

- Python ≥ 3.9
- OpenTopo API key (sign up at https://opentopography.org/developers)

## Install

1. Clone this repository
2. Install the package:

```bash
cd hyperlocalwx
pip install -e .
```

This will automatically install dependencies: `pandas`, `pyarrow`, `geopandas`, `shapely`

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

# Fetch data and build features for a region
python examples/01_fetch_and_build_region.py --region sf_coast_inland
```

**Available regions:**
- `sf_coast_inland` - San Francisco coast and inland areas
- `co_front_range` - Colorado Front Range
- `wa_rainshadow` - Washington rainshadow
- `lake_michigan` - Lake Michigan area
- `phoenix_uhi` - Phoenix urban heat island
- `cape_cod` - Cape Cod region

This creates: `data/features_hourly_<region>.parquet`

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

## Station index (one-time per region)

Build a canonical station index and alias map so any key (ICAO or USAF/WBAN) resolves to a single `station_id` (`USAF-WBAN`).

```python
import hyperlocalwx as wx

# meta_df = output of your station selection step (from NOAA catalog)
# static_df = per-station static features (elevation_m, dist_coast_m, etc.)
idx = wx.build_station_index(meta_df, region_key="sf_coast_inland", static_features_df=static_df)

# write alias map for quick resolution (KSFO -> 724940-23234, etc.)
wx.save_station_aliases(idx)
```

**Files produced**

```
data/station_index.parquet     # one row per station (metadata + optional static features)
data/station_aliases.json      # alias -> station_id map
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

### `ModuleNotFoundError: No module named 'hyperlocalwx'`
Make sure you've run `pip install -e .` from the project root directory.

### `OPENTOPO_KEY not set` warning
Get a free API key from https://opentopography.org/developers and set it:
```bash
export OPENTOPO_KEY=your_actual_key_here
```

### `FileNotFoundError` when running step 2
You must run Step 1 first to download and build the features. The workflow is sequential.

### Package installation fails with "Multiple top-level packages discovered"
This shouldn't happen with the current `pyproject.toml`, but if it does, make sure you have the `[tool.setuptools.packages.find]` section in `pyproject.toml`.

---

## Quick Reference: Complete Workflow

```bash
# 1. Install
pip install -e .

# 2. Set API key
export OPENTOPO_KEY=your_key_here

# 3. Fetch data and build features
python examples/01_fetch_and_build_region.py --region sf_coast_inland

# 4. Create targets and split
python examples/02_make_targets_and_split.py --region sf_coast_inland --q 0.8

# 5. Now you can train models using data/train.parquet and data/valid.parquet
```
