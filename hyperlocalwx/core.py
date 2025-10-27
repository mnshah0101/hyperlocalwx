from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict
import json
import pandas as pd
import numpy as np

DATA_DIR = Path("data")

# -------------------------
# I/O
# -------------------------


def load_features(region: str | Path) -> pd.DataFrame:
    """
    Load pipeline output parquet.
    region: region key (e.g., 'sf_coast_inland') or full path to a parquet
    """
    p = Path(region)
    if p.suffix == ".parquet" and p.exists():
        return pd.read_parquet(p)

    expected_path = DATA_DIR / f"features_hourly_{region}.parquet"
    if not expected_path.exists():
        raise FileNotFoundError(
            f"ERROR: Features file not found: {expected_path}\n"
            f"       Run 'python scripts/build_features.py --region {region}' first,\n"
            f"       or provide a full path to an existing parquet file."
        )
    return pd.read_parquet(expected_path)

# -------------------------
# Targets & Splits
# -------------------------


def create_next_hour_target(df: pd.DataFrame, drop_last=True) -> pd.DataFrame:
    """
    Adds `temp_c_tplus1` per station (Δt=+1h). Returns a new DataFrame.
    """
    # Validate required columns
    required = ["station", "time", "temp_c"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"ERROR: Missing required columns: {missing}\n"
            f"       DataFrame must have columns: {required}\n"
            f"       Current columns: {list(df.columns)}"
        )

    out = df.sort_values(["station", "time"]).copy()
    out["temp_c_tplus1"] = out.groupby("station")["temp_c"].shift(-1)
    if drop_last:
        out = out.dropna(subset=["temp_c_tplus1"]).reset_index(drop=True)
    return out


def time_split(df: pd.DataFrame, train_quantile: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split at a given quantile on 'time' column.
    Returns (train, valid).
    """
    if "time" not in df.columns:
        raise ValueError(
            f"ERROR: 'time' column not found in DataFrame.\n"
            f"       Available columns: {list(df.columns)}"
        )

    if not (0 < train_quantile < 1):
        raise ValueError(
            f"ERROR: train_quantile must be between 0 and 1, got: {train_quantile}"
        )

    cut = df["time"].quantile(train_quantile)
    train = df[df["time"] <= cut].copy()
    valid = df[df["time"] > cut].copy()

    if len(train) == 0 or len(valid) == 0:
        raise ValueError(
            f"ERROR: Split resulted in empty dataset!\n"
            f"       Train size: {len(train)}, Valid size: {len(valid)}\n"
            f"       Split time: {cut}, Quantile: {train_quantile}"
        )

    return train, valid

# -------------------------
# Station index & aliases
# -------------------------


def load_station_index(
    path: str | Path | None = None,
    region: str | None = None
) -> pd.DataFrame:
    """
    Load station index parquet file with proper error handling.

    Parameters:
    -----------
    path : str | Path | None
        Direct path to station index file. If None, will use region parameter.
    region : str | None
        Region name to load index for (e.g., 'sf_coast_inland').
        Creates path: data/station_index_{region}.parquet

    Returns:
    --------
    pd.DataFrame : Station index with metadata and static features
    """
    if path is None and region is None:
        # Default to generic station_index.parquet for backwards compatibility
        path = DATA_DIR / "station_index.parquet"
    elif path is None and region is not None:
        # Use region-specific file
        path = DATA_DIR / f"station_index_{region}.parquet"

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ERROR: Station index not found: {p}\n"
            f"       Build it first using one of these methods:\n"
            f"       1. Run: python scripts/build_station_index.py --region <your_region>\n"
            f"       2. Or in Python: wx.build_station_index(meta_df, region_key='...')\n"
            f"       Tip: Use region-specific files (station_index_<region>.parquet) to avoid conflicts."
        )
    return pd.read_parquet(p)


def build_station_index(
    meta_df: pd.DataFrame,
    region_key: str,
    static_features_df: Optional[pd.DataFrame] = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Create canonical station index from catalog metadata (and optional static features).

    Parameters:
    -----------
    meta_df : pd.DataFrame
        Station metadata from NOAA catalog.
        Expected columns: USAF, WBAN, ICAO, STATION_NAME, CTRY, STATE, LAT, LON, ELEV_M, BEGIN, END
    region_key : str
        Region identifier (e.g., 'sf_coast_inland')
    static_features_df : pd.DataFrame, optional
        Per-station static features (elevation_m, slope_deg, etc.)
    out_path : str | Path | None
        Output path. If None, defaults to data/station_index_{region_key}.parquet

    Returns:
    --------
    pd.DataFrame : Station index with all metadata and static features
    """
    # Default to region-specific output path
    if out_path is None:
        out_path = DATA_DIR / f"station_index_{region_key}.parquet"

    # Validate required columns
    required = ["USAF", "WBAN"]
    missing = [c for c in required if c not in meta_df.columns]
    if missing:
        raise ValueError(
            f"ERROR: Missing required columns in meta_df: {missing}\n"
            f"       Required: {required}\n"
            f"       Available: {list(meta_df.columns)}"
        )

    if meta_df.empty:
        raise ValueError("ERROR: meta_df is empty! Cannot build station index from empty DataFrame.")

    df = meta_df.copy()
    df["USAF"] = df["USAF"].astype(str).str.strip()
    df["WBAN"] = df["WBAN"].astype(str).str.strip()
    df["station_id"] = df["USAF"] + "-" + df["WBAN"]

    out = pd.DataFrame({
        "station_id": df["station_id"],
        "icao": df.get("ICAO"),
        "usaf": df["USAF"],
        "wban": df["WBAN"],
        "name": df.get("STATION_NAME"),
        "country": df.get("CTRY"),
        "state": df.get("STATE"),
        "lat": pd.to_numeric(df.get("LAT"), errors="coerce"),
        "lon": pd.to_numeric(df.get("LON"), errors="coerce"),
        "elev_m": pd.to_numeric(df.get("ELEV_M"), errors="coerce"),
        "begin": pd.to_numeric(df.get("BEGIN"), errors="coerce"),
        "end": pd.to_numeric(df.get("END"), errors="coerce"),
        "region": region_key,
    })

    if static_features_df is not None and not static_features_df.empty:
        sf = static_features_df.copy()
        # Try to key by station_id; if missing, map from ICAO
        if "station_id" not in sf.columns:
            if "station" in sf.columns:
                # map ICAO -> station_id from 'out'
                icao_to_id = dict(out[["icao", "station_id"]].dropna().values)
                sf["station_id"] = sf["station"].map(icao_to_id)

        # Drop columns that already exist in 'out' to avoid duplicates (_x, _y suffixes)
        # Keep lat/lon from metadata (more authoritative), but use elevation_m from terrain
        cols_to_drop = []
        for col in sf.columns:
            if col in out.columns and col not in ["station_id", "elev_m"]:
                cols_to_drop.append(col)
        if cols_to_drop:
            sf = sf.drop(columns=cols_to_drop)

        out = out.merge(sf.drop_duplicates("station_id"),
                        on="station_id", how="left")

    out.to_parquet(out_path, index=False)
    return out


def save_station_aliases(
    station_index: pd.DataFrame,
    out_path: str | Path | None = None,
    region: str | None = None
) -> Dict[str, str]:
    """
    Build and save alias map where keys like 'KSFO', '724940-23234', '724940', '23234'
    resolve to canonical 'USAF-WBAN'.

    Parameters:
    -----------
    station_index : pd.DataFrame
        Station index with station_id, icao, usaf, wban columns
    out_path : str | Path | None
        Output path. If None, will use region parameter or default.
    region : str | None
        Region name to create path: data/station_aliases_{region}.json

    Returns:
    --------
    Dict[str, str] : Alias mapping dictionary
    """
    # Determine output path
    if out_path is None and region is None:
        out_path = DATA_DIR / "station_aliases.json"  # backwards compatibility
    elif out_path is None and region is not None:
        out_path = DATA_DIR / f"station_aliases_{region}.json"

    alias: Dict[str, str] = {}
    for _, r in station_index.iterrows():
        sid = str(r["station_id"])
        alias[sid] = sid
        icao = r.get("icao")
        if pd.notna(icao):
            alias[str(icao).upper()] = sid
        usaf = r.get("usaf")
        if pd.notna(usaf):
            alias[str(usaf).zfill(6)] = sid
        wban = r.get("wban")
        if pd.notna(wban):
            alias[str(wban).zfill(5)] = sid

    Path(out_path).write_text(json.dumps(alias, indent=2))
    return alias


def resolve_station_id(
    key: str,
    alias_path: str | Path | None = None,
    region: str | None = None
) -> Optional[str]:
    """
    Resolve any key (ICAO, station_id, USAF, WBAN) to canonical station_id.

    Parameters:
    -----------
    key : str
        Station identifier (ICAO like 'KSFO', or USAF-WBAN like '724940-23234')
    alias_path : str | Path | None
        Direct path to aliases JSON. If None, will use region parameter.
    region : str | None
        Region name to load aliases for

    Returns:
    --------
    str | None : Canonical station_id, or None if not found
    """
    if alias_path is None and region is None:
        alias_path = DATA_DIR / "station_aliases.json"
    elif alias_path is None and region is not None:
        alias_path = DATA_DIR / f"station_aliases_{region}.json"

    p = Path(alias_path)
    if not p.exists():
        raise FileNotFoundError(
            f"ERROR: Station aliases file not found: {p}\n"
            f"       Build it first using:\n"
            f"       python scripts/build_station_index.py --region <your_region>\n"
            f"       This will create both station_index.parquet and station_aliases.json"
        )

    try:
        alias = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"ERROR: Invalid JSON in {p}: {e}\n"
            f"       The file may be corrupted. Rebuild it using:\n"
            f"       python scripts/build_station_index.py --region <your_region>"
        )

    return alias.get(str(key).upper(), None)


def stations_in_region(
    region: str,
    index_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Get all stations in a specific region.

    Parameters:
    -----------
    region : str
        Region identifier
    index_path : str | Path | None
        Path to station index. If None, will try region-specific file first.

    Returns:
    --------
    pd.DataFrame : Stations in the specified region
    """
    if index_path is None:
        # Try region-specific file first
        region_specific = DATA_DIR / f"station_index_{region}.parquet"
        if region_specific.exists():
            idx = load_station_index(path=region_specific)
        else:
            # Fall back to generic file
            idx = load_station_index()
    else:
        idx = load_station_index(index_path)

    return idx[idx["region"] == region].copy()


def stations_in_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    index_path: str | Path | None = None,
    region: str | None = None
) -> pd.DataFrame:
    """
    Get all stations within a bounding box.

    Parameters:
    -----------
    west, south, east, north : float
        Bounding box coordinates in degrees
    index_path : str | Path | None
        Path to station index. If None, will use region parameter.
    region : str | None
        Region name to load index for

    Returns:
    --------
    pd.DataFrame : Stations within the bounding box
    """
    idx = load_station_index(path=index_path, region=region)
    m = (idx["lon"].between(west, east)) & (idx["lat"].between(south, north))
    return idx[m].copy()


def enrich_with_station_meta(
    df_hourly: pd.DataFrame,
    index_path: str | Path | None = None,
    region: str | None = None
) -> pd.DataFrame:
    """
    Add station metadata + static features to an hourly dataframe.

    Parameters:
    -----------
    df_hourly : pd.DataFrame
        Hourly weather data with 'station' (ICAO) or 'station_id' column
    index_path : str | Path | None
        Direct path to station index. If None, will use region parameter.
    region : str | None
        Region name to load index for

    Returns:
    --------
    pd.DataFrame : Enriched dataframe with station metadata

    Notes:
    ------
    If columns already exist in df_hourly, they will NOT be overwritten.
    """
    # Validate input has station identifier
    if "station_id" not in df_hourly.columns and "station" not in df_hourly.columns:
        raise ValueError(
            f"ERROR: DataFrame must have either 'station' or 'station_id' column.\n"
            f"       Available columns: {list(df_hourly.columns)}"
        )

    idx = load_station_index(path=index_path, region=region)
    out = df_hourly.copy()

    if "station_id" not in out.columns and "station" in out.columns:
        icao_to_id = dict(idx[["icao", "station_id"]].dropna().values)
        out["station_id"] = out["station"].map(icao_to_id)

        # Check for unmapped stations
        unmapped = out["station_id"].isna().sum()
        if unmapped > 0:
            unique_unmapped = out[out["station_id"].isna()]["station"].unique()
            print(
                f"WARNING: {unmapped} rows have stations not found in index.\n"
                f"         Unmapped stations: {list(unique_unmapped)}\n"
                f"         These rows will have NaN for metadata fields."
            )

    keep = ["station_id", "icao", "name", "lat", "lon", "elev_m", "region"]
    # include common static features if present in index AND not already in df_hourly
    for c in ["elevation_m", "slope_deg", "aspect_deg", "relief_1km_m", "tpi_1km_m", "dist_coast_m", "dist_lake_m"]:
        if c in idx.columns and c not in out.columns:
            keep.append(c)

    return out.merge(idx[keep].drop_duplicates("station_id"), on="station_id", how="left", suffixes=("", "_from_index"))

# -------------------------
# Simple baselines
# -------------------------


def persistence_baseline(df_valid: pd.DataFrame, target_col: str = "temp_c_tplus1") -> Dict[str, float]:
    """
    Predict next hour = current hour. Returns RMSE/MAE/Bias.
    """
    required = ["temp_c", target_col]
    missing = [c for c in required if c not in df_valid.columns]
    if missing:
        raise ValueError(
            f"ERROR: Missing required columns: {missing}\n"
            f"       Available columns: {list(df_valid.columns)}\n"
            f"       Hint: Did you call create_next_hour_target() to add '{target_col}'?"
        )

    y = df_valid[target_col].to_numpy()
    yhat = df_valid["temp_c"].to_numpy()
    err = yhat - y

    n_valid = int(np.isfinite(err).sum())
    if n_valid == 0:
        raise ValueError(
            f"ERROR: No valid data points for baseline calculation!\n"
            f"       All {len(err)} rows have NaN/Inf values in temp_c or {target_col}."
        )

    return {
        "RMSE": float(np.sqrt(np.nanmean(err**2))),
        "MAE": float(np.nanmean(np.abs(err))),
        "Bias": float(np.nanmean(err)),
        "n": n_valid
    }


def climatology_baseline(df_train: pd.DataFrame, df_valid: pd.DataFrame,
                         target_col: str = "temp_c_tplus1") -> Dict[str, float]:
    """
    Per-(station,hour_utc) mean on train, applied to valid.
    """
    required = ["station", "hour_utc", target_col]
    for df_name, df in [("df_train", df_train), ("df_valid", df_valid)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"ERROR: Missing required columns in {df_name}: {missing}\n"
                f"       Required: {required}\n"
                f"       Available: {list(df.columns)}"
            )

    if df_train.empty:
        raise ValueError("ERROR: df_train is empty! Cannot compute climatology from empty training set.")

    clim = (df_train.groupby(["station", "hour_utc"])[target_col]
            .mean().rename("clim").reset_index())
    v = df_valid.merge(clim, on=["station", "hour_utc"], how="left")

    # Check for missing climatology
    missing_clim = v["clim"].isna().sum()
    if missing_clim > 0:
        print(
            f"WARNING: {missing_clim}/{len(v)} validation rows have no climatology match.\n"
            f"         This happens when valid has station-hour combinations not in train."
        )

    y = v[target_col].to_numpy()
    yhat = v["clim"].to_numpy()
    err = yhat - y

    n_valid = int(np.isfinite(err).sum())
    if n_valid == 0:
        raise ValueError("ERROR: No valid predictions! All climatology lookups resulted in NaN.")

    return {
        "RMSE": float(np.sqrt(np.nanmean(err**2))),
        "MAE": float(np.nanmean(np.abs(err))),
        "Bias": float(np.nanmean(err)),
        "n": n_valid
    }
