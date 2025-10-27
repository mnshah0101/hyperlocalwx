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
    return pd.read_parquet(DATA_DIR / f"features_hourly_{region}.parquet")

# -------------------------
# Targets & Splits
# -------------------------


def create_next_hour_target(df: pd.DataFrame, drop_last=True) -> pd.DataFrame:
    """
    Adds `temp_c_tplus1` per station (Δt=+1h). Returns a new DataFrame.
    """
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
    cut = df["time"].quantile(train_quantile)
    train = df[df["time"] <= cut].copy()
    valid = df[df["time"] > cut].copy()
    return train, valid

# -------------------------
# Station index & aliases
# -------------------------


def load_station_index(path: str | Path = DATA_DIR / "station_index.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def build_station_index(
    meta_df: pd.DataFrame,
    region_key: str,
    static_features_df: Optional[pd.DataFrame] = None,
    out_path: str | Path = DATA_DIR / "station_index.parquet",
) -> pd.DataFrame:
    """
    Create canonical station index from catalog metadata (and optional static features).
    Expected meta_df columns: USAF, WBAN, ICAO, STATION_NAME, CTRY, STATE, LAT, LON, ELEV_M, BEGIN, END
    """
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
        out = out.merge(sf.drop_duplicates("station_id"),
                        on="station_id", how="left")

    out.to_parquet(out_path, index=False)
    return out


def save_station_aliases(
    station_index: pd.DataFrame,
    out_path: str | Path = DATA_DIR / "station_aliases.json"
) -> Dict[str, str]:
    """
    Build and save alias map where keys like 'KSFO', '724940-23234', '724940', '23234'
    resolve to canonical 'USAF-WBAN'.
    """
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


def resolve_station_id(key: str, alias_path: str | Path = DATA_DIR / "station_aliases.json") -> Optional[str]:
    """
    Resolve any key (ICAO, station_id, USAF, WBAN) to canonical station_id.
    """
    alias = json.loads(Path(alias_path).read_text())
    return alias.get(str(key).upper(), None)


def stations_in_region(region: str, index_path: str | Path = DATA_DIR / "station_index.parquet") -> pd.DataFrame:
    idx = load_station_index(index_path)
    return idx[idx["region"] == region].copy()


def stations_in_bbox(
    west: float, south: float, east: float, north: float,
    index_path: str | Path = DATA_DIR / "station_index.parquet"
) -> pd.DataFrame:
    idx = load_station_index(index_path)
    m = (idx["lon"].between(west, east)) & (idx["lat"].between(south, north))
    return idx[m].copy()


def enrich_with_station_meta(
    df_hourly: pd.DataFrame,
    index_path: str | Path = DATA_DIR / "station_index.parquet"
) -> pd.DataFrame:
    """
    Add station metadata + static features to an hourly dataframe.
    Accepts df with 'station' (ICAO) or 'station_id'.
    """
    idx = load_station_index(index_path)
    out = df_hourly.copy()
    if "station_id" not in out.columns and "station" in out.columns:
        icao_to_id = dict(idx[["icao", "station_id"]].dropna().values)
        out["station_id"] = out["station"].map(icao_to_id)

    keep = ["station_id", "icao", "name", "lat", "lon", "elev_m", "region"]
    # include common static features if present
    for c in ["elevation_m", "slope_deg", "aspect_deg", "relief_1km_m", "tpi_1km_m", "dist_coast_m", "dist_lake_m"]:
        if c in idx.columns:
            keep.append(c)

    return out.merge(idx[keep].drop_duplicates("station_id"), on="station_id", how="left")

# -------------------------
# Simple baselines
# -------------------------


def persistence_baseline(df_valid: pd.DataFrame, target_col: str = "temp_c_tplus1") -> Dict[str, float]:
    """
    Predict next hour = current hour. Returns RMSE/MAE/Bias.
    """
    y = df_valid[target_col].to_numpy()
    yhat = df_valid["temp_c"].to_numpy()
    err = yhat - y
    return {
        "RMSE": float(np.sqrt(np.nanmean(err**2))),
        "MAE": float(np.nanmean(np.abs(err))),
        "Bias": float(np.nanmean(err)),
        "n": int(np.isfinite(err).sum())
    }


def climatology_baseline(df_train: pd.DataFrame, df_valid: pd.DataFrame,
                         target_col: str = "temp_c_tplus1") -> Dict[str, float]:
    """
    Per-(station,hour_utc) mean on train, applied to valid.
    """
    clim = (df_train.groupby(["station", "hour_utc"])[target_col]
            .mean().rename("clim").reset_index())
    v = df_valid.merge(clim, on=["station", "hour_utc"], how="left")
    y = v[target_col].to_numpy()
    yhat = v["clim"].to_numpy()
    err = yhat - y
    return {
        "RMSE": float(np.sqrt(np.nanmean(err**2))),
        "MAE": float(np.nanmean(np.abs(err))),
        "Bias": float(np.nanmean(err)),
        "n": int(np.isfinite(err).sum())
    }
