# scripts/build_features.py
from pathlib import Path
import argparse, gzip, numpy as np, pandas as pd
import geopandas as gpd, rioxarray as rxr, xarray as xr
from scipy.ndimage import sobel, maximum_filter, minimum_filter, uniform_filter

DATA = Path("data"); RAW = Path("raw"); NE = RAW/"natural_earth"

def parse_isdlite_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        raw = pd.read_csv(f, sep=r"\s+", engine="python", header=None)
    if raw.shape[1] == 11:
        raw.columns = ["year","mo","dy","hr","air_temp_c10","dewpt_c10","slp_hpa",
                       "wind_dir_deg","wind_spd_ms10","sky_code","precip_tenths_mm"]
    elif raw.shape[1] == 12:
        raw.columns = ["year","mo","dy","hr","air_temp_c10","dewpt_c10","slp_hpa",
                       "wind_dir_deg","wind_spd_ms10","sky_code","precip_1h_tenths_mm","precip_6h_tenths_mm"]
    else:
        return pd.DataFrame()
    for c in raw.columns: raw[c] = pd.to_numeric(raw[c], errors="coerce")
    def fix(col, scale=1.0, miss=-9999): return raw[col].replace(miss, np.nan)/scale
    out = pd.DataFrame()
    out["time"]         = pd.to_datetime(dict(year=raw.year, month=raw.mo, day=raw.dy, hour=raw.hr), utc=True)
    out["temp_c"]       = fix("air_temp_c10", 10.0)
    out["dewpoint_c"]   = fix("dewpt_c10", 10.0)
    out["wind_dir_deg"] = fix("wind_dir_deg")
    out["wind_spd_ms"]  = fix("wind_spd_ms10", 10.0)
    if "slp_hpa" in raw: out["slp_hpa"] = fix("slp_hpa", 10.0)
    if "precip_1h_tenths_mm" in raw: out["precip_1h_mm"] = fix("precip_1h_tenths_mm", 10.0)
    elif "precip_tenths_mm" in raw:  out["precip_1h_mm"] = fix("precip_tenths_mm", 10.0)
    if "precip_6h_tenths_mm" in raw: out["precip_6h_mm"] = fix("precip_6h_tenths_mm", 10.0)
    return out.dropna(subset=["temp_c"])

def terrain_derivatives(dem_path: Path) -> xr.Dataset:
    dem = rxr.open_rasterio(dem_path).squeeze()
    resx, resy = dem.rio.resolution()
    z = dem.values.astype(float)
    dzdx = sobel(z, axis=1) / (8*abs(resx))
    dzdy = sobel(z, axis=0) / (8*abs(resy))
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdy, -dzdx)) + 180.0) % 360.0
    win = 33
    relief = maximum_filter(z, size=win) - minimum_filter(z, size=win)
    mean1k = uniform_filter(z, size=win)
    tpi = z - mean1k
    ds = xr.Dataset(
        {"elevation_m": (("y","x"), z),
         "slope_deg": (("y","x"), slope),
         "aspect_deg": (("y","x"), aspect),
         "relief_1km_m": (("y","x"), relief),
         "tpi_1km_m": (("y","x"), tpi)},
        coords={"y": dem.y, "x": dem.x}
    ).rio.write_crs(dem.rio.crs)
    return ds

def sample_terrain(ds: xr.Dataset, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pts = gdf.to_crs(ds.rio.crs)
    xs, ys = pts.geometry.x, pts.geometry.y
    for var in ds.data_vars:
        pts[var] = [float(ds[var].sel(x=x, y=y, method="nearest").values) for x,y in zip(xs,ys)]
    return pts.to_crs(4326)

def add_distance_to_water(gdf_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf_points.to_crs(3857)
    coast = NE/"ne_10m_coastline.shp"
    lakes = NE/"ne_10m_lakes.shp"
    if coast.exists():
        c = gpd.read_file(str(coast)).to_crs(3857)
        g["dist_coast_m"] = g.distance(c.unary_union)
    else:
        g["dist_coast_m"] = np.nan
    if lakes.exists():
        l = gpd.read_file(str(lakes)).to_crs(3857)
        g["dist_lake_m"] = g.distance(l.unary_union)
    else:
        g["dist_lake_m"] = np.nan
    return g.to_crs(4326)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()

    # Inputs produced by get_data.py
    man = pd.read_csv(DATA/f"isd_manifest_{args.region}.csv")
    meta = pd.read_parquet(DATA/f"noaa_meta_{args.region}.parquet")

    # Parse hourly and tag station
    parts = []
    for _, r in man.iterrows():
        df = parse_isdlite_gz(Path(r["path"]))
        if df.empty: continue
        df["station"] = r["station"]       # ICAO
        df["station_id"] = r["station_id"] # USAF-WBAN
        parts.append(df)
    hourly = pd.concat(parts, ignore_index=True).dropna(subset=["temp_c"])
    hourly["hour_utc"] = hourly["time"].dt.hour
    hourly["hour_sin"] = np.sin(2*np.pi*hourly["hour_utc"]/24)
    hourly["hour_cos"] = np.cos(2*np.pi*hourly["hour_utc"]/24)

    # Station lat/lon from meta (NOAA catalog)
    st = meta[["ICAO","LAT","LON"]].dropna().rename(columns={"ICAO":"station","LAT":"lat","LON":"lon"})
    gst = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326)

    # Terrain from DEM
    dem = RAW/f"dem/srtm_{args.region}.tif"
    if not dem.exists():
        raise SystemExit(f"Missing DEM: {dem}. Run get_data.py with OPENTOPO_KEY set.")
    ds = terrain_derivatives(dem)
    gst = sample_terrain(ds, gst)
    gst = add_distance_to_water(gst)

    static = gst.drop(columns="geometry")
    feat = hourly.merge(static, on="station", how="left")

    out = DATA/f"features_hourly_{args.region}.parquet"
    feat.to_parquet(out, index=False)
    print(f"Saved {len(feat):,} rows → {out}")

if __name__ == "__main__":
    import numpy as np
    main()
