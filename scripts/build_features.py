# scripts/build_features.py
from pathlib import Path
import argparse
import gzip
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from scipy.ndimage import sobel, maximum_filter, minimum_filter, uniform_filter

DATA = Path("data")
RAW = Path("raw")
NE = RAW/"natural_earth"


def parse_isdlite_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        raw = pd.read_csv(f, sep=r"\s+", engine="python", header=None)
    if raw.shape[1] == 11:
        raw.columns = ["year", "mo", "dy", "hr", "air_temp_c10", "dewpt_c10", "slp_hpa",
                       "wind_dir_deg", "wind_spd_ms10", "sky_code", "precip_tenths_mm"]
    elif raw.shape[1] == 12:
        raw.columns = ["year", "mo", "dy", "hr", "air_temp_c10", "dewpt_c10", "slp_hpa",
                       "wind_dir_deg", "wind_spd_ms10", "sky_code", "precip_1h_tenths_mm", "precip_6h_tenths_mm"]
    else:
        return pd.DataFrame()
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    def fix(col, scale=1.0, miss=-
            9999): return raw[col].replace(miss, np.nan)/scale
    out = pd.DataFrame()
    out["time"] = pd.to_datetime(
        dict(year=raw.year, month=raw.mo, day=raw.dy, hour=raw.hr), utc=True)
    out["temp_c"] = fix("air_temp_c10", 10.0)
    out["dewpoint_c"] = fix("dewpt_c10", 10.0)
    out["wind_dir_deg"] = fix("wind_dir_deg")
    out["wind_spd_ms"] = fix("wind_spd_ms10", 10.0)
    if "slp_hpa" in raw:
        out["slp_hpa"] = fix("slp_hpa", 10.0)
    if "precip_1h_tenths_mm" in raw:
        out["precip_1h_mm"] = fix("precip_1h_tenths_mm", 10.0)
    elif "precip_tenths_mm" in raw:
        out["precip_1h_mm"] = fix("precip_tenths_mm", 10.0)
    if "precip_6h_tenths_mm" in raw:
        out["precip_6h_mm"] = fix("precip_6h_tenths_mm", 10.0)
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
        {"elevation_m": (("y", "x"), z),
         "slope_deg": (("y", "x"), slope),
         "aspect_deg": (("y", "x"), aspect),
         "relief_1km_m": (("y", "x"), relief),
         "tpi_1km_m": (("y", "x"), tpi)},
        coords={"y": dem.y, "x": dem.x}
    ).rio.write_crs(dem.rio.crs)
    return ds


def sample_terrain(ds: xr.Dataset, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pts = gdf.to_crs(ds.rio.crs)
    xs, ys = pts.geometry.x, pts.geometry.y
    for var in ds.data_vars:
        pts[var] = [float(ds[var].sel(x=x, y=y, method="nearest").values)
                    for x, y in zip(xs, ys)]
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

    print(f"Building features for region: {args.region}")
    print("=" * 60)

    # Inputs produced by get_data.py
    manifest_path = DATA / f"isd_manifest_{args.region}.csv"
    meta_path = DATA / f"noaa_meta_{args.region}.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"❌ Manifest file not found: {manifest_path}\n"
            f"   Run 'python scripts/get_data.py --region {args.region}' first."
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"ERROR Metadata file not found: {meta_path}\n"
            f"   Run 'python scripts/get_data.py --region {args.region}' first."
        )

    print(f"✓ Loading manifest from {manifest_path}")
    man = pd.read_csv(manifest_path)
    print(f"✓ Loading metadata from {meta_path}")
    meta = pd.read_parquet(meta_path)
    print(f"  Found {len(man)} stations in manifest")

    # Parse hourly and tag station
    print(f"\nParsing ISD-Lite files...")
    parts = []
    parsed_count = 0
    for i, r in man.iterrows():
        path = Path(r["path"])
        if not path.exists():
            print(f"  WARNING  Skipping {r['station']}: file not found at {path}")
            continue

        df = parse_isdlite_gz(path)
        if df.empty:
            print(f" WARNING Skipping {r['station']}: no valid data in file")
            continue

        df["station"] = r["station"]       # ICAO
        df["station_id"] = r["station_id"]  # USAF-WBAN
        parts.append(df)
        parsed_count += 1
        if (parsed_count % 5 == 0) or (i == len(man) - 1):
            print(f"  Parsed {parsed_count}/{len(man)} stations...")

    if not parts:
        raise ValueError(
            f"ERROR No valid data parsed from any station!\n"
            f"   Check that ISD-Lite files exist and contain valid data."
        )

    print(f"✓ Successfully parsed {len(parts)} stations")
    hourly = pd.concat(parts, ignore_index=True).dropna(subset=["temp_c"])
    print(f"✓ Combined into {len(hourly):,} hourly observations")

    hourly["hour_utc"] = hourly["time"].dt.hour
    hourly["hour_sin"] = np.sin(2*np.pi*hourly["hour_utc"]/24)
    hourly["hour_cos"] = np.cos(2*np.pi*hourly["hour_utc"]/24)

    # Station lat/lon from meta (NOAA catalog)
    print(f"\nExtracting station coordinates...")
    st = meta[["ICAO", "LAT", "LON"]].dropna().rename(
        columns={"ICAO": "station", "LAT": "lat", "LON": "lon"})
    print(f"✓ Found coordinates for {len(st)} stations")

    gst = gpd.GeoDataFrame(
        st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326)

    # Terrain from DEM
    print(f"\nProcessing terrain features...")
    dem = RAW/f"dem/srtm_{args.region}.tif"
    if not dem.exists():
        raise FileNotFoundError(
            f"ERROR DEM file not found: {dem}\n"
            f"   Run 'python scripts/get_data.py --region {args.region}' with OPENTOPO_KEY set.\n"
            f"   Get a free API key from: https://opentopography.org/developers"
        )

    print(f"✓ Computing terrain derivatives from {dem.name}...")
    ds = terrain_derivatives(dem)
    print(f"✓ Sampling terrain at station locations...")
    gst = sample_terrain(ds, gst)
    print(f"✓ Computing distance to water features...")
    gst = add_distance_to_water(gst)

    print(f"\nMerging hourly data with static features...")
    static = gst.drop(columns="geometry")
    feat = hourly.merge(static, on="station", how="left")

    # Check for stations without terrain features
    missing_terrain = feat["elevation_m"].isna().sum()
    if missing_terrain > 0:
        print(f"WARNING  Warning: {missing_terrain} rows missing terrain features (station not in meta or DEM)")

    out = DATA/f"features_hourly_{args.region}.parquet"
    print(f"\nSaving features...")
    feat.to_parquet(out, index=False)
    print(f"✓ Saved {len(feat):,} rows → {out}")

    # Build station index (for enrich_with_station_meta and station lookup)
    print(f"\nBuilding station index...")
    try:
        import hyperlocalwx as wx
        # Prepare static features with station_id as key
        static_for_index = static.copy()
        if "station_id" not in static_for_index.columns and "station" in static_for_index.columns:
            static_for_index["station_id"] = hourly[["station", "station_id"]].drop_duplicates("station").set_index("station")["station_id"]

        # Use region-specific filenames to avoid conflicts
        idx = wx.build_station_index(
            meta_df=meta,
            region_key=args.region,
            static_features_df=static_for_index
            # out_path defaults to data/station_index_{region}.parquet
        )
        print(f"✓ Built station index: {len(idx)} stations → data/station_index_{args.region}.parquet")

        # Also save station aliases for quick lookup
        wx.save_station_aliases(idx, region=args.region)
        print(f"✓ Saved station aliases → data/station_aliases_{args.region}.json")
    except Exception as e:
        print(f"WARNING  Warning: Could not build station index: {e}")
        print(f"   You can build it later with: python scripts/build_station_index.py --region {args.region}")

    print(f"\n{'='*60}")
    print(" Feature building complete!")
    print(f"   Output: {out}")
    print(f"   Next step: python examples/02_make_targets_and_split.py --region {args.region}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import numpy as np
    main()
