#!/usr/bin/env python3
"""
Build station_index.parquet and station_aliases.json from existing data.

This is a quick fix to generate the station index files without needing
to re-run the full build_features.py pipeline.

Usage:
  python scripts/build_station_index.py --region sf_coast_inland
"""

import argparse
from pathlib import Path
import pandas as pd
import hyperlocalwx as wx

DATA = Path("data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="Region key")
    args = ap.parse_args()

    # Load existing metadata
    meta_path = DATA / f"noaa_meta_{args.region}.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing metadata: {meta_path}\n"
            f"Run 'python scripts/get_data.py --region {args.region}' first."
        )

    meta = pd.read_parquet(meta_path)
    print(f"Loaded {len(meta)} stations from {meta_path}")

    # Try to load features parquet for static features (terrain, etc.)
    features_path = DATA / f"features_hourly_{args.region}.parquet"
    static_df = None

    if features_path.exists():
        print(f"Loading features from {features_path} to extract static features...")
        feat = pd.read_parquet(features_path)

        # Extract static features per station
        static_cols = []
        for c in ["elevation_m", "slope_deg", "aspect_deg", "relief_1km_m",
                  "tpi_1km_m", "dist_coast_m", "dist_lake_m", "lat", "lon"]:
            if c in feat.columns:
                static_cols.append(c)

        if "station" in feat.columns:
            # Group by station and take first value (all static features should be the same per station)
            static_df = feat[["station"] + static_cols].drop_duplicates("station")

            # Also add station_id if available
            if "station_id" in feat.columns:
                station_id_map = feat[["station", "station_id"]].drop_duplicates("station")
                static_df = static_df.merge(station_id_map, on="station", how="left")

            print(f"Extracted static features: {static_cols}")
    else:
        print(f"Features file not found at {features_path}, building index without static features.")

    # Build station index (region-specific filename to avoid conflicts)
    idx = wx.build_station_index(
        meta_df=meta,
        region_key=args.region,
        static_features_df=static_df
        # out_path defaults to data/station_index_{region}.parquet
    )
    print(f"✓ Built station index: {len(idx)} stations → data/station_index_{args.region}.parquet")

    # Save aliases (region-specific filename)
    wx.save_station_aliases(idx, region=args.region)
    print(f"✓ Saved station aliases → data/station_aliases_{args.region}.json")

    # Show sample
    print("\nSample station index rows:")
    print(idx.head())


if __name__ == "__main__":
    main()
