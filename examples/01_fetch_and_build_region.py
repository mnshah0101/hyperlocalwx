#!/usr/bin/env python3
"""
Fetch NOAA ISD-Lite + DEM and build features for a chosen region.

Usage:
  python examples/01_fetch_and_build_region.py --region sf_coast_inland
"""

import argparse
import os
import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True,
                    help="One of: sf_coast_inland, co_front_range, wa_rainshadow, lake_michigan, phoenix_uhi, cape_cod")
    args = ap.parse_args()

    # Ensure directories exist
    Path("data").mkdir(exist_ok=True, parents=True)
    Path("raw/natural_earth").mkdir(exist_ok=True, parents=True)

    # Warn if DEM key missing (you can still run; features build will fail if DEM is absent)
    if not os.environ.get("OPENTOPO_KEY"):
        print("NOTE: OPENTOPO_KEY not set. get_data will skip DEM fetch; set it to build terrain features.")

    # Step 1: Download station catalog + ISD-Lite + (optionally) DEM
    run(["python", "scripts/get_data.py", "--region", args.region])

    # Step 2: Build features parquet (needs DEM + Natural Earth shapefiles present)
    run(["python", "scripts/build_features.py", "--region", args.region])

    print("\nDone. Check data/features_hourly_{}.parquet".format(args.region))


if __name__ == "__main__":
    main()
