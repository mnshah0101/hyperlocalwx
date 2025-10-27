#!/usr/bin/env python3
"""
Create next-hour targets and make a time-based split for a region.

Usage:
  python examples/02_make_targets_and_split.py --region sf_coast_inland --q 0.8
"""

import argparse
from pathlib import Path
import hyperlocalwx as wx
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="Region key")
    ap.add_argument("--q", type=float, default=0.8,
                    help="Train quantile on time (default: 0.8)")
    args = ap.parse_args()

    Path("data").mkdir(exist_ok=True, parents=True)

    # Load features parquet produced by build_features
    df = wx.load_features(args.region)

    # Create next-hour target
    df = wx.create_next_hour_target(df)  # adds temp_c_tplus1

    # Save a train-ready file (optional)
    out_train_ready = Path("data") / f"train_ready_{args.region}.parquet"
    df.to_parquet(out_train_ready, index=False)
    print("Wrote:", out_train_ready)

    # Time-based split
    train, valid = wx.time_split(df, train_quantile=args.q)
    train_path = Path("data") / "train.parquet"
    valid_path = Path("data") / "valid.parquet"
    train.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)

    print("Train rows:", len(train), "→", train_path)
    print("Valid rows:", len(valid), "→", valid_path)


if __name__ == "__main__":
    main()
