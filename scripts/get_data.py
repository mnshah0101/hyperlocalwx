# scripts/get_data.py
from pathlib import Path
import os
import requests
import argparse
import pandas as pd
from tqdm import tqdm

# Handle both direct execution and module import
try:
    from .regions import REGIONS
except ImportError:
    from regions import REGIONS

DATA = Path("data")
DATA.mkdir(exist_ok=True, parents=True)
RAW = Path("raw")
RAW.mkdir(exist_ok=True, parents=True)
NOAA = RAW/"noaa"
NOAA.mkdir(exist_ok=True, parents=True)
DEM = RAW/"dem"
DEM.mkdir(exist_ok=True, parents=True)
NE = RAW/"natural_earth"  # put shapefiles here later


def ensure_isd_history_csv():
    out = NOAA/"isd-history.csv"
    if not out.exists():
        url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
    return out


def load_isd_history():
    p = ensure_isd_history_csv()
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df = df.rename(
        columns={"STATION NAME": "STATION_NAME", "ELEV(M)": "ELEV_M"})
    for c in ["LAT", "LON", "ELEV_M", "BEGIN", "END"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["USAF", "WBAN", "ICAO", "CTRY", "STATE", "STATION_NAME"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip().replace({"": pd.NA})
    return df


def resolve_icao_to_usaf_wban(icaos, years, bbox=None):
    df = load_isd_history()
    icaos = [c.upper() for c in icaos]
    y0, y1 = min(years), max(years)
    want_begin, want_end = y0*10000, y1*10000

    def score(r):
        b = r["BEGIN"] if pd.notna(r["BEGIN"]) else 0
        e = r["END"] if pd.notna(r["END"]) else 0
        return (1 if min(e, want_end) - max(b, want_begin) > 0 else 0, e)
    sub = df[df["ICAO"].isin(icaos)].copy()
    if not sub.empty:
        sub["__score__"] = sub.apply(score, axis=1)
        sub = sub.sort_values(["ICAO", "__score__"], ascending=[
                              True, False]).drop_duplicates("ICAO")
    missing = sorted(set(icaos) - set(sub["ICAO"].dropna().unique()))
    extras = pd.DataFrame()
    if missing and bbox:
        w, s, e, n = bbox
        in_box = df[df["LAT"].between(s, n) & df["LON"].between(w, e)].copy()
        in_box["__score__"] = in_box.apply(score, axis=1)
        in_box = in_box.sort_values(
            ["__score__", "END"], ascending=[False, False])
        extras = in_box.head(len(missing)+4)
    out = pd.concat([d for d in [sub, extras] if not d.empty], ignore_index=True) if (
        not sub.empty or not extras.empty) else pd.DataFrame()
    out = out.dropna(subset=["USAF", "WBAN", "LAT", "LON"]).copy()
    # save meta (lat/lon etc.) for later terrain sampling
    return out


def fetch_isdlite(usaf, wban, year):
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz"
    out = NOAA/f"isdlite_{usaf}_{wban}_{year}.gz"
    if not out.exists():
        r = requests.get(url, timeout=180)
        if r.status_code == 200:
            out.write_bytes(r.content)
    return out if out.exists() else None


def fetch_dem_srtm(region_key, bbox, api_key):
    w, s, e, n = bbox
    url = ("https://portal.opentopography.org/API/globaldem"
           f"?demtype=SRTMGL1&south={s}&north={n}&west={w}&east={e}"
           f"&outputFormat=GTiff&API_Key={api_key}")
    out = DEM/f"srtm_{region_key}.tif"
    if not out.exists():
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        out.write_bytes(r.content)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=REGIONS.keys())
    args = ap.parse_args()
    cfg = REGIONS[args.region]

    meta = resolve_icao_to_usaf_wban(
        cfg["icao"], cfg["years"], bbox=cfg["bbox"])
    meta["station"] = meta["ICAO"]
    meta["station_id"] = meta["USAF"].astype(str).str.zfill(
        6) + "-" + meta["WBAN"].astype(str).str.zfill(5)
    meta.to_parquet(DATA/f"noaa_meta_{args.region}.parquet", index=False)

    rows = []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="ISD-Lite"):
        for yr in cfg["years"]:
            p = fetch_isdlite(r["USAF"], r["WBAN"], yr)
            if p:
                rows.append(
                    {"station": r["ICAO"], "station_id": r["station_id"], "year": yr, "path": str(p)})
    pd.DataFrame(rows).to_csv(
        DATA/f"isd_manifest_{args.region}.csv", index=False)

    api_key = os.environ.get("OPENTOPO_KEY")
    if api_key:
        dem = fetch_dem_srtm(args.region, cfg["bbox"], api_key)
        print("DEM saved:", dem)
    else:
        print("NOTE: OPENTOPO_KEY not set → skipping DEM fetch.")

    print("Done. Put Natural Earth shapefiles into raw/natural_earth/: ne_10m_coastline.*, ne_10m_lakes.*")


if __name__ == "__main__":
    main()
