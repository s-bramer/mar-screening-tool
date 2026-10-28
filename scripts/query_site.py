"""
query_site.py — look up MAR-ST suitability scores for a BNG coordinate.

Usage:
    python scripts/query_site.py --easting 389900 --northing 281700 --name "Norton ASR"
    python scripts/query_site.py --easting 397953 --northing 357680 --name "Abbey Green"
    python scripts/query_site.py --csv sites.csv   # batch from CSV with columns: name,easting,northing

Reads:  processed/hover_base.csv  (pre-computed cell attributes)
        processed/grid.gpkg        (cell bounding boxes in EPSG:3857)

Output: prints a score summary table to stdout.
        With --out <file.csv>, writes all queried rows to CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

PROCESSED = Path(__file__).parent.parent / "processed"


def load_cells() -> pd.DataFrame:
    hover = PROCESSED / "hover_base.csv"
    if not hover.exists():
        sys.exit(f"ERROR: {hover} not found — run scripts/preprocess.py first.")
    cells = pd.read_csv(hover)
    for col in ("x0", "x1", "y0", "y1"):
        if col not in cells.columns:
            sys.exit(f"ERROR: hover_base.csv is missing column '{col}'. Rerun preprocess.py.")
    return cells


def query_one(cells: pd.DataFrame, east: int, north: int,
              transformer: Transformer) -> dict | None:
    wm_x, wm_y = transformer.transform(east, north)
    mask = (
        (cells["x0"] <= wm_x) & (wm_x <= cells["x1"]) &
        (cells["y0"] <= wm_y) & (wm_y <= cells["y1"])
    )
    if not mask.any():
        return None
    row = cells[mask].iloc[0]

    excl = int(row.get("constraint", 1)) == 0
    return {
        "easting":          east,
        "northing":         north,
        "aquifer":          row.get("aquifer", "?"),
        "geo_score (2.1)":  round(float(row.get("geo_score",      0)), 1),
        "sdtm_score (2.2)": round(float(row.get("sdtm_score",     0)), 1),
        "sdtm_m":           round(float(row.get("sdtm_mean_m",    0)), 1),
        "surf_geo (2.5)":   round(float(row.get("surf_geo_score", 0)), 1),
        "need_score":       round(float(row.get("need_score",     0)), 1),
        "water_score":      round(float(row.get("water_score",    0)), 1),
        "gwmu":             row.get("gwmu", "?"),
        "gwm":              row.get("gwm_name", "?"),
        "constraint":       "Excluded" if excl else "Feasible",
        "c_nonprod_ok":     "No"  if int(row.get("c_nonprod", 1)) == 0 else "Yes",
        "c_gwdte_ok":       "No"  if int(row.get("c_gwdte",   1)) == 0 else "Yes",
    }


def print_result(name: str, result: dict | None, east: int, north: int):
    sep = "-" * 60
    if result is None:
        print(f"\n[{name}]  E{east} N{north}  -- NOT IN STUDY AREA\n")
        return
    tag = "EXCLUDED" if result["constraint"] == "Excluded" else "FEASIBLE"
    print(f"\n{sep}")
    print(f"  {name}  (E{east} N{north})  [{tag}]")
    print(sep)
    print(f"  Aquifer        : {result['aquifer']}")
    print(f"  GWMU           : {result['gwmu']}")
    print(f"  GWM            : {result['gwm']}")
    print(f"  -- Scores ---------------------------------------------------")
    print(f"  2.1 geo_score  : {result['geo_score (2.1)']:.1f}  (Aquifer class)")
    print(f"  2.2 sdtm_score : {result['sdtm_score (2.2)']:.1f}  (SDTM {result['sdtm_m']:.1f} m)")
    print(f"  2.5 surf_geo   : {result['surf_geo (2.5)']:.1f}  (Surface geology permeability)")
    print(f"  need_score     : {result['need_score']:.1f}  [dummy]")
    print(f"  water_score    : {result['water_score']:.1f}  [dummy]")
    print(f"  -- Constraints ----------------------------------------------")
    print(f"  Non-productive aquifer ok : {result['c_nonprod_ok']}")
    print(f"  GWDTE exclusion ok        : {result['c_gwdte_ok']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Query MAR-ST suitability scores for a BNG location.")
    ap.add_argument("--easting",  type=int,  help="BNG Easting")
    ap.add_argument("--northing", type=int,  help="BNG Northing")
    ap.add_argument("--name",     type=str,  default="", help="Site name (label only)")
    ap.add_argument("--csv",      type=str,  help="Batch input CSV with columns: name,easting,northing")
    ap.add_argument("--out",      type=str,  help="Output CSV path for all queried rows")
    args = ap.parse_args()

    cells = load_cells()
    transformer = Transformer.from_crs(27700, 3857, always_xy=True)

    sites: list[tuple[str, int, int]] = []

    if args.csv:
        batch = pd.read_csv(args.csv)
        for _, r in batch.iterrows():
            sites.append((str(r.get("name", "")), int(r["easting"]), int(r["northing"])))
    elif args.easting and args.northing:
        sites.append((args.name or f"E{args.easting} N{args.northing}",
                      args.easting, args.northing))
    else:
        ap.print_help()
        sys.exit(1)

    rows = []
    for name, east, north in sites:
        result = query_one(cells, east, north, transformer)
        print_result(name, result, east, north)
        if result is not None:
            rows.append({"name": name, **result})

    if args.out and rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
