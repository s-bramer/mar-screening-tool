"""
Export the MAR-ST 1 km reference grid as a shapefile.

Colleagues can use this shapefile to spatially join their own datasets
to the MAR-ST grid cells (e.g. pipe network, reservoir data).

Output: exports/mar_st_grid_1km_BNG.shp  (EPSG:27700)

Usage:
    conda activate mar-st
    python scripts/export_grid_shapefile.py
"""

import sys
from pathlib import Path

import geopandas as gpd

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "processed"
OUT_DIR   = ROOT / "exports"

OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "mar_st_grid_1km_BNG.shp"

print("Reading grid...")
grid = gpd.read_file(PROCESSED / "grid.gpkg")

# Ensure BNG
if grid.crs is None or grid.crs.to_epsg() != 27700:
    print(f"  Reprojecting from {grid.crs} → EPSG:27700")
    grid = grid.to_crs(epsg=27700)
else:
    print(f"  CRS: {grid.crs} (OK)")

# Keep only cell identifier + geometry — strip scoring columns so colleagues
# get a clean reference grid without any preliminary/dummy score values.
id_cols = [c for c in grid.columns if c.lower() in ("cell_id", "id", "fid", "grid_id")]
if id_cols:
    export = grid[id_cols + ["geometry"]].copy()
else:
    # No ID column — create one from row index
    grid["cell_id"] = range(1, len(grid) + 1)
    export = grid[["cell_id", "geometry"]].copy()

print(f"  {len(export):,} cells")

# Compute cell centroid easting/northing for reference (useful for colleagues)
export["easting_m"]  = export.geometry.centroid.x.round(0).astype(int)
export["northing_m"] = export.geometry.centroid.y.round(0).astype(int)

print(f"Writing shapefile → {OUT_PATH}")
export.to_file(OUT_PATH)

# Also write as GeoPackage for colleagues who prefer it
gpkg_path = OUT_DIR / "mar_st_grid_1km_BNG.gpkg"
export.to_file(gpkg_path, driver="GPKG")
print(f"Writing GeoPackage → {gpkg_path}")

print("Done.")
print(f"\nColumns in output: {list(export.columns)}")
print(f"CRS: {export.crs}")
print(f"\nShare the contents of:  {OUT_DIR}")
print("Do NOT commit exports/ to git — treat as processed data output.")
