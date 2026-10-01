"""
Reconstruct geometries from Infoworks ICM CSV exports in data/DWMP/.

Each CSV stores vertex coordinates as flat JSON arrays:
  point_array    → [x1, y1, x2, y2, ...]  → LineString  (conduits, structures)
  boundary_array → [x1, y1, x2, y2, ...]  → Polygon     (subcatchments)
  x / y scalars                            → Point       (nodes)

Output: processed/dwmp.gpkg  (one layer per feature type, EPSG:27700)

Usage:
    python scripts/preprocess_dwmp.py
"""

import ast
import re
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.validation import make_valid

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mar_st.utils import get_logger

log = get_logger("preprocess_dwmp")

DWMP_DIR  = Path(__file__).parent.parent / "data" / "DWMP"
OUT_GPKG  = Path(__file__).parent.parent / "processed" / "dwmp.gpkg"
CRS       = "EPSG:27700"


# ---------------------------------------------------------------------------
# Geometry parsers
# ---------------------------------------------------------------------------

def _parse_coords(s) -> list[tuple] | None:
    """Parse '[x1,y1,x2,y2,...]' → [(x1,y1),(x2,y2),...]  or None."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s.startswith("["):
        return None
    try:
        vals = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        log.warning(f"Could not parse coord string: {s[:80]}")
        return None
    if len(vals) < 2 or len(vals) % 2 != 0:
        return None
    return list(zip(vals[::2], vals[1::2]))


def _to_linestring(s) -> LineString | None:
    coords = _parse_coords(s)
    if coords is None or len(coords) < 2:
        return None
    return LineString(coords)


def _to_polygon(s) -> Polygon | None:
    coords = _parse_coords(s)
    if coords is None or len(coords) < 3:
        return None
    # Remove consecutive duplicate vertices (ArcGIS chokes on them)
    deduped = [coords[0]]
    for c in coords[1:]:
        if c != deduped[-1]:
            deduped.append(c)
    coords = deduped
    if len(coords) < 3:
        return None
    # Close ring
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        # buffer(0) produces clean Polygon/MultiPolygon output, ArcGIS-safe
        poly = poly.buffer(0)
    # Unwrap GeometryCollection or MultiPolygon → largest polygon
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    elif hasattr(poly, "geoms"):  # GeometryCollection
        parts = [g for g in poly.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not parts:
            return None
        poly = max(parts, key=lambda p: p.area)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)
    if not isinstance(poly, Polygon) or poly.is_empty:
        return None
    return poly


def _to_point(x, y) -> Point | None:
    try:
        return Point(float(x), float(y))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-type loaders
# ---------------------------------------------------------------------------

# Columns to keep per type (drops *_flag columns and rarely-used fields)
_NODE_KEEP = [
    "node_id", "node_type", "system_type", "asset_id", "infonet_id",
    "x", "y", "ground_level", "flood_level", "flood_type",
    "chamber_area", "shaft_area",
]
_CONDUIT_KEEP = [
    "us_node_id", "link_suffix", "ds_node_id", "link_type", "system_type",
    "asset_id", "conduit_length", "conduit_width", "conduit_height",
    "shape", "conduit_material", "us_invert", "ds_invert",
    "gradient", "capacity", "critical_sewer_category", "point_array",
]
_SUBCATCH_KEEP = [
    "subcatchment_id", "system_type", "node_id", "link_suffix",
    "total_area", "contributing_area", "x", "y",
    "catchment_slope", "population", "connectivity",
    "land_use_id", "boundary_array",
]
_LINK_KEEP = [
    "us_node_id", "link_suffix", "ds_node_id", "system_type",
    "asset_id", "point_array",
]


def _keep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def load_nodes(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _keep(df, _NODE_KEEP)
    df["geometry"] = [_to_point(r["x"], r["y"]) for _, r in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)
    bad = gdf["geometry"].isna().sum()
    if bad:
        log.warning(f"nodes: {bad} rows have no geometry — dropped")
    return gdf[gdf["geometry"].notna()].reset_index(drop=True)


def load_linestrings(path: Path, id_col: str, keep: list[str]) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _keep(df, keep)
    df["geometry"] = df["point_array"].map(_to_linestring)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)
    bad = gdf["geometry"].isna().sum()
    if bad:
        log.warning(f"{path.stem}: {bad} rows have no geometry — dropped")
    return gdf[gdf["geometry"].notna()].reset_index(drop=True)


def load_subcatchments(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _keep(df, _SUBCATCH_KEEP)
    df["geometry"] = df["boundary_array"].map(_to_polygon)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS)
    bad = gdf["geometry"].isna().sum()
    if bad:
        log.warning(f"subcatchments: {bad} rows have no geometry — dropped")
    return gdf[gdf["geometry"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Maps layer name → (csv stem pattern, loader fn, id_col)
_LAYERS: dict[str, tuple] = {
    "node":        ("node",       load_nodes,        "node_id",        None),
    "conduit":     ("conduit",    load_linestrings,  "us_node_id",     _CONDUIT_KEEP),
    "subcatchment":("subcatchment", load_subcatchments, "subcatchment_id", None),
    "flap_valve":  ("flap valve", load_linestrings,  "us_node_id",     _LINK_KEEP),
    "flume":       ("flume",      load_linestrings,  "us_node_id",     _LINK_KEEP),
    "orifice":     ("orifice",    load_linestrings,  "us_node_id",     _LINK_KEEP),
    "pump":        ("pump",       load_linestrings,  "us_node_id",     _LINK_KEEP),
    "sluice":      ("sluice",     load_linestrings,  "us_node_id",     _LINK_KEEP),
    "weir":        ("weir",       load_linestrings,  "us_node_id",     _LINK_KEEP),
}


def main():
    if not DWMP_DIR.exists():
        log.error(f"DWMP data directory not found: {DWMP_DIR}")
        sys.exit(1)

    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale output so fiona doesn't append to old layers
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    csv_files = list(DWMP_DIR.glob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files in {DWMP_DIR}")

    written = []
    for layer_name, (stem_pattern, loader_fn, id_col, keep_cols) in _LAYERS.items():
        matches = [f for f in csv_files if stem_pattern in f.stem]
        if not matches:
            log.warning(f"No CSV found for layer '{layer_name}' (pattern: '{stem_pattern}')")
            continue
        if len(matches) > 1:
            log.warning(f"Multiple CSVs match '{stem_pattern}': {[f.name for f in matches]} — using first")
        path = matches[0]

        log.info(f"Processing {path.name} → layer '{layer_name}'")
        try:
            if loader_fn is load_linestrings:
                gdf = loader_fn(path, id_col, keep_cols)
            else:
                gdf = loader_fn(path)
        except Exception as e:
            log.error(f"  Failed: {e}")
            continue

        log.info(f"  {len(gdf):,} features, {gdf.geometry.geom_type.unique()}")
        gdf.to_file(OUT_GPKG, layer=layer_name, driver="GPKG")
        written.append(layer_name)

    log.info(f"Saved {len(written)} layers to {OUT_GPKG}")
    log.info(f"Layers: {written}")


if __name__ == "__main__":
    main()
