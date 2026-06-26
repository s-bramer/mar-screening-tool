"""
Data ingestion and preprocessing.

Each function loads a raw dataset, validates / assigns CRS, clips to the
study boundary, and optionally computes a suitability score column.
All outputs are in EPSG:27700 (BNG).
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from .utils import ensure_crs, get_logger

log = get_logger("ingest")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_layer(
    path: str,
    assumed_epsg: int | None = None,
    target_epsg: int = 27700,
    layer: str | None = None,
) -> gpd.GeoDataFrame:
    log.info(f"Loading: {path}")
    kwargs = {} if layer is None else {"layer": layer}
    gdf = gpd.read_file(path, **kwargs)
    return ensure_crs(gdf, path, assumed_epsg=assumed_epsg, target_epsg=target_epsg)


def clip_to_boundary(gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gpd.clip(gdf, boundary)


def area_weighted_join(
    grid: gpd.GeoDataFrame,
    source: gpd.GeoDataFrame,
    value_cols: list,
    output_cols: list,
    fill_value: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    For each grid cell compute the area-weighted mean of value_cols from source.
    """
    log.info(f"Area-weighted join: {value_cols} → {output_cols}")
    src = source[["geometry"] + value_cols].copy()
    overlay = gpd.overlay(
        grid[["cell_id", "geometry"]],
        src,
        how="intersection",
        keep_geom_type=False,
    )
    overlay["_area"] = overlay.geometry.area

    grid = grid.copy()
    for val_col, out_col in zip(value_cols, output_cols):
        num = (overlay[val_col] * overlay["_area"]).groupby(overlay["cell_id"]).sum()
        den = overlay.groupby("cell_id")["_area"].sum()
        grid[out_col] = grid["cell_id"].map(num / den).fillna(fill_value)

    return grid


def dominant_value_join(
    grid: gpd.GeoDataFrame,
    source: gpd.GeoDataFrame,
    value_col: str,
    output_col: str,
    fill_value=0,
) -> gpd.GeoDataFrame:
    """
    For each grid cell assign the value from the largest-intersecting polygon.
    Uses gpd.overlay (intersection) then picks the dominant value by area.
    """
    log.info(f"Spatial join: {value_col} → {output_col}")

    src = source[["geometry", value_col]].copy()

    overlay = gpd.overlay(
        grid[["cell_id", "geometry"]],
        src,
        how="intersection",
        keep_geom_type=False,
    )
    overlay["_area"] = overlay.geometry.area

    # Dominant value = that of the largest overlapping polygon fragment per cell
    idx = overlay.groupby("cell_id")["_area"].idxmax()
    dominant = overlay.loc[idx, ["cell_id", value_col]].set_index("cell_id")

    grid = grid.copy()
    grid[output_col] = grid["cell_id"].map(dominant[value_col]).fillna(fill_value)
    return grid


# ---------------------------------------------------------------------------
# Layer-specific loaders
# ---------------------------------------------------------------------------

def load_boundary(path: str) -> gpd.GeoDataFrame:
    gdf = load_layer(path)
    log.info(f"Study boundary: {len(gdf)} polygon(s), CRS=EPSG:{gdf.crs.to_epsg()}")
    return gdf


def load_hydrogeology(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src_cfg = cfg["data_sources"]["hydrogeology_625k"]
    gdf = load_layer(src_cfg["path"], assumed_epsg=src_cfg.get("assumed_crs"))
    gdf = clip_to_boundary(gdf, boundary)

    score_map = cfg["hydrogeology_scores"]
    char_field = src_cfg["character_field"]
    gdf["hydro_score"] = gdf[char_field].map(score_map).fillna(0).astype(float)

    log.info(
        f"Hydrogeology: {len(gdf)} polygons clipped, "
        f"score range {gdf['hydro_score'].min():.0f}–{gdf['hydro_score'].max():.0f}"
    )
    return gdf


def load_bedrock(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src_cfg = cfg["data_sources"]["bedrock_geology"]
    gdf = load_layer(src_cfg["path"], assumed_epsg=src_cfg.get("assumed_crs"))
    return clip_to_boundary(gdf, boundary)


def load_superficial(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src_cfg = cfg["data_sources"]["superficial_geology"]
    gdf = load_layer(src_cfg["path"], assumed_epsg=src_cfg.get("assumed_crs"))
    return clip_to_boundary(gdf, boundary)


def load_gwmu(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src_cfg = cfg["data_sources"]["gwmu"]
    gdf = load_layer(src_cfg["path"])
    return clip_to_boundary(gdf, boundary)


def load_sw_catchments(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src_cfg = cfg["data_sources"]["sw_catchments"]
    gdf = load_layer(src_cfg["path"])
    return clip_to_boundary(gdf, boundary)


def load_superficial_thickness(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load BGS Superficial Deposits Thickness Model and clip to study boundary."""
    src_cfg = cfg["data_sources"]["superficial_thickness"]
    gdf = load_layer(
        src_cfg["path"],
        layer=src_cfg.get("layer", "BGS_SDTM_1km_Coverage"),
    )
    gdf = clip_to_boundary(gdf, boundary)
    log.info(
        f"SDTM: {len(gdf)} hex cells clipped, "
        f"BSTM_MEAN {gdf['BSTM_MEAN'].min():.1f}–{gdf['BSTM_MEAN'].max():.1f} m, "
        f"COVER_PCT {gdf['COVER_PCT'].min():.2f}–{gdf['COVER_PCT'].max():.2f}"
    )
    return gdf


def load_rivers(cfg: dict, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    src_cfg = cfg["data_sources"]["os_rivers"]
    gdf = load_layer(src_cfg["path"], assumed_epsg=src_cfg.get("assumed_crs", 27700))
    return clip_to_boundary(gdf, boundary)


# ---------------------------------------------------------------------------
# Dummy score generators (placeholder until real data/models available)
# ---------------------------------------------------------------------------

def dummy_need_score(grid: gpd.GeoDataFrame, gwmu: gpd.GeoDataFrame, seed: int = 42) -> gpd.GeoDataFrame:
    """
    Placeholder 'Need for MAR' score based on GWMU membership + spatial noise.
    Replace with real CAMS ledger deficit data when available.
    """
    rng = np.random.default_rng(seed)
    grid = dominant_value_join(grid, gwmu, "gwmu_name", "gwmu_name", fill_value="Unknown")

    # Assign a random-but-spatially-consistent score per GWMU
    unique_gwmu = grid["gwmu_name"].unique()
    gwmu_scores = {g: rng.uniform(2, 9) for g in unique_gwmu}
    grid["need_score"] = grid["gwmu_name"].map(gwmu_scores).fillna(0)

    # Add smooth spatial noise (±1 unit) so the map isn't flat blocks
    noise = rng.uniform(-1, 1, size=len(grid))
    grid["need_score"] = np.clip(grid["need_score"] + noise, 0, 10)
    return grid


def dummy_water_score(grid: gpd.GeoDataFrame, seed: int = 99) -> gpd.GeoDataFrame:
    """
    Placeholder 'Water Availability' score.
    Replace with proximity-to-intake, rainfall, and DWMP data when available.
    """
    rng = np.random.default_rng(seed)
    cx = grid["cx"].values
    cy = grid["cy"].values

    # Gentle north–south gradient + random noise to give spatially coherent field
    cx_norm = (cx - cx.min()) / (cx.max() - cx.min())
    cy_norm = (cy - cy.min()) / (cy.max() - cy.min())
    base = 3 + 5 * (0.5 * cy_norm + 0.5 * (1 - cx_norm))
    noise = rng.uniform(-1.5, 1.5, size=len(grid))
    grid["water_score"] = np.clip(base + noise, 0, 10)
    return grid
