"""
Preprocessing pipeline — run once before launching the dashboard.

Outputs (written to processed/):
  grid.gpkg          — master 1 km grid with all scored columns
  hydrogeology.gpkg  — hydrogeology clipped to ST boundary
  bedrock.gpkg       — bedrock geology clipped to ST boundary
  superficial.gpkg   — superficial geology clipped to ST boundary
  gwmu.gpkg          — groundwater management units clipped
  gwdte.gpkg         — GWDTEs clipped to ST boundary (constraint + display)
  sw_catchments.gpkg — SW catchments clipped
  rivers.gpkg        — river network clipped (may take a moment)

Grid score columns:
  geo_score    — bedrock aquifer productivity (BGS 625K hydrogeology)
  sdtm_score   — superficial deposit infiltration suitability (BGS SDTM 1km)
  need_score   — need for MAR (dummy placeholder)
  water_score  — water availability (dummy placeholder)

Usage:
  conda activate mar-st
  python scripts/preprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.interpolate import griddata as scipy_griddata
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds as rio_from_bounds

from mar_st import config as cfg_mod
from mar_st.grid import create_grid
from mar_st.ingest import (
    load_boundary, load_hydrogeology, load_bedrock, load_superficial,
    load_gwmu, load_sw_catchments, load_rivers, load_layer,
    dominant_value_join, area_weighted_join,
    dummy_need_score, dummy_water_score,
    load_superficial_thickness,
)
from mar_st.mce import apply_constraints, weighted_sum, score_sdtm
from mar_st.utils import get_logger

log = get_logger("preprocess")


def run():
    cfg = cfg_mod.load()
    out_dir = cfg_mod.ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Study boundary
    # ------------------------------------------------------------------
    log.info("=== Study boundary ===")
    boundary = load_boundary(cfg_mod.ROOT / cfg["data_sources"]["study_boundary"]["path"])

    # ------------------------------------------------------------------
    # 2. Master grid
    # ------------------------------------------------------------------
    log.info("=== Master grid ===")
    resolution_m = cfg["grid"]["resolution_m"]
    grid = create_grid(boundary, resolution_m)

    # ------------------------------------------------------------------
    # 3. Hydrogeology — score and join to grid
    # ------------------------------------------------------------------
    log.info("=== Hydrogeology ===")
    hydro = load_hydrogeology(cfg, boundary)
    hydro.to_file(out_dir / "hydrogeology.gpkg", driver="GPKG")

    grid = dominant_value_join(grid, hydro, "hydro_score", "geo_score", fill_value=0)
    grid = dominant_value_join(grid, hydro, "CHARACTER", "hydro_character", fill_value="Unknown")

    # ------------------------------------------------------------------
    # 4. GWDTEs — load, clip, save
    # ------------------------------------------------------------------
    log.info("=== GWDTEs ===")
    gwdte_cfg = cfg["data_sources"]["gwdte"]
    gwdte = load_layer(cfg_mod.ROOT / gwdte_cfg["path"])
    gwdte = gpd.clip(gwdte, boundary)
    gwdte[["GWDTE_NAME", "site_name", "geometry"]].to_file(
        out_dir / "gwdte.gpkg", driver="GPKG"
    )
    log.info(f"GWDTEs: {len(gwdte)} polygons within study area")

    # ------------------------------------------------------------------
    # 5a. Superficial Deposits Thickness — area-weighted mean per cell
    # ------------------------------------------------------------------
    log.info("=== Superficial Deposits Thickness (BGS SDTM) ===")
    sdtm = load_superficial_thickness(cfg, boundary)
    grid = area_weighted_join(
        grid, sdtm,
        value_cols=["BSTM_MEAN", "COVER_PCT"],
        output_cols=["sdtm_mean_m", "sdtm_cover_pct"],
        fill_value=0.0,
    )

    # ------------------------------------------------------------------
    # 5. Constraints mask (non-productive aquifer + GWDTEs)
    # ------------------------------------------------------------------
    log.info("=== Constraints ===")
    grid = apply_constraints(grid, cfg, hydrogeology=hydro, gwdte=gwdte)

    # ------------------------------------------------------------------
    # 6. Dummy theme scores (placeholders)
    # ------------------------------------------------------------------
    log.info("=== Dummy scores (Need for MAR, Water Availability) ===")
    gwmu = load_gwmu(cfg, boundary)
    gwmu.to_file(out_dir / "gwmu.gpkg", driver="GPKG")

    grid = dummy_need_score(grid, gwmu)
    grid = dummy_water_score(grid)

    # ------------------------------------------------------------------
    # 6b. SDTM infiltration score
    # ------------------------------------------------------------------
    log.info("=== SDTM infiltration score ===")
    grid = score_sdtm(grid)

    # ------------------------------------------------------------------
    # 7. Composite suitability
    # ------------------------------------------------------------------
    log.info("=== Composite suitability ===")
    themes = cfg["mce"]["themes"]
    weights = {k: v["weight"] for k, v in themes.items()}
    score_cols = {
        "need_for_mar": "need_score",
        "geological_suitability": "geo_score",
        "water_availability": "water_score",
    }
    grid = weighted_sum(grid, weights, score_cols, constraint_col="constraint_mask")

    # ------------------------------------------------------------------
    # 8. Other display layers
    # ------------------------------------------------------------------
    log.info("=== Other display layers ===")
    bedrock = load_bedrock(cfg, boundary)
    bedrock[["LEX_D", "RCS_D", "geometry"]].to_file(out_dir / "bedrock.gpkg", driver="GPKG")

    superficial = load_superficial(cfg, boundary)
    superficial[["LEX_D", "geometry"]].to_file(out_dir / "superficial.gpkg", driver="GPKG")

    sw = load_sw_catchments(cfg, boundary)
    sw.to_file(out_dir / "sw_catchments.gpkg", driver="GPKG")

    log.info("=== Rivers (large file — this may take a minute) ===")
    rivers = load_rivers(cfg, boundary)
    rivers[["geometry"]].to_file(out_dir / "rivers.gpkg", driver="GPKG")

    # ------------------------------------------------------------------
    # 9. Save grid (BNG)
    # ------------------------------------------------------------------
    log.info("=== Saving grid ===")
    grid.to_file(out_dir / "grid.gpkg", driver="GPKG")

    # ------------------------------------------------------------------
    # 10. Pre-compute Web Mercator assets for the dashboard
    #     Doing this here means the dashboard only needs to load files —
    #     no scipy, rasterio, or reprojection at startup.
    # ------------------------------------------------------------------
    log.info("=== Pre-computing Web Mercator rasters ===")
    RASTER_W, RASTER_H = 300, 350
    _t = Transformer.from_crs(27700, 3857, always_xy=True)
    cx_wm, cy_wm = _t.transform(grid["cx"].values, grid["cy"].values)
    x_min, x_max = cx_wm.min(), cx_wm.max()
    y_min, y_max = cy_wm.min(), cy_wm.max()
    x_wm = np.linspace(x_min, x_max, RASTER_W)
    y_wm = np.linspace(y_min, y_max, RASTER_H)
    xx, yy = np.meshgrid(x_wm, y_wm)
    pts = np.column_stack([cx_wm, cy_wm])

    def _raster(col):
        return scipy_griddata(pts, grid[col].astype(float).values,
                              (xx, yy), method="nearest")

    R_NEED  = _raster("need_score")
    R_GEO   = _raster("geo_score")
    R_WATER = _raster("water_score")
    R_MASK  = _raster("constraint_mask")

    # Boundary mask in Web Mercator
    _bnd_wm = boundary.to_crs(3857)
    _tf = rio_from_bounds(x_min, y_min, x_max, y_max, RASTER_W, RASTER_H)
    _b = rio_rasterize(
        [(g, 1) for g in _bnd_wm.geometry],
        out_shape=(RASTER_H, RASTER_W), transform=_tf, fill=0, all_touched=True,
    ).astype(bool)
    B_MASK = np.flipud(_b)   # rasterio top-down → flip to match bottom-up y axis

    np.savez_compressed(
        out_dir / "rasters.npz",
        X_WM=x_wm, Y_WM=y_wm,
        R_NEED=R_NEED, R_GEO=R_GEO, R_WATER=R_WATER,
        R_MASK=R_MASK, B_MASK=B_MASK,
    )
    log.info("Rasters saved → processed/rasters.npz")

    # ------------------------------------------------------------------
    # 11. Pre-save hover data (cell bounding boxes + attributes)
    # ------------------------------------------------------------------
    log.info("=== GWM name join ===")
    gwm_bnd = load_layer(cfg["data_sources"]["gwm"]["path"])
    gwm_bnd = gpd.clip(gwm_bnd, boundary)
    grid = dominant_value_join(grid, gwm_bnd, "Name", "gwm_name", fill_value="")
    log.info(f"GWM names joined: {(grid['gwm_name'] != '').sum():,} cells within a model area")

    log.info("=== Pre-computing hover data ===")
    _grid_3857 = grid.to_crs(3857)
    _bounds = _grid_3857.geometry.bounds
    hover_df = pd.DataFrame({
        "x0":          _bounds["minx"].values,
        "y0":          _bounds["miny"].values,
        "x1":          _bounds["maxx"].values,
        "y1":          _bounds["maxy"].values,
        "geo_score":   grid["geo_score"].round(1).values,
        "need_score":  grid["need_score"].round(1).values,
        "water_score": grid["water_score"].round(1).values,
        "sdtm_score":  grid["sdtm_score"].round(1).values,
        "sdtm_mean_m": grid["sdtm_mean_m"].round(1).values,
        "aquifer":     grid["hydro_character"].values,
        "gwmu":        grid["gwmu_name"].values,
        "gwm_name":    grid["gwm_name"].values,
        "constraint":  grid["constraint_mask"].values,
    })
    hover_df.to_csv(out_dir / "hover_base.csv", index=False)
    log.info("Hover data saved → processed/hover_base.csv")

    # ------------------------------------------------------------------
    # 12. Pre-save Web Mercator display layers
    #     Rivers simplified at 1 km tolerance (matches grid resolution)
    # ------------------------------------------------------------------
    log.info("=== Pre-saving Web Mercator display layers ===")

    boundary.to_crs(3857).to_file(out_dir / "boundary_3857.gpkg", driver="GPKG")

    # gwm_bnd already loaded above — reuse here
    gwm_out = gwm_bnd[["Name", "geometry"]].to_crs(3857).copy()
    gwm_out.geometry = gwm_out.geometry.simplify(500)   # 500 m in Web Mercator
    gwm_out[~gwm_out.geometry.is_empty].to_file(out_dir / "gwm_3857.gpkg", driver="GPKG")
    log.info(f"GWM boundaries: {len(gwm_bnd)} model areas saved")

    hydro[["geometry", "CHARACTER"]].to_crs(3857).to_file(
        out_dir / "hydrogeology_3857.gpkg", driver="GPKG"
    )
    gwmu[["geometry", "gwmu_name"]].to_crs(3857).to_file(
        out_dir / "gwmu_3857.gpkg", driver="GPKG"
    )
    gwdte[["geometry", "GWDTE_NAME", "site_name"]].to_crs(3857).to_file(
        out_dir / "gwdte_3857.gpkg", driver="GPKG"
    )
    sw[["geometry", "mncat_name"]].to_crs(3857).to_file(
        out_dir / "sw_catchments_3857.gpkg", driver="GPKG"
    )

    rivers_simple = rivers.copy()
    rivers_simple.geometry = rivers_simple.geometry.simplify(1000)  # 1 km — matches grid resolution
    rivers_simple = rivers_simple[~rivers_simple.geometry.is_empty]
    rivers_simple[["geometry"]].to_crs(3857).to_file(
        out_dir / "rivers_3857.gpkg", driver="GPKG"
    )
    log.info("Web Mercator layers saved → processed/*_3857.gpkg")

    n_excl = (grid["constraint_mask"] == 0).sum()
    log.info(f"Preprocessing complete — {n_excl:,} cells excluded by constraints")
    log.info(f"Grid summary:\n{grid[['geo_score','need_score','water_score','composite_score']].describe().round(2)}")


if __name__ == "__main__":
    run()
