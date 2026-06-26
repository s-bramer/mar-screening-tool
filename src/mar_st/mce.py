"""
Multi-Criteria Evaluation engine.

Phase 1: simple weighted linear combination.
Phase 2 target: AHP pairwise comparison matrices with consistency check.
"""

import numpy as np
import geopandas as gpd

from .utils import get_logger

log = get_logger("mce")


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def exclude_by_overlap(
    grid: gpd.GeoDataFrame,
    exclusion_gdf: gpd.GeoDataFrame,
    label: str,
) -> gpd.GeoDataFrame:
    """
    Set constraint_mask = 0 for any cell that intersects the exclusion layer.
    Uses 'any overlap' rather than majority area — appropriate for hard constraints
    where even partial overlap with a protected area should exclude the cell.
    """
    if exclusion_gdf is None or exclusion_gdf.empty:
        log.info(f"Constraint [{label}]: exclusion layer empty, skipping")
        return grid

    joined = gpd.sjoin(
        grid[["cell_id", "geometry"]],
        exclusion_gdf[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    excluded_ids = joined["cell_id"].unique()
    grid = grid.copy()
    grid.loc[grid["cell_id"].isin(excluded_ids), "constraint_mask"] = 0
    log.info(f"Constraint [{label}]: {len(excluded_ids):,} cells excluded")
    return grid


# ---------------------------------------------------------------------------
# Main constraint builder
# ---------------------------------------------------------------------------

def apply_constraints(
    grid: gpd.GeoDataFrame,
    cfg: dict,
    hydrogeology: gpd.GeoDataFrame | None = None,
    gwdte: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Build a binary constraint_mask column (1 = feasible, 0 = excluded).
    Constraints are applied cumulatively; a cell excluded by any one constraint
    stays excluded.
    """
    grid = grid.copy()
    grid["constraint_mask"] = 1

    if cfg["constraints"].get("exclude_non_productive") and hydrogeology is not None:
        non_prod = hydrogeology[
            hydrogeology["CHARACTER"] == "Rocks with essentially no groundwater"
        ].copy()
        grid = exclude_by_overlap(grid, non_prod, "non-productive aquifer")

    if cfg["constraints"].get("exclude_gwdte") and gwdte is not None:
        grid = exclude_by_overlap(grid, gwdte, "GWDTE")

    total_excluded = (grid["constraint_mask"] == 0).sum()
    log.info(f"Constraints applied — {total_excluded:,} cells excluded in total")
    return grid


# ---------------------------------------------------------------------------
# Superficial deposits thickness scoring
# ---------------------------------------------------------------------------

def score_sdtm(
    grid: gpd.GeoDataFrame,
    thickness_col: str = "sdtm_mean_m",
    cover_col: str = "sdtm_cover_pct",
    output_col: str = "sdtm_score",
    breakpoints_m: list | None = None,
    scores: list | None = None,
) -> gpd.GeoDataFrame:
    """
    Score superficial deposits thickness for MAR infiltration suitability.

    Piecewise-linear between breakpoints; peaks at ~8 m (optimal filtration
    depth for infiltration MAR). Score is then weighted by fractional cell
    coverage (COVER_PCT), with a floor of 0.3 so sparsely-covered cells are
    penalised but not zeroed.

    Assumes all superficial deposits are permeable — revise once lithology
    (BGS 625K superficial geology) is integrated.
    """
    if breakpoints_m is None:
        breakpoints_m = [0, 1, 3, 8, 15, 25, 40, 50]
    if scores is None:
        scores = [4, 5, 8, 10, 8, 4, 2, 1]

    thickness = np.clip(grid[thickness_col].fillna(0).values, 0, breakpoints_m[-1])
    base = np.interp(thickness, breakpoints_m, scores)

    cover = np.clip(grid[cover_col].fillna(0).values, 0, 1)
    weight = 0.3 + 0.7 * cover

    grid = grid.copy()
    grid[output_col] = np.clip(base * weight, 0, 10).round(1)
    log.info(
        f"SDTM score — min={grid[output_col].min():.1f}, "
        f"mean={grid[output_col].mean():.1f}, "
        f"max={grid[output_col].max():.1f}"
    )
    return grid


# ---------------------------------------------------------------------------
# MCE weighted sum
# ---------------------------------------------------------------------------

def weighted_sum(
    grid: gpd.GeoDataFrame,
    theme_weights: dict[str, float],
    score_cols: dict[str, str],
    constraint_col: str | None = "constraint_mask",
    output_col: str = "composite_score",
) -> gpd.GeoDataFrame:
    """
    Compute composite suitability as a weighted sum of theme scores.

    Parameters
    ----------
    grid          : GeoDataFrame with a score column per theme
    theme_weights : {theme_key: weight}  — normalised if they don't sum to 1
    score_cols    : {theme_key: column_name_in_grid}
    constraint_col: binary column (1=feasible, 0=excluded); None to skip masking
    output_col    : name for the output composite score column
    """
    total_weight = sum(theme_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        log.warning(f"Theme weights sum to {total_weight:.3f} — normalising")
        theme_weights = {k: v / total_weight for k, v in theme_weights.items()}

    composite = np.zeros(len(grid))
    for theme, weight in theme_weights.items():
        col = score_cols[theme]
        if col not in grid.columns:
            raise KeyError(f"Score column '{col}' not found for theme '{theme}'")
        composite += weight * grid[col].fillna(0).values

    grid = grid.copy()
    grid[output_col] = np.clip(composite, 0, 10)

    if constraint_col and constraint_col in grid.columns:
        grid[output_col] = grid[output_col] * grid[constraint_col]

    log.info(
        f"Composite: min={grid[output_col].min():.2f}, "
        f"mean={grid[output_col].mean():.2f}, "
        f"max={grid[output_col].max():.2f}"
    )
    return grid
