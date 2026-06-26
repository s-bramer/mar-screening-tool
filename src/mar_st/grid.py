import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

from .utils import get_logger

log = get_logger("grid")


def create_grid(boundary_gdf: gpd.GeoDataFrame, resolution_m: int) -> gpd.GeoDataFrame:
    """
    Build a regular square grid over the bounding box of boundary_gdf,
    then keep only cells that intersect the boundary polygon.
    Returns a GeoDataFrame with columns: cell_id, geometry.
    """
    xmin, ymin, xmax, ymax = boundary_gdf.total_bounds

    # Snap extents to grid resolution
    xmin = np.floor(xmin / resolution_m) * resolution_m
    ymin = np.floor(ymin / resolution_m) * resolution_m
    xmax = np.ceil(xmax / resolution_m) * resolution_m
    ymax = np.ceil(ymax / resolution_m) * resolution_m

    xs = np.arange(xmin, xmax, resolution_m)
    ys = np.arange(ymin, ymax, resolution_m)

    log.info(
        f"Grid extents: ({xmin:.0f}, {ymin:.0f}) → ({xmax:.0f}, {ymax:.0f}), "
        f"potential cells: {len(xs) * len(ys):,}"
    )

    # Vectorised cell creation
    x_coords, y_coords = np.meshgrid(xs, ys)
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()

    cells = [
        box(x, y, x + resolution_m, y + resolution_m)
        for x, y in zip(x_flat, y_flat)
    ]

    grid = gpd.GeoDataFrame(geometry=cells, crs=boundary_gdf.crs)

    # Keep only cells that intersect the study boundary
    boundary_union = unary_union(boundary_gdf.geometry)
    grid = grid[grid.intersects(boundary_union)].copy().reset_index(drop=True)
    grid["cell_id"] = range(len(grid))

    # Cell centroid coordinates (BNG, metres)
    grid["cx"] = grid.geometry.centroid.x
    grid["cy"] = grid.geometry.centroid.y

    log.info(f"Grid built: {len(grid):,} cells at {resolution_m} m resolution")
    return grid
