import logging
import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def ensure_crs(
    gdf: gpd.GeoDataFrame,
    name: str,
    assumed_epsg: int | None = None,
    target_epsg: int = 27700,
) -> gpd.GeoDataFrame:
    log = get_logger("utils")

    if gdf.crs is None:
        if assumed_epsg is None:
            raise ValueError(f"{name}: dataset has no CRS and no assumed_epsg was provided.")
        log.warning(f"{name}: no CRS detected — assuming EPSG:{assumed_epsg} (check .prj)")
        gdf = gdf.set_crs(epsg=assumed_epsg)

    current_epsg = gdf.crs.to_epsg()
    if current_epsg != target_epsg:
        log.warning(f"{name}: reprojecting EPSG:{current_epsg} → EPSG:{target_epsg}")
        gdf = gdf.to_crs(epsg=target_epsg)

    return gdf
