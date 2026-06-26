# Project: ASR/MAR Suitability Screening & Dashboard — UK
## Overview
A spatial suitability screening tool and interactive dashboard to identify
suitable locations for Aquifer Storage Recovery (ASR) and Managed Aquifer
Recharge (MAR) across the UK. The tool ingests multiple spatial datasets,
applies user-defined weighting and ranking to each criterion layer, and
produces a composite suitability map with an interactive front-end.
## Objectives
- Ingest, process and visualise spatial datasets relevant to ASR/MAR suitability
- Apply a weighted multi-criteria evaluation (MCE) framework to rank locations
- Produce an interactive dashboard for exploring suitability outputs
- Allow users to adjust weights and thresholds dynamically and re-run analysis
- Export results as maps, reports, and spatial files for client/stakeholder use
## Tech Stack
- **Language**: Python 3.11
- **Environment**: Conda (env.yml) — always update env.yml when adding packages
- **Spatial analysis**: geopandas, rasterio, rasterstats, shapely, pyproj, fiona
- **Raster processing**: xarray, rioxarray, numpy, scipy
- **Data handling**: pandas, openpyxl
- **Visualisation / Dashboard**: panel, hvplot, holoviews (preferred) or streamlit
- **Mapping**: folium, leaflet via panel, or pydeck for 3D views
- **Plotting**: matplotlib, seaborn
- **Testing**: pytest
## Coordinate Reference Systems
- **Default CRS**: OSGB36 British National Grid (EPSG:27700) for all analysis
- **Web display**: reproject to WGS84 (EPSG:4326) only at the visualisation layer
- Always check and align CRS before any spatial join, clip, or overlay operation
- Log a warning if any input dataset is not in EPSG:27700 at ingest