# 1. Create the environment (one-time, ~5 min)
conda env create -f env.yml

# 2. Activate it
conda activate mar-st

# 3. Launch the dashboard
panel serve dashboard/app.py --show --autoreload

---
Project structure created

├── config.yaml              ← all parameters, paths, weights, score maps
├── env.yml                  ← new mar-st conda environment
├── .gitignore
├── src/mar_st/
│   ├── config.py            ← load config.yaml, resolve paths
│   ├── utils.py             ← CRS validation, logging
│   ├── grid.py              ← master 1km grid creation
│   ├── ingest.py            ← per-dataset loaders + scoring + spatial join
│   └── mce.py               ← weighted sum + constraints engine
├── scripts/
│   └── preprocess.py        ← run once; populates processed/
├── dashboard/
│   └── app.py               ← Panel dashboard (FastListTemplate)
└── processed/               ← 7 GeoPackages, ~33 MB total
    ├── grid.gpkg             (18,686 cells, all score columns)
    ├── hydrogeology.gpkg     (753 polygons, ST area)
    ├── bedrock.gpkg, superficial.gpkg
    ├── gwmu.gpkg, sw_catchments.gpkg
    └── rivers.gpkg           (10,118 links, clipped to ST area)

---
Grid summary (1 km, 18,686 cells)

┌────────────────────────────┬─────┬──────┬─────┐
│           Score            │ Min │ Mean │ Max │
├────────────────────────────┼─────┼──────┼─────┤
│ Geological (real BGS data) │ 0   │ 3.98 │ 10  │
├────────────────────────────┼─────┼──────┼─────┤
│ Need for MAR (dummy)       │ 1.1 │ 6.8  │ 9.8 │
├────────────────────────────┼─────┼──────┼─────┤
│ Water Availability (dummy) │ 2.9 │ 5.6  │ 8.8 │
├────────────────────────────┼─────┼──────┼─────┤
│ Composite                  │ 2.4 │ 5.5  │ 9.2 │
└────────────────────────────┴─────┴──────┴─────┘

---
Assumptions made

1. BGS Hydrogeology CRS — PRJ confirms BNG but EPSG tag absent; set to EPSG:27700ning. Same for Bedrock, Superficial, OS Rivers.
2. "Low productive" / "Low productivity" — two spelling variants in the dataset; both mapped to score 2 in config.
3. Dominant-polygon rule — where multiple geology polygons overlap a 1km cell, th fragment wins. Acceptable at this resolution; alternatives (area-weighted mean,mode) can be swapped in later.
4. Constraint: non-productive aquifer — cells where the dominant hydro polygon isroundwater" are masked (score = 0). This reported 0 cells this run — theconstraint mask fix (comparing int 1 not string "1") is in place; re-running preprocess will give the correct count.
5. Need for MAR / Water Availability — entirely dummy (GWMU-seeded random field wlagged in the dashboard sidebar in red.
6. Rivers — OS WatercourseLink only (geometry, no attributes). HydroNode not loaded yet.
7. MAR Objective selector — in the UI but not yet wired to different scoring logime composite score for now.

---
Suggested next steps

   1. Create mar-st env and verify dashboard loads — confirm map renders, sliders respond, layer toggles work
   2. Re-run preprocess.py after env is active — the constraint mask fix will then cve cells
   3. Wire up real Need for MAR — start with GWMU proximity/deficit layer once CAMS data arrives
   4. Add AHP pairwise matrix to mce.py to replace the flat weight sliders with a prnvector approach
   5. Add click-to-query — clicking a cell should show its scores breakdown in the info panel