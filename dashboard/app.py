"""
MAR-ST Phase 1 Dashboard

Suitability grid rendered as hv.Rectangles (Bokeh quad glyph).
Quad glyphs are never triangulated — no QuadMesh conversion, no NaN-to-zero
WebGL artefacts, no fan spikes.

Static overlay layers are pre-built once and served from cache.
The reactive hot-path only recomputes composite scores (~1 ms numpy) and
rebuilds the 12 k-row Rectangles element.

Launch:
    conda activate mar-st
    panel serve dashboard/app.py --show --autoreload

Requires:
    python scripts/preprocess.py   (populates processed/)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import geopandas as gpd

import panel as pn
import hvplot.pandas        # noqa: F401
import holoviews as hv
import geoviews.tile_sources as gts
from holoviews import opts
from bokeh.models import HoverTool, FixedTicker, CustomJSTickFormatter
from pyproj import Transformer

from mar_st import config as cfg_mod
from mar_st.utils import get_logger

log = get_logger("dashboard")

pn.extension(throttled=True)
hv.extension("bokeh")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED = cfg_mod.ROOT / "processed"
CFG       = cfg_mod.load()
THEMES    = CFG["mce"]["themes"]
MAP_H     = CFG["dashboard"]["map_height"]
SIDEBAR_W = CFG["dashboard"]["sidebar_width"]
SUIT_CMAP = "RdYlGn"

# ---------------------------------------------------------------------------
# BNG axis tick positions — pre-computed once at startup
# Converts nice round BNG values to EPSG:3857 for FixedTicker placement.
# ---------------------------------------------------------------------------
_T = Transformer.from_crs(27700, 3857, always_xy=True)
_E_BNG = [300_000, 340_000, 380_000, 420_000, 460_000, 500_000]   # Easting
_N_BNG = [200_000, 240_000, 280_000, 320_000, 360_000, 400_000]   # Northing
# Use a central representative coordinate from the other axis for conversion
_E_WM  = list(_T.transform(_E_BNG, [295_000] * len(_E_BNG))[0])
_N_WM  = list(_T.transform([410_000] * len(_N_BNG), _N_BNG)[1])


def _bng_axes_hook(plot, element):
    """Replace Web Mercator axis ticks/labels with BNG Easting/Northing (km)."""
    fig = plot.state

    def _fmt(wm_vals, bng_vals):
        keys_js = str([round(v) for v in wm_vals])
        # Labels in km (e.g. 300 → "300")
        vals_js = str([str(b // 1000) for b in bng_vals])
        return CustomJSTickFormatter(code=f"""
            var keys = {keys_js};
            var vals = {vals_js};
            for (var i = 0; i < keys.length; i++) {{
                if (Math.abs(keys[i] - tick) < 5000) return vals[i];
            }}
            return "";
        """)

    fig.xaxis.ticker      = FixedTicker(ticks=_E_WM)
    fig.xaxis.formatter   = _fmt(_E_WM, _E_BNG)
    fig.xaxis.axis_label  = "Easting (BNG km)"
    fig.yaxis.ticker      = FixedTicker(ticks=_N_WM)
    fig.yaxis.formatter   = _fmt(_N_WM, _N_BNG)
    fig.yaxis.axis_label  = "Northing (BNG km)"


# ---------------------------------------------------------------------------
# Cached startup loader — runs once per server process
# ---------------------------------------------------------------------------

@pn.cache
def _load_all() -> dict:
    """Load pre-computed assets and build static HoloViews layers once."""

    def _gpkg(name):
        p = PROCESSED / f"{name}.gpkg"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun:  python scripts/preprocess.py"
            )
        return gpd.read_file(p)

    log.info("Loading pre-computed assets (first connection — will be cached)...")

    # Per-cell data: bounding boxes + scores in EPSG:3857 (matches Bokeh figure CRS)
    cells = pd.read_csv(PROCESSED / "hover_base.csv")

    # Pre-filter feasible cells once — hot path only touches this slice
    feasible_base = cells[cells["constraint"] == 1][
        ["x0", "y0", "x1", "y1",
         "geo_score", "need_score", "water_score",
         "sdtm_score", "sdtm_mean_m",
         "aquifer", "gwmu", "gwm_name"]
    ].copy().reset_index(drop=True)

    # Excluded cells — built once, never changes with weight sliders
    # geo_score == 0  → non-productive aquifer
    # geo_score  > 0  → within GWDTE (or other hard constraint)
    excl = cells[cells["constraint"] == 0][
        ["x0", "y0", "x1", "y1", "geo_score", "aquifer", "gwmu"]
    ].copy().reset_index(drop=True)
    excl["reason"] = np.where(
        excl["geo_score"] == 0,
        "Non-productive aquifer",
        "GWDTE",
    )

    HOVER_EXCL = HoverTool(tooltips=[
        ("GWMU",        "@gwmu"),
        ("Aquifer",     "@aquifer"),
        ("Geo Score",   "@geo_score{0.0}"),
        ("──────────", ""),
        ("Excluded:",   "@reason"),
    ])

    EXCL_RECTS = hv.Rectangles(
        excl,
        kdims=["x0", "y0", "x1", "y1"],
        vdims=["geo_score", "aquifer", "gwmu", "reason"],
    ).opts(
        opts.Rectangles(
            color="#de1212",
            fill_alpha=0.40,
            line_alpha=0,
            tools=[HOVER_EXCL],
        )
    )

    # Static overlay layers (EPSG:3857 → geo=False passes coords as-is to Bokeh)
    # NOTE: gwdte_3857.gpkg is intentionally NOT loaded — its complex polygon
    # geometries produce degenerate WebGL vertices that cause fan/spike artefacts.
    # GWDTE exclusion is already visible as uncoloured gaps in the suitability grid.
    bnd    = _gpkg("boundary_3857")
    hydro  = _gpkg("hydrogeology_3857")
    gwmu   = _gpkg("gwmu_3857")
    gwm    = _gpkg("gwm_3857")
    sw     = _gpkg("sw_catchments_3857")
    rivers = _gpkg("rivers_3857")

    # Simplify rivers further at load time: 2 km tolerance in EPSG:3857 metres
    # (≈1.25 km ground at 52°N) — appropriate for a 1 km analysis grid.
    rivers = rivers.copy()
    rivers.geometry = rivers.geometry.simplify(2000, preserve_topology=False)
    rivers = rivers[~rivers.geometry.is_empty].reset_index(drop=True)

    # Convert polygon layers to their exterior rings (LineStrings) before
    # rendering. Polygon fills — even with fill_alpha=0 — still go through
    # WebGL triangulation and produce the same fan/spike artefacts seen with
    # the GWDTE layer. Rendering only the ring lines avoids this entirely.
    def _rings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = gdf.copy()
        g.geometry = g.geometry.boundary
        return g[~g.geometry.is_empty].reset_index(drop=True)

    _s = dict(geo=False, legend=False, hover=False)

    L_BOUNDARY = _rings(bnd).hvplot(line_color="white",   line_width=2.5, **_s)
    L_GWMU     = _rings(gwmu).hvplot(line_color="#e67e00", line_width=1.5, **_s)
    L_SW       = _rings(sw).hvplot(line_color="#2980b9",  line_width=1.2, **_s)
    L_RIVERS   = rivers.hvplot(line_color="#5dade2", line_width=0.8, **_s)
    gwm_rings = _rings(gwm[["Name", "geometry"]])
    gwm_rings.geometry = gwm_rings.geometry.simplify(1000, preserve_topology=False)
    gwm_rings = gwm_rings[~gwm_rings.geometry.is_empty].reset_index(drop=True)
    HOVER_GWM = HoverTool(tooltips=[("GW Model", "@Name")])
    L_GWM = gwm_rings.hvplot(
        line_color="#9b59b6", line_width=2.0,
        tools=[HOVER_GWM],
        geo=False, legend=False, hover=False,
    )
    L_HYDRO    = hydro.hvplot(
        c="CHARACTER",
        cmap=["#e0e0e0", "#c6dff0", "#57a0ce", "#1a6faf", "#c6dff0"],
        alpha=0.45, line_width=0, **_s,
    )

    TILE_BASE = gts.CartoDark.opts(
        opts.WMTS(
            width=900, height=MAP_H,
            active_tools=["wheel_zoom", "pan"],
            toolbar="above",
        )
    )

    log.info("Startup complete — subsequent connections served from cache.")

    return dict(
        CELLS=cells,
        FEASIBLE_BASE=feasible_base,
        EXCL_RECTS=EXCL_RECTS,
        TILE_BASE=TILE_BASE,
        L_BOUNDARY=L_BOUNDARY,
        L_GWMU=L_GWMU,
        L_GWM=L_GWM,
        L_SW=L_SW,
        L_RIVERS=L_RIVERS,
        L_HYDRO=L_HYDRO,
    )


try:
    _d = _load_all()
    CELLS         = _d["CELLS"]
    FEASIBLE_BASE = _d["FEASIBLE_BASE"]
    EXCL_RECTS    = _d["EXCL_RECTS"]
    TILE_BASE     = _d["TILE_BASE"]
    L_BOUNDARY    = _d["L_BOUNDARY"]
    L_GWMU        = _d["L_GWMU"]
    L_GWM         = _d["L_GWM"]
    L_SW          = _d["L_SW"]
    L_RIVERS      = _d["L_RIVERS"]
    L_HYDRO       = _d["L_HYDRO"]
    DATA_LOADED = True
except FileNotFoundError as e:
    DATA_LOADED = False
    LOAD_ERROR  = str(e)
    log.error(LOAD_ERROR)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

w_need = pn.widgets.FloatSlider(
    name="Need for MAR",
    value=THEMES["need_for_mar"]["weight"],
    start=0.0, end=1.0, step=0.01,
)
w_geo = pn.widgets.FloatSlider(
    name="Geological Suitability",
    value=THEMES["geological_suitability"]["weight"],
    start=0.0, end=1.0, step=0.01,
)
w_water = pn.widgets.FloatSlider(
    name="Water Availability",
    value=THEMES["water_availability"]["weight"],
    start=0.0, end=1.0, step=0.01,
)

layer_checks = pn.widgets.CheckBoxGroup(
    name="Overlay Layers",
    value=["Suitability Grid", "Excluded Cells", "Study Boundary", "GW Mgmt Units"],
    options=[
        "Suitability Grid",
        "Excluded Cells",
        "Study Boundary",
        "Hydrogeology (BGS)",
        "GW Mgmt Units",
        "GW Models",
        "SW Catchments",
        "Rivers",
    ],
)

colour_by = pn.widgets.Select(
    name="Colour grid by",
    value="composite_score",
    options={
        "Composite Suitability":       "composite_score",
        "Need for MAR (dummy)":        "need_score",
        "Geological Suitability":      "geo_score",
        "Water Availability (dummy)":  "water_score",
        "Infiltration (SDTM)":         "sdtm_score",
    },
)

mar_objective = pn.widgets.Select(
    name="MAR Objective",
    options=[
        "All objectives",
        "Environmental Destination",
        "Aquifer Water Balance",
        "Storm Water (DWMP)",
        "Water Supply / ASR",
    ],
    value="All objectives",
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

if not DATA_LOADED:
    sidebar = pn.Column(
        pn.pane.Markdown("## MAR-ST Tool\n*Phase 1 — Screening*"),
        pn.layout.Divider(),
        pn.pane.Alert(
            "Preprocessed data not found.\n\n"
            "Run:  `python scripts/preprocess.py`",
            alert_type="danger",
        ),
        width=SIDEBAR_W,
    )
    main_content = pn.pane.Markdown(
        "## Awaiting preprocessed data — see sidebar."
    )

else:
    # -----------------------------------------------------------------------
    # Column map: score selector → vdim name in FEASIBLE_BASE / CELLS
    # -----------------------------------------------------------------------
    _COL_MAP = {
        "composite_score": "composite",
        "need_score":      "need_score",
        "geo_score":       "geo_score",
        "water_score":     "water_score",
        "sdtm_score":      "sdtm_score",
    }

    # -----------------------------------------------------------------------
    # Reactive map
    #
    # hv.Rectangles → Bokeh quad glyph → zero triangulation.
    # Coordinates are in EPSG:3857 (metres), matching gts.CartoDark's CRS.
    # Hot path: numpy weighted sum on ~12 k feasible rows (~1 ms)
    # -----------------------------------------------------------------------

    def _build_map(w_n, w_g, w_w, layers, col_by):
        total = (w_n + w_g + w_w) or 1.0

        feasible = FEASIBLE_BASE.copy()
        feasible["composite"] = np.clip(
            (w_n / total * feasible["need_score"]
             + w_g / total * feasible["geo_score"]
             + w_w / total * feasible["water_score"]),
            0, 10,
        ).round(1)

        score_col = _COL_MAP[col_by]

        # Build hover tooltip conditionally — layer-specific rows only appear
        # when that overlay layer is active.
        _tt = []
        if "GW Models"    in layers: _tt += [("GW Model",  "@gwm_name")]
        if "GW Mgmt Units" in layers: _tt += [("GWMU",      "@gwmu")]
        _tt += [
            ("Aquifer",         "@aquifer"),
            ("──────────────", ""),
            ("Composite",       "@composite{0.0}"),
            ("Geological",      "@geo_score{0.0}"),
            ("Need (dummy)",    "@need_score{0.0}"),
            ("Water (dummy)",   "@water_score{0.0}"),
            ("──────────────", ""),
            ("Infiltration",    "@sdtm_score{0.0}"),
            ("SDTM depth (m)",  "@sdtm_mean_m{0.0}"),
        ]
        hover_tool = HoverTool(tooltips=_tt)

        rects = hv.Rectangles(
            feasible,
            kdims=["x0", "y0", "x1", "y1"],
            vdims=["composite", "geo_score", "need_score",
                   "water_score", "sdtm_score", "sdtm_mean_m",
                   "aquifer", "gwmu", "gwm_name"],
        ).opts(
            opts.Rectangles(
                color=score_col,
                cmap=SUIT_CMAP,
                clim=(0, 10),
                colorbar=True,
                clabel="Score (0–10)",
                line_alpha=0,
                fill_alpha=0.72,
                tools=[hover_tool],
            )
        )

        plot = TILE_BASE
        if "Excluded Cells"     in layers: plot = plot * EXCL_RECTS
        if "Suitability Grid"   in layers: plot = plot * rects
        if "Hydrogeology (BGS)" in layers: plot = plot * L_HYDRO
        if "GW Mgmt Units"      in layers: plot = plot * L_GWMU
        if "GW Models"          in layers: plot = plot * L_GWM
        if "SW Catchments"      in layers: plot = plot * L_SW
        if "Rivers"             in layers: plot = plot * L_RIVERS
        if "Study Boundary"     in layers: plot = plot * L_BOUNDARY

        return plot.opts(
            opts.Overlay(
                active_tools=["wheel_zoom", "pan"],
                toolbar="above",
                hooks=[_bng_axes_hook],
            )
        )

    map_pane = pn.panel(
        pn.bind(_build_map, w_need, w_geo, w_water, layer_checks, colour_by),
        sizing_mode="stretch_both",
    )

    # -----------------------------------------------------------------------
    # Stats panel
    # -----------------------------------------------------------------------

    def _stats(w_n, w_g, w_w):
        total = (w_n + w_g + w_w) or 1.0
        composite = np.clip(
            (w_n / total * CELLS["need_score"].values
             + w_g / total * CELLS["geo_score"].values
             + w_w / total * CELLS["water_score"].values)
            * CELLS["constraint"].values,
            0, 10,
        )
        mask     = CELLS["constraint"].values == 1
        feasible = composite[mask]
        excl     = int((~mask).sum())
        hi       = int((feasible >= 7).sum())
        med      = int(((feasible >= 4) & (feasible < 7)).sum())
        lo       = int((feasible < 4).sum())
        n        = len(feasible) or 1
        pct      = [100 * w / total for w in (w_n, w_g, w_w)]

        return pn.pane.Markdown(f"""
**Weights:** Need {pct[0]:.0f}% | Geo {pct[1]:.0f}% | Water {pct[2]:.0f}%

---

| Category | Cells | % of feasible |
|---|---:|---:|
| High (≥7) | {hi:,} | {100*hi/n:.1f}% |
| Medium (4–7) | {med:,} | {100*med/n:.1f}% |
| Low (<4) | {lo:,} | {100*lo/n:.1f}% |
| Excluded | {excl:,} | — |

**Mean score (feasible):** {feasible.mean():.2f} / 10
""", width=SIDEBAR_W - 20)

    stats_pane = pn.panel(pn.bind(_stats, w_need, w_geo, w_water))

    sidebar = pn.Column(
        pn.pane.Markdown("## MAR-ST Tool\n*Phase 1 — Screening*"),
        pn.layout.Divider(),
        pn.pane.Markdown("### MAR Objective"),
        mar_objective,
        pn.layout.Divider(),
        pn.pane.Markdown("### MCE Weights"),
        pn.pane.Markdown(
            "_Normalised automatically. Updates on slider release._",
            styles={"color": "#888", "font-size": "0.82em"},
        ),
        w_need, w_geo, w_water,
        pn.layout.Divider(),
        pn.pane.Markdown("### Display"),
        colour_by,
        layer_checks,
        pn.layout.Divider(),
        stats_pane,
        pn.layout.Divider(),
        pn.pane.Markdown(
            "⚠ *Need for MAR* and *Water Availability* are **dummy scores**. "
            "Geological Suitability uses real BGS 625K data.\n\n",
            styles={"color": "#c0392b", "font-size": "0.80em"},
        ),
        width=SIDEBAR_W,
    )
    main_content = pn.Column(map_pane, sizing_mode="stretch_both")


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

template = pn.template.FastListTemplate(
    title=CFG["dashboard"]["title"],
    sidebar=[sidebar],
    main=[main_content],
    accent_base_color="#1a6faf",
    header_background="#1a3a5c",
)

template.servable()
