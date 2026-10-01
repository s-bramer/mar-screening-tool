"""
MAR-ST Phase 1 Dashboard

Suitability grid rendered as hv.Rectangles (Bokeh quad glyph).
Quad glyphs are never triangulated - no QuadMesh conversion, no NaN-to-zero
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

import re
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
import plotly.graph_objects as go
import yaml

from mar_st import config as cfg_mod
from mar_st.utils import get_logger

log = get_logger("dashboard")

pn.extension("plotly", throttled=True)
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

# CSS injected into Bokeh slider shadow-DOM to control label appearance
_THEME_SLIDER_SS = ["""
    .bk-slider-title {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a3a5c !important;
        letter-spacing: 0.01em;
    }
"""]
_SUB_SLIDER_SS = ["""
    .bk-slider-title {
        font-size: 11.5px !important;
        font-weight: 400 !important;
        color: #55687a !important;
    }
"""]
_SUB_SLIDER_PLACEHOLDER_SS = ["""
    .bk-slider-title {
        font-size: 11.5px !important;
        font-weight: 400 !important;
        color: #c0392b !important;
    }
"""]
_CHECKBOX_SS = ["""
    :host {
        --type-ramp-base-font-size: 11.5px;
        color: #55687a;
    }
"""]
_CHECKBOX_DISABLED_SS = ["""
    :host {
        --type-ramp-base-font-size: 11.5px;
        color: #b0bec5;
    }
"""]

# ---------------------------------------------------------------------------
# MAR decision tree 
# ---------------------------------------------------------------------------

MAR_PATHS = {
    "Environmental Destination (ED)": {
        "River Water":         ["Surface Infiltration"],
        "Drinking Water":      ["Borehole Recharge", "Surface Infiltration"],
        "Storm Water":         ["Surface Infiltration"],
    },
    "Aquifer Water Balance (AWB)": {
        "River Water":         ["Surface Infiltration"],
        "Drinking Water":      ["Surface Infiltration", "Borehole Recharge"],
        "Storm Water":         ["Surface Infiltration"],
        "Treated Waste Water": ["Surface Infiltration"],
    },
    "DWMP": {
        "Storm Water":         ["Surface Infiltration"],
        "Treated Waste Water": ["Surface Infiltration"],
    },
    "WRMP": {
        "Drinking Water":      ["ASR", "Borehole Recharge"],
        "Treated Waste Water": ["Surface Infiltration"],
    },
}

_ALL_OBJECTIVES  = list(MAR_PATHS.keys())
_ALL_WATER_TYPES = sorted({wt for obj in MAR_PATHS.values() for wt in obj})
_ALL_METHODS     = sorted({m for obj in MAR_PATHS.values()
                             for ms in obj.values() for m in ms})

# ---------------------------------------------------------------------------
# Sankey data  (node indices / geometry)
# ---------------------------------------------------------------------------

_OBJ_NODE    = {"Environmental Destination (ED)": 0,
                "Aquifer Water Balance (AWB)": 1, "DWMP": 2, "WRMP": 3}
_WATER_NODE  = {"River Water": 4, "Drinking Water": 5,
                "Storm Water": 6, "Treated Waste Water": 7}
_METHOD_NODE = {"Surface Infiltration": 8, "Borehole Recharge": 9, "ASR": 10}

# All valid (obj, water, method) triples from chart.csv
_SANKEY_PATHS = [
    (0, 4, 8), (0, 5, 8), (0, 5, 9), (0, 6, 8),           # ED
    (1, 4, 8), (1, 5, 8), (1, 5, 9), (1, 6, 8), (1, 7, 8), # AWB
    (2, 6, 8), (2, 7, 8),                                   # DWMP
    (3, 5, 9), (3, 5, 10), (3, 7, 8),                      # WRMP
]

# Sankey link arrays  (22.5° = 2 units; equal fractions per water type)
_SNK_SRC = [0, 0, 0,   1, 1, 1, 1,   2, 2,   3, 3,    4,  5, 5,  5,   6,  7]
_SNK_TGT = [6, 5, 4,   5, 4, 6, 7,   6, 7,   7, 5,    8,  8, 9, 10,   8,  8]
_SNK_VAL = [4, 4, 4,   2, 2, 2, 2,   4, 4,   2, 2,    6,  3, 4,  1,  10,  8]

_SNK_LABELS = [
    "ED", "AWB", "DWMP", "WRMP",
    "River Water", "Drinking Water", "Storm Water", "Treated Waste Water",
    "Surface Infiltration", "Borehole Recharge", "ASR",
]
_SNK_X = [0.02] * 4 + [0.45] * 4 + [0.92] * 3
_SNK_Y = [0.14, 0.48, 0.70, 0.88,
           0.13, 0.31, 0.55, 0.81,
           0.43, 0.88, 0.96]

_SNK_NODE_RGB = {
    0: (39, 174, 96),    1: (26, 92, 26),
    2: (130, 224, 170),  3: (200, 240, 200),
    4: (93, 173, 226),   5: (26, 82, 118),
    6: (189, 195, 199),  7: (113, 125, 126),
    8: (245, 166, 35),   9: (211, 84, 0),   10: (123, 36, 28),
}

_SNK_DEFAULT_LINK_ALPHA = 0.35   # no filter active
_SNK_ACTIVE_LINK_ALPHA  = 0.72   # on the chosen path
_SNK_DIM_LINK_ALPHA     = 0.06   # not on the chosen path


def _active_links(obj_val: str, water_val: str, method_val: str) -> set[tuple]:
    """Return the set of (src, tgt) pairs that are on at least one active path."""
    obj_n    = _OBJ_NODE.get(obj_val)
    water_n  = _WATER_NODE.get(water_val)
    method_n = _METHOD_NODE.get(method_val)

    active = set()
    for (o, w, m) in _SANKEY_PATHS:
        if (obj_n    is None or o == obj_n)    and \
           (water_n  is None or w == water_n)  and \
           (method_n is None or m == method_n):
            active.add((o, w))
            active.add((w, m))
    return active


def _build_sankey_fig(obj_val: str, water_val: str, method_val: str) -> go.Figure:
    """Plotly Sankey with active path highlighted; dims all other links."""
    any_filter = not (
        obj_val.startswith("All") and
        water_val.startswith("All") and
        method_val.startswith("All")
    )
    active = _active_links(obj_val, water_val, method_val)
    active_nodes = {n for pair in active for n in pair}

    # Link colours
    link_colors = []
    for s, t in zip(_SNK_SRC, _SNK_TGT):
        r, g, b = _SNK_NODE_RGB[s]
        if (s, t) in active:
            alpha = _SNK_ACTIVE_LINK_ALPHA
        elif any_filter:
            alpha = _SNK_DIM_LINK_ALPHA
        else:
            alpha = _SNK_DEFAULT_LINK_ALPHA
        link_colors.append(f"rgba({r},{g},{b},{alpha})")

    # Node colours
    node_colors = []
    for i in range(11):
        r, g, b = _SNK_NODE_RGB[i]
        alpha = 1.0 if (not any_filter or i in active_nodes) else 0.22
        node_colors.append(f"rgba({r},{g},{b},{alpha})")

    # Subtitle showing current selection path
    parts = []
    if not obj_val.startswith("All"):    parts.append(obj_val)
    if not water_val.startswith("All"):  parts.append(water_val)
    if not method_val.startswith("All"): parts.append(method_val)
    subtitle = " → ".join(parts) if parts else "All combinations active"

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=18, thickness=22,
            line=dict(color="white", width=0.5),
            label=_SNK_LABELS,
            color=node_colors,
            x=_SNK_X,
            y=_SNK_Y,
        ),
        link=dict(
            source=_SNK_SRC,
            target=_SNK_TGT,
            value=_SNK_VAL,
            color=link_colors,
        ),
    ))

    for col_label, xpos in [
        ("MAR Objective", 0.02),
        ("Water Source",  0.45),
        ("MAR Method",    0.92),
    ]:
        fig.add_annotation(
            x=xpos, y=1.18, xref="paper", yref="paper",
            text=f"<b>{col_label}</b>",
            showarrow=False,
            font=dict(size=12, color="#1C2833", family="Arial"),
            align="center",
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>MAR Decision Tree</b>"
                f"<br><sup style='color:#555'>Selected path: {subtitle}</sup>"
            ),
            font=dict(size=14, family="Arial", color="#1C2833"),
            x=0.5, xanchor="center", y=0.98,
        ),
        font=dict(size=11, family="Arial", color="#2C3E50"),
        paper_bgcolor="#F8F9FA",
        margin=dict(l=20, r=20, t=140, b=20),
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# BNG axis tick positions - pre-computed once at startup
# ---------------------------------------------------------------------------

_T = Transformer.from_crs(27700, 3857, always_xy=True)
_E_BNG = [300_000, 340_000, 380_000, 420_000, 460_000, 500_000]
_N_BNG = [200_000, 240_000, 280_000, 320_000, 360_000, 400_000]
_E_WM  = list(_T.transform(_E_BNG, [295_000] * len(_E_BNG))[0])
_N_WM  = list(_T.transform([410_000] * len(_N_BNG), _N_BNG)[1])


def _bng_axes_hook(plot, element):
    fig = plot.state

    def _fmt(wm_vals, bng_vals):
        keys_js = str([round(v) for v in wm_vals])
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
# Cached startup loader - runs once per server process
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

    log.info("Loading pre-computed assets (first connection - will be cached)...")

    cells = pd.read_csv(PROCESSED / "hover_base.csv")

    rng = np.random.default_rng(42)
    n   = len(cells)
    x   = cells["x0"].values
    y   = cells["y0"].values
    x_n = (x - x.min()) / max(x.max() - x.min(), 1)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1)

    slope_raw = 0.55 * x_n - 0.25 * np.abs(y_n - 0.45) + 0.20 * rng.standard_normal(n)
    cells["slope_score"] = np.clip(5.5 + 3.8 * slope_raw, 1.0, 10.0).round(1)

    depth_proxy       = 0.55 * y_n + 0.45 * rng.uniform(0, 1, n)
    depth_m           = 2 + 26 * depth_proxy
    cells["depth_water_score"] = np.interp(
        depth_m, [0, 3, 8, 15, 25, 40], [2, 7, 10, 9, 5, 2]
    ).round(1)
    cells["depth_water_m"] = depth_m.round(1)

    sg_base = np.interp(
        cells["sdtm_mean_m"].fillna(0).values,
        [0, 0.5, 2, 6, 15, 30], [4, 6, 10, 9, 6, 3],
    )
    cells["surf_geo_score"] = np.clip(
        sg_base + 1.2 * rng.standard_normal(n), 1.0, 10.0
    ).round(1)

    bnd  = _gpkg("boundary_3857")
    gwmu = _gpkg("gwmu_3857")
    gwm  = _gpkg("gwm_3857")

    def _rings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        g = gdf.copy()
        g.geometry = g.geometry.boundary
        return g[~g.geometry.is_empty].reset_index(drop=True)

    _s = dict(geo=False, legend=False, hover=False)
    L_BOUNDARY = _rings(bnd).hvplot(line_color="#1a3a5c", line_width=2.5, **_s)
    L_GWMU     = _rings(gwmu).hvplot(line_color="#e67e00", line_width=1.5, **_s)

    gwm_rings = _rings(gwm[["Name", "geometry"]])
    gwm_rings.geometry = gwm_rings.geometry.simplify(1000, preserve_topology=False)
    gwm_rings = gwm_rings[~gwm_rings.geometry.is_empty].reset_index(drop=True)
    L_GWM = gwm_rings.hvplot(
        line_color="#9b59b6", line_width=2.0,
        tools=[HoverTool(tooltips=[("GW Model", "@Name")])],
        geo=False, legend=False, hover=False,
    )

    TILE_BASE = (gts.EsriWorldLightGrayBase * gts.EsriWorldLightGrayReference).opts(
        opts.WMTS(
            width=900, height=MAP_H,
            active_tools=["wheel_zoom", "pan"],
            toolbar="above",
        )
    )

    log.info("Startup complete - subsequent connections served from cache.")

    return dict(
        CELLS=cells,
        TILE_BASE=TILE_BASE,
        L_BOUNDARY=L_BOUNDARY,
        L_GWMU=L_GWMU,
        L_GWM=L_GWM,
    )


try:
    _d = _load_all()
    CELLS         = _d["CELLS"]
    TILE_BASE     = _d["TILE_BASE"]
    L_BOUNDARY    = _d["L_BOUNDARY"]
    L_GWMU        = _d["L_GWMU"]
    L_GWM         = _d["L_GWM"]
    # Individual constraint columns - added to hover_base.csv when preprocessing
    # is re-run with updated mce.py.  Derive from existing data if missing so the
    # dashboard works without a preprocess rerun (and survives @pn.cache hits).
    if "c_nonprod" not in CELLS.columns:
        CELLS["c_nonprod"] = np.where(
            (CELLS["constraint"] == 0) & (CELLS["geo_score"] <= 0.0), 0, 1
        )
        CELLS["c_gwdte"] = np.where(
            (CELLS["constraint"] == 0) & (CELLS["geo_score"] > 0.0), 0, 1
        )
    DATA_LOADED = True
except FileNotFoundError as e:
    DATA_LOADED = False
    LOAD_ERROR  = str(e)
    log.error(LOAD_ERROR)


# ---------------------------------------------------------------------------
# Widgets - MAR selection (dependent dropdowns)
# ---------------------------------------------------------------------------

mar_objective = pn.widgets.Select(
    name="MAR Objective",
    options=["All objectives"] + _ALL_OBJECTIVES,
    value="All objectives",
)

mar_water_type = pn.widgets.Select(
    name="Water Type",
    options=["All water types"] + _ALL_WATER_TYPES,
    value="All water types",
    disabled=False,
)

mar_method_sel = pn.widgets.Select(
    name="MAR Method",
    options=["All methods"] + _ALL_METHODS,
    value="All methods",
    disabled=False,
)


def _update_water_opts(event):
    obj = event.new
    if obj == "All objectives":
        new_opts = ["All water types"] + _ALL_WATER_TYPES
    else:
        new_opts = ["All water types"] + sorted(MAR_PATHS[obj].keys())
    mar_water_type.options = new_opts
    if mar_water_type.value not in new_opts:
        mar_water_type.value = "All water types"


def _update_method_opts(*_):
    obj = mar_objective.value
    wt  = mar_water_type.value

    if obj == "All objectives" and wt == "All water types":
        new_opts = ["All methods"] + _ALL_METHODS
    elif obj == "All objectives":
        methods = sorted({m for o in MAR_PATHS.values()
                          for w, ms in o.items() if w == wt for m in ms})
        new_opts = ["All methods"] + methods
    elif wt == "All water types":
        methods = sorted({m for ms in MAR_PATHS[obj].values() for m in ms})
        new_opts = ["All methods"] + methods
    else:
        new_opts = ["All methods"] + MAR_PATHS.get(obj, {}).get(wt, [])

    mar_method_sel.options = new_opts
    if mar_method_sel.value not in new_opts:
        mar_method_sel.value = "All methods"


mar_objective.param.watch(_update_water_opts,  "value")
mar_objective.param.watch(_update_method_opts, "value")
mar_water_type.param.watch(_update_method_opts,"value")


# ---------------------------------------------------------------------------
# Widgets - MCE theme weights
# ---------------------------------------------------------------------------

_v_need  = THEMES["need_for_mar"]["weight"]
_v_geo   = THEMES["geological_suitability"]["weight"]
_v_water = THEMES["water_availability"]["weight"]

w_need = pn.widgets.FloatSlider(
    name="1. Demand: Need for MAR",
    value=_v_need,
    start=0.0, end=1.0, step=0.01,
    stylesheets=_THEME_SLIDER_SS,
)
w_geo = pn.widgets.FloatSlider(
    name="2. Feasibility: Hydrogeology",
    value=_v_geo,
    start=0.0, end=1.0, step=0.01,
    stylesheets=_THEME_SLIDER_SS,
)
w_water = pn.widgets.FloatSlider(
    name="3. Supply: Water Availability",
    value=_v_water,
    start=0.0, end=1.0, step=0.01,
    stylesheets=_THEME_SLIDER_SS,
)

# ── Need for MAR sub-criteria (dummy - data pending) ──────────────────────
_ph = _SUB_SLIDER_PLACEHOLDER_SS   # shorthand: placeholder (red label)
_ac = _SUB_SLIDER_SS               # shorthand: active (normal label)

w_need_gwdte    = pn.widgets.FloatSlider(
    name="1.1  GWDTE Proximity  [ED]",              value=0.25, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_need_sw_prio  = pn.widgets.FloatSlider(
    name="1.2  Surface Water Body Priority  [ED]",  value=0.25, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_need_abs_risk = pn.widgets.FloatSlider(
    name="1.3  GW Abstractions at Risk  [ED]",      value=0.10, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_need_cams     = pn.widgets.FloatSlider(
    name="1.4  CAMS Ledger Deficit  [AWB]",         value=0.20, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_need_cso      = pn.widgets.FloatSlider(
    name="1.5  CSO Overflow Proximity  [DWMP]",      value=0.15, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_need_wwtw_prox = pn.widgets.FloatSlider(
    name="1.6  WwTW / pDWF Proximity  [DWMP]",      value=0.10, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))

# ── Geological sub-criteria (real BGS data for aquifer + SDTM) ───────────
w_aquifer     = pn.widgets.FloatSlider(
    name="2.1  Aquifer Classification",             value=0.50, start=0, end=1, step=0.01,
    stylesheets=_ac, margin=(5, 10, 5, 10))
w_sdtm        = pn.widgets.FloatSlider(
    name="2.2  SDTM Thickness  [infiltration]",     value=0.20, start=0, end=1, step=0.01,
    stylesheets=_ac, margin=(5, 10, 5, 10))
w_slope       = pn.widgets.FloatSlider(
    name="2.3  Topography / Slope  [infiltration]", value=0.10, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_depth_water = pn.widgets.FloatSlider(
    name="2.4  Depth to Water",                     value=0.10, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_surf_geo    = pn.widgets.FloatSlider(
    name="2.5  Surface Geology  [infiltration]",    value=0.10, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))

# ── Water Availability sub-criteria (dummy - data pending) ────────────────
w_water_storm  = pn.widgets.FloatSlider(
    name="3.1  Storm Water at Network Points  [DWMP]", value=0.25, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_water_drink  = pn.widgets.FloatSlider(
    name="3.2  Drinking Water Availability",            value=0.25, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_water_tww    = pn.widgets.FloatSlider(
    name="3.3  Treated Waste Water  (WwTW proximity)", value=0.25, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))
w_water_river  = pn.widgets.FloatSlider(
    name="3.4  River Water  (untreated)",               value=0.25, start=0, end=1, step=0.01,
    stylesheets=_ph, margin=(5, 10, 5, 10))

# ── Hard constraint toggles ───────────────────────────────────────────────
c_nonprod_chk = pn.widgets.Checkbox(
    name="Non-productive aquifer", value=True,
    stylesheets=_CHECKBOX_SS, margin=(3, 10, 3, 10))
c_gwdte_chk   = pn.widgets.Checkbox(
    name="GWDTE overlap", value=True,
    stylesheets=_CHECKBOX_SS, margin=(3, 10, 3, 10))
c_sw_land_chk = pn.widgets.Checkbox(
    name="Land use - surface water  [coming soon]", value=False,
    disabled=True, stylesheets=_CHECKBOX_DISABLED_SS, margin=(3, 10, 3, 10))
c_flood_chk   = pn.widgets.Checkbox(
    name="Surface water flood risk  [coming soon]", value=False,
    disabled=True, stylesheets=_CHECKBOX_DISABLED_SS, margin=(3, 10, 3, 10))
c_contam_chk  = pn.widgets.Checkbox(
    name="Contaminated land / landfill  [coming soon]", value=False,
    disabled=True, stylesheets=_CHECKBOX_DISABLED_SS, margin=(3, 10, 3, 10))
c_spz_chk     = pn.widgets.Checkbox(
    name="Source Protection Zone SPZ1  [coming soon]", value=False,
    disabled=True, stylesheets=_CHECKBOX_DISABLED_SS, margin=(3, 10, 3, 10))

_INFIL_ONLY_WIDGETS = [w_sdtm, w_slope, w_surf_geo]


def _on_method_change(event):
    is_borehole = event.new in ("Borehole Recharge", "ASR")
    for w in _INFIL_ONLY_WIDGETS:
        w.disabled = is_borehole


mar_method_sel.param.watch(_on_method_change, "value")


# ---------------------------------------------------------------------------
# Widgets - display controls
# ---------------------------------------------------------------------------

layer_checks = pn.widgets.CheckBoxGroup(
    name="Overlay Layers",
    value=["Suitability Grid", "Excluded Cells", "Study Boundary", "GW Mgmt Units"],
    options=[
        "Suitability Grid",
        "Excluded Cells",
        "Study Boundary",
        "GW Mgmt Units",
        "GW Models",
        "Hydrogeology (BGS)  - coming soon",
        "SW Catchments  - coming soon",
        "Rivers  - coming soon",
    ],
)

colour_by = pn.widgets.Select(
    name="Colour grid by",
    value="composite_score",
    options={
        "Composite Suitability":             "composite_score",
        "Geological Sub-composite":          "geo_composite",
        "Need for MAR (dummy)":              "need_score",
        "Geological Suitability (aquifer)":  "geo_score",
        "Water Availability (dummy)":        "water_score",
        "SDTM Thickness":                    "sdtm_score",
        "Topography / Slope (dummy)":        "slope_score",
        "Depth to Water (dummy)":            "depth_water_score",
        "Surface Geology (dummy)":           "surf_geo_score",
    },
)

# ---------------------------------------------------------------------------
# Reactive Sankey pane (always visible, outside DATA_LOADED block)
# ---------------------------------------------------------------------------

sankey_pane = pn.panel(
    pn.bind(_build_sankey_fig, mar_objective, mar_water_type, mar_method_sel),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------

_INFIL_COLS = {"sdtm_score", "slope_score", "surf_geo_score"}

_GEO_SUB_COLS = {
    "aquifer":  "geo_score",
    "sdtm":     "sdtm_score",
    "slope":    "slope_score",
    "depth":    "depth_water_score",
    "surf_geo": "surf_geo_score",
}


def _compute_geo_composite(df, method, w_aq, w_sd, w_sl, w_dw, w_sg):
    is_infil = method not in ("Borehole Recharge", "ASR")
    active = {
        "geo_score":         w_aq,
        "sdtm_score":        w_sd if is_infil else 0.0,
        "slope_score":       w_sl if is_infil else 0.0,
        "depth_water_score": w_dw,
        "surf_geo_score":    w_sg if is_infil else 0.0,
    }
    total = sum(active.values()) or 1.0
    return np.clip(
        sum((w / total) * df[col].values for col, w in active.items()),
        0, 10,
    )


# ---------------------------------------------------------------------------
# Data Sources tab
# ---------------------------------------------------------------------------

_DS_YAML = Path(__file__).parent / "data_sources.yaml"

_STATUS_BADGE = {
    "active":      ("✔ Active",      "#27ae60"),
    "placeholder": ("⚠ Placeholder", "#e67e00"),
    "coming_soon": ("🔜 Coming soon", "#7f8c8d"),
    "overlay":     ("⬚ Overlay",     "#2980b9"),
}


def _format_field(label: str, value) -> str:
    """Render one YAML field as Markdown for the Data Sources tab.

    Single-line → **Label:** value
    Multi-line  → **Label:** on its own line, then forced <br> per content line.
                  Leading '>' is escaped so it doesn't render as a blockquote.
                  Trailing empty lines (YAML | scalar artifact) are stripped.
    """
    v = str(value).strip()
    if "\n" not in v:
        return f"**{label}:** {v}  "

    lines = v.split("\n")
    # Strip trailing empty lines produced by YAML | block scalars.
    while lines and not lines[-1].strip():
        lines.pop()
    # Escape '>' at the very start of a line (Markdown blockquote syntax).
    lines = [re.sub(r"^(\s*)>", r"\1&gt;", l) for l in lines]

    body = "  \n".join(lines)
    return f"**{label}:**  \n{body}  "


def _build_data_sources_tab() -> pn.Column:
    try:
        with open(_DS_YAML, encoding="utf-8") as f:
            ds = yaml.safe_load(f)
    except FileNotFoundError:
        return pn.Column(pn.pane.Alert(
            f"data_sources.yaml not found at {_DS_YAML}", alert_type="warning"
        ))

    _h1 = [":host h2 { font-size:15px; font-weight:700; color:#1a3a5c; margin:8px 0 2px 0; }"]
    _h2 = [":host h3 { font-size:12px; font-weight:600; color:#1a3a5c; margin:4px 0 1px 0; }"]

    rows = [pn.pane.Markdown(
        "Data sources, licences and processing steps for each criterion. "
        "Edit `dashboard/data_sources.yaml` to update this tab.\n\n---",
        styles={"color": "#555", "font-size": "0.85em"},
        sizing_mode="stretch_width",
    )]

    # ── MCE themes ────────────────────────────────────────────────────────────
    for i_theme, theme in enumerate(ds.get("themes", []), 1):
        rows.append(pn.pane.Markdown(
            f"## {i_theme}. {theme['label']}",
            stylesheets=_h1, sizing_mode="stretch_width",
        ))
        if theme.get("description"):
            rows.append(pn.pane.Markdown(
                f"_{theme['description'].strip()}_",
                styles={"color": "#555", "font-size": "0.83em"},
                stylesheets=[":host p { margin: 0; } :host { margin: 0 0 4px 0; }"],
                sizing_mode="stretch_width",
            ))

        for i_crit, crit in enumerate(theme.get("criteria", []), 1):
            label, colour = _STATUS_BADGE.get(
                crit.get("status", "placeholder"), ("?", "#888")
            )
            badge = (
                f"<span style='background:{colour};color:white;"
                f"padding:1px 7px;border-radius:3px;font-size:0.78em;"
                f"font-weight:600;margin-left:6px'>{label}</span>"
            )
            number = f"{i_theme}.{i_crit}"
            name_html = (
                f"<span style='color:#c0392b'>{crit['name']}</span>"
                if crit.get("status") == "placeholder"
                else crit["name"]
            )
            md_lines = [f"### {number}  {name_html} {badge}"]
            if crit.get("sidebar_slider"):
                md_lines.append(
                    f"_Sidebar slider: **{crit['sidebar_slider']}**_"
                )
            md_lines.append("")

            fields = [
                ("Source",      crit.get("source")),
                ("Licence",     crit.get("licence")),
                ("Accessed",    crit.get("date_accessed")),
                ("Key field",   crit.get("key_field")),
                ("Gridding",    crit.get("gridding", "").strip()),
                ("Scoring",     crit.get("scoring", "").strip()),
                ("Notes",       crit.get("notes")),
            ]
            for k, v in fields:
                if v:
                    md_lines.append(_format_field(k, v))

            rows.append(pn.pane.Markdown(
                "\n".join(md_lines),
                stylesheets=_h2,
                sizing_mode="stretch_width",
                styles={"border-left": f"3px solid {colour}",
                        "padding-left": "10px", "margin": "6px 0 10px 0"},
            ))

        rows.append(pn.layout.Divider())

    # ── Hard Constraints ──────────────────────────────────────────────────────
    rows.append(pn.pane.Markdown("## Hard Constraints", stylesheets=_h1,
                                  sizing_mode="stretch_width"))
    rows.append(pn.pane.Markdown(
        "_Cells intersecting these layers are hard-excluded (score = 0) regardless of MCE weights. "
        "Active constraints are toggled via checkboxes in the sidebar._",
        styles={"color": "#555", "font-size": "0.83em"},
        sizing_mode="stretch_width",
    ))

    for i_con, con in enumerate(ds.get("constraints", []), 1):
        label, colour = _STATUS_BADGE.get(
            con.get("status", "coming_soon"), ("?", "#888")
        )
        badge = (
            f"<span style='background:{colour};color:white;"
            f"padding:1px 7px;border-radius:3px;font-size:0.78em;"
            f"font-weight:600;margin-left:6px'>{label}</span>"
        )
        md_lines = [f"### C.{i_con}  {con['name']} {badge}"]
        if con.get("dashboard_toggle"):
            md_lines.append(f"_Dashboard toggle: **{con['dashboard_toggle']}**_")
        md_lines.append("")
        for k, v in [("Source", con.get("source")),
                     ("Licence", con.get("licence")),
                     ("Accessed", con.get("date_accessed")),
                     ("Key field", con.get("key_field")),
                     ("Notes", con.get("notes"))]:
            if v:
                md_lines.append(_format_field(k, v))

        rows.append(pn.pane.Markdown(
            "\n".join(md_lines),
            stylesheets=_h2,
            sizing_mode="stretch_width",
            styles={"border-left": "3px solid #c0392b",
                    "padding-left": "10px", "margin": "6px 0 10px 0"},
        ))

    rows.append(pn.layout.Divider())

    # ── Overlay layers ────────────────────────────────────────────────────────
    rows.append(pn.pane.Markdown("## Overlay Layers", stylesheets=_h1,
                                  sizing_mode="stretch_width"))
    rows.append(pn.pane.Markdown(
        "_Context layers toggled via the Layer checkboxes. Not used in MCE scoring._",
        styles={"color": "#555", "font-size": "0.83em"},
        sizing_mode="stretch_width",
    ))

    for i_ov, ov in enumerate(ds.get("overlays", []), 1):
        label, colour = _STATUS_BADGE.get(
            ov.get("status", "overlay"), ("?", "#888")
        )
        badge = (
            f"<span style='background:{colour};color:white;"
            f"padding:1px 7px;border-radius:3px;font-size:0.78em;"
            f"font-weight:600;margin-left:6px'>{label}</span>"
        )
        md_lines = [f"### O.{i_ov}  {ov['name']} {badge}"]
        if ov.get("layer_checkbox"):
            md_lines.append(f"_Layer checkbox: **{ov['layer_checkbox']}**_")
        md_lines.append("")
        for k, v in [("Source", ov.get("source")),
                     ("Licence", ov.get("licence")),
                     ("Accessed", ov.get("date_accessed")),
                     ("Notes", ov.get("notes"))]:
            if v:
                md_lines.append(_format_field(k, v))

        rows.append(pn.pane.Markdown(
            "\n".join(md_lines),
            stylesheets=_h2,
            sizing_mode="stretch_width",
            styles={"border-left": f"3px solid {colour}",
                    "padding-left": "10px", "margin": "6px 0 10px 0"},
        ))

    return pn.Column(*rows, sizing_mode="stretch_width",
                     styles={"padding": "12px 20px"})


_data_sources_tab = _build_data_sources_tab()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# ── Reactive selection summary ─────────────────────────────────────────────

def _sel_summary(obj, wt, method):
    parts = []
    if not obj.startswith("All"):    parts.append(f"**{obj}**")
    if not wt.startswith("All"):     parts.append(f"**{wt}**")
    if not method.startswith("All"): parts.append(f"**{method}**")
    if parts:
        text = "→ " + " → ".join(parts)
        color = "#27AE60"
    else:
        text  = "_No filter - showing all combinations_"
        color = "#888"
    return pn.pane.Markdown(
        text,
        styles={"color": color, "font-size": "0.82em"},
        width=SIDEBAR_W - 20,
    )

sel_summary_pane = pn.panel(
    pn.bind(_sel_summary, mar_objective, mar_water_type, mar_method_sel)
)

# ── Reusable helper text style ─────────────────────────────────────────────
_hint = {"color": "#888", "font-size": "0.80em"}
_warn = {"color": "#c0392b", "font-size": "0.80em"}


if not DATA_LOADED:
    sidebar = pn.Column(
        pn.pane.Alert(
            "Preprocessed data not found.\n\n"
            "Run:  `python scripts/preprocess.py`",
            alert_type="danger",
        ),
        width=SIDEBAR_W,
    )
    main_content = pn.Tabs(
        ("MAR Decision Tree", pn.Column(
            sankey_pane,
            sizing_mode="stretch_both",
        )),
        sizing_mode="stretch_both",
    )

else:
    # -----------------------------------------------------------------------
    # Column map: colour_by selector → column in CELLS
    # -----------------------------------------------------------------------
    _COL_MAP = {
        "composite_score":   "composite",
        "geo_composite":     "geo_composite",
        "need_score":        "need_score",
        "geo_score":         "geo_score",
        "water_score":       "water_score",
        "sdtm_score":        "sdtm_score",
        "slope_score":       "slope_score",
        "depth_water_score": "depth_water_score",
        "surf_geo_score":    "surf_geo_score",
    }

    # -----------------------------------------------------------------------
    # Reactive map
    #
    # Range-save hook: Bokeh on_change callbacks write the current pan/zoom
    # into _view whenever the user moves the map.  _build_map re-applies
    # those values as xlim/ylim on the new figure so zoom is preserved.
    # -----------------------------------------------------------------------

    _view = {"x_range": None, "y_range": None}

    def _range_save_hook(plot, element):
        """Attach Bokeh callbacks to persist the current viewport."""
        fig = plot.state

        def _save(attr, old, new):
            xs, xe = fig.x_range.start, fig.x_range.end
            ys, ye = fig.y_range.start, fig.y_range.end
            if None not in (xs, xe, ys, ye):
                _view["x_range"] = (xs, xe)
                _view["y_range"] = (ys, ye)

        fig.x_range.on_change("start", _save)
        fig.x_range.on_change("end",   _save)
        fig.y_range.on_change("start", _save)
        fig.y_range.on_change("end",   _save)

    _SEL_LABEL = {
        "composite_score":   "Overall Suitability",
        "geo_composite":     "Feasibility: Hydrogeology",
        "need_score":        "Demand: Need for MAR",
        "geo_score":         "Feasibility: Hydrogeology (aquifer)",
        "water_score":       "Supply: Water Availability",
        "sdtm_score":        "Feasibility: SDTM Thickness",
        "slope_score":       "Feasibility: Topography / Slope",
        "depth_water_score": "Feasibility: Depth to Water",
        "surf_geo_score":    "Feasibility: Surface Geology",
    }
    _SUBCRIT = {"geo_score", "sdtm_score", "slope_score",
                "depth_water_score", "surf_geo_score"}

    def _build_map(w_n, w_g, w_w, w_aq, w_sd, w_sl, w_dw, w_sg,
                   layers, col_by, method, objective,
                   c_nonprod_on, c_gwdte_on):
        total = (w_n + w_g + w_w) or 1.0

        # Build combined constraint mask from active toggles
        feas_mask = np.ones(len(CELLS), dtype=bool)
        if c_nonprod_on:
            feas_mask &= (CELLS["c_nonprod"].values == 1)
        if c_gwdte_on:
            feas_mask &= (CELLS["c_gwdte"].values == 1)

        feasible = CELLS[feas_mask].copy().reset_index(drop=True)
        feasible["geo_composite"] = _compute_geo_composite(
            feasible, method, w_aq, w_sd, w_sl, w_dw, w_sg,
        ).round(1)
        feasible["composite"] = np.clip(
            (w_n / total * feasible["need_score"]
             + w_g / total * feasible["geo_composite"]
             + w_w / total * feasible["water_score"]),
            0, 10,
        ).round(1)

        # Build excluded-cells layer dynamically
        excl_df = CELLS[~feas_mask][
            ["x0", "y0", "x1", "y1", "geo_score", "aquifer", "gwmu",
             "c_nonprod", "c_gwdte"]
        ].copy().reset_index(drop=True)
        excl_df["reason"] = np.select(
            [
                (excl_df["c_nonprod"] == 0) & (excl_df["c_gwdte"] == 0),
                excl_df["c_nonprod"] == 0,
            ],
            ["Non-productive aquifer + GWDTE", "Non-productive aquifer"],
            default="GWDTE",
        )
        _hover_excl = HoverTool(tooltips=[
            ("GWMU",        "@gwmu"),
            ("Aquifer",     "@aquifer"),
            ("──────────", ""),
            ("Excluded:",   "@reason"),
        ])
        excl_rects = hv.Rectangles(
            excl_df,
            kdims=["x0", "y0", "x1", "y1"],
            vdims=["geo_score", "aquifer", "gwmu", "reason"],
        ).opts(opts.Rectangles(
            color="#de1212", fill_alpha=0.40, line_alpha=0, tools=[_hover_excl],
        ))

        score_col = _COL_MAP[col_by]
        sel_lbl   = _SEL_LABEL.get(col_by, "Selected Score")

        _tt = []
        if "GW Models"     in layers: _tt += [("GW Model", "@gwm_name")]
        if "GW Mgmt Units" in layers: _tt += [("GWMU",     "@gwmu")]
        _tt += [("Aquifer", "@aquifer"), ("──────────", "")]
        if col_by in _SUBCRIT:
            _tt += [(sel_lbl, f"@{score_col}{{0.0}}"), ("──────────", "")]
        _tt += [
            ("Demand (Need for MAR)",        "@need_score{0.0}"),
            ("Feasibility (Hydrogeology)",   "@geo_composite{0.0}"),
            ("Supply (Water Availability)",  "@water_score{0.0}"),
            ("──────────",                   ""),
            ("Overall Score",                "@composite{0.0}"),
        ]

        rects = hv.Rectangles(
            feasible,
            kdims=["x0", "y0", "x1", "y1"],
            vdims=[
                "composite", "geo_composite",
                "geo_score", "need_score", "water_score",
                "sdtm_score", "sdtm_mean_m",
                "slope_score", "depth_water_score", "depth_water_m", "surf_geo_score",
                "aquifer", "gwmu", "gwm_name",
            ],
        ).opts(opts.Rectangles(
            color=score_col,
            cmap=SUIT_CMAP,
            clim=(0, 10),
            colorbar=True,
            clabel="Score (0–10)",
            line_alpha=0,
            fill_alpha=0.72,
            tools=[HoverTool(tooltips=_tt)],
        ))

        plot = TILE_BASE
        if "Excluded Cells"   in layers: plot = plot * excl_rects
        if "Suitability Grid" in layers: plot = plot * rects
        if "GW Mgmt Units"    in layers: plot = plot * L_GWMU
        if "GW Models"        in layers: plot = plot * L_GWM
        if "Study Boundary"   in layers: plot = plot * L_BOUNDARY

        _ov_opts = dict(
            active_tools=["wheel_zoom", "pan"],
            toolbar="above",
            hooks=[_bng_axes_hook, _range_save_hook],
        )
        if _view["x_range"] is not None:
            _ov_opts["xlim"] = _view["x_range"]
            _ov_opts["ylim"] = _view["y_range"]

        return plot.opts(opts.Overlay(**_ov_opts))

    map_pane = pn.panel(
        pn.bind(
            _build_map,
            w_need, w_geo, w_water,
            w_aquifer, w_sdtm, w_slope, w_depth_water, w_surf_geo,
            layer_checks, colour_by,
            mar_method_sel, mar_objective,
            c_nonprod_chk, c_gwdte_chk,
        ),
        sizing_mode="stretch_both",
    )

    # -----------------------------------------------------------------------
    # Stats panel
    # -----------------------------------------------------------------------

    def _stats(w_n, w_g, w_w, w_aq, w_sd, w_sl, w_dw, w_sg, method,
               c_nonprod_on, c_gwdte_on):
        total = (w_n + w_g + w_w) or 1.0

        feas_mask = np.ones(len(CELLS), dtype=bool)
        if c_nonprod_on:
            feas_mask &= (CELLS["c_nonprod"].values == 1)
        if c_gwdte_on:
            feas_mask &= (CELLS["c_gwdte"].values == 1)

        geo_comp = _compute_geo_composite(CELLS, method, w_aq, w_sd, w_sl, w_dw, w_sg)

        composite = np.clip(
            (w_n / total * CELLS["need_score"].values
             + w_g / total * geo_comp
             + w_w / total * CELLS["water_score"].values)
            * feas_mask.astype(float),
            0, 10,
        )
        feasible = composite[feas_mask]
        excl     = int((~feas_mask).sum())
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
| Excluded | {excl:,} | - |

**Mean score (feasible):** {feasible.mean():.2f} / 10
""", width=SIDEBAR_W - 20)

    stats_pane = pn.panel(
        pn.bind(
            _stats,
            w_need, w_geo, w_water,
            w_aquifer, w_sdtm, w_slope, w_depth_water, w_surf_geo,
            mar_method_sel,
            c_nonprod_chk, c_gwdte_chk,
        )
    )

    # -----------------------------------------------------------------------
    # Sidebar assembly
    # -----------------------------------------------------------------------

    _card_ss = ["""
        :host .card-header {
            background: transparent !important;
            border: none !important;
            padding: 3px 6px !important;
            min-height: 22px !important;
            border-radius: 0 !important;
        }
        :host .card-header button,
        :host .card-header h3 {
            background: transparent !important;
            border: none !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0.85;
        }
        :host .card-header button:hover,
        :host .card-header h3:hover {
            opacity: 1.0;
        }
    """]

    def _card_styles(accent):
        return {
            "border-radius": "0",
            "box-shadow": "none",
            "border": "none",
            "border-left": f"3px solid {accent}",
            "border-top": "1px solid #e8edf0",
            "margin-top": "0",
        }

    _card_title = (
        "<span style='font-size:13px;font-weight:400;color:#3d5a73;"
        "letter-spacing:0.04em'>"
        "▸  Sub-criteria weights</span>"
    )

    _need_card = pn.Card(
        w_need_gwdte, w_need_sw_prio, w_need_abs_risk, w_need_cams, w_need_cso, w_need_wwtw_prox,
        title=_card_title,
        collapsed=True, collapsible=True,
        sizing_mode="stretch_width",
        styles=_card_styles("#2980b9"),
        stylesheets=_card_ss,
        margin=(0, 5, 8, 5),
    )
    _geo_card = pn.Card(
        pn.pane.Markdown(
            "_\\[infiltration\\] criteria auto-disabled for Borehole / ASR._",
            styles={"color": "#777", "font-size": "0.78em"},
        ),
        w_aquifer, w_sdtm, w_slope, w_depth_water, w_surf_geo,
        title=_card_title,
        collapsed=True, collapsible=True,
        sizing_mode="stretch_width",
        styles=_card_styles("#27ae60"),
        stylesheets=_card_ss,
        margin=(0, 5, 8, 5),
    )
    _water_card = pn.Card(
        w_water_storm, w_water_drink, w_water_tww, w_water_river,
        title=_card_title,
        collapsed=True, collapsible=True,
        sizing_mode="stretch_width",
        styles=_card_styles("#e67e22"),
        stylesheets=_card_ss,
        margin=(0, 5, 8, 5),
    )

    _constraint_card_title = (
        "<span style='font-size:13px;font-weight:400;color:#3d5a73;"
        "letter-spacing:0.04em'>"
        "▸  Active constraints</span>"
    )
    _constraint_card = pn.Card(
        c_nonprod_chk, c_gwdte_chk,
        pn.layout.Divider(margin=(4, 0, 4, 0)),
        c_sw_land_chk, c_flood_chk, c_contam_chk, c_spz_chk,
        title=_constraint_card_title,
        collapsed=True, collapsible=True,
        sizing_mode="stretch_width",
        styles=_card_styles("#c0392b"),
        stylesheets=_card_ss,
        margin=(0, 5, 8, 5),
    )

    _sh_ss = [":host h3 { font-size: 16px; font-weight: 700; color: #1a3a5c; margin: 0 0 2px 0; }"]

    def _sh(label):
        return pn.pane.Markdown(f"### {label}", stylesheets=_sh_ss, margin=(10, 0, 0, 0))

    def _div():
        return pn.layout.Divider(margin=(4, 0, 4, 0))

    sidebar = pn.Column(
        # ── MAR Selection ─────────────────────────────────────────────────
        _sh("MAR Selection"),
        pn.pane.Markdown(
            "_Select an Objective to narrow available Water Types and Methods._",
            styles=_hint, margin=(0, 0, 2, 0),
        ),
        mar_objective,
        mar_water_type,
        mar_method_sel,
        sel_summary_pane,
        _div(),

        # ── MCE Theme Weights ─────────────────────────────────────────────
        _sh("MCE Theme Weights"),
        pn.pane.Markdown(
            "_Normalised automatically. Updates on slider release._",
            styles=_hint, margin=(0, 0, 2, 0),
        ),

        w_need,
        _need_card,

        w_geo,
        _geo_card,

        w_water,
        _water_card,
        _div(),

        # ── Hard Constraints ──────────────────────────────────────────────
        _sh("Constraints"),
        pn.pane.Markdown(
            "_Cell with contraints will get a score of 0._",
            styles=_hint, margin=(0, 0, 2, 0),
        ),
        _constraint_card,
        _div(),

        # ── Display ───────────────────────────────────────────────────────
        _sh("Display and Overlays"),
        colour_by,
        layer_checks,
        _div(),

        # ── Stats ─────────────────────────────────────────────────────────
        stats_pane,
        _div(),

        pn.pane.Markdown(
            "⚠ **Placeholder data:** Need for MAR, Water Availability, "
            "Topography/Slope, Depth to Water, Surface Geology, and all "
            "Need / Water sub-criteria scores are spatially coherent dummies. "
            "Aquifer Classification and SDTM Thickness use real BGS data.",
            styles=_warn,
        ),
        width=SIDEBAR_W,
    )

    main_content = pn.Tabs(
        ("Suitability Map",   pn.Column(map_pane,           sizing_mode="stretch_both")),
        ("MAR Decision Tree", pn.Column(sankey_pane,        sizing_mode="stretch_both")),
        ("Data Sources",      pn.Column(_data_sources_tab,  sizing_mode="stretch_both",
                                        scroll=True)),
        sizing_mode="stretch_both",
    )


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_subtitle = "Phase 1 - Screening"
template = pn.template.FastListTemplate(
    title=(
        f"{CFG['dashboard']['title']}"
        f"&ensp;<span style='font-size:0.60em;font-weight:300;opacity:0.70'>{_subtitle}</span>"
    ),
    sidebar=[sidebar],
    main=[main_content],
    accent_base_color="#1a6faf",
    header_background="#1a3a5c",
)

template.servable()
