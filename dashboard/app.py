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
_E_BNG = [300_000, 340_000, 380_000, 420_000, 460_000, 500_000]
_N_BNG = [200_000, 240_000, 280_000, 320_000, 360_000, 400_000]
_E_WM  = list(_T.transform(_E_BNG, [295_000] * len(_E_BNG))[0])
_N_WM  = list(_T.transform([410_000] * len(_N_BNG), _N_BNG)[1])


def _bng_axes_hook(plot, element):
    """Replace Web Mercator axis ticks/labels with BNG Easting/Northing (km)."""
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

    cells = pd.read_csv(PROCESSED / "hover_base.csv")

    # -----------------------------------------------------------------------
    # Dummy placeholder scores — spatially coherent, reproducible seed.
    # All three are flagged as placeholders in the UI; real data replaces them
    # in later phases.
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n   = len(cells)
    x   = cells["x0"].values
    y   = cells["y0"].values
    x_n = (x - x.min()) / max(x.max() - x.min(), 1)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1)

    # Slope / topography: eastern Midlands flatter → higher MAR suitability
    slope_raw = 0.55 * x_n - 0.25 * np.abs(y_n - 0.45) + 0.20 * rng.standard_normal(n)
    cells["slope_score"] = np.clip(5.5 + 3.8 * slope_raw, 1.0, 10.0).round(1)

    # Depth to water: northward proxy for elevation; piecewise score peaks at 8–15 m
    depth_proxy       = 0.55 * y_n + 0.45 * rng.uniform(0, 1, n)
    depth_m           = 2 + 26 * depth_proxy
    cells["depth_water_score"] = np.interp(
        depth_m, [0, 3, 8, 15, 25, 40], [2, 7, 10, 9, 5, 2]
    ).round(1)
    cells["depth_water_m"] = depth_m.round(1)

    # Surface geology suitability: moderate superficial deposits (2–6 m) = best
    # infiltration path; scores derived from SDTM proxy + noise.
    sg_base = np.interp(
        cells["sdtm_mean_m"].fillna(0).values,
        [0, 0.5, 2, 6, 15, 30], [4, 6, 10, 9, 6, 3],
    )
    cells["surf_geo_score"] = np.clip(
        sg_base + 1.2 * rng.standard_normal(n), 1.0, 10.0
    ).round(1)

    # Pre-filter feasible cells once — hot path only touches this slice
    feasible_base = cells[cells["constraint"] == 1][[
        "x0", "y0", "x1", "y1",
        "geo_score", "need_score", "water_score",
        "sdtm_score", "sdtm_mean_m",
        "slope_score", "depth_water_score", "depth_water_m", "surf_geo_score",
        "aquifer", "gwmu", "gwm_name",
    ]].copy().reset_index(drop=True)

    # Excluded cells — built once, never changes with weight sliders
    excl = cells[cells["constraint"] == 0][
        ["x0", "y0", "x1", "y1", "geo_score", "aquifer", "gwmu"]
    ].copy().reset_index(drop=True)
    excl["reason"] = np.where(
        excl["geo_score"] == 0, "Non-productive aquifer", "GWDTE",
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
    ).opts(opts.Rectangles(
        color="#de1212", fill_alpha=0.40, line_alpha=0, tools=[HOVER_EXCL],
    ))

    # Static overlay layers (EPSG:3857)
    # NOTE: gwdte_3857.gpkg intentionally NOT loaded — complex polygon geometries
    # produce degenerate WebGL vertices (fan/spike artefacts). GWDTE exclusion is
    # visible as uncoloured gaps in the suitability grid.
    bnd    = _gpkg("boundary_3857")
    hydro  = _gpkg("hydrogeology_3857")
    gwmu   = _gpkg("gwmu_3857")
    gwm    = _gpkg("gwm_3857")
    sw     = _gpkg("sw_catchments_3857")
    rivers = _gpkg("rivers_3857")

    rivers = rivers.copy()
    rivers.geometry = rivers.geometry.simplify(2000, preserve_topology=False)
    rivers = rivers[~rivers.geometry.is_empty].reset_index(drop=True)

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
    L_GWM = gwm_rings.hvplot(
        line_color="#9b59b6", line_width=2.0,
        tools=[HoverTool(tooltips=[("GW Model", "@Name")])],
        geo=False, legend=False, hover=False,
    )
    L_HYDRO = hydro.hvplot(
        c="CHARACTER",
        cmap=["#e0e0e0", "#c6dff0", "#57a0ce", "#1a6faf", "#c6dff0"],
        alpha=0.45, line_width=0, **_s,
    )

    TILE_BASE = gts.CartoDark.opts(opts.WMTS(
        width=900, height=MAP_H,
        active_tools=["wheel_zoom", "pan"],
        toolbar="above",
    ))

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

mar_method = pn.widgets.Select(
    name="MAR Method",
    options=["All methods", "Surface Infiltration", "Deep Borehole Recharge"],
    value="All methods",
)

# MCE theme weights
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

# Geological sub-criteria weights
w_aquifer     = pn.widgets.FloatSlider(
    name="Aquifer Classification",          value=0.50, start=0.0, end=1.0, step=0.01,
)
w_sdtm        = pn.widgets.FloatSlider(
    name="SDTM Thickness  [infiltration]",  value=0.20, start=0.0, end=1.0, step=0.01,
)
w_slope       = pn.widgets.FloatSlider(
    name="Topography / Slope  [infiltration]", value=0.10, start=0.0, end=1.0, step=0.01,
)
w_depth_water = pn.widgets.FloatSlider(
    name="Depth to Water",                  value=0.10, start=0.0, end=1.0, step=0.01,
)
w_surf_geo    = pn.widgets.FloatSlider(
    name="Surface Geology  [infiltration]", value=0.10, start=0.0, end=1.0, step=0.01,
)

_INFIL_ONLY_WIDGETS = [w_sdtm, w_slope, w_surf_geo]


def _on_method_change(event):
    """Disable infiltration-only sub-criteria when deep borehole is selected."""
    is_borehole = event.new == "Deep Borehole Recharge"
    for w in _INFIL_ONLY_WIDGETS:
        w.disabled = is_borehole


mar_method.param.watch(_on_method_change, "value")

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
# Shared computation helpers
# ---------------------------------------------------------------------------

_INFIL_COLS = {"sdtm_score", "slope_score", "surf_geo_score"}

_GEO_SUB_COLS = {
    "aquifer":   "geo_score",
    "sdtm":      "sdtm_score",
    "slope":     "slope_score",
    "depth":     "depth_water_score",
    "surf_geo":  "surf_geo_score",
}


def _compute_geo_composite(df, method, w_aq, w_sd, w_sl, w_dw, w_sg):
    """
    Weighted sub-composite for Geological Suitability.
    Infiltration-only criteria (SDTM, slope, surface geology) are zeroed when
    the selected method is Deep Borehole Recharge.
    """
    is_infil = method != "Deep Borehole Recharge"
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
    # Column map: colour_by selector value → column name in FEASIBLE_BASE
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

    # Objective descriptions shown below the dropdown
    _OBJ_DESC = {
        "All objectives": (
            "Screening all four MAR objectives simultaneously. "
            "Use the objective selector to focus on a specific use case."
        ),
        "Environmental Destination": (
            "Support groundwater levels beneath wetlands and surface water flows. "
            "Key criteria: proximity to GWDTEs and surface water prioritisation layer."
        ),
        "Aquifer Water Balance": (
            "Support aquifer WFD balance and protect existing abstractions. "
            "Key criteria: GWMU deficit status (CAMS ledger — data pending)."
        ),
        "Storm Water (DWMP)": (
            "Recharge storm water close to source; reduce CSO spill frequency. "
            "Key criteria: CSO locations and WwTW catchment proximity."
        ),
        "Water Supply / ASR": (
            "Meet summer peak demand via aquifer storage and recovery. "
            "Key criteria: deep confined aquifer suitability; "
            "select 'Deep Borehole Recharge' method."
        ),
    }

    # -----------------------------------------------------------------------
    # Reactive map
    # -----------------------------------------------------------------------

    def _build_map(w_n, w_g, w_w, w_aq, w_sd, w_sl, w_dw, w_sg,
                   layers, col_by, method, objective):
        total = (w_n + w_g + w_w) or 1.0

        feasible = FEASIBLE_BASE.copy()

        # Geological sub-composite
        feasible["geo_composite"] = _compute_geo_composite(
            feasible, method, w_aq, w_sd, w_sl, w_dw, w_sg,
        ).round(1)

        # Top-level composite
        feasible["composite"] = np.clip(
            (w_n / total * feasible["need_score"]
             + w_g / total * feasible["geo_composite"]
             + w_w / total * feasible["water_score"]),
            0, 10,
        ).round(1)

        score_col = _COL_MAP[col_by]

        _tt = []
        if "GW Models"     in layers: _tt += [("GW Model",  "@gwm_name")]
        if "GW Mgmt Units" in layers: _tt += [("GWMU",      "@gwmu")]
        _tt += [
            ("Aquifer",               "@aquifer"),
            ("──────────────",        ""),
            ("Composite",             "@composite{0.0}"),
            ("  Geo sub-composite",   "@geo_composite{0.0}"),
            ("  ├ Aquifer class.",     "@geo_score{0.0}"),
            ("  ├ SDTM thickness",     "@sdtm_score{0.0}  (@sdtm_mean_m{0.0} m)"),
            ("  ├ Topography/slope",   "@slope_score{0.0}"),
            ("  ├ Depth to water",     "@depth_water_score{0.0}  (@depth_water_m{0.0} m)"),
            ("  └ Surface geology",    "@surf_geo_score{0.0}"),
            ("──────────────",        ""),
            ("  Need (dummy)",         "@need_score{0.0}"),
            ("  Water (dummy)",        "@water_score{0.0}"),
        ]
        hover_tool = HoverTool(tooltips=_tt)

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
            tools=[hover_tool],
        ))

        plot = TILE_BASE
        if "Excluded Cells"     in layers: plot = plot * EXCL_RECTS
        if "Suitability Grid"   in layers: plot = plot * rects
        if "Hydrogeology (BGS)" in layers: plot = plot * L_HYDRO
        if "GW Mgmt Units"      in layers: plot = plot * L_GWMU
        if "GW Models"          in layers: plot = plot * L_GWM
        if "SW Catchments"      in layers: plot = plot * L_SW
        if "Rivers"             in layers: plot = plot * L_RIVERS
        if "Study Boundary"     in layers: plot = plot * L_BOUNDARY

        return plot.opts(opts.Overlay(
            active_tools=["wheel_zoom", "pan"],
            toolbar="above",
            hooks=[_bng_axes_hook],
        ))

    map_pane = pn.panel(
        pn.bind(
            _build_map,
            w_need, w_geo, w_water,
            w_aquifer, w_sdtm, w_slope, w_depth_water, w_surf_geo,
            layer_checks, colour_by,
            mar_method, mar_objective,
        ),
        sizing_mode="stretch_both",
    )

    # -----------------------------------------------------------------------
    # Stats panel
    # -----------------------------------------------------------------------

    def _stats(w_n, w_g, w_w, w_aq, w_sd, w_sl, w_dw, w_sg, method):
        total = (w_n + w_g + w_w) or 1.0

        geo_comp = _compute_geo_composite(CELLS, method, w_aq, w_sd, w_sl, w_dw, w_sg)

        composite = np.clip(
            (w_n / total * CELLS["need_score"].values
             + w_g / total * geo_comp
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

    stats_pane = pn.panel(
        pn.bind(
            _stats,
            w_need, w_geo, w_water,
            w_aquifer, w_sdtm, w_slope, w_depth_water, w_surf_geo,
            mar_method,
        )
    )

    # -----------------------------------------------------------------------
    # Objective description (reactive)
    # -----------------------------------------------------------------------

    def _obj_desc(objective):
        return pn.pane.Markdown(
            f"_{_OBJ_DESC[objective]}_",
            styles={"color": "#aaa", "font-size": "0.80em"},
            width=SIDEBAR_W - 20,
        )

    obj_desc_pane = pn.panel(pn.bind(_obj_desc, mar_objective))

    # -----------------------------------------------------------------------
    # Sidebar assembly
    # -----------------------------------------------------------------------

    sidebar = pn.Column(
        pn.pane.Markdown("## MAR-ST Tool\n*Phase 1 — Screening*"),
        pn.layout.Divider(),

        pn.pane.Markdown("### MAR Objective"),
        mar_objective,
        # obj_desc_pane,
        pn.layout.Divider(),

        pn.pane.Markdown("### MAR Method"),
        mar_method,
        pn.pane.Markdown(
            "Selects which geological sub-criteria are active. "
            "Infiltration-only criteria are greyed out for deep borehole.",
            styles={"color": "#888", "font-size": "0.80em"},
        ),
        pn.layout.Divider(),

        pn.pane.Markdown("### MCE Theme Weights"),
        pn.pane.Markdown(
            "_Normalised automatically. Updates on slider release._",
            styles={"color": "#888", "font-size": "0.82em"},
        ),
        w_need, w_geo, w_water,
        pn.layout.Divider(),

        pn.pane.Markdown("### Geological Sub-criteria"),
        pn.pane.Markdown(
            "_Weights within the Geological Suitability theme. "
            "\\[infiltration\\] criteria disabled for Deep Borehole._",
            styles={"color": "#888", "font-size": "0.80em"},
        ),
        w_aquifer, w_sdtm, w_slope, w_depth_water, w_surf_geo,
        pn.layout.Divider(),

        pn.pane.Markdown("### Display"),
        colour_by,
        layer_checks,
        pn.layout.Divider(),

        stats_pane,
        pn.layout.Divider(),

        pn.pane.Markdown(
            "⚠ **Placeholder data:** Need for MAR, Water Availability, "
            "Topography/Slope, Depth to Water, and Surface Geology scores "
            "are spatially coherent dummies. "
            "Aquifer Classification and SDTM Thickness use real BGS data.",
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
