"""
Prototype Sankey: MAR Objective -> Water Source -> Method.
Output: data/img/sankey.html  (open in any browser)

Run:
    conda run -n mar-st python scripts/sankey_prototype.py
"""

import plotly.graph_objects as go
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "img" / "sankey.html"

# ── Nodes ─────────────────────────────────────────────────────────────────────
#  0-3  MAR Objectives
#  4-7  Water Types
#  8-10 MAR Methods

labels = [
    # Objectives
    "Environmental Destination (ED)",
    "Aquifer Water Balance",
    "Storm Water Management (DWMP)",
    "Water Supply / Peak Demand (WRMP)",
    # Water Types
    "River Water",
    "Drinking Water (treated / partial)",
    "Storm Water",
    "Treated Waste Water",
    # Methods
    "Surface Infiltration (Unsaturated Zone)",
    "Borehole Recharge",
    "Aquifer Storage & Recovery (ASR)",
]

node_colors = [
    # Objectives — greens matching original chart
    "#27AE60",  # ED
    "#1A5C1A",  # Aquifer Water Balance
    "#82E0AA",  # DWMP
    "#D5F5E3",  # WRMP
    # Water Types — blues/greys matching original chart
    "#5DADE2",  # River Water
    "#1A5276",  # Drinking Water
    "#BDC3C7",  # Storm Water
    "#717D7E",  # Treated Waste Water
    # Methods — yellows/oranges/browns matching original chart
    "#F5A623",  # Surface Infiltration
    "#E87722",  # Borehole Recharge
    "#8B2500",  # ASR
]

# Fixed column positions: objectives=left, water types=centre, methods=right
x = [0.02] * 4 + [0.45] * 4 + [0.92] * 3
# y positions sized to reflect node flow totals (22.5° = 2 units; total = 32)
y = [
    0.06, 0.44, 0.67, 0.88,   # objectives  (ED=12, AWB=8, DWMP=8, WRMP=4)
    0.05, 0.25, 0.52, 0.80,   # water types (River=6, Drink=8, Storm=10, TrWW=8)
    0.38, 0.88, 0.96,         # methods     (Surf=27, BH=4, ASR=1)
]

# ── Links ─────────────────────────────────────────────────────────────────────
# Source of truth: data/img/chart.csv  (22.5° = 2 units; equal fractions per water type)
#
# Objective → Water Type
#   ED   (12) → Storm Water (4), Drinking Water (4), River Water (4)
#   AWB   (8) → Drinking Water (2), River Water (2), Storm Water (2), Treated WW (2)
#   DWMP  (8) → Storm Water (4), Treated WW (4)
#   WRMP  (4) → Treated WW (2), Drinking Water (2)
#
# Water Type → Method
#   River Water  (6)  → Surface Infiltration (6)
#   Drinking Water(8) → Surface Infiltration (3), Borehole (4), ASR (1)
#   Storm Water  (10) → Surface Infiltration (10)
#   Treated WW   (8)  → Surface Infiltration (8)
#
# DW→Surf = ED(2)+AWB(1)=3 ; DW→BH = ED(2)+AWB(1)+WRMP(1)=4 ; DW→ASR = WRMP(1)=1

src = [0, 0, 0,   1, 1, 1, 1,   2, 2,   3, 3,    4,  5, 5,  5,   6,  7]
tgt = [6, 5, 4,   5, 4, 6, 7,   6, 7,   7, 5,    8,  8, 9, 10,   8,  8]
val = [4, 4, 4,   2, 2, 2, 2,   4, 4,   2, 2,    6,  3, 4,  1,  10,  8]

link_rgba = {
    0: "rgba(39,174,96,0.35)",    # ED
    1: "rgba(26,92,26,0.35)",     # AWB
    2: "rgba(130,224,170,0.40)",  # DWMP
    3: "rgba(213,245,227,0.55)",  # WRMP
    4: "rgba(93,173,226,0.35)",   # River Water
    5: "rgba(26,82,118,0.35)",    # Drinking Water
    6: "rgba(189,195,199,0.45)",  # Storm Water
    7: "rgba(113,125,126,0.45)",  # Treated Waste Water
}
link_colors = [link_rgba[s] for s in src]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure(go.Sankey(
    arrangement="fixed",
    node=dict(
        pad=20,
        thickness=24,
        line=dict(color="#ffffff", width=0.5),
        label=labels,
        color=node_colors,
        x=x,
        y=y,
    ),
    link=dict(
        source=src,
        target=tgt,
        value=val,
        color=link_colors,
    ),
))

# Column header annotations
for label, xpos in [
    ("MAR Objective", 0.02),
    ("Water Source", 0.45),
    ("MAR Method", 0.92),
]:
    fig.add_annotation(
        x=xpos, y=1.06, xref="paper", yref="paper",
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=13, color="#1C2833", family="Arial"),
        align="center",
    )

fig.update_layout(
    title=dict(
        text="MAR Objective, Water Source and Method Combinations",
        font=dict(size=16, family="Arial Black", color="#1C2833"),
        x=0.5,
        xanchor="center",
        y=0.97,
    ),
    font=dict(size=12, family="Arial", color="#2C3E50"),
    paper_bgcolor="#F8F9FA",
    margin=dict(l=30, r=30, t=80, b=30),
    width=1150,
    height=620,
)

fig.write_html(str(OUT), include_plotlyjs="cdn")
print(f"Saved → {OUT}")
