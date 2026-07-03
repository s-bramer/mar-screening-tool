"""
chart_v2.py — Enhanced concentric ring chart
Output: data/img/chart_v2.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "img" / "chart_v2.png"

# ── Ring geometry ─────────────────────────────────────────────────────────────
R0I, R0O = 0.18, 0.40   # Objective  (inner)
R1I, R1O = 0.44, 0.66   # Water Type (middle)
R2I, R2O = 0.70, 0.97   # Method     (outer)
GAP = 0.7               # white gap in degrees at each sector edge

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "ED":    "#27AE60",   "AWB":   "#1A5C1A",
    "DWMP":  "#82E0AA",   "WRMP":  "#C8F0C8",
    "River": "#5DADE2",   "Drink": "#1A5276",
    "Storm": "#BDC3C7",   "TrWW":  "#717D7E",
    "Surf":  "#F5A623",   "BH":    "#D35400",   "ASR":   "#7B241C",
}

WATER_TXT  = {"River": "#0A2E5E", "Drink": "white", "Storm": "#333", "TrWW": "white"}
METHOD_TXT = {"Surf":  "#3B270A", "BH":    "white", "ASR":   "white"}

METHOD_LBL = {
    "Surf": "Surface\nInfiltration",
    "BH":   "Borehole\nRecharge",
    "ASR":  "Aquifer Storage\n& Recovery (ASR)",
}

# ── Chart structure ───────────────────────────────────────────────────────────
# Objectives: four equal 90° quadrants (CCW from east / 3 o'clock).
# water_types: (fraction_of_obj_arc, water_key, display_label, methods)
# methods    : (method_key, fraction_of_water_arc)

STRUCTURE = [
    # Source of truth: data/img/chart.csv
    # Arc positions (CCW math degrees):
    #   DWMP  135–225° (left,       centre 180° / 9 o'clock)
    #   AWB   225–315° (bottom,     centre 270° / 6 o'clock)
    #   WRMP   90–135° (upper-left, centre 112° / 10:30 o'clock)
    #   ED    315–450° (right,      centre   0° / 3 o'clock, wraps through 360°)
    #
    # Within each sector water types are listed CCW from t1 → t2 in equal fractions.
    # In clock terms that reads in reverse (t2 end is the "top" of each sector).
    #
    # DWMP: Storm Water, Treated Waste Water (both → Surface Infiltration)
    ("DWMP", 135, 225, "#1A3A1A", "DWMP", [
        (0.5, "Storm", "Storm\nWater",         [("Surf", 1.0)]),
        (0.5, "TrWW",  "Treated\nWaste Water", [("Surf", 1.0)]),
    ]),
    # AWB: River Water, Drinking Water (→ Surf + BH), Storm Water, Treated Waste Water
    ("AWB",  225, 315, "white",   "Aquifer\nWater\nBalance", [
        (0.25, "Drink", "Drinking\nWater",      [("Surf", 0.5), ("BH", 0.5)]),
        (0.25, "River", "River\nWater",         [("Surf", 1.0)]),
        (0.25, "Storm", "Storm\nWater",         [("Surf", 1.0)]),
        (0.25, "TrWW",  "Treated\nWaste Water", [("Surf", 1.0)]),
    ]),
    # WRMP: Treated Waste Water → Surf; Drinking Water → BH + ASR
    ("WRMP",  90, 135, "#1A3A1A", "WRMP", [
        (0.5, "TrWW",  "Treated\nWaste Water", [("Surf", 1.0)]),
        (0.5, "Drink", "Drinking\nWater",      [("BH", 0.5), ("ASR", 0.5)]),
    ]),
    # ED: Storm Water, Drinking Water (→ BH + Surf), River Water  (equal thirds)
    ("ED",   315, 450, "black",   "ED", [
        (1/3, "Storm", "Storm\nWater",    [("Surf", 1.0)]),
        (1/3, "Drink", "Drinking\nWater", [("BH", 0.5), ("Surf", 0.5)]),
        (1/3, "River", "River\nWater",    [("Surf", 1.0)]),
    ]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_wedge(ax, r_in, r_out, t1, t2, color):
    g = GAP / 2
    ax.add_patch(Wedge((0, 0), r_out, t1 + g, t2 - g,
                       width=r_out - r_in, facecolor=color,
                       edgecolor="white", linewidth=1.5, zorder=2))


def polar_xy(r, deg):
    a = np.radians(deg)
    return r * np.cos(a), r * np.sin(a)


def draw_label(ax, r, t1, t2, text, fontsize, color, bold=False):
    arc = abs(t2 - t1)
    if arc < 10:
        return
    fs = max(5.5, fontsize * min(1.0, arc / 18))
    x, y = polar_xy(r, (t1 + t2) / 2)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color,
            fontweight="bold" if bold else "normal",
            multialignment="center", zorder=4, rotation=0)


def draw_boundary(ax, angle_deg):
    """Bold white radial line separating objectives through all three rings."""
    a = np.radians(angle_deg)
    ax.plot([R0I * np.cos(a), R2O * np.cos(a)],
            [R0I * np.sin(a), R2O * np.sin(a)],
            color="white", linewidth=3.5, zorder=5)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set(xlim=(-1.15, 1.15), ylim=(-0.99, 1.05), aspect="equal")
ax.axis("off")

ax.add_patch(plt.Circle((0, 0), R0I - 0.01, color="white", zorder=5))

# ── Draw rings ────────────────────────────────────────────────────────────────
for obj_key, obj_t1, obj_t2, obj_txt_col, obj_lbl, waters in STRUCTURE:
    span = obj_t2 - obj_t1

    draw_wedge(ax, R0I, R0O, obj_t1, obj_t2, C[obj_key])
    draw_label(ax, (R0I + R0O) / 2, obj_t1, obj_t2, obj_lbl,
               fontsize=9, color=obj_txt_col, bold=True)

    cursor = obj_t1
    for frac, wat_key, wat_lbl, methods in waters:
        w_t1, w_t2 = cursor, cursor + frac * span

        draw_wedge(ax, R1I, R1O, w_t1, w_t2, C[wat_key])
        draw_label(ax, (R1I + R1O) / 2, w_t1, w_t2, wat_lbl,
                   fontsize=7.5, color=WATER_TXT[wat_key])

        m_cursor = w_t1
        for met_key, m_frac in methods:
            m_t1 = m_cursor
            m_t2 = m_cursor + m_frac * (w_t2 - w_t1)
            draw_wedge(ax, R2I, R2O, m_t1, m_t2, C[met_key])
            draw_label(ax, (R2I + R2O) / 2, m_t1, m_t2,
                       METHOD_LBL[met_key], fontsize=7, color=METHOD_TXT[met_key])
            m_cursor = m_t2
        cursor = w_t2

    draw_boundary(ax, obj_t1)
# All four boundary lines are already drawn by the loop above (at 90, 135, 225, 315).

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title("MAR Objective, Water Source and Method Combinations",
             fontsize=13, fontweight="bold", color="#1C2833", pad=14)

# ── Legends ───────────────────────────────────────────────────────────────────
def make_legend(ax, handles, title, loc, anchor):
    return ax.legend(handles=handles, title=title, title_fontsize=8.5,
                     loc=loc, bbox_to_anchor=anchor,
                     fontsize=8, frameon=True, framealpha=0.95,
                     edgecolor="#CCCCCC", borderpad=0.8)

leg_obj = [
    mpatches.Patch(color=C["ED"],   label="ED — Environmental Destination"),
    mpatches.Patch(color=C["AWB"],  label="Aquifer Water Balance"),
    mpatches.Patch(color=C["DWMP"], label="DWMP — Storm Water Management"),
    mpatches.Patch(color=C["WRMP"], label="WRMP — Water Supply / Peak Demand"),
]
leg_water = [
    mpatches.Patch(color=C["River"], label="River Water"),
    mpatches.Patch(color=C["Drink"], label="Drinking Water (treated / partial)"),
    mpatches.Patch(color=C["Storm"], label="Storm Water"),
    mpatches.Patch(color=C["TrWW"],  label="Treated Waste Water"),
]
leg_meth = [
    mpatches.Patch(color=C["Surf"], label="Surface Infiltration (Unsaturated Zone)"),
    mpatches.Patch(color=C["BH"],   label="Borehole Recharge"),
    mpatches.Patch(color=C["ASR"],  label="Aquifer Storage & Recovery (ASR)"),
]

l1 = make_legend(ax, leg_obj,   "MAR Objective",  "upper left",   (-0.04, -0.02))
l2 = make_legend(ax, leg_water, "Water Source",   "upper center", (0.50,  -0.02))
l3 = make_legend(ax, leg_meth,  "MAR Method",     "upper right",  (1.04,  -0.02))
ax.add_artist(l1)
ax.add_artist(l2)

plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
