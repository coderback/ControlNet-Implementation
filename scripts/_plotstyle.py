"""Shared, polished matplotlib styling for the README figures — no extra dependencies.

`apply_style()` sets clean rcParams (left-aligned bold titles, muted labels, soft grid,
no top/right spines); the palette constants keep ON/OFF colours consistent across plots.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#22303c"        # primary text
MUTED = "#6b7a89"      # axis labels / secondary text
OFF_C = "#c4cdd6"      # control OFF (neutral grey-blue)
ON_C = "#2f8f6b"       # control ON (green)
LINE_COLORS = ["#3b6ea5", "#e0863c", "#2f8f6b"]  # down / mid / cond-embed
GRID = "#eef1f4"
CONNECT = "#d7dde3"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d7de",
        "axes.linewidth": 1.0,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })


def clean_axes(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
