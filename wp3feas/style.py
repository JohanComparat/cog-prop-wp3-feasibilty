"""Figure style of the proposal (erc-cog-prop/scripts/make_proposal_figures.py): the dataviz reference palette."""

import matplotlib.pyplot as plt

INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
# categorical slots 1-4 of the reference palette, in their fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BAND = "#f1f0ec"  # neutral tint for tolerance bands

WIDTH_FULL = 6.9   # inches: full text width of Part B2
WIDTH_HALF = 3.35


def set_style():
    plt.rcParams.update({
        "font.size": 8, "font.family": "sans-serif",
        "text.color": INK, "axes.labelcolor": SEC, "axes.titlecolor": INK,
        "axes.edgecolor": BASE, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.color": MUT, "ytick.color": MUT,
        "xtick.labelcolor": SEC, "ytick.labelcolor": SEC,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 6.5, "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "lines.linewidth": 1.4,
    })


def title(ax, text):
    ax.set_title(text, fontsize=8, color=SEC, loc="left")
