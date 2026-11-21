"""One chart style for every figure in the report.

Colours are fixed roles, not per-figure choices: a categorical order that stays
distinguishable under colour-vision deficiency, one grey for context series, and
recessive axes. Text is always drawn in ink colours, never in a series colour.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#b9b8b1"  # de-emphasised series
# Fixed categorical order; assign in sequence, never cycle.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "text.color": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 11,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.labelcolor": INK_SECONDARY,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def subtitle(ax, text: str) -> None:
    """Secondary line under the title: what was measured, in plain words."""
    ax.text(
        0.0,
        1.015,
        text,
        transform=ax.transAxes,
        fontsize=10,
        color=INK_SECONDARY,
        va="bottom",
        ha="left",
    )
