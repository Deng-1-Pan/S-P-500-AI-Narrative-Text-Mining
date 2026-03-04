"""Shared plotting style helpers (professional light theme)."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

PROFESSIONAL_COLORS = {
    "background": "#FFFFFF",
    "panel": "#FAFAFA",
    "fg": "#222222",
    "muted": "#555555",
    "accent": "#0B8E8A",
    "negative": "#C62828",
    "warning": "#B8860B",
    "grid": "#E0E0E0",
    "blue": "#1F449C",
}

# Backward-compatible alias used across analysis modules.
SPOTIFY_COLORS = PROFESSIONAL_COLORS


def apply_spotify_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="white",
        palette=[
            SPOTIFY_COLORS["blue"],
            SPOTIFY_COLORS["accent"],
            SPOTIFY_COLORS["negative"],
            SPOTIFY_COLORS["warning"],
        ],
    )
    mpl.rcParams.update(
        {
            "figure.facecolor": SPOTIFY_COLORS["background"],
            "axes.facecolor": SPOTIFY_COLORS["panel"],
            "savefig.facecolor": SPOTIFY_COLORS["background"],
            "savefig.edgecolor": SPOTIFY_COLORS["background"],
            "text.color": SPOTIFY_COLORS["fg"],
            "axes.labelcolor": SPOTIFY_COLORS["fg"],
            "axes.edgecolor": SPOTIFY_COLORS["grid"],
            "axes.titlecolor": SPOTIFY_COLORS["fg"],
            "xtick.color": SPOTIFY_COLORS["muted"],
            "ytick.color": SPOTIFY_COLORS["muted"],
            "grid.color": SPOTIFY_COLORS["grid"],
            "grid.linewidth": 0.9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.grid": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.facecolor": "none",
            "legend.edgecolor": "none",
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def style_axes(ax, grid_axis: Optional[str] = None, grid_alpha: float = 0.12):
    ax.set_facecolor(SPOTIFY_COLORS["panel"])
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in {"top", "right"}:
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color("#A6A6A6")
            spine.set_linewidth(0.9)
    ax.tick_params(axis="both", colors=SPOTIFY_COLORS["muted"], labelcolor=SPOTIFY_COLORS["muted"])

    effective_grid_axis = "y" if grid_axis is None else grid_axis
    if effective_grid_axis:
        ax.grid(
            True,
            axis=effective_grid_axis,
            alpha=grid_alpha,
            color=SPOTIFY_COLORS["grid"],
            linewidth=0.9,
        )
    else:
        ax.grid(False)
    return ax


def style_legend(ax):
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("none")
        frame.set_edgecolor("none")
        for text in leg.get_texts():
            text.set_color(SPOTIFY_COLORS["fg"])
        title = leg.get_title()
        if title is not None:
            title.set_color(SPOTIFY_COLORS["fg"])
    return leg


def save_figure(fig, output_path: str, dpi: int = 180) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=SPOTIFY_COLORS["background"],
        edgecolor=SPOTIFY_COLORS["background"],
    )
    plt.close(fig)


apply_professional_theme = apply_spotify_theme
