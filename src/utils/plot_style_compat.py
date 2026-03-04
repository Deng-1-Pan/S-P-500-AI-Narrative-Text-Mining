"""Style loader with resilient fallbacks and profile-specific defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotStyleBundle:
    colors: Dict[str, str]
    apply_theme: Callable[[], None]
    style_axes: Callable
    save_figure: Callable
    style_legend: Callable


_FALLBACK_PROFILES: Dict[str, Dict[str, str]] = {
    "analysis-dark": {
        "background": "#121212",
        "fg": "#F5F5F5",
        "muted": "#B3B3B3",
        "accent": "#1DB954",
        "negative": "#FF5A5F",
        "grid": "#2A2A2A",
        "blue": "#4EA1FF",
        "warning": "#F4C542",
        "panel": "#121212",
    },
    "analysis-dark-minimal": {
        "background": "#121212",
        "accent": "#1DB954",
        "grid": "#2A2A2A",
        "muted": "#B3B3B3",
        "blue": "#4EA1FF",
        "negative": "#FF5A5F",
    },
    "analysis-dark-eda-legacy": {
        "background": "#121212",
        "panel": "#181818",
        "fg": "#F5F5F5",
        "muted": "#B3B3B3",
        "accent": "#1DB954",
        "negative": "#FF5A5F",
        "warning": "#F4C542",
        "grid": "#2A2A2A",
        "blue": "#4EA1FF",
    },
    "stage16-light": {
        "background": "#FFFFFF",
        "fg": "#222222",
        "muted": "#555555",
        "accent": "#0B8E8A",
        "negative": "#C62828",
        "grid": "#E0E0E0",
        "blue": "#1F449C",
        "panel": "#FAFAFA",
    },
}


def load_plot_style(profile: str = "analysis-dark") -> PlotStyleBundle:
    try:
        from src.utils.visual_style import (
            SPOTIFY_COLORS,
            apply_spotify_theme,
            save_figure,
            style_axes,
            style_legend,
        )

        return PlotStyleBundle(
            colors=SPOTIFY_COLORS,
            apply_theme=apply_spotify_theme,
            style_axes=style_axes,
            save_figure=save_figure,
            style_legend=style_legend,
        )
    except Exception:  # pragma: no cover
        colors = dict(_FALLBACK_PROFILES.get(profile, _FALLBACK_PROFILES["analysis-dark"]))

        def apply_theme():
            return None

        def fallback_style_axes(ax, **_kwargs):
            return ax

        def fallback_style_legend(ax):
            return ax.get_legend()

        def fallback_save_figure(fig, output_path: str, dpi: int = 180):
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        return PlotStyleBundle(
            colors=colors,
            apply_theme=apply_theme,
            style_axes=fallback_style_axes,
            save_figure=fallback_save_figure,
            style_legend=fallback_style_legend,
        )
