from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.constants import GICS_SECTOR_MAP
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="stage16-light")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


GSECTOR_NAME_MAP = GICS_SECTOR_MAP


def _data_csv_path(output_path: str) -> str:
    return output_path.replace(".png", "_data.csv")


def _sector_label_series(df: pd.DataFrame) -> pd.Series:
    label = pd.Series(index=df.index, dtype="object")
    if "sector" in df.columns:
        s = df["sector"].astype(str).str.strip()
        valid = s.notna() & s.ne("") & s.ne("nan")
        label.loc[valid] = s.loc[valid]
    if "gsector" in df.columns:
        gsec_num = pd.to_numeric(df["gsector"], errors="coerce")
        mapped = gsec_num.map(GSECTOR_NAME_MAP)
        label = label.where(label.notna(), mapped)
        fallback = "GICS " + df["gsector"].astype(str)
        label = label.where(label.notna(), fallback)
    return label.fillna("Unknown Sector")


def plot_wrds_distribution(df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    dist_cols = [c for c in ["log_mkcap", "rd_intensity_mkcap", "eps", "ret_q"] if c in df.columns]
    df[dist_cols].to_csv(_data_csv_path(output_path), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    pairs = [
        ("log_mkcap", "log_mkcap"),
        ("rd_intensity_mkcap", "rd_intensity_mkcap"),
        ("eps", "eps"),
        ("ret_q", "ret_q"),
    ]
    for ax, (col, title) in zip(axes.flatten(), pairs):
        if col in df.columns:
            sns.histplot(df[col].dropna(), bins=40, kde=False, ax=ax, color=SPOTIFY_COLORS.get("blue", "#1F449C"))
        ax.set_title(title)
        style_axes(ax, grid_axis="y", grid_alpha=0.08)
    fig.suptitle("Stage16 WRDS Core Variable Distributions", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_ai_zero_by_sector(df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    work = df.copy()
    work["sector_label"] = _sector_label_series(work)
    agg = (
        work.groupby("sector_label", as_index=False)
        .agg(n=("doc_id", "size"), zero_share=("overall_ai_ratio", lambda s: float((s.fillna(0) == 0).mean())))
        .sort_values("zero_share", ascending=False)
    )
    agg.to_csv(_data_csv_path(output_path), index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    bars = ax.bar(agg["sector_label"], agg["zero_share"], color=SPOTIFY_COLORS.get("accent", "#0B8E8A"))
    ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in agg["zero_share"]], padding=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sector")
    ax.set_ylabel("Share of calls with overall_ai_ratio = 0")
    ax.set_title("AI Zero-Proportion by Sector")
    ax.tick_params(axis="x", rotation=30)
    style_axes(ax, grid_axis="y", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_ai_nonzero_by_sector(df: pd.DataFrame, output_path: str) -> None:
    """Opposite of plot_ai_zero_by_sector: share of calls with overall_ai_ratio != 0."""
    _ensure_parent(output_path)
    apply_spotify_theme()
    work = df.copy()
    work["sector_label"] = _sector_label_series(work)
    agg = (
        work.groupby("sector_label", as_index=False)
        .agg(n=("doc_id", "size"), nonzero_share=("overall_ai_ratio", lambda s: float((s.fillna(0) != 0).mean())))
        .sort_values("nonzero_share", ascending=False)
    )
    agg.to_csv(_data_csv_path(output_path), index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    bars = ax.bar(agg["sector_label"], agg["nonzero_share"], color=SPOTIFY_COLORS.get("blue", "#1F449C"))
    ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in agg["nonzero_share"]], padding=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Sector")
    ax.set_ylabel("Share of calls with overall_ai_ratio ≠ 0")
    ax.set_title("AI Non-Zero Proportion by Sector")
    ax.tick_params(axis="x", rotation=30)
    style_axes(ax, grid_axis="y", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_ai_trend_by_size(df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    work = df.dropna(subset=["yearq", "overall_ai_ratio", "log_mkcap"]).copy()
    if len(work) == 0:
        work = df.copy()
        work["size_bucket"] = "All"
    else:
        work["size_bucket"] = pd.qcut(work["log_mkcap"].rank(method="first"), q=3, labels=["Small", "Mid", "Large"])

    trend = (
        work.groupby(["yearq", "size_bucket"], as_index=False)["overall_ai_ratio"]
        .mean()
        .sort_values("yearq")
    )
    trend.to_csv(_data_csv_path(output_path), index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    for bucket, sub in trend.groupby("size_bucket"):
        ax.plot(sub["yearq"], sub["overall_ai_ratio"], marker="o", label=str(bucket))

    # --- Mark ChatGPT launch (2022-11-30 -> 2022Q4) ---
    all_quarters = sorted(trend["yearq"].unique())
    chatgpt_q = "2022Q4"
    if chatgpt_q in all_quarters:
        x_idx = all_quarters.index(chatgpt_q)
        ax.axvline(x=x_idx, color="#FF8C00", linewidth=1.8, linestyle="--", zorder=5)
        y_max = trend["overall_ai_ratio"].max()
        ax.annotate(
            "ChatGPT Launch\n(Nov 2022)",
            xy=(x_idx, y_max * 0.88),
            xytext=(x_idx + 0.7, y_max * 0.96),
            fontsize=8.5,
            color="#FF8C00",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#FF8C00", lw=1.4),
            ha="left",
            va="top",
        )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Mean overall_ai_ratio")
    ax.set_title("AI Intensity Trend by Size Quantile")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", ncol=3)
    style_axes(ax, grid_axis="y", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_assoc_bar(assoc_df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    assoc_df.to_csv(_data_csv_path(output_path), index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    plot_df = assoc_df.sort_values("spearman_corr").copy()
    colors = np.where(plot_df["spearman_corr"] >= 0, SPOTIFY_COLORS.get("accent", "#0B8E8A"), SPOTIFY_COLORS.get("negative", "#C62828"))
    ax.barh(plot_df["pair"], plot_df["spearman_corr"], color=colors)
    ax.axvline(0, color=SPOTIFY_COLORS.get("muted", "#555555"), linewidth=1)
    ax.set_xlabel("Spearman correlation")
    ax.set_title("Stage16 Association (Descriptive, Non-causal)")
    style_axes(ax, grid_axis="x", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_coefplot(regression_table: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    plot_df = regression_table.copy()
    core = plot_df[plot_df["x"].isin(["log_mkcap", "rd_intensity_mkcap", "eps", "capex_intensity_mkcap", "ret_q"])].copy()
    core.to_csv(_data_csv_path(output_path), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    for ax, (model, sub) in zip(axes, core.groupby("model", sort=False)):
        if len(sub) == 0:
            continue
        sub = sub.sort_values("coef")
        ax.errorbar(sub["coef"], sub["x"], xerr=1.96 * sub["se"], fmt="o", color=SPOTIFY_COLORS.get("blue", "#1F449C"))
        ax.axvline(0, color=SPOTIFY_COLORS.get("muted", "#555555"), linewidth=1)
        ax.set_title(model)
        style_axes(ax, grid_axis="x", grid_alpha=0.1)
    fig.suptitle("Stage16 Coefficient Plot (95% CI)")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_quadrant_finance(df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    order = ["Aligned", "Passive", "Self-Promoting", "Silent"]
    vars_ = ["log_mkcap", "rd_intensity_mkcap", "eps", "ret_q", "speech_ai_ratio", "qa_ai_ratio"]
    finance_long = (
        df[["quadrant"] + [c for c in vars_ if c in df.columns]]
        .melt(id_vars=["quadrant"], var_name="metric", value_name="value")
    )
    finance_long.to_csv(_data_csv_path(output_path), index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    vars_ = ["log_mkcap", "rd_intensity_mkcap", "eps", "ret_q"]
    for ax, col in zip(axes.flatten(), vars_):
        if {"quadrant", col}.issubset(df.columns):
            sns.boxplot(data=df, x="quadrant", y=col, order=order, ax=ax, color=SPOTIFY_COLORS.get("blue", "#1F449C"))
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
        style_axes(ax, grid_axis="y", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_quadrant_sector_heatmap(df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    order = ["Aligned", "Passive", "Self-Promoting", "Silent"]
    work = df.copy()
    work["sector_label"] = _sector_label_series(work)
    ctab = pd.crosstab(work["sector_label"], work["quadrant"], normalize="index").reindex(columns=order).fillna(0)
    ctab.to_csv(_data_csv_path(output_path), index=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    sns.heatmap(ctab, cmap="Blues", annot=True, fmt=".1%", ax=ax)
    ax.set_title("Sector × Quadrant Share")
    ax.set_xlabel("Quadrant")
    ax.set_ylabel("Sector")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_gap_by_quadrant(df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    order = ["Aligned", "Passive", "Self-Promoting", "Silent"]
    df[["quadrant", "narrative_invest_gap"]].to_csv(_data_csv_path(output_path), index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    sns.violinplot(data=df, x="quadrant", y="narrative_invest_gap", order=order, ax=ax, color=SPOTIFY_COLORS.get("accent", "#0B8E8A"))
    ax.set_title("Narrative-Investment Gap by Quadrant")
    ax.set_xlabel("Quadrant")
    ax.set_ylabel("z(overall_ai_ratio) - z(rd_intensity_mkcap)")
    ax.tick_params(axis="x", rotation=20)
    style_axes(ax, grid_axis="y", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def plot_model_compare(metrics_df: pd.DataFrame, output_path: str) -> None:
    _ensure_parent(output_path)
    apply_spotify_theme()
    metrics_df.to_csv(_data_csv_path(output_path), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    plot_df = metrics_df.sort_values("r2", ascending=False).copy()
    axes[0].barh(plot_df["model"], plot_df["r2"], color=SPOTIFY_COLORS.get("blue", "#1F449C"))
    axes[0].invert_yaxis()
    axes[0].set_title("R² (test)")
    style_axes(axes[0], grid_axis="x", grid_alpha=0.1)
    axes[1].barh(plot_df["model"], plot_df["mae"], color=SPOTIFY_COLORS.get("negative", "#C62828"))
    axes[1].invert_yaxis()
    axes[1].set_title("MAE (test)")
    style_axes(axes[1], grid_axis="x", grid_alpha=0.1)
    fig.suptitle("Stage15 Minimal Model Comparison")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def default_takeaways() -> Dict[str, Dict[str, str]]:
    return {
        "wrds_distribution.png": {
            "takeaway": "WRDS core variables are skewed and heavy-tailed, supporting robust methods.",
            "mechanism": "Firm size and investment intensity vary strongly across the panel.",
            "caution": "Tail behavior can drive linear sensitivity without winsorization/robust errors.",
        },
        "ai_zero_by_sector.png": {
            "takeaway": "Zero AI discussion share differs by sector.",
            "mechanism": "Industry exposure likely shapes AI salience in calls.",
            "caution": "Sector composition effects are descriptive, not causal.",
        },
        "ai_nonzero_by_sector.png": {
            "takeaway": "AI engagement (non-zero AI ratio) is highest in tech and communication sectors.",
            "mechanism": "Firms in innovation-driven sectors are more likely to discuss AI in earnings calls.",
            "caution": "Sector composition effects are descriptive, not causal.",
        },
        "ai_trend_by_size_or_sector.png": {
            "takeaway": "AI narrative intensity evolves differently by size groups over time.",
            "mechanism": "Large firms may adopt new disclosure language earlier.",
            "caution": "Trend shifts may include vocabulary drift rather than pure behavior change.",
        },
        "assoc_bar.png": {
            "takeaway": "AI metrics co-move with financial metadata in monotonic association tests.",
            "mechanism": "Size and innovation proxies align with narrative intensity differences.",
            "caution": "Spearman correlations are non-causal and omit many strategic confounders.",
        },
        "coefplot_ai.png": {
            "takeaway": "Controlled regressions isolate partial associations under sector and quarter FE.",
            "mechanism": "Coefficients summarize within-sector-time variation in narrative metadata.",
            "caution": "Even with controls, estimates remain associational and specification-dependent.",
        },
    }
