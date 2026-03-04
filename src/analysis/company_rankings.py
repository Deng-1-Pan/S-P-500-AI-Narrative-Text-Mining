"""
Company Ranking Visualizations

Generates yearly Top/Bottom 10 company charts based on AI intensity metrics.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import os

import pandas as pd

from src.utils.doc_id import parse_doc_id as parse_doc_id_shared
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark-minimal")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


def _parse_doc_id(doc_id: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    parsed = parse_doc_id_shared(
        doc_id,
        allow_ticker_without_q=True,
        allow_ticker_on_invalid=True,
    )
    return parsed.ticker, parsed.year, parsed.quarter


def _maybe_build_doc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure document-level metrics. If section-level metrics are provided,
    aggregate them to document-level.
    """
    if "overall_kw_ai_ratio" in df.columns:
        return df

    # If section-level metrics were passed in, aggregate to doc-level
    if "section" in df.columns and "doc_id" in df.columns:
        try:
            from src.metrics.ai_intensity import compute_document_intensity
            print("Detected section-level metrics. Aggregating to document-level metrics...")
            return compute_document_intensity(df)
        except Exception as e:
            print(f"Failed to aggregate section-level metrics: {e}")
            return df

    return df


def _ensure_overall_kw_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if "overall_kw_ai_ratio" in df.columns:
        return df

    required = {"speech_kw_ai_sentences", "qa_kw_ai_sentences", "speech_total_sentences", "qa_total_sentences"}
    if required.issubset(df.columns):
        total = df["speech_total_sentences"].fillna(0) + df["qa_total_sentences"].fillna(0)
        kw_total = df["speech_kw_ai_sentences"].fillna(0) + df["qa_kw_ai_sentences"].fillna(0)
        df = df.copy()
        df["overall_kw_ai_ratio"] = kw_total / total.replace(0, pd.NA)
        df["overall_kw_ai_ratio"] = df["overall_kw_ai_ratio"].fillna(0.0)
        return df

    return df


def _aggregate_company_year(doc_metrics: pd.DataFrame) -> pd.DataFrame:
    doc_metrics = _maybe_build_doc_metrics(doc_metrics)
    doc_metrics = _ensure_overall_kw_ratio(doc_metrics)
    parsed = doc_metrics["doc_id"].apply(_parse_doc_id)
    doc_metrics = doc_metrics.copy()
    doc_metrics["ticker"] = [p[0] for p in parsed]
    doc_metrics["year"] = [p[1] for p in parsed]
    doc_metrics["quarter"] = [p[2] for p in parsed]
    doc_metrics = doc_metrics.dropna(subset=["ticker", "year"])

    agg_map = {
        "overall_kw_ai_ratio": "mean",
        "speech_kw_ai_ratio": "mean",
        "qa_kw_ai_ratio": "mean",
        "speech_kw_ai_sentences": "sum",
        "qa_kw_ai_sentences": "sum",
        "speech_total_sentences": "sum",
        "qa_total_sentences": "sum",
        "doc_id": "count",
    }

    agg_map = {k: v for k, v in agg_map.items() if k in doc_metrics.columns}
    agg_df = doc_metrics.groupby(["ticker", "year"]).agg(agg_map).reset_index()

    agg_df = agg_df.rename(columns={"doc_id": "num_calls"})
    if "speech_kw_ai_sentences" in agg_df.columns and "qa_kw_ai_sentences" in agg_df.columns:
        agg_df["overall_kw_ai_sentences"] = agg_df["speech_kw_ai_sentences"].fillna(0) + agg_df["qa_kw_ai_sentences"].fillna(0)
    if "speech_total_sentences" in agg_df.columns and "qa_total_sentences" in agg_df.columns:
        agg_df["overall_total_sentences"] = agg_df["speech_total_sentences"].fillna(0) + agg_df["qa_total_sentences"].fillna(0)

    return agg_df


def _plot_top(
    year_df: pd.DataFrame,
    metric_col: str,
    year: int,
    output_path: str,
    title: str,
    top_n: int = 10
) -> None:
    import matplotlib.pyplot as plt

    apply_spotify_theme()
    df = year_df.dropna(subset=[metric_col]).copy()
    if len(df) == 0:
        print(f"No data for {metric_col} in {year}. Skipping.")
        return

    top = df.nlargest(top_n, metric_col).sort_values(metric_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    ax.barh(top["ticker"], top[metric_col], color=SPOTIFY_COLORS.get("accent", "#1DB954"), alpha=0.9)
    ax.set_title(f"Top {top_n}")
    ax.set_xlabel("AI Ratio")
    style_axes(ax, grid_axis="x", grid_alpha=0.08)
    fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved ranking plot to {output_path}")


def run_company_ranking_analysis(
    doc_metrics_path: str,
    output_dir: str = "outputs/figures",
    start_year: int = 2020,
    end_year: int = 2025
) -> pd.DataFrame:
    """
    Generate yearly top 10 charts for overall, speech, and Q&A AI intensity (dictionary-based).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading document metrics...")
    doc_metrics = pd.read_parquet(doc_metrics_path)
    agg_df = _aggregate_company_year(doc_metrics)

    rankings_rows: List[Dict] = []

    metrics = [
        ("overall_kw_ai_ratio", "Overall AI Discussion (Dictionary)"),
        ("speech_kw_ai_ratio", "Management AI Discussion (Speech, Dictionary)"),
        ("qa_kw_ai_ratio", "Q&A AI Discussion (Dictionary)")
    ]

    for year in range(start_year, end_year + 1):
        year_df = agg_df[agg_df["year"] == year]
        if len(year_df) == 0:
            print(f"No company-year data for {year}. Skipping year.")
            continue

        for metric_col, title_prefix in metrics:
            if metric_col not in year_df.columns:
                print(f"Missing column {metric_col} for {year}. Available columns: {sorted(year_df.columns.tolist())}")
                continue
            title = f"{title_prefix} - Top 10 Companies ({year})"
            filename = f"{metric_col}_top10_{year}.png"
            output_path = os.path.join(output_dir, filename)
            _plot_top(year_df, metric_col, year, output_path, title)

            # Save ranking rows
            top = year_df.nlargest(10, metric_col).copy()
            for rank, row in enumerate(top.sort_values(metric_col, ascending=False).itertuples(), start=1):
                rankings_rows.append({
                    "year": year,
                    "metric": metric_col,
                    "rank_type": "top",
                    "rank": rank,
                    "ticker": row.ticker,
                    "value": getattr(row, metric_col)
                })

    rankings_df = pd.DataFrame(rankings_rows)
    rankings_path = os.path.join(output_dir, "company_rankings_top10.csv")
    rankings_df.to_csv(rankings_path, index=False)
    print(f"Saved ranking table to {rankings_path}")

    return rankings_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Company top 10 AI rankings")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)

    args = parser.parse_args()

    run_company_ranking_analysis(
        args.metrics,
        args.output_dir,
        args.start_year,
        args.end_year
    )
