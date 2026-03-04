"""
Industry-Level AI Intensity Analysis

Analyzes AI intensity trends by industry sector for Top 100 AI-discussing companies.
Each year, the top 100 companies by AI intensity are selected, then aggregated by industry.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.constants import GICS_SECTOR_MAP
from src.utils.doc_id import parse_doc_id as parse_doc_id_shared
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
style_legend = _STYLE.style_legend
save_figure = _STYLE.save_figure


def _parse_doc_id(doc_id: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse doc_id (format: TICKER_YYYYQX) into (ticker, year, quarter).

    Reused from company_rankings.py.
    """
    parsed = parse_doc_id_shared(
        doc_id,
        allow_ticker_without_q=True,
        allow_ticker_on_invalid=True,
    )
    return parsed.ticker, parsed.year, parsed.quarter


def get_industry_mapping(final_dataset_path: str) -> pd.DataFrame:
    """
    Extract ticker -> (gsector, sector) mapping from final_dataset.csv.

    Args:
        final_dataset_path: Path to final_dataset.csv

    Returns:
        DataFrame with columns [ticker, gsector, sector], deduplicated by ticker
    """
    path = Path(final_dataset_path)
    required_cols = ["ticker", "gsector", "sector"]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, usecols=required_cols)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=required_cols)
    else:
        raise ValueError(
            f"Unsupported dataset format for industry mapping: {path.suffix or '[no extension]'}. "
            "Expected .csv or .parquet"
        )

    # Deduplicate by ticker (take first occurrence)
    mapping = df.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    return mapping[["ticker", "gsector", "sector"]]


def select_top100_by_ai_intensity_per_year(
    doc_metrics: pd.DataFrame,
    start_year: int = 2020,
    end_year: int = 2025,
    top_n: int = 100
) -> pd.DataFrame:
    """
    Select Top N companies by AI Intensity for each year independently.

    Args:
        doc_metrics: Document-level metrics with doc_id and overall_kw_ai_ratio
        start_year: Start year for analysis
        end_year: End year for analysis
        top_n: Number of top companies to select per year

    Returns:
        DataFrame with columns [ticker, year, avg_ai_intensity, rank]
    """
    # Parse doc_id to extract ticker and year
    doc_metrics = doc_metrics.copy()
    parsed = doc_metrics["doc_id"].apply(_parse_doc_id)
    doc_metrics["ticker"] = [p[0] for p in parsed]
    doc_metrics["year"] = [p[1] for p in parsed]

    # Filter valid rows
    doc_metrics = doc_metrics.dropna(subset=["ticker", "year", "overall_kw_ai_ratio"])
    doc_metrics["year"] = doc_metrics["year"].astype(int)

    # Aggregate to company-year level
    company_year = doc_metrics.groupby(["ticker", "year"]).agg({
        "overall_kw_ai_ratio": "mean",
        "doc_id": "count"
    }).reset_index()
    company_year = company_year.rename(columns={
        "overall_kw_ai_ratio": "avg_ai_intensity",
        "doc_id": "num_calls"
    })

    # Filter to year range
    company_year = company_year[
        (company_year["year"] >= start_year) &
        (company_year["year"] <= end_year)
    ]

    # Select Top N per year
    top100_rows = []
    for year in range(start_year, end_year + 1):
        year_df = company_year[company_year["year"] == year].copy()
        if len(year_df) == 0:
            continue

        # Sort by AI intensity descending and select top N
        year_df = year_df.nlargest(min(top_n, len(year_df)), "avg_ai_intensity")
        year_df["rank"] = range(1, len(year_df) + 1)
        top100_rows.append(year_df)

    if not top100_rows:
        return pd.DataFrame(columns=["ticker", "year", "avg_ai_intensity", "num_calls", "rank"])

    top100_df = pd.concat(top100_rows, ignore_index=True)
    return top100_df


def aggregate_industry_year(
    top100_df: pd.DataFrame,
    industry_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate AI intensity by industry and year.

    Args:
        top100_df: Top 100 companies per year with avg_ai_intensity
        industry_mapping: Ticker to (gsector, sector) mapping

    Returns:
        DataFrame with columns [gsector, sector, year, industry_ai_intensity, num_companies]
    """
    # Join industry info
    merged = top100_df.merge(industry_mapping, on="ticker", how="left")

    # Drop rows without industry mapping
    merged = merged.dropna(subset=["gsector"])
    merged["gsector"] = merged["gsector"].astype(int)

    # Aggregate by industry and year
    industry_year = merged.groupby(["gsector", "sector", "year"]).agg({
        "avg_ai_intensity": "mean",
        "ticker": "nunique"
    }).reset_index()

    industry_year = industry_year.rename(columns={
        "avg_ai_intensity": "industry_ai_intensity",
        "ticker": "num_companies"
    })

    return industry_year


def plot_industry_ai_trends(
    industry_year_df: pd.DataFrame,
    output_path: str,
    title: str = "AI Intensity by Industry (Top 100 AI Companies per Year)"
) -> None:
    """
    Create line plot of AI intensity trends by industry over years.

    Args:
        industry_year_df: Industry-year aggregated data
        output_path: Path to save the figure
        title: Plot title
    """
    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    # Get unique sectors and create color mapping
    sectors = industry_year_df["sector"].unique()
    colors = plt.cm.tab10(range(len(sectors)))
    color_map = dict(zip(sectors, colors))

    # Plot each industry as a line
    for sector in sorted(sectors):
        sector_df = industry_year_df[industry_year_df["sector"] == sector].sort_values("year")
        ax.plot(
            sector_df["year"],
            sector_df["industry_ai_intensity"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=sector,
            color=color_map[sector]
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("AI Intensity (Mean Ratio)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Set x-axis to show integer years
    years = sorted(industry_year_df["year"].unique())
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)

    style_axes(ax, grid_axis="y", grid_alpha=0.08)
    style_legend(ax)

    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved industry AI trends plot to {output_path}")


def run_industry_analysis(
    doc_metrics_path: str,
    final_dataset_path: str = "final_dataset.csv",
    output_dir: str = "outputs/figures",
    start_year: int = 2020,
    end_year: int = 2025,
    top_n: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for industry-level AI analysis.

    Args:
        doc_metrics_path: Path to document_metrics.parquet
        final_dataset_path: Path to final_dataset.csv for industry mapping
        output_dir: Output directory for figures and CSVs
        start_year: Start year for analysis
        end_year: End year for analysis
        top_n: Number of top companies to select per year

    Returns:
        Tuple of (top100_per_year_df, industry_year_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading document metrics...")
    doc_metrics = pd.read_parquet(doc_metrics_path)

    print("Loading industry mapping...")
    industry_mapping = get_industry_mapping(final_dataset_path)
    print(f"Found {len(industry_mapping)} unique companies with industry info")

    print(f"Selecting Top {top_n} companies by AI Intensity per year...")
    top100_df = select_top100_by_ai_intensity_per_year(
        doc_metrics, start_year, end_year, top_n
    )
    print(f"Selected {len(top100_df)} company-year observations")

    # Save Top 100 per year
    top100_path = os.path.join(output_dir, "top100_ai_companies_by_year.csv")
    top100_df.to_csv(top100_path, index=False)
    print(f"Saved Top {top_n} AI companies to {top100_path}")

    print("Aggregating by industry and year...")
    industry_year_df = aggregate_industry_year(top100_df, industry_mapping)
    print(f"Generated {len(industry_year_df)} industry-year observations")

    # Save industry-year data
    industry_path = os.path.join(output_dir, "industry_ai_by_year.csv")
    industry_year_df.to_csv(industry_path, index=False)
    print(f"Saved industry AI data to {industry_path}")

    # Generate plot
    print("Generating industry AI trends plot...")
    plot_path = os.path.join(output_dir, "industry_ai_trends_top100.png")
    plot_industry_ai_trends(industry_year_df, plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Industry AI Intensity Summary (Top 100 AI Companies per Year)")
    print("=" * 60)

    for year in sorted(industry_year_df["year"].unique()):
        year_data = industry_year_df[industry_year_df["year"] == year]
        print(f"\n{year}:")
        for _, row in year_data.sort_values("industry_ai_intensity", ascending=False).iterrows():
            print(f"  {row['sector']}: {row['industry_ai_intensity']:.4f} ({row['num_companies']} companies)")

    return top100_df, industry_year_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Industry-level AI intensity analysis")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet",
                        help="Path to document metrics parquet file")
    parser.add_argument("--dataset", default="final_dataset.csv",
                        help="Path to final_dataset.csv for industry mapping")
    parser.add_argument("--output-dir", default="outputs/figures",
                        help="Output directory for figures and CSVs")
    parser.add_argument("--start-year", type=int, default=2020,
                        help="Start year for analysis")
    parser.add_argument("--end-year", type=int, default=2025,
                        help="End year for analysis")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top companies to select per year")

    args = parser.parse_args()

    run_industry_analysis(
        args.metrics,
        args.dataset,
        args.output_dir,
        args.start_year,
        args.end_year,
        args.top_n
    )
