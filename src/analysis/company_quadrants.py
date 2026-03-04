"""
Company Quadrants Module

Classifies companies into 4 types based on AI narrative patterns:
1. Aligned: High Speech + High Q&A (genuine AI focus)
2. Passive: Low Speech + High Q&A (responding to analyst pressure)
3. Self-Promoting: High Speech + Low Q&A (AI-washing)
4. Silent: Low Speech + Low Q&A (not engaging with AI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Dict, Tuple
import os

from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
style_legend = _STYLE.style_legend
save_figure = _STYLE.save_figure


def classify_companies(
    doc_metrics_df: pd.DataFrame,
    speech_col: str = 'speech_kw_ai_ratio',
    qa_col: str = 'qa_kw_ai_ratio',
    threshold_method: str = 'mean'
) -> pd.DataFrame:
    """
    Classify companies into quadrants based on AI intensity.
    
    Args:
        doc_metrics_df: Document-level metrics
        speech_col: Column for speech AI intensity
        qa_col: Column for Q&A AI intensity
        threshold_method: 'mean' or 'median_nonzero' for cutoff
        
    Returns:
        DataFrame with quadrant labels
    """
    df = doc_metrics_df.copy()
    
    # Compute thresholds (avoid zero-inflation bias)
    if threshold_method == 'median_nonzero':
        speech_vals = df.loc[df[speech_col] > 0, speech_col]
        qa_vals = df.loc[df[qa_col] > 0, qa_col]
        speech_threshold = float(speech_vals.median()) if len(speech_vals) > 0 else 0.0
        qa_threshold = float(qa_vals.median()) if len(qa_vals) > 0 else 0.0
    elif threshold_method == 'mean':
        speech_threshold = float(df[speech_col].mean())
        qa_threshold = float(df[qa_col].mean())
    else:
        raise ValueError("threshold_method must be one of: mean, median_nonzero")

    print(f"Thresholds ({threshold_method}): Speech={speech_threshold:.4f}, Q&A={qa_threshold:.4f}")
    
    # Classify
    def classify_row(row):
        # High = strictly above the threshold (prevents zeros from being "High")
        high_speech = row[speech_col] > speech_threshold
        high_qa = row[qa_col] > qa_threshold
        
        if high_speech and high_qa:
            return 'Aligned'
        elif not high_speech and high_qa:
            return 'Passive'
        elif high_speech and not high_qa:
            return 'Self-Promoting'
        else:
            return 'Silent'
    
    df['quadrant'] = df.apply(classify_row, axis=1)
    
    return df, speech_threshold, qa_threshold


def aggregate_to_company(
    doc_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate document-level metrics to company level.
    
    Args:
        doc_metrics_df: Document-level data
        
    Returns:
        Company-level aggregated data
    """
    # Parse ticker from doc_id
    doc_metrics_df = doc_metrics_df.copy()
    doc_metrics_df['ticker'] = doc_metrics_df['doc_id'].apply(
        lambda x: str(x).rsplit('_', 1)[0] if '_' in str(x) else x
    )
    
    agg_df = doc_metrics_df.groupby('ticker').agg({
        'speech_kw_ai_ratio': 'mean',
        'qa_kw_ai_ratio': 'mean',
        'overall_kw_ai_ratio': 'mean',
        'doc_id': 'count'  # Number of earnings calls
    }).reset_index()
    
    agg_df = agg_df.rename(columns={'doc_id': 'num_calls'})
    
    return agg_df


def plot_quadrant_scatter(
    df: pd.DataFrame,
    speech_col: str,
    qa_col: str,
    speech_threshold: float,
    qa_threshold: float,
    output_path: str,
    title: str = "Company AI Narrative Quadrants"
):
    """
    Create scatter plot with quadrant visualization.
    """
    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    
    # Color by quadrant
    colors = {
        'Aligned': SPOTIFY_COLORS.get("accent", "#1DB954"),
        'Passive': SPOTIFY_COLORS.get("blue", "#4EA1FF"),
        'Self-Promoting': SPOTIFY_COLORS.get("warning", "#F4C542"),
        'Silent': SPOTIFY_COLORS.get("muted", "#B3B3B3"),
    }
    
    for quadrant, color in colors.items():
        subset = df[df['quadrant'] == quadrant]
        ax.scatter(
            subset[speech_col], subset[qa_col],
            c=color, label=f"{quadrant} (n={len(subset)})",
            alpha=0.6, s=50
        )
    
    # Draw threshold lines
    ax.axvline(x=speech_threshold, color=SPOTIFY_COLORS.get("grid", "#2A2A2A"), linestyle='--', alpha=0.9)
    ax.axhline(y=qa_threshold, color=SPOTIFY_COLORS.get("grid", "#2A2A2A"), linestyle='--', alpha=0.9)
    
    # Labels for quadrants
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    offset = 0.02
    ax.text(xlim[1]-offset, ylim[1]-offset, 'Aligned\n(Genuine Focus)', 
            ha='right', va='top', fontsize=10, color=colors['Aligned'], weight='bold')
    ax.text(xlim[0]+offset, ylim[1]-offset, 'Passive\n(Responding)', 
            ha='left', va='top', fontsize=10, color=colors['Passive'], weight='bold')
    ax.text(xlim[1]-offset, ylim[0]+offset, 'Self-Promoting\n(AI-Washing?)', 
            ha='right', va='bottom', fontsize=10, color=colors['Self-Promoting'], weight='bold')
    ax.text(xlim[0]+offset, ylim[0]+offset, 'Silent\n(Disengaged)', 
            ha='left', va='bottom', fontsize=10, color=colors['Silent'], weight='bold')
    
    ax.set_xlabel('Speech AI Intensity (Management Prepared Remarks)', fontsize=12)
    ax.set_ylabel('Q&A AI Intensity (Analyst Interaction)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    style_axes(ax, grid_axis="both", grid_alpha=0.08)
    style_legend(ax)
    
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved quadrant plot to {output_path}")


def plot_quadrant_distribution(
    df: pd.DataFrame,
    output_path: str
):
    """
    Create bar chart of quadrant distribution.
    """
    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    
    counts = df['quadrant'].value_counts()
    colors = [
        SPOTIFY_COLORS.get("accent", "#1DB954"),
        SPOTIFY_COLORS.get("blue", "#4EA1FF"),
        SPOTIFY_COLORS.get("warning", "#F4C542"),
        SPOTIFY_COLORS.get("muted", "#B3B3B3"),
    ]
    order = ['Aligned', 'Passive', 'Self-Promoting', 'Silent']
    
    counts = counts.reindex(order)
    
    bars = ax.bar(counts.index, counts.values, color=colors)
    
    # Add count labels
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=12)
    
    ax.set_xlabel('Quadrant', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)
    ax.set_title('Distribution of Companies by AI Narrative Pattern', fontsize=14)
    
    style_axes(ax, grid_axis="y", grid_alpha=0.10)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved distribution plot to {output_path}")


def compare_quadrant_financials(
    company_classified: pd.DataFrame,
    wrds_data_path: str,
    output_dir: str = "outputs/figures",
) -> pd.DataFrame:
    """
    Merge WRDS financial data with quadrant labels and compare financial
    characteristics across quadrants using pairwise t-tests and boxplots.

    Economic narrative:
      - Aligned   : Genuine AI focus — likely high R&D, tech-heavy
      - Self-Promoting: High speech, low Q&A — possible AI-Washing?
      - Passive   : Responding to analyst pressure rather than proactively disclosing
      - Silent    : Not engaging with AI at all

    Args:
        company_classified: DataFrame with 'ticker' and 'quadrant' columns.
        wrds_data_path:     Path to WRDS CSV with financial data.
        output_dir:         Directory to save plots and results.

    Returns:
        Merged DataFrame with quadrant + financial variables.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        wrds = pd.read_csv(wrds_data_path, low_memory=False)
    except Exception as e:
        print(f"[compare_quadrant_financials] Cannot load WRDS data: {e}")
        return company_classified

    wrds = wrds.rename(columns={'tic': 'ticker'})

    # Compute financial metrics
    wrds['rd_intensity'] = wrds['xrdq'] / wrds['mkvaltq'].replace(0, np.nan)
    wrds['log_mktcap']   = np.log(wrds['mkvaltq'].replace(0, np.nan))
    wrds['eps_positive'] = (wrds['epspxq'] > 0).astype(float)

    # Use latest available WRDS record per ticker
    fin_cols = ['ticker', 'rd_intensity', 'log_mktcap', 'eps_positive', 'mkvaltq']
    fin_cols = [c for c in fin_cols if c in wrds.columns]
    if 'datadate' in wrds.columns:
        wrds_latest = (
            wrds.sort_values('datadate')
            .drop_duplicates('ticker', keep='last')[fin_cols]
        )
    else:
        wrds_latest = wrds.drop_duplicates('ticker', keep='last')[fin_cols]

    merged = company_classified.merge(wrds_latest, on='ticker', how='left')

    metrics = [m for m in ['rd_intensity', 'log_mktcap', 'eps_positive'] if m in merged.columns]
    quadrant_order = ['Aligned', 'Passive', 'Self-Promoting', 'Silent']

    # ── Statistical comparison ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("Quadrant Financial Comparison (mean ± std)")
    print("="*60)
    summary_rows = []
    for q in quadrant_order:
        sub = merged[merged['quadrant'] == q]
        row = {'quadrant': q, 'n': len(sub)}
        for m in metrics:
            row[m + '_mean'] = sub[m].mean()
            row[m + '_std']  = sub[m].std()
        summary_rows.append(row)
    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))
    summary_table.to_csv(os.path.join(output_dir, 'quadrant_financial_summary.csv'), index=False)

    # ── Pairwise t-tests (Aligned vs Self-Promoting is most interesting) ────
    print("\n--- Pairwise t-tests (Aligned vs Self-Promoting) ---")
    aligned_df = merged[merged['quadrant'] == 'Aligned']
    self_prom_df = merged[merged['quadrant'] == 'Self-Promoting']
    for m in metrics:
        a = aligned_df[m].dropna()
        s = self_prom_df[m].dropna()
        if len(a) >= 2 and len(s) >= 2:
            t_stat, p_val = stats.ttest_ind(a, s, equal_var=False)
            print(f"  {m:25s}: t={t_stat:+.3f}, p={p_val:.4f}"
                  f"  (Aligned μ={a.mean():.4f}, Self-Promoting μ={s.mean():.4f})")

    # ANOVA across all quadrants
    print("\n--- One-way ANOVA across all quadrants ---")
    for m in metrics:
        groups = [merged[merged['quadrant'] == q][m].dropna().values
                  for q in quadrant_order if len(merged[merged['quadrant'] == q][m].dropna()) >= 2]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"  {m:25s}: F={f_stat:.3f}, p={p_val:.4f}")

    # ── Economic narrative ──────────────────────────────────────────────────
    print("\n--- Economic Narrative ---")
    print("  Aligned    : Companies that discuss AI consistently in both prepared"
          " remarks and Q&A — likely genuine R&D investors.")
    print("  Self-Promoting: High management speech but low analyst follow-up —"
          " potential AI-Washing. Watch for high speech_kw_ai_ratio with low rd_intensity.")
    print("  Passive    : Management only engages when analysts push — reactive strategy.")
    print("  Silent     : Minimal AI engagement — legacy businesses or conservative disclosers.")

    # ── Boxplots ────────────────────────────────────────────────────────────
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        palette = {'Aligned': 'seagreen', 'Passive': 'steelblue',
                   'Self-Promoting': 'darkorange', 'Silent': 'gray'}
        for ax, m in zip(axes, metrics):
            plot_df = merged[merged['quadrant'].isin(quadrant_order)].copy()
            sns.boxplot(
                data=plot_df, x='quadrant', y=m, order=quadrant_order,
                palette=palette, ax=ax, showfliers=False,
            )
            ax.set_title(m.replace('_', ' ').title(), fontsize=12)
            ax.set_xlabel('')
            ax.set_xticklabels(quadrant_order, rotation=15, ha='right')
            ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle("Financial Characteristics by AI Narrative Quadrant", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(output_dir, 'quadrant_financial_boxplots.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved financial boxplots → {out_path}")

    return merged


def run_quadrant_analysis(
    doc_metrics_path: str,
    output_dir: str = "outputs/figures",
    wrds_data_path: Optional[str] = None,
    features_output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full quadrant analysis pipeline.

    Args:
        doc_metrics_path: Path to document metrics
        output_dir:       Output directory
        wrds_data_path:   Optional path to WRDS CSV for financial comparison

    Returns:
        Tuple of (document-level df, company-level df)
    """
    os.makedirs(output_dir, exist_ok=True)
    if features_output_dir is None:
        features_output_dir = os.path.normpath(os.path.join(output_dir, "..", "features"))
    os.makedirs(features_output_dir, exist_ok=True)

    print("Loading metrics...")
    doc_metrics = pd.read_parquet(doc_metrics_path)

    # Document-level analysis
    print("\nDocument-level quadrant analysis...")
    doc_classified, speech_th, qa_th = classify_companies(doc_metrics, threshold_method='mean')

    print("\nQuadrant Distribution (Documents):")
    print(doc_classified['quadrant'].value_counts())

    plot_quadrant_scatter(
        doc_classified, 'speech_kw_ai_ratio', 'qa_kw_ai_ratio',
        speech_th, qa_th,
        f"{output_dir}/quadrant_scatter_documents.png",
        "Document-Level AI Narrative Quadrants"
    )

    # Company-level analysis
    print("\nCompany-level quadrant analysis...")
    company_agg = aggregate_to_company(doc_metrics)
    company_classified, comp_speech_th, comp_qa_th = classify_companies(
        company_agg, threshold_method='mean'
    )

    print("\nQuadrant Distribution (Companies):")
    print(company_classified['quadrant'].value_counts())

    plot_quadrant_scatter(
        company_classified, 'speech_kw_ai_ratio', 'qa_kw_ai_ratio',
        comp_speech_th, comp_qa_th,
        f"{output_dir}/quadrant_scatter_companies.png",
        "Company-Level AI Narrative Quadrants"
    )

    plot_quadrant_distribution(company_classified, f"{output_dir}/quadrant_distribution.png")

    # Financial comparison across quadrants (t-tests + boxplots)
    if wrds_data_path and os.path.exists(wrds_data_path):
        print("\nRunning financial comparison across quadrants...")
        company_classified = compare_quadrant_financials(
            company_classified, wrds_data_path, output_dir
        )

    # Save results
    doc_classified.to_parquet(
        os.path.join(features_output_dir, "documents_with_quadrants.parquet"), index=False
    )
    company_classified.to_csv(f"{output_dir}/company_quadrants.csv", index=False)

    return doc_classified, company_classified


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quadrant analysis")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--wrds", default=None, help="Path to WRDS CSV for financial comparison")

    args = parser.parse_args()

    run_quadrant_analysis(args.metrics, args.output_dir, wrds_data_path=args.wrds)
