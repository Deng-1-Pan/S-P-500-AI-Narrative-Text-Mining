"""
AI Intensity Metrics Module

Computes AI intensity scores at the document and section level.
"""

from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor

from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark-minimal")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


def _section_groupby_compute(args) -> pd.DataFrame:
    """
    Top-level helper for multiprocessing (must be picklable on Windows).
    """
    df, doc_id_col, section_col, kw_pred_col = args
    group_cols = [doc_id_col, section_col]
    grouped = df.groupby(group_cols, dropna=False)
    base = grouped.size().rename("total_sentences").reset_index()

    if kw_pred_col in df.columns:
        kw_stats = grouped[kw_pred_col].agg(['sum', 'mean']).reset_index()
        kw_stats = kw_stats.rename(columns={'sum': 'kw_ai_sentences', 'mean': 'kw_ai_ratio'})
        base = base.merge(kw_stats, on=group_cols, how="left")

    return base


def compute_section_intensity(
    sentences_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    section_col: str = 'section',
    kw_pred_col: str = 'kw_is_ai',
    num_workers: Optional[int] = None,
    chunk_size: int = 1000
) -> pd.DataFrame:
    """
    Compute AI intensity metrics per document and section (dictionary-based).
    
    Args:
        sentences_df: DataFrame with sentence-level keyword flags
        doc_id_col: Column for document ID
        section_col: Column for section (speech/qa)
        kw_pred_col: Column for keyword predictions
        
    Returns:
        DataFrame with document-section level metrics
    """
    if sentences_df.empty:
        return pd.DataFrame()

    # Decide on parallelism
    doc_ids = sentences_df[doc_id_col].unique()
    total_docs = len(doc_ids)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    if num_workers <= 1 or total_docs < chunk_size:
        return _section_groupby_compute((
            sentences_df,
            doc_id_col,
            section_col,
            kw_pred_col
        ))

    print(f"Computing section intensities with multiprocessing ({num_workers} workers)")
    chunks = [doc_ids[i:i + chunk_size] for i in range(0, total_docs, chunk_size)]
    sub_dfs = [sentences_df[sentences_df[doc_id_col].isin(chunk)].copy() for chunk in chunks]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        args_iter = [
            (sub_df, doc_id_col, section_col, kw_pred_col)
            for sub_df in sub_dfs
        ]
        for part in tqdm(ex.map(_section_groupby_compute, args_iter), total=len(sub_dfs), desc="Computing intensities"):
            results.append(part)

    return pd.concat(results, ignore_index=True)


def compute_document_intensity(
    section_metrics_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate section metrics to document level.
    
    Args:
        section_metrics_df: DataFrame from compute_section_intensity
        
    Returns:
        DataFrame with document-level metrics
    """
    results = []
    
    for doc_id in section_metrics_df['doc_id'].unique():
        doc_df = section_metrics_df[section_metrics_df['doc_id'] == doc_id]
        
        result = {'doc_id': doc_id}
        
        for section in ['speech', 'qa']:
            section_row = doc_df[doc_df['section'] == section]
            
            if len(section_row) == 0:
                result[f'{section}_total_sentences'] = 0
                result[f'{section}_kw_ai_ratio'] = 0.0
            else:
                row = section_row.iloc[0]
                result[f'{section}_total_sentences'] = row.get('total_sentences', 0)
                result[f'{section}_kw_ai_ratio'] = row.get('kw_ai_ratio', 0.0)
                result[f'{section}_kw_ai_sentences'] = row.get('kw_ai_sentences', 0)
        
        # Compute overall metrics
        total_sents = result.get('speech_total_sentences', 0) + result.get('qa_total_sentences', 0)
        if total_sents > 0:
            kw_ai_total = result.get('speech_kw_ai_sentences', 0) + result.get('qa_kw_ai_sentences', 0)
            result['overall_kw_ai_ratio'] = kw_ai_total / total_sents
        else:
            result['overall_kw_ai_ratio'] = 0.0
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_all_metrics(
    sentences_df: pd.DataFrame,
    output_dir: str = "outputs/features",
    figures_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    write_section_metrics: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Compute all AI intensity metrics and save (dictionary-based).
    
    Args:
        sentences_df: Sentence-level data with keyword flags
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with metric DataFrames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(output_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Computing section-level metrics...")
    section_metrics = compute_section_intensity(sentences_df, num_workers=num_workers)
    if write_section_metrics:
        section_metrics.to_parquet(f"{output_dir}/section_metrics.parquet", index=False)
    
    print("Computing document-level metrics...")
    doc_metrics = compute_document_intensity(section_metrics)
    doc_metrics.to_parquet(f"{output_dir}/document_metrics.parquet", index=False)
    
    print(f"\n=== AI Intensity Summary ===")
    print(f"Documents analyzed: {len(doc_metrics)}")
    print(f"Avg Speech AI Ratio (KW): {doc_metrics['speech_kw_ai_ratio'].mean():.3f}")
    print(f"Avg Q&A AI Ratio (KW): {doc_metrics['qa_kw_ai_ratio'].mean():.3f}")

    # Visualizations
    if len(doc_metrics) > 0:
        try:
            plot_intensity_distributions(doc_metrics, figures_dir)
            plot_intensity_scatter(doc_metrics, figures_dir)
        except Exception as e:
            print(f"Warning: failed to generate AI intensity plots: {e}")
    
    return {
        'section_metrics': section_metrics,
        'document_metrics': doc_metrics
    }


def plot_intensity_distributions(
    doc_metrics_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot distributions of AI intensity (Speech vs Q&A).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    sns.histplot(doc_metrics_df["speech_kw_ai_ratio"], bins=30, kde=True, ax=axes[0], color=SPOTIFY_COLORS.get("blue", "#4EA1FF"))
    axes[0].set_title("Speech AI Intensity Distribution (Dictionary)")
    axes[0].set_xlabel("AI Ratio")
    axes[0].set_ylabel("Count")

    style_axes(axes[0], grid_axis="y", grid_alpha=0.08)
    sns.histplot(doc_metrics_df["qa_kw_ai_ratio"], bins=30, kde=True, ax=axes[1], color=SPOTIFY_COLORS.get("negative", "#FF5A5F"))
    axes[1].set_title("Q&A AI Intensity Distribution (Dictionary)")
    axes[1].set_xlabel("AI Ratio")
    axes[1].set_ylabel("Count")
    style_axes(axes[1], grid_axis="y", grid_alpha=0.08)
    fig.tight_layout()
    output_path = os.path.join(output_dir, "ai_intensity_distributions.png")
    save_figure(fig, output_path, dpi=180)
    print(f"Saved AI intensity distribution plot to {output_path}")


def plot_intensity_scatter(
    doc_metrics_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot Speech vs Q&A AI intensity scatter with overall intensity color.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    scatter = ax.scatter(
        doc_metrics_df["speech_kw_ai_ratio"],
        doc_metrics_df["qa_kw_ai_ratio"],
        c=doc_metrics_df["overall_kw_ai_ratio"],
        cmap="viridis",
        alpha=0.6,
        s=40
    )
    ax.set_xlabel("Speech AI Intensity (Dictionary)")
    ax.set_ylabel("Q&A AI Intensity (Dictionary)")
    ax.set_title("Speech vs Q&A AI Intensity (Color = Overall AI Ratio)")
    style_axes(ax, grid_axis="both", grid_alpha=0.08)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Overall AI Ratio")
    cbar.ax.tick_params(colors=SPOTIFY_COLORS.get("muted", "#B3B3B3"))

    fig.tight_layout()
    output_path = os.path.join(output_dir, "ai_intensity_scatter.png")
    save_figure(fig, output_path, dpi=180)
    print(f"Saved AI intensity scatter plot to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute AI intensity metrics")
    parser.add_argument("--input", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    
    args = parser.parse_args()
    
    sentences_df = pd.read_parquet(args.input)
    compute_all_metrics(sentences_df, args.output_dir)
