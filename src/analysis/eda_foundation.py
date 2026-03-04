"""Foundational EDA reports and charts for AI discussion in earnings calls."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark-eda-legacy")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


DEFAULT_SENTENCES_PATH = os.path.join("outputs", "features", "sentences.parquet")
DEFAULT_DOC_METRICS_PATH = os.path.join("outputs", "features", "document_metrics.parquet")
DEFAULT_INITIATION_PATH = os.path.join("outputs", "features", "initiation_scores.parquet")
DEFAULT_PARSED_PATH = os.path.join("outputs", "features", "parsed_transcripts.parquet")
DEFAULT_FIGURE_DIR = os.path.join("outputs", "figures", "eda")
DEFAULT_REPORT_DIR = os.path.join("outputs", "report", "eda")
RATIO_COLUMNS = ("speech_kw_ai_ratio", "qa_kw_ai_ratio", "overall_kw_ai_ratio")


def _safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _normalize_section(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"qa", "q&a", "question_answer", "questions_and_answers"}:
        return "qa"
    if text in {"speech", "prepared_remarks", "prepared remarks"}:
        return "speech"
    if "q&a" in text or "qa" == text or "question" in text:
        return "qa"
    if "speech" in text or "prepared" in text:
        return "speech"
    return text


def compute_data_funnel(
    parsed_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    initiation_df: Optional[pd.DataFrame] = None,
    section_col: str = "section",
) -> Dict[str, float]:
    """Compute top-line funnel counts and speech/Q&A sentence shares."""
    if section_col not in sentences_df.columns:
        raise ValueError(f"Missing '{section_col}' in sentences dataframe")

    section_normalized = sentences_df[section_col].map(_normalize_section)
    speech_count = int((section_normalized == "speech").sum())
    qa_count = int((section_normalized == "qa").sum())
    total_sentences = int(len(sentences_df))

    return {
        "total_parsed_documents": int(len(parsed_df)),
        "total_sentences": total_sentences,
        "speech_sentences": speech_count,
        "qa_sentences": qa_count,
        "speech_sentence_share": _safe_divide(speech_count, total_sentences),
        "qa_sentence_share": _safe_divide(qa_count, total_sentences),
        "tracked_initiation_documents": int(len(initiation_df)) if initiation_df is not None else 0,
    }


def summarize_ratio_columns(
    doc_metrics_df: pd.DataFrame,
    columns: Iterable[str] = RATIO_COLUMNS,
) -> pd.DataFrame:
    """Return descriptive summary statistics for each ratio column."""
    rows = []
    for col in columns:
        if col not in doc_metrics_df.columns:
            continue
        series = pd.to_numeric(doc_metrics_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "metric": col,
                "n": int(series.shape[0]),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std(ddof=0)),
                "p90": float(series.quantile(0.90)),
                "p95": float(series.quantile(0.95)),
                "zero_share": float((series == 0).mean()),
                "nonzero_share": float((series > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_ai_exchange_zero_split(
    initiation_df: pd.DataFrame,
    exchange_col: str = "total_ai_exchanges",
) -> Dict[str, float]:
    """Compute zero vs non-zero split for AI-related Q&A exchanges."""
    if exchange_col not in initiation_df.columns:
        raise ValueError(f"Missing '{exchange_col}' in initiation dataframe")

    exchanges = pd.to_numeric(initiation_df[exchange_col], errors="coerce").fillna(0)
    total_docs = int(exchanges.shape[0])
    zero_count = int((exchanges == 0).sum())
    nonzero_count = int((exchanges > 0).sum())
    return {
        "total_documents": total_docs,
        "zero_ai_exchanges_count": zero_count,
        "nonzero_ai_exchanges_count": nonzero_count,
        "zero_ai_exchanges_share": _safe_divide(zero_count, total_docs),
        "nonzero_ai_exchanges_share": _safe_divide(nonzero_count, total_docs),
    }


def _set_plot_style() -> None:
    apply_spotify_theme()
    sns.set_style("whitegrid")


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_funnel_reports(
    funnel: Dict[str, float],
    report_dir: str,
    export_formats: Iterable[str] = ("json",),
) -> Dict[str, Optional[str]]:
    os.makedirs(report_dir, exist_ok=True)
    selected = {fmt.strip().lower() for fmt in export_formats}
    paths: Dict[str, Optional[str]] = {
        "funnel_json_path": None,
        "funnel_csv_path": None,
        "funnel_txt_path": None,
    }

    if "json" in selected:
        json_path = os.path.join(report_dir, "funnel_summary.json")
        _save_json(json_path, funnel)
        paths["funnel_json_path"] = json_path

    if "csv" in selected:
        csv_path = os.path.join(report_dir, "funnel_summary.csv")
        pd.DataFrame([funnel]).to_csv(csv_path, index=False)
        paths["funnel_csv_path"] = csv_path

    if "txt" in selected:
        txt_path = os.path.join(report_dir, "funnel_summary.txt")
        lines = [
            "EDA Funnel Summary",
            f"Total parsed earnings call documents: {funnel['total_parsed_documents']:,}",
            f"Total sentences processed: {funnel['total_sentences']:,}",
            f"Speech sentences: {funnel['speech_sentences']:,} ({funnel['speech_sentence_share']:.2%})",
            f"Q&A sentences: {funnel['qa_sentences']:,} ({funnel['qa_sentence_share']:.2%})",
            f"Tracked initiation documents: {funnel['tracked_initiation_documents']:,}",
        ]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        paths["funnel_txt_path"] = txt_path

    return paths


def _draw_zero_heavy_hist(ax: plt.Axes, series: pd.Series, color: str, zero_color: str, label: str) -> Dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    zero_count = int((clean == 0).sum())
    nonzero = clean[clean > 0]
    median = float(clean.median()) if not clean.empty else 0.0
    zero_share = float((clean == 0).mean()) if not clean.empty else 0.0

    x_cap = float(clean.quantile(0.995)) if not clean.empty else 0.01
    x_cap = max(x_cap, 0.01)
    bins = np.linspace(0.0, x_cap, 60)
    bin_width = bins[1] - bins[0]

    if not nonzero.empty:
        ax.hist(nonzero, bins=bins, color=color, alpha=0.70, edgecolor="none", label=label)
    ax.bar(0.0, zero_count, width=bin_width * 0.92, color=zero_color, alpha=0.95, align="edge", label="Exact zero")
    ax.axvline(median, color=SPOTIFY_COLORS["warning"], linestyle="--", linewidth=2)
    ax.set_xlim(0, x_cap)
    style_axes(ax, grid_axis="y", grid_alpha=0.12)

    return {"median": median, "zero_share": zero_share}


def plot_overall_zero_inflation(doc_metrics_df: pd.DataFrame, output_path: str) -> Dict[str, float]:
    """Create a zero-inflation histogram for overall AI ratio."""
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(SPOTIFY_COLORS["background"])

    stats = _draw_zero_heavy_hist(
        ax=ax,
        series=doc_metrics_df["overall_kw_ai_ratio"],
        color=SPOTIFY_COLORS["blue"],
        zero_color=SPOTIFY_COLORS["negative"],
        label="Non-zero call distribution",
    )
    ax.set_title("Zero-Inflation in Overall AI Mention Intensity", fontsize=16, weight="bold")
    ax.set_xlabel("overall_kw_ai_ratio")
    ax.set_ylabel("Number of earnings calls")
    ax.text(
        0.01,
        0.98,
        (
            f"Median = {stats['median']:.4f}\n"
            f"Zero bucket = {stats['zero_share']:.1%} of calls"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=SPOTIFY_COLORS["fg"],
        bbox={
            "facecolor": SPOTIFY_COLORS.get("panel", SPOTIFY_COLORS["background"]),
            "edgecolor": SPOTIFY_COLORS["grid"],
            "alpha": 0.9,
        },
    )
    fig.text(
        0.012,
        0.02,
        "Takeaway: AI mention intensity is highly sparse with a large mass at exactly zero.",
        color=SPOTIFY_COLORS["muted"],
        fontsize=10,
    )
    ax.legend(loc="upper right")

    save_figure(fig, output_path, dpi=200)
    return stats


def plot_speech_vs_qa_zero_hist(doc_metrics_df: pd.DataFrame, output_path: str) -> Dict[str, float]:
    """Create side-by-side histograms for speech vs Q&A AI intensity."""
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True, sharey=True)
    fig.patch.set_facecolor(SPOTIFY_COLORS["background"])

    stats_speech = _draw_zero_heavy_hist(
        ax=axes[0],
        series=doc_metrics_df["speech_kw_ai_ratio"],
        color=SPOTIFY_COLORS["accent"],
        zero_color=SPOTIFY_COLORS["negative"],
        label="Non-zero speech calls",
    )
    axes[0].set_title("Speech (Prepared Remarks)", fontsize=13, weight="bold")
    axes[0].set_xlabel("speech_kw_ai_ratio")
    axes[0].set_ylabel("Number of earnings calls")

    stats_qa = _draw_zero_heavy_hist(
        ax=axes[1],
        series=doc_metrics_df["qa_kw_ai_ratio"],
        color=SPOTIFY_COLORS["blue"],
        zero_color=SPOTIFY_COLORS["negative"],
        label="Non-zero Q&A calls",
    )
    axes[1].set_title("Q&A (Interactive Dialogue)", fontsize=13, weight="bold")
    axes[1].set_xlabel("qa_kw_ai_ratio")

    axes[0].text(
        0.02,
        0.97,
        f"Median={stats_speech['median']:.4f}\nZero={stats_speech['zero_share']:.1%}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=SPOTIFY_COLORS["fg"],
    )
    axes[1].text(
        0.02,
        0.97,
        f"Median={stats_qa['median']:.4f}\nZero={stats_qa['zero_share']:.1%}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=SPOTIFY_COLORS["fg"],
    )
    fig.suptitle("Speech vs Q&A AI Mention Distributions (Same Scale)", fontsize=16, weight="bold", y=0.98)
    fig.text(
        0.012,
        0.02,
        "Takeaway: Speech is heavily zero-concentrated, while Q&A carries a slightly thicker non-zero tail.",
        color=SPOTIFY_COLORS["muted"],
        fontsize=10,
    )
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")

    save_figure(fig, output_path, dpi=200)
    return {
        "speech_median": stats_speech["median"],
        "qa_median": stats_qa["median"],
        "speech_zero_share": stats_speech["zero_share"],
        "qa_zero_share": stats_qa["zero_share"],
    }


def plot_total_ai_exchanges_zero_split(initiation_df: pd.DataFrame, output_path: str) -> Dict[str, float]:
    """Create a bar chart of calls with 0 AI exchanges vs >=1."""
    split = compute_ai_exchange_zero_split(initiation_df)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.patch.set_facecolor(SPOTIFY_COLORS["background"])

    labels = ["0 AI Exchanges", ">=1 AI Exchange"]
    values = [split["zero_ai_exchanges_count"], split["nonzero_ai_exchanges_count"]]
    colors = [SPOTIFY_COLORS["negative"], SPOTIFY_COLORS["accent"]]

    bars = ax.bar(labels, values, color=colors, alpha=0.92, width=0.62)
    for idx, bar in enumerate(bars):
        share = split["zero_ai_exchanges_share"] if idx == 0 else split["nonzero_ai_exchanges_share"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{int(bar.get_height()):,}\n({share:.1%})",
            ha="center",
            va="bottom",
            fontsize=11,
            color=SPOTIFY_COLORS["fg"],
            weight="bold",
        )

    ax.set_title("Distribution of Total AI Exchanges per Call", fontsize=16, weight="bold")
    ax.set_ylabel("Number of earnings calls")
    style_axes(ax, grid_axis="y", grid_alpha=0.12)
    fig.text(
        0.012,
        0.02,
        "Takeaway: A large share of calls still have zero AI-related Q&A exchanges.",
        color=SPOTIFY_COLORS["muted"],
        fontsize=10,
    )

    save_figure(fig, output_path, dpi=200)
    return split


def _load_required_parquet(path: str, dataset_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file for {dataset_name}: {path}")
    return pd.read_parquet(path)


def _load_optional_parquet(path: str, dataset_name: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] Optional file not found for {dataset_name}: {path}")
        return None
    return pd.read_parquet(path)


def run_eda_foundation(
    sentences_path: str = DEFAULT_SENTENCES_PATH,
    document_metrics_path: str = DEFAULT_DOC_METRICS_PATH,
    initiation_scores_path: str = DEFAULT_INITIATION_PATH,
    parsed_transcripts_path: str = DEFAULT_PARSED_PATH,
    figure_dir: str = DEFAULT_FIGURE_DIR,
    report_dir: str = DEFAULT_REPORT_DIR,
    funnel_export_formats: Iterable[str] = ("json",),
) -> Dict[str, Any]:
    """
    Run foundational EDA outputs:
    - Funnel stats (JSON by default; CSV/TXT optional for backward compatibility)
    - Ratio summaries (CSV)
    - Zero-inflation plots
    """
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print("Loading required parquet files...")
    sentences_df = _load_required_parquet(sentences_path, "sentences")
    doc_metrics_df = _load_required_parquet(document_metrics_path, "document metrics")
    parsed_df = _load_required_parquet(parsed_transcripts_path, "parsed transcripts")

    print("Loading optional initiation file...")
    initiation_df = _load_optional_parquet(initiation_scores_path, "initiation scores")

    print("Computing funnel metrics...")
    funnel = compute_data_funnel(parsed_df, sentences_df, initiation_df)
    funnel_paths = _save_funnel_reports(funnel, report_dir, export_formats=funnel_export_formats)

    print("Computing ratio summaries...")
    ratio_summary = summarize_ratio_columns(doc_metrics_df)
    ratio_summary_path = os.path.join(report_dir, "ratio_summary.csv")
    ratio_summary.to_csv(ratio_summary_path, index=False)

    print("Generating zero-inflation figures...")
    overall_hist_path = os.path.join(figure_dir, "overall_kw_ai_ratio_zero_inflation.png")
    overall_stats = plot_overall_zero_inflation(doc_metrics_df, overall_hist_path)

    speech_qa_hist_path = os.path.join(figure_dir, "speech_vs_qa_zero_inflation.png")
    speech_qa_stats = plot_speech_vs_qa_zero_hist(doc_metrics_df, speech_qa_hist_path)

    ai_exchange_split = None
    exchange_split_path = None
    exchange_plot_path = None
    if initiation_df is not None and "total_ai_exchanges" in initiation_df.columns:
        exchange_plot_path = os.path.join(figure_dir, "total_ai_exchanges_zero_split.png")
        ai_exchange_split = plot_total_ai_exchanges_zero_split(initiation_df, exchange_plot_path)
        exchange_split_path = os.path.join(report_dir, "ai_exchange_zero_split.json")
        _save_json(exchange_split_path, ai_exchange_split)
        pd.DataFrame([ai_exchange_split]).to_csv(
            os.path.join(report_dir, "ai_exchange_zero_split.csv"),
            index=False,
        )
    elif initiation_df is not None:
        print("[WARN] initiation_scores.parquet loaded but missing 'total_ai_exchanges'; skipping exchange plot.")
    else:
        print("[WARN] initiation_scores.parquet not available; skipping initiation exchange outputs.")

    print("\n=== EDA Funnel (Slide 1) ===")
    print(f"Parsed earnings call documents: {funnel['total_parsed_documents']:,}")
    print(f"Total sentences processed: {funnel['total_sentences']:,}")
    print(f"Q&A sentence share: {funnel['qa_sentence_share']:.2%}")
    print(f"Speech sentence share: {funnel['speech_sentence_share']:.2%}")
    print(f"Tracked initiation documents: {funnel['tracked_initiation_documents']:,}")

    print("\n=== Sparsity Signals (Slide 2) ===")
    print(f"overall_kw_ai_ratio median: {overall_stats['median']:.4f}")
    print(f"overall_kw_ai_ratio zero share: {overall_stats['zero_share']:.2%}")
    print(f"speech_kw_ai_ratio median: {speech_qa_stats['speech_median']:.4f}")
    print(f"qa_kw_ai_ratio median: {speech_qa_stats['qa_median']:.4f}")
    if ai_exchange_split is not None:
        print(f"total_ai_exchanges == 0 share: {ai_exchange_split['zero_ai_exchanges_share']:.2%}")

    return {
        "funnel": funnel,
        **funnel_paths,
        "ratio_summary_path": ratio_summary_path,
        "overall_hist_path": overall_hist_path,
        "speech_qa_hist_path": speech_qa_hist_path,
        "exchange_plot_path": exchange_plot_path,
        "exchange_split_path": exchange_split_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foundational EDA for earnings-call AI analysis.")
    parser.add_argument("--sentences", default=DEFAULT_SENTENCES_PATH, help="Path to sentences.parquet")
    parser.add_argument("--document-metrics", default=DEFAULT_DOC_METRICS_PATH, help="Path to document_metrics.parquet")
    parser.add_argument("--initiation-scores", default=DEFAULT_INITIATION_PATH, help="Path to initiation_scores.parquet")
    parser.add_argument("--parsed-transcripts", default=DEFAULT_PARSED_PATH, help="Path to parsed_transcripts.parquet")
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR, help="Directory for output figures")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="Directory for output summary files")
    parser.add_argument(
        "--funnel-formats",
        default="json",
        help="Comma-separated export formats for funnel summary: json,csv,txt",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    funnel_formats = [x.strip() for x in str(args.funnel_formats).split(",") if x.strip()]
    run_eda_foundation(
        sentences_path=args.sentences,
        document_metrics_path=args.document_metrics,
        initiation_scores_path=args.initiation_scores,
        parsed_transcripts_path=args.parsed_transcripts,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
        funnel_export_formats=funnel_formats,
    )
