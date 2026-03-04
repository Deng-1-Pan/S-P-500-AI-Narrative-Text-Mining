from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.company_quadrants import classify_companies
from src.utils.doc_id import parse_doc_id as parse_doc_id_shared
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="stage16-light")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


def parse_doc_id(doc_id: str) -> Tuple[str | None, int | None, int | None, str | None]:
    parsed = parse_doc_id_shared(
        doc_id,
        allow_ticker_without_q=False,
        allow_ticker_on_invalid=True,
    )
    return parsed.ticker, parsed.year, parsed.quarter, parsed.yearq


def _attach_doc_keys(df: pd.DataFrame, doc_col: str = "doc_id") -> pd.DataFrame:
    out = df.copy()
    parsed = out[doc_col].apply(parse_doc_id)
    out["ticker"] = [x[0] for x in parsed]
    out["year"] = [x[1] for x in parsed]
    out["quarter"] = [x[2] for x in parsed]
    out["yearq"] = [x[3] for x in parsed]
    return out


def _fallback_quadrants(document_metrics: pd.DataFrame) -> pd.DataFrame:
    quad_df, _, _ = classify_companies(document_metrics, threshold_method="mean")
    return quad_df[["doc_id", "quadrant"]].copy()


def _make_merge_funnel(
    diagnostics: Dict[str, float],
    output_path: str,
) -> None:
    apply_spotify_theme()
    labels = [
        "Text docs",
        "Parsable key",
        "Matched key",
        "Matched + core vars",
    ]
    values = [
        diagnostics["n_total_docs"],
        diagnostics["n_parsable_docs"],
        diagnostics["n_key_matched_docs"],
        diagnostics["n_core_matched_docs"],
    ]
    pd.DataFrame({"stage": labels, "count": values}).to_csv(
        output_path.replace(".png", "_data.csv"), index=False
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#FFFFFF"))
    bars = ax.bar(labels, values, color=[SPOTIFY_COLORS.get("blue", "#1F449C"), "#3E72BC", "#7399D0", SPOTIFY_COLORS.get("accent", "#0B8E8A")])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(v)}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Documents")
    ax.set_title("Stage15 Merge Funnel")
    style_axes(ax, grid_axis="y", grid_alpha=0.1)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=200)


def build_text_wrds_panel(
    document_metrics_path: str,
    initiation_scores_path: str,
    quadrants_path: str | None,
    final_dataset_path: str,
    wrds_feature_store_path: str,
    output_features_dir: str = "outputs/features",
    output_figures_dir: str = "outputs/figures",
) -> Dict[str, object]:
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_figures_dir, exist_ok=True)

    doc = pd.read_parquet(document_metrics_path)
    init = pd.read_parquet(initiation_scores_path)
    final_dataset = pd.read_parquet(final_dataset_path)
    wrds = pd.read_parquet(wrds_feature_store_path)

    if quadrants_path and os.path.exists(quadrants_path):
        quadrants = pd.read_parquet(quadrants_path)[["doc_id", "quadrant"]].copy()
    else:
        quadrants = _fallback_quadrants(doc)

    docs = _attach_doc_keys(doc, "doc_id")
    if docs[["ticker", "year", "quarter", "yearq"]].isna().any().any():
        raise ValueError("Failed parsing doc_id into ticker/year/quarter/yearq.")

    init = _attach_doc_keys(init, "doc_id")
    init_keep = [
        "doc_id",
        "analyst_initiated_ratio",
        "management_pivot_ratio",
        "ai_initiation_score",
        "total_ai_exchanges",
    ]
    init_keep = [c for c in init_keep if c in init.columns]
    docs = docs.merge(init[init_keep], on="doc_id", how="left")
    docs = docs.merge(quadrants, on="doc_id", how="left")

    final = final_dataset.copy()
    if "yearq" not in final.columns:
        final["yearq"] = final["year"].astype(int).astype(str) + "Q" + final["quarter"].astype(int).astype(str)
    meta_keep = [c for c in ["ticker", "yearq", "date", "gsector", "sector", "industry", "industry_name"] if c in final.columns]
    final_meta = final[meta_keep].drop_duplicates(["ticker", "yearq"])
    docs = docs.merge(final_meta, on=["ticker", "yearq"], how="left", suffixes=("", "_call"))

    docs["overall_ai_ratio"] = docs.get("overall_kw_ai_ratio")
    docs["speech_ai_ratio"] = docs.get("speech_kw_ai_ratio")
    docs["qa_ai_ratio"] = docs.get("qa_kw_ai_ratio")
    docs["ai_sentence_count"] = docs.get("speech_kw_ai_sentences", 0).fillna(0) + docs.get("qa_kw_ai_sentences", 0).fillna(0)
    docs["total_sentence_count"] = docs.get("speech_total_sentences", 0).fillna(0) + docs.get("qa_total_sentences", 0).fillna(0)
    docs["analyst_initiated_share"] = docs.get("analyst_initiated_ratio")
    docs["management_pivot_share"] = docs.get("management_pivot_ratio")
    docs["initiation_score"] = docs.get("ai_initiation_score")

    wrds_merge = wrds.copy()
    wrds_merge["_wrds_row_exists"] = 1
    docs = docs.merge(wrds_merge, on=["ticker", "yearq"], how="left", suffixes=("", "_wrds"))

    ticker_universe = set(wrds["ticker"].astype(str).unique())
    pair_universe = set(zip(wrds["ticker"].astype(str), wrds["yearq"].astype(str)))
    core_cols = [c for c in ["mkcap", "price", "shares", "eps", "rd_intensity_mkcap"] if c in docs.columns]
    docs["_core_ok"] = docs[core_cols].notna().all(axis=1) if core_cols else True

    def classify_reason(row: pd.Series) -> str:
        key = (str(row["ticker"]), str(row["yearq"]))
        if key[0] not in ticker_universe:
            return "ticker_not_in_wrds_universe"
        if key not in pair_universe:
            return "ticker_exists_but_quarter_missing"
        if not bool(row["_core_ok"]):
            return "wrds_row_missing_core_values"
        return "matched"

    docs["merge_reason"] = docs.apply(classify_reason, axis=1)
    docs["_key_matched"] = docs["merge_reason"].isin(["matched", "wrds_row_missing_core_values"])
    docs["_core_matched"] = docs["merge_reason"].eq("matched")

    diagnostics = pd.DataFrame(
        docs["merge_reason"].value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="count")
    )
    diagnostics["rate"] = diagnostics["count"] / float(len(docs)) if len(docs) else np.nan
    diag_path = os.path.join(output_features_dir, "merge_diagnostics.csv")
    diagnostics.to_csv(diag_path, index=False)

    funnel_stats = {
        "n_total_docs": float(len(docs)),
        "n_parsable_docs": float(docs[["ticker", "yearq"]].notna().all(axis=1).sum()),
        "n_key_matched_docs": float(docs["_key_matched"].sum()),
        "n_core_matched_docs": float(docs["_core_matched"].sum()),
    }
    funnel_path = os.path.join(output_figures_dir, "merge_funnel.png")
    _make_merge_funnel(funnel_stats, funnel_path)

    if int(docs["doc_id"].duplicated().sum()) > 0:
        raise ValueError("Duplicate doc_id detected in text-wrds panel.")
    if int(docs.duplicated(["ticker", "yearq"]).sum()) > 0:
        raise ValueError("Duplicate ticker+yearq detected in text-wrds panel.")

    docs["quarter_index"] = docs["year"].astype(int) * 4 + docs["quarter"].astype(int)
    qdiff = docs.sort_values(["ticker", "quarter_index"]).groupby("ticker")["quarter_index"].diff().dropna()
    if bool((qdiff < 0).any()):
        raise ValueError("Quarter ordering is not monotonic within ticker.")

    # lead/lag cross-firm validation is enforced at WRDS feature-store construction.

    panel_path = os.path.join(output_features_dir, "text_wrds_panel.parquet")
    docs.to_parquet(panel_path, index=False)

    return {
        "panel": docs,
        "panel_path": panel_path,
        "merge_diagnostics": diagnostics,
        "merge_diagnostics_path": diag_path,
        "merge_funnel_path": funnel_path,
    }
