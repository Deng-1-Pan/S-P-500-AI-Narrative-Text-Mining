"""Shared ML helper functions used by multiple stages."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fallback: float = np.nan,
) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return float(fallback)
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float(fallback)


def aggregate_doc_text(
    sentences_df: pd.DataFrame,
    section: Optional[str] = None,
    ai_only: bool = False,
    mask_non_ai: bool = False,
    output_col: str = "doc_text",
    sort: bool = False,
) -> pd.DataFrame:
    if sentences_df is None or len(sentences_df) == 0:
        return pd.DataFrame(columns=["doc_id", output_col])

    cols = ["doc_id", "text"] + (["section"] if "section" in sentences_df.columns else [])
    df = sentences_df[cols].copy()

    if ai_only and "kw_is_ai" in sentences_df.columns:
        df = df[sentences_df["kw_is_ai"].fillna(False).astype(bool)].copy()
    elif mask_non_ai and "kw_is_ai" in sentences_df.columns:
        df.loc[~sentences_df["kw_is_ai"].fillna(False).astype(bool), "text"] = ""

    if section and "section" in df.columns:
        sec = df[df["section"] == section].copy()
        if len(sec) > 0:
            df = sec

    df["text"] = df["text"].fillna("").astype(str)
    grouped = df.groupby("doc_id", sort=sort)["text"].agg(" ".join).reset_index()
    return grouped.rename(columns={"text": output_col})
