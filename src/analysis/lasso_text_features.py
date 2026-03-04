"""
Stage 11 rewrite: AI sentence sentiment + forward R&D prediction.

This module now focuses on targets that management can influence directly:
- Continuous: y_next_rd_intensity_change
- Binary:     rd_increased_next_quarter

Text features are built from AI-tagged sentences and combined with simple
Loughran-McDonald style positive/negative tone ratios.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.utils.doc_id import attach_doc_keys
from src.utils.ml_helpers import safe_roc_auc
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
style_legend = _STYLE.style_legend
save_figure = _STYLE.save_figure


POSITIVE_WORDS = {
    "benefit",
    "confidence",
    "efficient",
    "efficiency",
    "gain",
    "growth",
    "improve",
    "innovation",
    "lead",
    "margin",
    "opportunity",
    "optimize",
    "progress",
    "strong",
    "upgrade",
    "value",
}
NEGATIVE_WORDS = {
    "challenge",
    "concern",
    "decline",
    "difficult",
    "headwind",
    "loss",
    "pressure",
    "risk",
    "slowdown",
    "uncertain",
    "uncertainty",
    "volatile",
    "weak",
    "worse",
}
TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _parse_doc_id(df: pd.DataFrame) -> pd.DataFrame:
    if "doc_id" not in df.columns:
        return df.copy()
    return attach_doc_keys(
        df,
        doc_id_col="doc_id",
        ticker_col="ticker",
        year_col="year",
        quarter_col="quarter",
        yearq_col="",
        keep_existing=True,
        allow_ticker_without_q=False,
        allow_ticker_on_invalid=False,
    )


def _build_doc_corpus(sentences_df: pd.DataFrame, section: Optional[str] = None) -> pd.DataFrame:
    cols = ["doc_id", "text"] + (["section"] if "section" in sentences_df.columns else [])
    s = sentences_df[cols].copy()

    if "kw_is_ai" in sentences_df.columns:
        s = s[sentences_df["kw_is_ai"].fillna(False).astype(bool)].copy()

    if section and "section" in s.columns:
        sec = s[s["section"] == section].copy()
        if len(sec) > 0:
            s = sec

    s["text"] = s["text"].fillna("").astype(str)
    return s.groupby("doc_id", sort=False)["text"].agg(" ".join).reset_index()


def compute_ai_sentiment_features(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """Compute positive/negative tone ratios only on AI-tagged sentences."""
    if sentences_df is None or len(sentences_df) == 0:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "ai_sentiment_positive_ratio",
                "ai_sentiment_negative_ratio",
                "ai_positive_count",
                "ai_negative_count",
                "ai_sentiment_token_count",
            ]
        )

    s = sentences_df[["doc_id", "text"]].copy()
    if "kw_is_ai" in sentences_df.columns:
        s = s[sentences_df["kw_is_ai"].fillna(False).astype(bool)].copy()

    s["text"] = s["text"].fillna("").astype(str)

    rows: List[Dict[str, float]] = []
    for doc_id, grp in s.groupby("doc_id", sort=False):
        text = " ".join(grp["text"].tolist()).lower()
        toks = TOKEN_RE.findall(text)
        if not toks:
            rows.append(
                {
                    "doc_id": doc_id,
                    "ai_sentiment_positive_ratio": 0.0,
                    "ai_sentiment_negative_ratio": 0.0,
                    "ai_positive_count": 0,
                    "ai_negative_count": 0,
                    "ai_sentiment_token_count": 0,
                }
            )
            continue

        pos = sum(t in POSITIVE_WORDS for t in toks)
        neg = sum(t in NEGATIVE_WORDS for t in toks)
        denom = pos + neg
        rows.append(
            {
                "doc_id": doc_id,
                "ai_sentiment_positive_ratio": float(pos / denom) if denom > 0 else 0.0,
                "ai_sentiment_negative_ratio": float(neg / denom) if denom > 0 else 0.0,
                "ai_positive_count": int(pos),
                "ai_negative_count": int(neg),
                "ai_sentiment_token_count": int(len(toks)),
            }
        )

    return pd.DataFrame(rows)


def _load_forward_rd_targets(
    doc_metrics_path: str,
    doc_metrics_df: pd.DataFrame,
    regression_dataset_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build forward R&D targets from regression dataset in the same feature directory."""
    candidates: List[str] = []
    if regression_dataset_path:
        candidates.append(regression_dataset_path)

    doc_dir = os.path.dirname(doc_metrics_path)
    candidates.append(os.path.join(doc_dir, "regression_dataset.parquet"))

    # Stage-folder layout fallback: features/stage05/document_metrics.parquet -> features/stage09/regression_dataset.parquet
    stage_dir = os.path.basename(doc_dir)
    if stage_dir.startswith("stage"):
        features_root = os.path.dirname(doc_dir)
        candidates.append(os.path.join(features_root, "stage09", "regression_dataset.parquet"))

    candidates.append(os.path.join("outputs", "features", "regression_dataset.parquet"))

    src = None
    for path in candidates:
        if os.path.exists(path):
            src = pd.read_parquet(path)
            break

    if src is None:
        src = doc_metrics_df[["doc_id"]].copy()

    src = _parse_doc_id(src)

    if "rd_intensity" in src.columns and "y_next_rd_intensity_change" not in src.columns:
        if {"ticker", "year", "quarter"}.issubset(src.columns):
            src = src.sort_values(["ticker", "year", "quarter"]).copy()
            src["y_next_rd_intensity_change"] = src.groupby("ticker", sort=False)["rd_intensity"].shift(-1) - src["rd_intensity"]

    if "y_next_rd_intensity_change" not in src.columns:
        raise ValueError(
            "Unable to build 'y_next_rd_intensity_change'. Expected rd_intensity data in regression_dataset.parquet."
        )

    src["y_next_rd_intensity_change"] = pd.to_numeric(src["y_next_rd_intensity_change"], errors="coerce")
    src["rd_increased_next_quarter"] = np.where(
        src["y_next_rd_intensity_change"].notna(),
        (src["y_next_rd_intensity_change"] > 0).astype(int),
        np.nan,
    )

    keep_cols = ["doc_id", "y_next_rd_intensity_change", "rd_increased_next_quarter"]
    for c in ["log_mktcap", "eps_positive", "rd_intensity", "year", "quarter", "ticker"]:
        if c in src.columns:
            keep_cols.append(c)

    out = src[keep_cols].drop_duplicates("doc_id")
    out = doc_metrics_df[["doc_id"]].drop_duplicates().merge(out, on="doc_id", how="left")
    return out


def _prepare_extra_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if not feature_cols:
        return sparse.csr_matrix((len(train_df), 0)), sparse.csr_matrix((len(test_df), 0))

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_df[feature_cols])
    X_test = imputer.transform(test_df[feature_cols])

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return sparse.csr_matrix(X_train), sparse.csr_matrix(X_test)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return safe_roc_auc(y_true, y_score, fallback=np.nan)


def fit_lasso_ngram(
    corpus_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_col: str,
    text_col: str = "text",
    doc_id_col: str = "doc_id",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    cv: int = 5,
    random_state: int = 42,
    precomputed_features: Optional[Dict] = None,
    compute_cv_predictions: bool = True,
    extra_features_df: Optional[pd.DataFrame] = None,
    task_type: Optional[str] = None,
) -> Dict:
    """Fit sparse text model with optional sentiment/structured features."""
    del precomputed_features

    merged = corpus_df[[doc_id_col, text_col]].merge(
        target_df[[doc_id_col, target_col]].dropna(),
        on=doc_id_col,
        how="inner",
    )
    if extra_features_df is not None and len(extra_features_df) > 0:
        merged = merged.merge(extra_features_df, on=doc_id_col, how="left")

    if len(merged) < 12:
        return {}

    merged[text_col] = merged[text_col].fillna("").astype(str)
    y = pd.to_numeric(merged[target_col], errors="coerce").to_numpy()
    valid = ~np.isnan(y)
    merged = merged.loc[valid].reset_index(drop=True)
    y = y[valid]

    if len(merged) < 12:
        return {}

    if task_type is None:
        uniq = set(pd.Series(y).dropna().astype(int).unique().tolist())
        task_type = "classification" if uniq.issubset({0, 1}) else "regression"

    extra_cols = []
    if extra_features_df is not None:
        extra_cols = [c for c in merged.columns if c not in {doc_id_col, text_col, target_col}]

    splitter = (
        StratifiedKFold(n_splits=min(max(2, cv), len(merged)), shuffle=True, random_state=random_state)
        if task_type == "classification"
        else KFold(n_splits=min(max(2, cv), len(merged)), shuffle=True, random_state=random_state)
    )

    oof_score = np.full(len(merged), np.nan, dtype=float)

    if compute_cv_predictions:
        for train_idx, test_idx in splitter.split(merged, y if task_type == "classification" else None):
            tr = merged.iloc[train_idx]
            te = merged.iloc[test_idx]
            y_tr = y[train_idx]

            vect = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                min_df=2,
                max_df=0.95,
                stop_words="english",
            )
            Xtr_text = vect.fit_transform(tr[text_col].tolist())
            Xte_text = vect.transform(te[text_col].tolist())

            if extra_cols:
                Xtr_extra, Xte_extra = _prepare_extra_matrix(tr, te, extra_cols)
                Xtr = sparse.hstack([Xtr_text, Xtr_extra], format="csr")
                Xte = sparse.hstack([Xte_text, Xte_extra], format="csr")
            else:
                Xtr, Xte = Xtr_text, Xte_text

            if task_type == "classification":
                if np.unique(y_tr).size < 2:
                    oof_score[test_idx] = float(np.mean(y_tr))
                    continue
                model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.8,
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=3000,
                )
                model.fit(Xtr, y_tr.astype(int))
                oof_score[test_idx] = model.predict_proba(Xte)[:, 1]
            else:
                model = Lasso(alpha=1e-3, max_iter=5000)
                model.fit(Xtr, y_tr)
                oof_score[test_idx] = model.predict(Xte)

    vect_full = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    X_text = vect_full.fit_transform(merged[text_col].tolist())

    if extra_cols:
        X_extra, _ = _prepare_extra_matrix(merged, merged, extra_cols)
        X_full = sparse.hstack([X_text, X_extra], format="csr")
    else:
        X_full = X_text

    if task_type == "classification":
        final_model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.8,
            class_weight="balanced",
            random_state=random_state,
            max_iter=3000,
        )
        final_model.fit(X_full, y.astype(int))
        if not compute_cv_predictions:
            oof_score = final_model.predict_proba(X_full)[:, 1]
        y_hat = (oof_score >= 0.5).astype(int)
        metrics = {
            "ROC-AUC": _safe_roc_auc(y.astype(int), oof_score),
            "Accuracy": float(accuracy_score(y.astype(int), y_hat)),
            "Precision": float(precision_score(y.astype(int), y_hat, zero_division=0)),
            "Recall": float(recall_score(y.astype(int), y_hat, zero_division=0)),
            "F1-Score": float(f1_score(y.astype(int), y_hat, zero_division=0)),
        }
    else:
        final_model = Lasso(alpha=1e-3, max_iter=5000)
        final_model.fit(X_full, y)
        if not compute_cv_predictions:
            oof_score = final_model.predict(X_full)
        mse = float(np.mean((y - oof_score) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        metrics = {
            "MSE": mse,
            "R2": float(1.0 - np.sum((y - oof_score) ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        }

    text_features = vect_full.get_feature_names_out().tolist()
    feature_names = text_features + extra_cols
    coefs = final_model.coef_.ravel()

    doc_freq = np.asarray((X_text > 0).sum(axis=0)).ravel().tolist()
    extra_doc_freq = [np.nan] * len(extra_cols)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "doc_frequency": doc_freq + extra_doc_freq,
            "log_doc_frequency": np.log1p(np.array(doc_freq + [0] * len(extra_cols), dtype=float)),
            "feature_type": ["ngram"] * len(text_features) + ["sentiment_or_structured"] * len(extra_cols),
        }
    )
    coef_df = coef_df[coef_df["coefficient"] != 0].copy().sort_values("coefficient", ascending=False)

    return {
        "coef_df": coef_df,
        "vectorizer": vect_full,
        "lasso": final_model,
        "y_true": y,
        "y_pred": oof_score,
        "alpha": getattr(final_model, "alpha", np.nan),
        "r2": metrics.get("R2", np.nan),
        "r2_train": metrics.get("R2", np.nan),
        "r2_oof": metrics.get("R2", np.nan),
        "kendall_tau": None,
        "kendall_tau_oof": None,
        "kendall_p": None,
        "target_col": target_col,
        "n_docs": len(merged),
        "metrics": metrics,
        "task_type": task_type,
    }


def plot_volcano(
    coef_df: pd.DataFrame,
    output_path: str,
    target_col: str = "target",
    top_n_labels: int = 15,
) -> None:
    if coef_df is None or len(coef_df) == 0:
        return

    apply_spotify_theme()
    df = coef_df[coef_df["feature_type"] == "ngram"].copy()
    if len(df) == 0:
        return

    df["color"] = np.where(df["coefficient"] >= 0, SPOTIFY_COLORS.get("accent", "#1DB954"), SPOTIFY_COLORS.get("negative", "#FF5A5F"))
    df["abs_coef"] = df["coefficient"].abs()
    to_label = pd.concat([df.nlargest(top_n_labels, "abs_coef"), df.nsmallest(top_n_labels, "abs_coef")]).drop_duplicates("feature")

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    ax.scatter(df["coefficient"], df["log_doc_frequency"], c=df["color"], alpha=0.65, s=28, linewidths=0)
    for _, row in to_label.iterrows():
        ax.annotate(row["feature"], xy=(row["coefficient"], row["log_doc_frequency"]), xytext=(4, 2), textcoords="offset points", fontsize=7, color=row["color"], alpha=0.85)

    ax.axvline(0, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("log(Document Frequency + 1)")
    ax.set_title(f"Volcano Plot (AI N-grams)\nTarget: {target_col}")
    style_axes(ax, grid_axis="y", grid_alpha=0.10)
    style_legend(ax)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)


def plot_top_coefficients(
    coef_df: pd.DataFrame,
    output_path: str,
    target_col: str = "target",
    top_n: int = 20,
) -> None:
    if coef_df is None or len(coef_df) == 0:
        return

    apply_spotify_theme()
    top = coef_df.assign(abs_coef=lambda x: x["coefficient"].abs()).nlargest(top_n * 2, "abs_coef")
    top = top.sort_values("coefficient")
    colors = [SPOTIFY_COLORS.get("negative", "#FF5A5F") if x < 0 else SPOTIFY_COLORS.get("accent", "#1DB954") for x in top["coefficient"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.32)))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    ax.barh(top["feature"], top["coefficient"], color=colors, alpha=0.85)
    ax.axvline(0, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linewidth=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Top Sparse Coefficients\nTarget: {target_col}")
    style_axes(ax, grid_axis="x", grid_alpha=0.10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, output_path: str, target_col: str) -> None:
    if np.unique(y_true).size < 2:
        return
    apply_spotify_theme()

    auc = _safe_roc_auc(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    ax.plot(fpr, tpr, color=SPOTIFY_COLORS.get("blue", "#4EA1FF"), linewidth=2.0, label=f"ROC-AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Forward R&D Increase ROC\nTarget: {target_col}")
    ax.legend(loc="lower right", fontsize=9)
    style_axes(ax, grid_axis="both", grid_alpha=0.10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)


def run_lasso_text_analysis(
    sentences_path: str,
    doc_metrics_path: str,
    initiation_scores_path: Optional[str] = None,
    regression_dataset_path: Optional[str] = None,
    output_dir: str = "outputs/figures",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    cv: int = 5,
    compute_cv_predictions: bool = True,
) -> Dict[str, Dict]:
    """End-to-end sparse text model for forward R&D investment prediction."""
    del initiation_scores_path

    os.makedirs(output_dir, exist_ok=True)

    sentences_df = pd.read_parquet(sentences_path)
    doc_metrics = pd.read_parquet(doc_metrics_path)

    corpus_df = _build_doc_corpus(sentences_df)
    sentiment_df = compute_ai_sentiment_features(sentences_df)

    target_df = _load_forward_rd_targets(
        doc_metrics_path=doc_metrics_path,
        doc_metrics_df=doc_metrics,
        regression_dataset_path=regression_dataset_path,
    )

    sentiment_path = os.path.join(output_dir, "ai_sentiment_features.csv")
    sentiment_df.to_csv(sentiment_path, index=False)

    targets: List[Tuple[str, str]] = []
    if "rd_increased_next_quarter" in target_df.columns:
        non_missing = target_df["rd_increased_next_quarter"].dropna()
        if len(non_missing) >= 12 and non_missing.nunique() >= 2:
            targets.append(("rd_increased_next_quarter", "classification"))
    if "y_next_rd_intensity_change" in target_df.columns:
        non_missing = target_df["y_next_rd_intensity_change"].dropna()
        if len(non_missing) >= 12:
            targets.append(("y_next_rd_intensity_change", "regression"))

    if not targets:
        raise ValueError("No valid forward-R&D targets available for Stage 11.")

    all_results: Dict[str, Dict] = {}
    summary_rows: List[Dict[str, float]] = []

    for target_col, task in targets:
        res = fit_lasso_ngram(
            corpus_df=corpus_df,
            target_df=target_df,
            target_col=target_col,
            max_features=max_features,
            ngram_range=ngram_range,
            cv=cv,
            compute_cv_predictions=compute_cv_predictions,
            extra_features_df=sentiment_df,
            task_type=task,
        )
        if not res:
            continue

        safe_name = target_col.replace("/", "_")
        coef_df = res["coef_df"]
        coef_csv = os.path.join(output_dir, f"lasso_coefs_{safe_name}.csv")
        coef_df.to_csv(coef_csv, index=False)

        plot_volcano(coef_df, os.path.join(output_dir, f"volcano_{safe_name}.png"), target_col=target_col)
        plot_top_coefficients(coef_df, os.path.join(output_dir, f"lasso_coef_bar_{safe_name}.png"), target_col=target_col)

        row = {
            "target": target_col,
            "task_type": task,
            "n_docs": res["n_docs"],
            "nonzero_features": int(len(coef_df)),
        }
        row.update(res["metrics"])

        if task == "classification":
            _plot_roc(
                y_true=res["y_true"].astype(int),
                y_score=res["y_pred"],
                output_path=os.path.join(output_dir, f"lasso_roc_{safe_name}.png"),
                target_col=target_col,
            )

        summary_rows.append(row)
        all_results[target_col] = res

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "lasso_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forward R&D text/sentiment sparse modeling")
    parser.add_argument("--sentences", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--initiation", default="outputs/features/initiation_scores.parquet")
    parser.add_argument("--regression-dataset", default=None)
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--skip-cv-pred", action="store_true")
    args = parser.parse_args()

    run_lasso_text_analysis(
        sentences_path=args.sentences,
        doc_metrics_path=args.metrics,
        initiation_scores_path=args.initiation,
        regression_dataset_path=args.regression_dataset,
        output_dir=args.output_dir,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        cv=args.cv,
        compute_cv_predictions=not args.skip_cv_pred,
    )
