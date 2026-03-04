"""
Stage 10 benchmark comparison rewritten as outperformance classification.

Core idea:
- Build a binary target `beats_sector_median` from next-quarter outcomes.
- Compare classification models under shared OOS folds.
- Report ROC-AUC / Accuracy / Precision / Recall / F1 instead of regression errors.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.doc_id import attach_doc_keys
from src.utils.ml_helpers import aggregate_doc_text, safe_roc_auc
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


DEFAULT_METADATA_FEATURES = [
    "log_mktcap",
    "rd_intensity",
    "eps_positive",
    "stock_price",
    "year",
    "quarter",
    "sector",
]
DEFAULT_TEXT_RATIO_FEATURES = [
    "speech_kw_ai_ratio",
    "qa_kw_ai_ratio",
    "overall_kw_ai_ratio",
]


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return safe_roc_auc(y_true, y_prob, fallback=0.5)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _aggregate_doc_text(sentences_df: pd.DataFrame, section: Optional[str] = "qa") -> pd.DataFrame:
    return aggregate_doc_text(
        sentences_df,
        section=section,
        ai_only=False,
        mask_non_ai=False,
        output_col="doc_text",
        sort=False,
    )


def _choose_metadata_columns(df: pd.DataFrame, requested: Optional[Sequence[str]]) -> List[str]:
    candidates = list(requested) if requested is not None else DEFAULT_METADATA_FEATURES
    return [c for c in candidates if c in df.columns]


def _iter_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    group_col: Optional[str],
    n_splits: int,
    random_state: int,
) -> Tuple[Iterable[Tuple[np.ndarray, np.ndarray]], str]:
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("__MISSING_GROUP__").astype(str)
        n_groups = groups.nunique()
        if n_groups >= n_splits and n_splits >= 2:
            splitter = GroupKFold(n_splits=n_splits)
            return splitter.split(df, y=y, groups=groups), f"GroupKFold({group_col})"

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return splitter.split(df, y), "StratifiedKFold"


def _parse_doc_id_quarter(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_binary_target(regression_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str]:
    base = _parse_doc_id_quarter(regression_df.copy())

    if target_col in base.columns:
        candidate = pd.to_numeric(base[target_col], errors="coerce")
        uniq = sorted(set(candidate.dropna().astype(int).tolist()))
        if set(uniq).issubset({0, 1}) and len(uniq) >= 1:
            base[target_col] = candidate.astype(float)
            return base[base[target_col].notna()].copy(), f"target_preexisting_{target_col}"

    if "y_next_mktcap_growth" not in base.columns:
        if {"mkvaltq", "ticker", "year", "quarter"}.issubset(base.columns):
            base = base.sort_values(["ticker", "year", "quarter"]).copy()
            base["y_next_mktcap_growth"] = (
                base.groupby("ticker", sort=False)["mkvaltq"].shift(-1) / base["mkvaltq"] - 1.0
            )

    if "y_next_mktcap_growth" in base.columns and {"year", "quarter", "sector"}.issubset(base.columns):
        growth = pd.to_numeric(base["y_next_mktcap_growth"], errors="coerce")
        med = base.assign(_growth=growth).groupby(["year", "quarter", "sector"])["_growth"].transform("median")
        beats = np.where(growth.notna() & med.notna(), (growth > med).astype(float), np.nan)
        base["beats_sector_median"] = beats
        out = base[base["beats_sector_median"].notna()].copy()
        if len(out) > 0 and out["beats_sector_median"].nunique() >= 2:
            return out, "target_growth_vs_sector_median"

    if "y_next_eps_growth_yoy" in base.columns:
        y_eps = pd.to_numeric(base["y_next_eps_growth_yoy"], errors="coerce")
        base["beats_sector_median"] = np.where(y_eps.notna(), (y_eps > 0).astype(float), np.nan)
        out = base[base["beats_sector_median"].notna()].copy()
        if len(out) > 0 and out["beats_sector_median"].nunique() >= 2:
            return out, "target_eps_growth_positive"

    raise ValueError(
        "Cannot construct binary outperformance target. Need one of: "
        "precomputed binary target, or mkvaltq+ticker+year+quarter+sector, or y_next_eps_growth_yoy."
    )


def _build_metadata_pipeline(X_train_meta: pd.DataFrame, model_kind: str, random_state: int = 42) -> Pipeline:
    num_cols = [c for c in X_train_meta.columns if pd.api.types.is_numeric_dtype(X_train_meta[c])]
    cat_cols = [c for c in X_train_meta.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    if model_kind == "logit_l1":
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.6,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
    elif model_kind == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    return Pipeline([("pre", pre), ("model", model)])


def _predict_text_logit(
    train_text: Sequence[str],
    y_train: np.ndarray,
    test_text: Sequence[str],
    text_max_features: int,
    random_state: int,
) -> np.ndarray:
    train_text = pd.Series(train_text).fillna("").astype(str)
    test_text = pd.Series(test_text).fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=text_max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train_text.tolist())
    X_test = vectorizer.transform(test_text.tolist())

    if X_train.shape[1] == 0 or np.unique(y_train).size < 2:
        return np.full(len(test_text), float(np.mean(y_train)))

    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.8,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def _predict_text_ratio_logit(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    test_df: pd.DataFrame,
    text_ratio_features: Sequence[str],
    random_state: int,
) -> np.ndarray:
    cols = [c for c in text_ratio_features if c in train_df.columns]
    if not cols or np.unique(y_train).size < 2:
        return np.full(len(test_df), float(np.mean(y_train)))

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=0.8,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(train_df[cols], y_train)
    return pipe.predict_proba(test_df[cols])[:, 1]


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "ROC-AUC": _safe_roc_auc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def evaluate_benchmark_models(
    regression_df: pd.DataFrame,
    sentences_df: Optional[pd.DataFrame] = None,
    target_col: str = "beats_sector_median",
    group_col: str = "ticker",
    n_splits: int = 5,
    random_state: int = 42,
    text_section: Optional[str] = "qa",
    text_max_features: int = 3000,
    metadata_features: Optional[Sequence[str]] = None,
    include_metadata_elasticnet: bool = True,
    text_model_mode: str = "raw",
    filter_non_ai_initiation: bool = True,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate classification benchmarks using shared OOS folds."""
    del include_metadata_elasticnet, filter_non_ai_initiation  # kept for backward-compatible signature

    if "doc_id" not in regression_df.columns:
        raise ValueError("regression_df must include 'doc_id'")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    base, target_source = _build_binary_target(regression_df, target_col=target_col)
    base = base[base["doc_id"].notna()].copy().reset_index(drop=True)
    base[target_col] = base["beats_sector_median"].astype(int)

    if text_model_mode == "raw":
        text_df = _aggregate_doc_text(sentences_df, section=text_section) if sentences_df is not None else pd.DataFrame(columns=["doc_id", "doc_text"])
        base = base.merge(text_df, on="doc_id", how="left")
        base["doc_text"] = base.get("doc_text", "").fillna("").astype(str)
    elif text_model_mode != "ratios":
        raise ValueError("text_model_mode must be 'raw' or 'ratios'")

    if len(base) < n_splits:
        raise ValueError(f"Not enough rows ({len(base)}) for n_splits={n_splits}")
    if base[target_col].nunique() < 2:
        raise ValueError("Binary target has only one class after filtering")

    meta_cols = _choose_metadata_columns(base, metadata_features)
    if len(meta_cols) == 0:
        raise ValueError("No metadata features available for benchmark evaluation")

    y_full = base[target_col].to_numpy(dtype=int)
    split_iter, split_method = _iter_splits(base, y_full, group_col=group_col, n_splits=n_splits, random_state=random_state)

    fold_records: List[Dict] = []
    roc_payload: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"y_true": [], "y_score": []})

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_df = base.iloc[train_idx].copy()
        test_df = base.iloc[test_idx].copy()
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)

        if verbose:
            print(f"[Benchmark] Fold {fold_idx}/{n_splits}: train={len(train_df)} test={len(test_df)}")

        X_train_meta = train_df[meta_cols].copy()
        X_test_meta = test_df[meta_cols].copy()

        train_groups = set(train_df[group_col].fillna("__MISSING_GROUP__").astype(str)) if group_col in train_df.columns else set()
        test_groups = set(test_df[group_col].fillna("__MISSING_GROUP__").astype(str)) if group_col in test_df.columns else set()
        group_overlap_count = len(train_groups & test_groups) if group_col in base.columns else 0

        model_probs: Dict[str, np.ndarray] = {}

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_df), 1)), y_train)
        model_probs["Dummy Majority"] = dummy.predict_proba(np.zeros((len(test_df), 1)))[:, 1]

        try:
            m_logit = _build_metadata_pipeline(X_train_meta, model_kind="logit_l1", random_state=random_state)
            m_logit.fit(X_train_meta, y_train)
            model_probs["Metadata Logistic-L1"] = m_logit.predict_proba(X_test_meta)[:, 1]
        except Exception:
            model_probs["Metadata Logistic-L1"] = np.full(len(test_df), float(np.mean(y_train)))

        try:
            m_rf = _build_metadata_pipeline(X_train_meta, model_kind="rf", random_state=random_state)
            m_rf.fit(X_train_meta, y_train)
            model_probs["Metadata RandomForest"] = m_rf.predict_proba(X_test_meta)[:, 1]
        except Exception:
            model_probs["Metadata RandomForest"] = np.full(len(test_df), float(np.mean(y_train)))

        text_label = "Text TF-IDF Logistic-L1" if text_model_mode == "raw" else "Text-Ratio Logistic-L1"
        try:
            if text_model_mode == "raw":
                model_probs[text_label] = _predict_text_logit(
                    train_text=train_df["doc_text"],
                    y_train=y_train,
                    test_text=test_df["doc_text"],
                    text_max_features=text_max_features,
                    random_state=random_state,
                )
            else:
                model_probs[text_label] = _predict_text_ratio_logit(
                    train_df=train_df,
                    y_train=y_train,
                    test_df=test_df,
                    text_ratio_features=DEFAULT_TEXT_RATIO_FEATURES,
                    random_state=random_state,
                )
        except Exception:
            model_probs[text_label] = np.full(len(test_df), float(np.mean(y_train)))

        for model_name, y_prob in model_probs.items():
            y_prob = np.asarray(y_prob, dtype=float)
            metrics = _classification_metrics(y_test, y_prob)
            fold_records.append(
                {
                    "Fold": fold_idx,
                    "Model": model_name,
                    "ROC-AUC": metrics["ROC-AUC"],
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1-Score": metrics["F1-Score"],
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "group_overlap_count": int(group_overlap_count),
                    "split_method": split_method,
                    "target_col": target_col,
                    "target_source": target_source,
                    "positive_rate_test": float(np.mean(y_test)),
                }
            )
            roc_payload[model_name]["y_true"].extend(y_test.tolist())
            roc_payload[model_name]["y_score"].extend(y_prob.tolist())

    folds_df = pd.DataFrame(fold_records)
    if len(folds_df) == 0:
        raise RuntimeError("No fold results were produced")

    summary = (
        folds_df.groupby("Model", as_index=False)
        .agg(
            **{
                "ROC-AUC_mean": ("ROC-AUC", "mean"),
                "ROC-AUC_std": ("ROC-AUC", "std"),
                "Accuracy_mean": ("Accuracy", "mean"),
                "Accuracy_std": ("Accuracy", "std"),
                "Precision_mean": ("Precision", "mean"),
                "Precision_std": ("Precision", "std"),
                "Recall_mean": ("Recall", "mean"),
                "Recall_std": ("Recall", "std"),
                "F1-Score_mean": ("F1-Score", "mean"),
                "F1-Score_std": ("F1-Score", "std"),
                "Folds": ("Fold", "count"),
            }
        )
        .sort_values(["ROC-AUC_mean", "F1-Score_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
    std_cols = [c for c in summary.columns if c.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0.0)
    summary.attrs["roc_payload"] = {k: {"y_true": np.array(v["y_true"]), "y_score": np.array(v["y_score"])} for k, v in roc_payload.items()}

    return folds_df, summary


def _plot_benchmark_comparison(summary_df: pd.DataFrame, output_png: str) -> None:
    apply_spotify_theme()
    plot_df = summary_df.copy()
    if len(plot_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    y_pos = np.arange(len(plot_df))

    axes[0].barh(
        y_pos,
        plot_df["ROC-AUC_mean"].values,
        xerr=plot_df["ROC-AUC_std"].values,
        color=SPOTIFY_COLORS.get("blue", "#4EA1FF"),
        alpha=0.9,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["Model"].tolist())
    axes[0].invert_yaxis()
    axes[0].set_title("ROC-AUC (higher is better)")
    axes[0].set_xlabel("ROC-AUC")
    style_axes(axes[0], grid_axis="x", grid_alpha=0.12)

    axes[1].barh(
        y_pos,
        plot_df["F1-Score_mean"].values,
        xerr=plot_df["F1-Score_std"].values,
        color=SPOTIFY_COLORS.get("accent", "#1DB954"),
        alpha=0.9,
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(plot_df["Model"].tolist())
    axes[1].invert_yaxis()
    axes[1].set_title("F1-Score (higher is better)")
    axes[1].set_xlabel("F1-Score")
    style_axes(axes[1], grid_axis="x", grid_alpha=0.12)

    fig.suptitle("Benchmark Classification Comparison")
    fig.tight_layout()
    save_figure(fig, output_png, dpi=180)


def _plot_roc_curves(summary_df: pd.DataFrame, output_png: str) -> None:
    apply_spotify_theme()
    payload = summary_df.attrs.get("roc_payload", {}) or {}
    if not payload:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    for model in summary_df["Model"].tolist():
        y_true = np.asarray(payload.get(model, {}).get("y_true", []), dtype=int)
        y_score = np.asarray(payload.get(model, {}).get("y_score", []), dtype=float)
        if y_true.size == 0 or np.unique(y_true).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = _safe_roc_auc(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{model} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    style_axes(ax, grid_axis="both", grid_alpha=0.12)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_png, dpi=180)


def write_benchmark_outputs(
    folds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    prefix: str = "benchmark_comparison",
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    folds_csv = os.path.join(output_dir, f"{prefix}_folds.csv")
    summary_csv = os.path.join(output_dir, f"{prefix}_summary.csv")
    plot_png = os.path.join(output_dir, f"{prefix}.png")
    roc_png = os.path.join(output_dir, f"{prefix}_roc.png")

    folds_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _plot_benchmark_comparison(summary_df, plot_png)
    _plot_roc_curves(summary_df, roc_png)

    return {
        "folds_csv": folds_csv,
        "summary_csv": summary_csv,
        "plot_png": plot_png,
        "roc_plot_png": roc_png,
    }


def run_benchmark_comparison(
    regression_dataset_path: str,
    sentences_path: str,
    output_dir: str = "outputs/figures",
    target_col: str = "beats_sector_median",
    group_col: str = "ticker",
    n_splits: int = 5,
    random_state: int = 42,
    text_section: Optional[str] = "qa",
    text_max_features: int = 3000,
    include_metadata_elasticnet: bool = True,
    text_model_mode: str = "ratios",
    filter_non_ai_initiation: bool = True,
    verbose: bool = True,
) -> Dict[str, str]:
    reg_df = pd.read_parquet(regression_dataset_path)
    sentences_df = pd.read_parquet(sentences_path) if os.path.exists(sentences_path) else None
    folds_df, summary_df = evaluate_benchmark_models(
        regression_df=reg_df,
        sentences_df=sentences_df,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        random_state=random_state,
        text_section=text_section,
        text_max_features=text_max_features,
        include_metadata_elasticnet=include_metadata_elasticnet,
        text_model_mode=text_model_mode,
        filter_non_ai_initiation=filter_non_ai_initiation,
        verbose=verbose,
    )
    paths = write_benchmark_outputs(folds_df, summary_df, output_dir=output_dir)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark classification for outperformance")
    parser.add_argument("--regression-dataset", default="outputs/features/regression_dataset.parquet")
    parser.add_argument("--sentences", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--target-col", default="beats_sector_median")
    parser.add_argument("--group-col", default="ticker")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-section", default="qa", choices=["qa", "speech", "all"])
    parser.add_argument("--text-max-features", type=int, default=3000)
    parser.add_argument("--text-model", default="ratios", choices=["ratios", "raw"])
    args = parser.parse_args()

    section = None if args.text_section == "all" else args.text_section
    paths = run_benchmark_comparison(
        regression_dataset_path=args.regression_dataset,
        sentences_path=args.sentences,
        output_dir=args.output_dir,
        target_col=args.target_col,
        group_col=args.group_col,
        n_splits=args.cv_folds,
        random_state=args.seed,
        text_section=section,
        text_max_features=args.text_max_features,
        text_model_mode=args.text_model,
    )
    print("Saved benchmark outputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
