"""
Regression Analysis Module

Cross-sectional regression analysis:
- DVs: AI Initiation Score (management proactiveness), Overall AI keyword ratio
- IVs: Financial and metadata features (no target-component leakage)

Evaluation:
- R² (in-sample, descriptive fit only)
- Kendall's Tau (out-of-sample fold-based rank correlation) — professor's preferred metric
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import kendalltau
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.iolib.summary2 import summary_col

from src.utils.doc_id import attach_doc_keys
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


def prepare_regression_data(
    initiation_scores_path: str,
    doc_metrics_path: str,
    wrds_data_path: str,
) -> pd.DataFrame:
    """
    Prepare data for regression analysis by merging metrics with financial data.
    """
    print("Loading data...")

    initiation = pd.read_parquet(initiation_scores_path)
    doc_metrics = pd.read_parquet(doc_metrics_path)
    wrds = pd.read_csv(wrds_data_path, low_memory=False)

    if len(initiation) == 0 or "doc_id" not in initiation.columns:
        print("Warning: No initiation scores available. Using doc_metrics only.")
        initiation = doc_metrics[["doc_id"]].copy()
        initiation["ai_initiation_score"] = 0.5
        initiation["total_ai_exchanges"] = 0
        initiation["analyst_initiated_ratio"] = 0.0
        initiation["management_pivot_ratio"] = 0.0

    initiation = attach_doc_keys(
        initiation,
        doc_id_col="doc_id",
        ticker_col="ticker",
        year_col="year",
        quarter_col="quarter",
        yearq_col="",
        keep_existing=False,
        raise_on_invalid=True,
    )

    doc_metrics = attach_doc_keys(
        doc_metrics,
        doc_id_col="doc_id",
        ticker_col="ticker",
        year_col="year",
        quarter_col="quarter",
        yearq_col="",
        keep_existing=False,
        raise_on_invalid=True,
    )

    init_cols = ["doc_id", "ticker", "year", "quarter"]
    for col in [
        "ai_initiation_score",
        "total_ai_exchanges",
        "analyst_initiated_ratio",
        "management_pivot_ratio",
        "analyst_only_count",
        "analyst_initiated_count",
        "management_pivot_count",
    ]:
        if col in initiation.columns:
            init_cols.append(col)
    merged = pd.merge(
        initiation[init_cols],
        doc_metrics[["doc_id", "speech_kw_ai_ratio", "qa_kw_ai_ratio", "overall_kw_ai_ratio"]],
        on="doc_id",
        how="left",
    )

    wrds = wrds.rename(columns={"tic": "ticker"})
    if "datadate" in wrds.columns:
        wrds["datadate"] = pd.to_datetime(wrds["datadate"], errors="coerce")

    qtr_col = "datacqtr" if "datacqtr" in wrds.columns else ("datafqtr" if "datafqtr" in wrds.columns else None)
    if qtr_col:
        wrds["wrds_year"] = wrds[qtr_col].astype(str).str[:4].astype(int)
        wrds["wrds_quarter"] = wrds[qtr_col].astype(str).str[-1].astype(int)

    wrds["rd_intensity"] = wrds["xrdq"] / wrds["mkvaltq"]
    wrds["rd_intensity"] = wrds["rd_intensity"].replace([np.inf, -np.inf], np.nan)
    wrds["log_mktcap"] = np.log(wrds["mkvaltq"].replace(0, np.nan))
    wrds["stock_price"] = wrds["prccq"]
    wrds["eps_positive"] = (wrds["epspxq"] > 0).astype(int)
    if "gsector" in wrds.columns:
        wrds["sector"] = wrds["gsector"].astype(str)

    wrds_cols = [
        "ticker",
        "wrds_year",
        "wrds_quarter",
        "rd_intensity",
        "log_mktcap",
        "stock_price",
        "eps_positive",
        "sector",
        "mkvaltq",
        "xrdq",
        "epspxq",
    ]
    wrds_subset = wrds[wrds_cols + (["datadate"] if "datadate" in wrds.columns else [])].copy()
    if "datadate" in wrds_subset.columns:
        wrds_subset = (
            wrds_subset.sort_values(["datadate"])
            .drop_duplicates(subset=["ticker", "wrds_year", "wrds_quarter"], keep="last")
            .drop(columns=["datadate"])
        )
    else:
        wrds_subset = wrds_subset.drop_duplicates(
            subset=["ticker", "wrds_year", "wrds_quarter"], keep="last"
        )

    final = pd.merge(
        merged,
        wrds_subset,
        left_on=["ticker", "year", "quarter"],
        right_on=["ticker", "wrds_year", "wrds_quarter"],
        how="left",
    )

    print(f"Final regression dataset: {len(final)} observations")
    print(f"Missing rd_intensity: {final['rd_intensity'].isna().sum()}")
    print(f"Missing log_mktcap: {final['log_mktcap'].isna().sum()}")

    return final


def _prepare_model_frame(
    df: pd.DataFrame,
    dv: str,
    ivs: Sequence[str],
    filter_non_ai_initiation: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a model-ready frame and a transparent attrition summary."""
    work = df.copy()
    total_rows = int(len(work))
    attrition: Dict[str, Any] = {
        "target": dv,
        "ivs": list(ivs),
        "rows_total": total_rows,
        "rows_removed_no_ai_filter": 0,
        "rows_after_no_ai_filter": total_rows,
        "rows_missing_dv": 0,
        "rows_final": 0,
        "missing_by_column": {},
    }

    if filter_non_ai_initiation and dv == "ai_initiation_score" and "total_ai_exchanges" in work.columns:
        mask = work["total_ai_exchanges"].fillna(0) > 0
        attrition["rows_removed_no_ai_filter"] = int((~mask).sum())
        work = work.loc[mask].copy()
        attrition["rows_after_no_ai_filter"] = int(len(work))

    if dv in work.columns:
        attrition["rows_missing_dv"] = int(work[dv].isna().sum())

    needed_cols = [dv] + list(ivs)
    for col in needed_cols:
        attrition["missing_by_column"][col] = int(work[col].isna().sum()) if col in work.columns else int(len(work))

    reg_df = work.dropna(subset=needed_cols).copy()
    attrition["rows_final"] = int(len(reg_df))
    return reg_df, attrition


def _print_attrition(label: str, attrition: Dict[str, Any]) -> None:
    print(f"  [{label}] Sample attrition for target={attrition['target']}")
    print(f"    total rows: {attrition['rows_total']}")
    print(f"    removed no-AI initiation rows: {attrition['rows_removed_no_ai_filter']}")
    print(f"    rows after no-AI filter: {attrition['rows_after_no_ai_filter']}")
    print(f"    missing DV rows: {attrition['rows_missing_dv']}")
    for col, miss in attrition.get("missing_by_column", {}).items():
        print(f"    missing {col}: {miss}")
    print(f"    final regression sample: {attrition['rows_final']}")


def _iter_oos_splits(
    reg_df: pd.DataFrame,
    group_col: Optional[str] = "ticker",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    n_splits = max(2, int(n_splits))
    if len(reg_df) < n_splits:
        n_splits = max(2, len(reg_df))

    if group_col and group_col in reg_df.columns:
        groups = reg_df[group_col].fillna("__MISSING_GROUP__").astype(str)
        n_groups = groups.nunique()
        if n_groups >= n_splits and n_splits >= 2:
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(reg_df, groups=groups)), f"GroupKFold({group_col})"
        print(
            f"  [OOS] Falling back to KFold: insufficient groups for {group_col} "
            f"(groups={n_groups}, n_splits={n_splits})"
        )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(reg_df)), "KFold"


def _fit_statsmodels_ols(
    train_df: pd.DataFrame,
    dv: str,
    ivs: Sequence[str],
    add_constant: bool = True,
    robust: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    y_train = train_df[dv]
    X_train = train_df[list(ivs)]
    if add_constant:
        X_train = sm.add_constant(X_train, has_constant="add")
    model = sm.OLS(y_train, X_train)
    return model.fit(cov_type="HC1") if robust else model.fit()


def compute_kendall_tau_oos(
    df: pd.DataFrame,
    dv: str,
    ivs: Sequence[str],
    add_constant: bool = True,
    robust: bool = True,
    group_col: Optional[str] = "ticker",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute OOS Kendall's tau using fold-based held-out predictions."""
    reg_df = df.dropna(subset=[dv] + list(ivs)).copy()
    if len(reg_df) < 3:
        return {
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "n_obs": int(len(reg_df)),
            "split_method": "insufficient-data",
            "oof_predictions": np.array([], dtype=float),
        }

    effective_splits = min(max(2, n_splits), len(reg_df))
    splits, split_method = _iter_oos_splits(
        reg_df, group_col=group_col, n_splits=effective_splits, random_state=random_state
    )

    y_true = reg_df[dv].to_numpy(dtype=float)
    y_oof = np.full(len(reg_df), np.nan, dtype=float)

    for train_idx, test_idx in splits:
        train_df = reg_df.iloc[train_idx]
        test_df = reg_df.iloc[test_idx]
        if len(train_df) < max(3, len(ivs) + 1):
            y_oof[test_idx] = float(train_df[dv].mean()) if len(train_df) else float(np.nanmean(y_true))
            continue

        try:
            fitted = _fit_statsmodels_ols(train_df, dv=dv, ivs=ivs, add_constant=add_constant, robust=robust)
            X_test = test_df[list(ivs)]
            if add_constant:
                X_test = sm.add_constant(X_test, has_constant="add")
                X_test = X_test.reindex(columns=fitted.model.exog_names, fill_value=0.0)
                if "const" in X_test.columns:
                    X_test["const"] = 1.0
            preds = np.asarray(fitted.predict(X_test), dtype=float)
        except Exception:
            preds = np.full(len(test_df), float(train_df[dv].mean()), dtype=float)

        y_oof[test_idx] = preds

    valid = ~np.isnan(y_oof)
    if valid.sum() < 2:
        tau = np.nan
        p_val = np.nan
    else:
        tau, p_val = kendalltau(y_true[valid], y_oof[valid])

    return {
        "kendall_tau": float(tau) if tau is not None and not np.isnan(tau) else np.nan,
        "kendall_p": float(p_val) if p_val is not None and not np.isnan(p_val) else np.nan,
        "n_obs": int(valid.sum()),
        "split_method": split_method,
        "oof_predictions": y_oof,
    }


def run_regression(
    df: pd.DataFrame,
    dv: str,
    ivs: List[str],
    add_constant: bool = True,
    robust: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run OLS regression on an already-filtered model frame."""
    reg_df = df[[dv] + ivs].dropna()
    print(f"Regression sample size: {len(reg_df)}")
    return _fit_statsmodels_ols(reg_df, dv=dv, ivs=ivs, add_constant=add_constant, robust=robust)


def run_regression_analysis(
    initiation_scores_path: str,
    doc_metrics_path: str,
    wrds_data_path: str,
    output_dir: str = "outputs/figures",
    features_output_dir: Optional[str] = None,
    oos_group_col: Optional[str] = "ticker",
    oos_cv_folds: int = 5,
    filter_non_ai_initiation: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Full regression analysis pipeline with OOS Kendall Tau and attrition logging."""
    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    reg_df = prepare_regression_data(initiation_scores_path, doc_metrics_path, wrds_data_path)
    features_out_dir = (
        os.path.normpath(features_output_dir)
        if features_output_dir is not None
        else os.path.normpath(os.path.join(output_dir, "..", "features"))
    )
    os.makedirs(features_out_dir, exist_ok=True)
    reg_df.to_parquet(os.path.join(features_out_dir, "regression_dataset.parquet"), index=False)

    results: Dict[str, Any] = {}
    ivs_base = ["log_mktcap", "rd_intensity"]
    ivs_fin = ["log_mktcap", "rd_intensity", "eps_positive"]

    attrition_records: List[Dict[str, Any]] = []
    oos_metrics: Dict[str, Dict[str, Any]] = {}

    def _run_model(label: str, result_key: str, dv: str, ivs: List[str]) -> None:
        print("\n" + "=" * 60)
        print(f"{label}: {dv} ~ {', '.join(ivs)}")
        print("=" * 60)

        model_df, attrition = _prepare_model_frame(
            reg_df,
            dv=dv,
            ivs=ivs,
            filter_non_ai_initiation=filter_non_ai_initiation,
        )
        attrition_row = {
            "model_label": label,
            "target": dv,
            "ivs": "|".join(ivs),
            "rows_total": attrition["rows_total"],
            "rows_removed_no_ai_filter": attrition["rows_removed_no_ai_filter"],
            "rows_after_no_ai_filter": attrition["rows_after_no_ai_filter"],
            "rows_missing_dv": attrition["rows_missing_dv"],
            "rows_final": attrition["rows_final"],
        }
        for miss_col, miss_val in attrition.get("missing_by_column", {}).items():
            attrition_row[f"missing_{miss_col}"] = miss_val
        attrition_records.append(attrition_row)
        _print_attrition(label, attrition)

        if len(model_df) < max(10, len(ivs) + 2):
            print(f"  [Skipping {label}] insufficient rows after filtering/dropna: {len(model_df)}")
            return

        model = run_regression(model_df, dv=dv, ivs=ivs)
        print(model.summary())
        results[result_key] = model

        oos = compute_kendall_tau_oos(
            model_df,
            dv=dv,
            ivs=ivs,
            group_col=oos_group_col,
            n_splits=oos_cv_folds,
            random_state=random_state,
        )
        oos_metrics[result_key] = oos
        tau = oos.get("kendall_tau")
        p_val = oos.get("kendall_p")
        split_method = oos.get("split_method")
        print(
            f"  [{label}] OOS Kendall's Tau = "
            f"{(f'{tau:.4f}' if pd.notna(tau) else 'nan')} "
            f"(p = {(f'{p_val:.4f}' if pd.notna(p_val) else 'nan')}, {split_method})"
        )

    _run_model("Model 1", "model1", dv="ai_initiation_score", ivs=ivs_base)
    _run_model("Model 2", "model2", dv="ai_initiation_score", ivs=ivs_fin)
    _run_model("Model 3", "model3", dv="overall_kw_ai_ratio", ivs=ivs_fin)

    model_list = [results[k] for k in ["model1", "model2", "model3"] if k in results]
    model_labels = [
        lab for key, lab in [("model1", "AI Init (1)"), ("model2", "AI Init (2)"), ("model3", "AI Ratio (3)")] if key in results
    ]
    if model_list:
        summary = summary_col(model_list, stars=True, float_format="%0.4f", model_names=model_labels)
        summary_path = f"{output_dir}/regression_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(str(summary))
            if oos_metrics:
                f.write("\n\nOOS Kendall Tau (Group-aware CV where possible)\n")
                for key, metrics in oos_metrics.items():
                    f.write(
                        f"{key}: tau={metrics['kendall_tau']}, p={metrics['kendall_p']}, "
                        f"n={metrics['n_obs']}, split={metrics['split_method']}\n"
                    )
        print(f"\nSaved regression summary -> {summary_path}")

    attrition_df = pd.DataFrame(attrition_records)
    if len(attrition_df) > 0:
        attrition_path = f"{output_dir}/regression_sample_attrition.csv"
        attrition_df.to_csv(attrition_path, index=False)
        print(f"Saved regression attrition summary -> {attrition_path}")

    if "model2" in results:
        plot_coefficients(results, f"{output_dir}/regression_coefficients.png")

    results["oos_metrics"] = oos_metrics
    results["attrition"] = attrition_df
    return results


def plot_coefficients(results: Dict[str, Any], output_path: str) -> None:
    """Plot regression coefficients with confidence intervals."""
    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    model = results["model2"]
    params = model.params.drop("const", errors="ignore")
    conf_int = model.conf_int().drop("const", errors="ignore")
    if len(params) == 0:
        plt.close(fig)
        return

    y_pos = np.arange(len(params))
    colors = [SPOTIFY_COLORS.get("accent", "#1DB954") if v >= 0 else SPOTIFY_COLORS.get("negative", "#FF5A5F") for v in params.values]

    ax.barh(
        y_pos,
        params.values,
        xerr=[params.values - conf_int[0].values, conf_int[1].values - params.values],
        capsize=4,
        color=colors,
        edgecolor=SPOTIFY_COLORS.get("grid", "#2A2A2A"),
        alpha=0.95,
    )
    ax.axvline(x=0, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linestyle="-", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params.index)
    ax.set_xlabel("Coefficient Estimate")
    ax.set_title("Regression Coefficients: AI Initiation Score (Model 2)")
    style_axes(ax, grid_axis="x", grid_alpha=0.15)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved coefficient plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regression analysis")
    parser.add_argument("--scores", default="outputs/features/initiation_scores.parquet", help="Initiation scores file")
    parser.add_argument("--metrics", default="outputs/features/document_metrics.parquet", help="Document metrics file")
    parser.add_argument("--wrds", default="data/wrds.csv")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--oos-group-col", default="ticker")
    parser.add_argument("--oos-cv-folds", type=int, default=5)
    parser.add_argument("--no-filter-non-ai-initiation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_regression_analysis(
        args.initiation,
        args.doc_metrics,
        args.wrds,
        args.output_dir,
        oos_group_col=(None if str(args.oos_group_col).lower() in {"none", ""} else args.oos_group_col),
        oos_cv_folds=args.oos_cv_folds,
        filter_non_ai_initiation=not args.no_filter_non_ai_initiation,
        random_state=args.seed,
    )
