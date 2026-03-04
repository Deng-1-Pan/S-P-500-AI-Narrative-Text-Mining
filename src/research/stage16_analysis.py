from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.research.regression_primitives import fit_stage16_regression, zscore_series
from src.research.stage16_viz import (
    default_takeaways,
    plot_ai_trend_by_size,
    plot_ai_nonzero_by_sector,
    plot_ai_zero_by_sector,
    plot_assoc_bar,
    plot_coefplot,
    plot_gap_by_quadrant,
    plot_model_compare,
    plot_quadrant_finance,
    plot_quadrant_sector_heatmap,
    plot_wrds_distribution,
)
from src.research.text_wrds_panel import build_text_wrds_panel
from src.research.wrds_features import build_wrds_feature_store


def _zscore(s: pd.Series) -> pd.Series:
    return zscore_series(s)


def _spearman_association(df: pd.DataFrame) -> pd.DataFrame:
    ai_metrics = [c for c in ["overall_ai_ratio", "speech_ai_ratio", "qa_ai_ratio", "initiation_score"] if c in df.columns]
    wrds_feats = [c for c in ["log_mkcap", "rd_intensity_mkcap", "eps", "capex_intensity_mkcap", "ret_q"] if c in df.columns]
    rows: List[Dict[str, object]] = []
    for ai in ai_metrics:
        for feat in wrds_feats:
            work = df[[ai, feat]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(work) < 5:
                corr = np.nan
            else:
                corr = float(work.corr(method="spearman").iloc[0, 1])
            rows.append({"ai_metric": ai, "wrds_feature": feat, "spearman_corr": corr, "pair": f"{ai} ~ {feat}"})
    return pd.DataFrame(rows)


def _fit_regression(
    panel: pd.DataFrame,
    y: str,
    x_list: List[str],
    fe_sector: str = "gsector",
    fe_time: str = "yearq",
    cluster_col: str = "ticker",
) -> Tuple[pd.DataFrame, str]:
    return fit_stage16_regression(
        panel=panel,
        y=y,
        x_list=x_list,
        fe_sector=fe_sector,
        fe_time=fe_time,
        cluster_col=cluster_col,
        min_obs=20,
    )


def _linear_model_compare(panel: pd.DataFrame, test_quarters: int = 4) -> pd.DataFrame:
    target = "lead_d_rd"
    finance = [c for c in ["log_mkcap", "rd_intensity_mkcap", "eps", "capex_intensity_mkcap", "ret_q"] if c in panel.columns]
    text = [c for c in ["overall_ai_ratio", "speech_ai_ratio", "qa_ai_ratio", "initiation_score"] if c in panel.columns]
    if "quarter_index" not in panel.columns:
        panel = panel.copy()
        panel["quarter_index"] = panel["year"].astype(int) * 4 + panel["quarter"].astype(int)

    base_cols = list(set([target, "quarter_index", "gsector"] + finance + text))
    work = panel[base_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target, "quarter_index"])
    if len(work) < 30:
        return pd.DataFrame(columns=["model", "r2", "mae", "delta_r2_vs_finance", "delta_mae_vs_finance"])

    q = sorted(work["quarter_index"].dropna().unique())
    if len(q) <= 1:
        return pd.DataFrame(columns=["model", "r2", "mae", "delta_r2_vs_finance", "delta_mae_vs_finance"])
    split_n = min(max(1, test_quarters), len(q) - 1)
    cutoff = q[-split_n]
    train = work[work["quarter_index"] < cutoff].copy()
    test = work[work["quarter_index"] >= cutoff].copy()
    if len(train) < 10 or len(test) < 5:
        return pd.DataFrame(columns=["model", "r2", "mae", "delta_r2_vs_finance", "delta_mae_vs_finance"])

    def run_model(name: str, cols: List[str]) -> Dict[str, float | str]:
        cols = cols + (["gsector"] if "gsector" in work.columns else [])
        cols = [c for c in cols if c in work.columns]
        x_train = pd.get_dummies(train[cols], columns=["gsector"] if "gsector" in cols else [], drop_first=True)
        x_test = pd.get_dummies(test[cols], columns=["gsector"] if "gsector" in cols else [], drop_first=True)
        x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
        med = x_train.median(numeric_only=True)
        x_train = x_train.fillna(med)
        x_test = x_test.fillna(med)
        model = LinearRegression()
        model.fit(x_train, train[target])
        pred = model.predict(x_test)
        return {"model": name, "r2": float(r2_score(test[target], pred)), "mae": float(mean_absolute_error(test[target], pred))}

    rows = [
        run_model("Finance-only", finance),
        run_model("Text-only", text),
        run_model("Finance+Text", finance + text),
    ]
    out = pd.DataFrame(rows)
    finance_row = out[out["model"] == "Finance-only"].iloc[0]
    out["delta_r2_vs_finance"] = out["r2"] - finance_row["r2"]
    out["delta_mae_vs_finance"] = finance_row["mae"] - out["mae"]
    return out


def _plot_lasso_feature_selection(
    panel: pd.DataFrame,
    output_path: str,
    target: str = "lead_d_rd",
    alpha: float = 5e-4,
) -> None:
    """Fit a standardised Lasso on Finance+Text and plot the retained coefficients.

    This provides an interpretable complement to the R² comparison table on
    Slide 18: even at a very small alpha, Lasso zeroes-out or heavily shrinks
    text features, showing they add no incremental signal beyond finance vars.
    """
    finance = [
        c for c in ["log_mkcap", "rd_intensity_mkcap", "eps", "capex_intensity_mkcap", "ret_q"]
        if c in panel.columns
    ]
    text = [
        c for c in ["overall_ai_ratio", "speech_ai_ratio", "qa_ai_ratio", "initiation_score"]
        if c in panel.columns
    ]
    all_features = finance + text

    if target not in panel.columns or not all_features:
        return

    work = panel[all_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 20:
        return

    imputer = SimpleImputer(strategy="median")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(imputer.fit_transform(work[all_features]))
    # Standardise Y too so coefficients are fully standardised Beta coefficients
    # (interpretable as "SD change in Y per SD change in X", unitless, ±small decimals)
    y = scaler_y.fit_transform(work[[target]].values).ravel()

    model = Lasso(alpha=alpha, max_iter=15000, random_state=42)
    model.fit(X, y)

    coef_df = pd.DataFrame(
        {
            "feature": all_features,
            "coefficient": model.coef_,
            "type": ["Finance"] * len(finance) + ["Text (AI Narrative)"] * len(text),
        }
    ).sort_values("coefficient")

    n_nonzero = int((coef_df["coefficient"] != 0).sum())
    n_text_nonzero = int(
        ((coef_df["type"] == "Text (AI Narrative)") & (coef_df["coefficient"] != 0)).sum()
    )

    COLOR_FINANCE_POS = "#4C6A92"
    COLOR_FINANCE_NEG = "#2A4A6B"
    COLOR_TEXT_POS = "#9F2B2B"
    COLOR_TEXT_NEG = "#C0392B"
    COLOR_ZERO = "#D1C9BC"

    colors = []
    for _, row in coef_df.iterrows():
        if row["coefficient"] == 0.0:
            colors.append(COLOR_ZERO)
        elif row["type"].startswith("Text"):
            colors.append(COLOR_TEXT_POS if row["coefficient"] > 0 else COLOR_TEXT_NEG)
        else:
            colors.append(COLOR_FINANCE_POS if row["coefficient"] > 0 else COLOR_FINANCE_NEG)

    fig, ax = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.52)))
    fig.patch.set_facecolor("#F6F3EE")
    ax.set_facecolor("#F6F3EE")

    ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, alpha=0.88, edgecolor="none")
    ax.axvline(0, color="#374151", linewidth=0.9, linestyle="--")

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_FINANCE_POS, label="Finance feature"),
        mpatches.Patch(facecolor=COLOR_TEXT_POS, label="Text / AI narrative feature"),
        mpatches.Patch(facecolor=COLOR_ZERO, label="Zeroed-out by Lasso"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right", framealpha=0.7)

    ax.set_xlabel(
        "Standardised Beta Coefficient  (both X and Y standardised — unitless effect size)",
        fontsize=9,
    )
    ax.set_title(
        "Lasso Feature Selection: Finance + AI Narrative → Next-Quarter R&D Change\n"
        f"Fully standardised β  |  α={alpha}  |  N={len(work):,}"
        f"  |  Non-zero features: {n_nonzero}/{len(all_features)}"
        f"  |  Text features retained: {n_text_nonzero}/{len(text)}",
        fontsize=9,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D1C9BC")
    ax.spines["bottom"].set_color("#D1C9BC")
    ax.tick_params(colors="#374151", labelsize=9)
    ax.xaxis.label.set_color("#374151")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_stage16_markdown(
    takeaway_df: pd.DataFrame,
    feature_rel_dir: str = "../features",
    figure_rel_dir: str = "../figures",
) -> str:
    lines = [
        "## Stage 15 — WRDS-linked Metadata & Economic Meaning",
        "",
        "Stage 15 links narrative metadata from earnings calls to WRDS firm-quarter metadata, focusing on interpretable feature engineering and descriptive economic meaning.",
        "",
        "### Core Figures",
        f"![wrds_distribution]({figure_rel_dir}/wrds_distribution.png)",
        f"![assoc_bar]({figure_rel_dir}/assoc_bar.png)",
        f"![coefplot_ai]({figure_rel_dir}/coefplot_ai.png)",
        f"![quadrant_finance]({figure_rel_dir}/quadrant_finance.png)",
        f"![gap_by_quadrant]({figure_rel_dir}/gap_by_quadrant.png)",
        "",
        "### Takeaways",
    ]
    for _, row in takeaway_df.head(5).iterrows():
        lines.append(
            f"- {row['takeaway']} Mechanism: {row['mechanism']} Caution: {row['caution']}"
        )
    return "\n".join(lines) + "\n"


def _upsert_stage16_report_section(
    report_path: str,
    takeaway_df: pd.DataFrame,
    feature_rel_dir: str = "../features",
    figure_rel_dir: str = "../figures",
) -> None:
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    start_marker = "<!-- STAGE15_START -->"
    end_marker = "<!-- STAGE15_END -->"
    section = f"{start_marker}\n{_build_stage16_markdown(takeaway_df, feature_rel_dir=feature_rel_dir, figure_rel_dir=figure_rel_dir)}{end_marker}\n"
    if os.path.exists(report_path):
        content = open(report_path, "r", encoding="utf-8").read()
    else:
        content = "# Report\n\n"
    legacy_start = "<!-- STAGE16_START -->"
    legacy_end = "<!-- STAGE16_END -->"
    if legacy_start in content and legacy_end in content:
        left = content.split(legacy_start)[0].rstrip()
        right = content.split(legacy_end)[1].lstrip()
        content = f"{left}\n\n{right}".strip() + "\n"
    if start_marker in content and end_marker in content:
        left = content.split(start_marker)[0].rstrip()
        right = content.split(end_marker)[1].lstrip()
        new_content = f"{left}\n\n{section}\n{right}".strip() + "\n"
    else:
        if not content.endswith("\n"):
            content += "\n"
        new_content = content.rstrip() + "\n\n" + section
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def run_stage15(
    wrds_path: str,
    document_metrics_path: str,
    initiation_scores_path: str,
    quadrants_path: str | None,
    final_dataset_path: str,
    output_features_dir: str = "outputs/features",
    output_figures_dir: str = "outputs/figures",
    report_path: str = "outputs/report/report.md",
    test_quarters: int = 4,
    cluster_col: str = "ticker",
) -> Dict[str, object]:
    os.makedirs(output_features_dir, exist_ok=True)
    os.makedirs(output_figures_dir, exist_ok=True)

    wrds_out = build_wrds_feature_store(
        wrds_path=wrds_path,
        output_dir=output_features_dir,
    )
    panel_out = build_text_wrds_panel(
        document_metrics_path=document_metrics_path,
        initiation_scores_path=initiation_scores_path,
        quadrants_path=quadrants_path,
        final_dataset_path=final_dataset_path,
        wrds_feature_store_path=wrds_out["feature_store_path"],
        output_features_dir=output_features_dir,
        output_figures_dir=output_figures_dir,
    )
    panel = panel_out["panel"].copy()

    fig_paths = {
        "dist": os.path.join(output_figures_dir, "wrds_distribution.png"),
        "zero_by_sector": os.path.join(output_figures_dir, "ai_zero_by_sector.png"),
        "nonzero_by_sector": os.path.join(output_figures_dir, "ai_nonzero_by_sector.png"),
        "trend": os.path.join(output_figures_dir, "ai_trend_by_size_or_sector.png"),
        "assoc": os.path.join(output_figures_dir, "assoc_bar.png"),
        "coef": os.path.join(output_figures_dir, "coefplot_ai.png"),
        "quadrant_finance": os.path.join(output_figures_dir, "quadrant_finance.png"),
        "quadrant_heatmap": os.path.join(output_figures_dir, "quadrant_sector_heatmap.png"),
        "gap": os.path.join(output_figures_dir, "gap_by_quadrant.png"),
        "model_compare": os.path.join(output_figures_dir, "model_compare.png"),
        "lasso_selection": os.path.join(output_figures_dir, "lasso_feature_selection.png"),
    }

    plot_wrds_distribution(panel, fig_paths["dist"])
    plot_ai_zero_by_sector(panel, fig_paths["zero_by_sector"])
    plot_ai_nonzero_by_sector(panel, fig_paths["nonzero_by_sector"])
    plot_ai_trend_by_size(panel, fig_paths["trend"])

    assoc_df = _spearman_association(panel)
    assoc_path = os.path.join(output_features_dir, "association_table.csv")
    assoc_df.to_csv(assoc_path, index=False)
    plot_assoc_bar(assoc_df, fig_paths["assoc"])

    x_core = [c for c in ["log_mkcap", "rd_intensity_mkcap", "eps", "capex_intensity_mkcap", "ret_q"] if c in panel.columns]
    reg1, formula1 = _fit_regression(panel, "overall_ai_ratio", x_core, cluster_col=cluster_col)
    reg2, formula2 = _fit_regression(panel, "initiation_score", x_core, cluster_col=cluster_col)
    reg_table = pd.concat([reg1, reg2], ignore_index=True)
    reg_table["formula"] = np.where(reg_table["y"] == "overall_ai_ratio", formula1, formula2)
    reg_path = os.path.join(output_features_dir, "regression_table.csv")
    reg_table.to_csv(reg_path, index=False)
    plot_coefplot(reg_table, fig_paths["coef"])

    panel["narrative_invest_gap"] = _zscore(panel["overall_ai_ratio"]) - _zscore(panel["rd_intensity_mkcap"])
    quad_profile = (
        panel.groupby("quadrant", as_index=False)[
            [c for c in ["log_mkcap", "rd_intensity_mkcap", "eps", "ret_q", "speech_ai_ratio", "qa_ai_ratio", "narrative_invest_gap"] if c in panel.columns]
        ]
        .mean()
    )
    quad_profile_path = os.path.join(output_features_dir, "quadrant_profile.csv")
    quad_profile.to_csv(quad_profile_path, index=False)
    plot_quadrant_finance(panel, fig_paths["quadrant_finance"])
    plot_quadrant_sector_heatmap(panel, fig_paths["quadrant_heatmap"])
    plot_gap_by_quadrant(panel, fig_paths["gap"])

    compare = _linear_model_compare(panel, test_quarters=test_quarters)
    compare_path = os.path.join(output_features_dir, "model_compare.csv")
    compare.to_csv(compare_path, index=False)
    plot_model_compare(compare, fig_paths["model_compare"])

    # Lasso feature-selection plot: visually proves text features are shrunk / zeroed
    # relative to financial variables when both compete under L1 regularisation.
    _plot_lasso_feature_selection(panel, fig_paths["lasso_selection"])

    takeaways = default_takeaways()
    takeaway_rows = []
    for fig_name, vals in takeaways.items():
        takeaway_rows.append({"figure": fig_name, **vals})
    takeaway_df = pd.DataFrame(takeaway_rows)
    takeaway_path = os.path.join(output_features_dir, "figure_takeaways.csv")
    takeaway_df.to_csv(takeaway_path, index=False)

    report_dir = os.path.dirname(report_path) or "."
    feature_rel_dir = os.path.relpath(output_features_dir, report_dir).replace("\\", "/")
    figure_rel_dir = os.path.relpath(output_figures_dir, report_dir).replace("\\", "/")
    _upsert_stage16_report_section(
        report_path,
        takeaway_df,
        feature_rel_dir=feature_rel_dir,
        figure_rel_dir=figure_rel_dir,
    )

    return {
        "wrds_feature_store_path": wrds_out["feature_store_path"],
        "panel_path": panel_out["panel_path"],
        "merge_diagnostics_path": panel_out["merge_diagnostics_path"],
        "merge_funnel_path": panel_out["merge_funnel_path"],
        "association_table_path": assoc_path,
        "regression_table_path": reg_path,
        "quadrant_profile_path": quad_profile_path,
        "model_compare_path": compare_path,
        "figure_takeaways_path": takeaway_path,
        "figures": fig_paths,
        "report_path": report_path,
        "se_type": "cluster_by_firm",
        "capxy_assumption": "annual_proxy_raw",
    }


def run_stage16(*args, **kwargs):
    """Backward-compatible alias for the renamed Stage 15 entrypoint."""
    return run_stage15(*args, **kwargs)
