"""Stage 13 rewrite: dual-mechanism path analysis report."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.research.data import build_research_dataset, prepare_wrds_features, run_basic_sanity_checks
from src.research.regression_primitives import fit_path_regression as fit_path_regression_shared
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


TOKEN_RE = re.compile(r"[a-zA-Z]+")
EFFICIENCY_KEYWORDS = {"cost", "reduce", "efficiency", "automate", "automation", "margin", "save", "productivity"}
GROWTH_KEYWORDS = {"product", "revenue", "research", "growth", "new", "opportunity", "innovation", "expand"}


def _read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _build_mechanism_ratios(sentences_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["doc_id", "text"] + (["kw_is_ai"] if "kw_is_ai" in sentences_df.columns else [])
    s = sentences_df[cols].copy()

    if "kw_is_ai" in s.columns:
        s = s[s["kw_is_ai"].fillna(False).astype(bool)].copy()
    else:
        s = s[s["text"].fillna("").str.contains(r"\bai\b", case=False, regex=True)].copy()

    s["text"] = s["text"].fillna("").astype(str).str.lower()

    rows: List[Dict[str, float]] = []
    for doc_id, grp in s.groupby("doc_id", sort=False):
        tokens = TOKEN_RE.findall(" ".join(grp["text"].tolist()))
        n = len(tokens)
        if n == 0:
            rows.append(
                {
                    "doc_id": doc_id,
                    "ai_token_count": 0,
                    "efficiency_ai_ratio": 0.0,
                    "growth_ai_ratio": 0.0,
                }
            )
            continue

        eff = sum(tok in EFFICIENCY_KEYWORDS for tok in tokens)
        gro = sum(tok in GROWTH_KEYWORDS for tok in tokens)
        rows.append(
            {
                "doc_id": doc_id,
                "ai_token_count": int(n),
                "efficiency_ai_ratio": float(eff / n),
                "growth_ai_ratio": float(gro / n),
            }
        )

    return pd.DataFrame(rows)


def _fit_path_regression(
    df: pd.DataFrame,
    mechanism_name: str,
    x_var: str,
    y_var: str,
    controls: List[str],
    fe_var: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    return fit_path_regression_shared(
        df=df,
        mechanism_name=mechanism_name,
        x_var=x_var,
        y_var=y_var,
        controls=controls,
        fe_var=fe_var,
        min_obs=8,
    )


def _plot_scatter(df: pd.DataFrame, x: str, y: str, output_path: str, title: str, color: str) -> None:
    if len(df) == 0:
        return

    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    xvals = pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=float)
    yvals = pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xvals) & np.isfinite(yvals)
    xvals, yvals = xvals[mask], yvals[mask]

    if len(xvals) == 0:
        plt.close(fig)
        return

    ax.scatter(xvals, yvals, s=24, alpha=0.65, color=color)

    if len(xvals) >= 2:
        try:
            slope, intercept = np.polyfit(xvals, yvals, deg=1)
            xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 100)
            ax.plot(xs, slope * xs + intercept, linewidth=1.5, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"))
        except Exception:
            pass

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    style_axes(ax, grid_axis="both", grid_alpha=0.12)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)


def _plot_mechanism_strength(reg_df: pd.DataFrame, output_path: str) -> None:
    if reg_df is None or len(reg_df) == 0:
        return

    apply_spotify_theme()
    plot_df = reg_df.copy()
    plot_df["abs_coef"] = plot_df["coef"].abs()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    ax.bar(plot_df["mechanism"], plot_df["abs_coef"], color=[SPOTIFY_COLORS.get("blue", "#4EA1FF"), SPOTIFY_COLORS.get("accent", "#1DB954")])
    ax.set_ylabel("|Coefficient| on Path Feature")
    ax.set_title("Dual-Path Mechanism Strength")
    style_axes(ax, grid_axis="y", grid_alpha=0.12)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)


def _write_report(
    report_path: str,
    dataset: pd.DataFrame,
    regression_table: pd.DataFrame,
    figure_files: List[str],
) -> None:
    n_obs = len(dataset)
    n_firms = int(dataset["ticker"].nunique()) if "ticker" in dataset.columns else 0
    n_quarters = int(dataset["year_quarter"].nunique()) if "year_quarter" in dataset.columns else 0

    lines: List[str] = []
    lines.append("# S&P 500 AI Narrative: Dual-Mechanism Path Analysis")
    lines.append("")
    lines.append("## Design Shift")
    lines.append("- Stage 13 now tests two mechanism paths instead of a generic explanatory OLS stack.")
    lines.append("- Path A (Efficiency-AI): AI language around cost reduction and automation predicts short-term EPS growth.")
    lines.append("- Path B (Growth-AI): AI language around product/research opportunity predicts forward R&D intensity change.")
    lines.append("")

    lines.append("## Dataset")
    lines.append(f"- Observations: {n_obs}")
    lines.append(f"- Firms: {n_firms}")
    lines.append(f"- Year-quarters: {n_quarters}")
    lines.append("")

    lines.append("## Dual-Path Regression Results")
    lines.append("```csv")
    lines.append(regression_table.to_csv(index=False).strip())
    lines.append("```")
    lines.append("")

    lines.append("## Figures")
    for fig in figure_files:
        name = os.path.basename(fig)
        lines.append(f"### {name}")
        lines.append(f"![{name}](figures/{name})")
        lines.append("")

    lines.append("## Economic Interpretation")
    lines.append("- A larger positive coefficient for `efficiency_ai_ratio` supports a short-horizon profitability channel.")
    lines.append("- A larger positive coefficient for `growth_ai_ratio` on forward R&D change supports a long-horizon innovation channel.")
    lines.append("- If either path is weak or insignificant, AI narrative appears more rhetorical than operational for that channel.")

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_research_report(
    sentences_with_keywords_path: str,
    document_metrics_path: str,
    initiation_scores_path: str,
    parsed_transcripts_path: str,
    final_dataset_path: str,
    wrds_path: str,
    output_dir: str = "outputs/report",
    features_output_dir: str = os.path.join("outputs", "features"),
    model_target: str = "y_next_mktcap_growth",
    test_quarters: int = 4,
) -> Dict[str, Any]:
    del model_target, test_quarters

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    tab_dir = os.path.join(output_dir, "tables")
    lasso_dir = os.path.join(output_dir, "lasso")
    case_dir = os.path.join(output_dir, "cases")
    for d in [fig_dir, tab_dir, lasso_dir, case_dir]:
        os.makedirs(d, exist_ok=True)

    sentences_kw = _read_table(sentences_with_keywords_path)
    doc_metrics = _read_table(document_metrics_path)
    initiation = _read_table(initiation_scores_path)
    parsed = _read_table(parsed_transcripts_path)
    final_dataset = _read_table(final_dataset_path)

    wrds = prepare_wrds_features(wrds_path)

    build = build_research_dataset(
        document_metrics=doc_metrics,
        initiation_scores=initiation,
        sentences_with_keywords=sentences_kw,
        parsed_transcripts=parsed,
        final_dataset=final_dataset,
        wrds_features=wrds,
    )

    research_df = build.dataset.copy()
    data_dict = build.data_dictionary.copy()
    run_basic_sanity_checks(research_df)

    mech = _build_mechanism_ratios(sentences_kw)
    research_df = research_df.merge(mech, on="doc_id", how="left")
    research_df["efficiency_ai_ratio"] = research_df["efficiency_ai_ratio"].fillna(0.0)
    research_df["growth_ai_ratio"] = research_df["growth_ai_ratio"].fillna(0.0)
    research_df["ai_token_count"] = research_df.get("ai_token_count", 0).fillna(0.0)

    # Small samples may not have 4-quarter history for YoY EPS growth; fallback to next-quarter EPS growth.
    if "y_next_eps_growth_yoy" in research_df.columns and "epspxq" in research_df.columns:
        if int(research_df["y_next_eps_growth_yoy"].notna().sum()) < 8:
            tmp = research_df.sort_values(["ticker", "year", "quarter"]).copy()
            next_eps = tmp.groupby("ticker", sort=False)["epspxq"].shift(-1)
            tmp["y_next_eps_growth_yoy"] = np.where(
                tmp["epspxq"].notna() & (tmp["epspxq"] != 0),
                next_eps / tmp["epspxq"] - 1.0,
                np.nan,
            )
            research_df["y_next_eps_growth_yoy"] = tmp["y_next_eps_growth_yoy"].values

    if "y_next_rd_intensity_change" in research_df.columns and "rd_intensity" in research_df.columns:
        if int(research_df["y_next_rd_intensity_change"].notna().sum()) < 8:
            tmp = research_df.sort_values(["ticker", "year", "quarter"]).copy()
            tmp["y_next_rd_intensity_change"] = tmp.groupby("ticker", sort=False)["rd_intensity"].shift(-1) - tmp["rd_intensity"]
            research_df["y_next_rd_intensity_change"] = tmp["y_next_rd_intensity_change"].values

    research_dataset_path = os.path.join(features_output_dir, "research_dataset.parquet")
    data_dict_path = os.path.join(features_output_dir, "data_dictionary.csv")
    os.makedirs(os.path.dirname(research_dataset_path), exist_ok=True)
    research_df.to_parquet(research_dataset_path, index=False)
    data_dict.to_csv(data_dict_path, index=False)

    controls = ["overall_kw_ai_ratio", "qa_kw_ai_ratio", "speech_kw_ai_ratio", "log_mktcap", "rd_intensity", "eps_positive"]
    fe_var = "sector" if "sector" in research_df.columns else ("gsector" if "gsector" in research_df.columns else "ticker")

    reg_a, work_a = _fit_path_regression(
        research_df,
        mechanism_name="Efficiency-AI",
        x_var="efficiency_ai_ratio",
        y_var="y_next_eps_growth_yoy",
        controls=controls,
        fe_var=fe_var,
    )
    reg_b, work_b = _fit_path_regression(
        research_df,
        mechanism_name="Growth-AI",
        x_var="growth_ai_ratio",
        y_var="y_next_rd_intensity_change",
        controls=controls,
        fe_var=fe_var,
    )

    reg_table = pd.DataFrame([reg_a, reg_b])
    dual_path_csv = os.path.join(tab_dir, "dual_path_regressions.csv")
    reg_table.to_csv(dual_path_csv, index=False)

    fig_a = os.path.join(fig_dir, "mechanism_efficiency_eps.png")
    fig_b = os.path.join(fig_dir, "mechanism_growth_rd.png")
    fig_strength = os.path.join(fig_dir, "mechanism_path_strength.png")

    _plot_scatter(
        work_a,
        x="efficiency_ai_ratio",
        y="y_next_eps_growth_yoy",
        output_path=fig_a,
        title="Mechanism A: Efficiency-AI vs Next EPS Growth",
        color=SPOTIFY_COLORS.get("blue", "#4EA1FF"),
    )
    _plot_scatter(
        work_b,
        x="growth_ai_ratio",
        y="y_next_rd_intensity_change",
        output_path=fig_b,
        title="Mechanism B: Growth-AI vs Next R&D Intensity Change",
        color=SPOTIFY_COLORS.get("accent", "#1DB954"),
    )
    _plot_mechanism_strength(reg_table, fig_strength)

    report_path = os.path.join(output_dir, "report.md")
    _write_report(
        report_path=report_path,
        dataset=research_df,
        regression_table=reg_table,
        figure_files=[fig_a, fig_b, fig_strength],
    )

    return {
        "research_dataset_path": research_dataset_path,
        "data_dictionary_path": data_dict_path,
        "report_path": report_path,
        "figure_dir": fig_dir,
        "table_dir": tab_dir,
        "lasso_dir": lasso_dir,
        "case_dir": case_dir,
    }


if __name__ == "__main__":
    out = run_research_report(
        sentences_with_keywords_path="outputs/features/sentences_with_keywords.parquet",
        document_metrics_path="outputs/features/document_metrics.parquet",
        initiation_scores_path="outputs/features/initiation_scores.parquet",
        parsed_transcripts_path="outputs/features/parsed_transcripts.parquet",
        final_dataset_path="data/final_dataset.parquet",
        wrds_path="data/wrds.csv",
        output_dir="outputs/report",
    )
    print("Research report outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")
