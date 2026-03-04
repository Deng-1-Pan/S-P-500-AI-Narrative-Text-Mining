from __future__ import annotations

import os
from typing import Dict, Iterable, List

import pandas as pd


def _fmt_df(df: pd.DataFrame, n: int = 12) -> str:
    if df is None or len(df) == 0:
        return "(no rows)"
    show = df.head(n).copy()
    return show.to_csv(index=False)


def write_report(
    report_path: str,
    dataset: pd.DataFrame,
    data_dictionary_path: str,
    figure_notes: List[Dict[str, str]],
    fe_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    lasso_metrics: pd.DataFrame,
    lasso_top_terms: pd.DataFrame,
    deep_dive_cases: pd.DataFrame,
) -> None:
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    n_calls = len(dataset)
    n_firms = int(dataset["ticker"].nunique()) if "ticker" in dataset.columns else 0
    n_quarters = int(dataset["year_quarter"].nunique()) if "year_quarter" in dataset.columns else 0

    zero_overall = float((dataset["overall_kw_ai_ratio"] == 0).mean()) if "overall_kw_ai_ratio" in dataset.columns else float("nan")
    zero_qa = float((dataset["qa_kw_ai_ratio"] == 0).mean()) if "qa_kw_ai_ratio" in dataset.columns else float("nan")

    lines: List[str] = []
    lines.append("# S&P 500 Earnings Calls AI Narrative: Research Report")
    lines.append("")
    lines.append("## Research Questions and Hypotheses")
    lines.append("1. Higher AI narrative intensity predicts next-quarter R&D intensity change.")
    lines.append("2. AI narrative structure in Q&A predicts next-quarter market-cap growth.")
    lines.append("3. AI narrative variables add incremental predictive value beyond finance-only baselines.")
    lines.append("")

    lines.append("## Dataset Overview")
    lines.append(f"- Observations (firm-quarter calls): {n_calls}")
    lines.append(f"- Firms: {n_firms}")
    lines.append(f"- Quarters: {n_quarters}")
    lines.append(f"- Zero share (overall AI ratio): {zero_overall:.3f}")
    lines.append(f"- Zero share (Q&A AI ratio): {zero_qa:.3f}")
    lines.append(f"- Data dictionary: `{data_dictionary_path}`")
    lines.append("")

    lines.append("## Initial Findings")
    lines.append("- AI mentions remain concentrated in a subset of calls and sectors.")
    lines.append("- Structural Q&A metadata captures variation not seen in raw intensity alone.")
    lines.append("")

    lines.append("## Figure Takeaways")
    for note in figure_notes:
        fig = note.get("figure", "")
        lines.append(f"### {fig}")
        lines.append(f"![{fig}](figures/{fig})")
        lines.append(f"- Takeaway: {note.get('takeaway', '')}")
        lines.append(f"- Mechanism: {note.get('mechanism', '')}")
        lines.append(f"- Caution: {note.get('caution', '')}")
        lines.append("")

    lines.append("## Feature-Level Metadata")
    lines.append("Regression-style metadata association coefficients are saved in `tables/metadata_association.csv`.")
    lines.append("")

    lines.append("## Structural Metadata")
    lines.append("Structural variables include Q&A share, analyst-vs-management AI share, and first-AI-turn position.")
    lines.append("")

    lines.append("## Econometric Link (WRDS)")
    lines.append("Main FE specifications include industry FE and quarter FE with HC1 robust errors.")
    lines.append("")
    lines.append("```csv")
    lines.append(_fmt_df(fe_summary, n=20).rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Model Comparison")
    lines.append("Time-split out-of-sample comparison (last quarters as test):")
    lines.append("```csv")
    lines.append(_fmt_df(model_comparison, n=20).rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Lasso / ElasticNet Interpretation")
    lines.append("Economic target text model metrics:")
    lines.append("```csv")
    lines.append(_fmt_df(lasso_metrics, n=10).rstrip())
    lines.append("```")
    lines.append("Top signed terms with stability and examples:")
    lines.append("```csv")
    lines.append(_fmt_df(lasso_top_terms, n=30).rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Deep Dive Cases")
    lines.append("Two representative firm-quarter cases and model rationale:")
    lines.append("```csv")
    lines.append(_fmt_df(deep_dive_cases, n=10).rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Limitations")
    lines.append("- No CRSP/IBES event-window returns or analyst coverage in current data package.")
    lines.append("- WRDS fields are narrow (missing assets/sales/capex/debt), so some classic controls are unavailable.")
    lines.append("- R&D outcomes have substantial missingness; interpreted as conditional evidence.")
    lines.append("- Keyword-based AI intensity remains a proxy and is not a causal treatment variable.")
    lines.append("")

    lines.append("## Next Steps")
    lines.append("1. Integrate CRSP daily return windows for CAR/volatility tests.")
    lines.append("2. Integrate richer Compustat controls (assets, sales, capex, debt) and firm FE main specs.")
    lines.append("3. Add sentence-level semantic embeddings for narrative dimensions beyond dictionary mentions.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
