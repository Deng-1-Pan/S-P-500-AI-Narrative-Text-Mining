from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_dataset_overview(df: pd.DataFrame, output_path: str) -> Dict[str, str]:
    _ensure_parent(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sector_counts = (
        df["gsector"].astype(str).value_counts().head(10).sort_values(ascending=True)
        if "gsector" in df.columns
        else pd.Series(dtype=float)
    )
    if len(sector_counts):
        axes[0, 0].barh(sector_counts.index.astype(str), sector_counts.values, color="#3b7ddd")
        sector_counts.to_frame("count").to_csv(output_path.replace(".png", "_sector_counts.csv"), index_label="sector")
    axes[0, 0].set_title("Sample by GICS Sector (Top 10)")

    if "total_sentences" in df.columns:
        sns.histplot(df["total_sentences"], bins=40, ax=axes[0, 1], color="#ef8a62")
        df[["total_sentences"]].dropna().to_csv(output_path.replace(".png", "_total_sentences.csv"), index=False)
    axes[0, 1].set_title("Call Length Distribution (Sentences)")

    if "overall_kw_ai_ratio" in df.columns:
        sns.histplot(df["overall_kw_ai_ratio"], bins=50, ax=axes[1, 0], color="#4daf4a")
        df[["overall_kw_ai_ratio"]].dropna().to_csv(output_path.replace(".png", "_ai_intensity_dist.csv"), index=False)
    axes[1, 0].set_title("Overall AI Intensity Distribution")

    zero_metrics = {
        "Overall AI=0": float((df.get("overall_kw_ai_ratio", pd.Series(dtype=float)) == 0).mean()) if "overall_kw_ai_ratio" in df.columns else np.nan,
        "Speech AI=0": float((df.get("speech_kw_ai_ratio", pd.Series(dtype=float)) == 0).mean()) if "speech_kw_ai_ratio" in df.columns else np.nan,
        "Q&A AI=0": float((df.get("qa_kw_ai_ratio", pd.Series(dtype=float)) == 0).mean()) if "qa_kw_ai_ratio" in df.columns else np.nan,
        "AI exchanges=0": float((df.get("total_ai_exchanges", pd.Series(dtype=float)) == 0).mean()) if "total_ai_exchanges" in df.columns else np.nan,
    }
    zm = pd.Series(zero_metrics).dropna()
    axes[1, 1].bar(zm.index, zm.values, color="#7b3294")
    zm.to_frame("ratio").to_csv(output_path.replace(".png", "_zero_metrics.csv"), index_label="metric")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].set_title("Zero-Inflation Diagnostics")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": os.path.basename(output_path),
        "takeaway": "AI discussion is highly zero-inflated and uneven across sectors.",
        "mechanism": "Most calls still have no AI mentions, but a subset of sectors drives the narrative mass.",
        "caution": "Distributional skew implies linear models need robustness checks and temporal validation.",
    }


def plot_metadata_association(df: pd.DataFrame, output_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    _ensure_parent(output_path)

    numeric = [c for c in ["log_mktcap", "rd_intensity", "eps_growth_yoy", "price_growth_yoy", "ln_price"] if c in df.columns]
    cols = ["overall_kw_ai_ratio", "gsector", "year_quarter"] + numeric
    work = df[cols].dropna().copy()
    if "gsector" in work.columns:
        work["gsector"] = work["gsector"].astype(str)
    if "year_quarter" in work.columns:
        work["year_quarter"] = work["year_quarter"].astype(str)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if len(work) < 300:
        assoc = pd.DataFrame(columns=["feature", "coef", "p_value"])
    else:
        for col in ["overall_kw_ai_ratio"] + numeric:
            work[f"z_{col}"] = (work[col] - work[col].mean()) / (work[col].std() + 1e-9)
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        if len(work) < 300:
            assoc = pd.DataFrame(columns=["feature", "coef", "p_value"])
        else:
            rhs = " + ".join([f"z_{c}" for c in numeric] + ["C(gsector)", "C(year_quarter)"])
            model = smf.ols(f"z_overall_kw_ai_ratio ~ {rhs}", data=work).fit(cov_type="HC1")
            assoc = pd.DataFrame(
                {
                    "feature": numeric,
                    "coef": [model.params.get(f"z_{c}", np.nan) for c in numeric],
                    "p_value": [model.pvalues.get(f"z_{c}", np.nan) for c in numeric],
                }
            ).dropna().sort_values("coef")

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(assoc):
        colors = ["#1b9e77" if x > 0 else "#d95f02" for x in assoc["coef"]]
        ax.barh(assoc["feature"], assoc["coef"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
    ax.set_title("AI Intensity vs Metadata (Std. FE Coefficients)")
    ax.set_xlabel("Standardized coefficient")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return assoc, {
        "figure": os.path.basename(output_path),
        "takeaway": "AI intensity co-moves with firm characteristics after sector and quarter controls.",
        "mechanism": "Size and innovation proxies shape how much AI narrative appears in calls.",
        "caution": "Associations are descriptive and may reflect omitted strategic disclosure incentives.",
    }


def plot_structural_metadata(df: pd.DataFrame, output_path: str) -> Dict[str, str]:
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x1 = "qa_sentence_share"
    y1 = "overall_kw_ai_ratio"
    if {x1, y1}.issubset(df.columns):
        sns.regplot(data=df, x=x1, y=y1, scatter_kws={"s": 10, "alpha": 0.2}, line_kws={"color": "red"}, ax=axes[0])
        df[[x1, y1]].dropna().to_csv(output_path.replace(".png", "_qa_share_scatter.csv"), index=False)
    axes[0].set_title("Q&A Share vs AI Intensity")

    x2 = "analyst_ai_share"
    y2 = "y_next_mktcap_growth"
    if {x2, y2}.issubset(df.columns):
        temp = df[[x2, y2]].dropna()
        if len(temp):
            temp["bin"] = pd.qcut(temp[x2].rank(method="first"), q=10, labels=False)
            agg = temp.groupby("bin", as_index=False)[[x2, y2]].mean()
            axes[1].plot(agg[x2], agg[y2], marker="o")
            agg.to_csv(output_path.replace(".png", "_analyst_ai_binned.csv"), index=False)
    axes[1].set_title("Analyst AI Share vs Next Mktcap Growth (binned)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": os.path.basename(output_path),
        "takeaway": "Conversation structure matters: AI concentration in Q&A and analyst-led mentions correlate with outcomes.",
        "mechanism": "Interactive Q&A may reveal higher-information AI narratives than scripted remarks.",
        "caution": "Analyst share can proxy sector coverage intensity, not purely firm-level fundamentals.",
    }


def plot_time_series(df: pd.DataFrame, output_path: str) -> Dict[str, str]:
    _ensure_parent(output_path)

    ts = (
        df.dropna(subset=["year_quarter", "overall_kw_ai_ratio"])
        .groupby("year_quarter", as_index=False)["overall_kw_ai_ratio"]
        .mean()
        .sort_values("year_quarter")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts["year_quarter"], ts["overall_kw_ai_ratio"], marker="o", label="All firms")
    ts.to_csv(output_path.replace(".png", "_all_firms.csv"), index=False)

    if {"log_mktcap", "quarter_index"}.issubset(df.columns):
        tmp = df.dropna(subset=["log_mktcap", "overall_kw_ai_ratio", "year_quarter", "quarter_index"]).copy()
        tmp["size_group"] = np.where(tmp["log_mktcap"] >= tmp["log_mktcap"].median(), "Large", "Small")
        grp = tmp.groupby(["year_quarter", "size_group"], as_index=False)["overall_kw_ai_ratio"].mean()
        grp.to_csv(output_path.replace(".png", "_size_grouped.csv"), index=False)
        for g, sub in grp.groupby("size_group"):
            ax.plot(sub["year_quarter"], sub["overall_kw_ai_ratio"], marker=".", alpha=0.8, label=f"{g} cap")

    if "2022Q4" in ts["year_quarter"].tolist():
        idx = ts.index[ts["year_quarter"] == "2022Q4"][0]
        ax.axvline(ts.loc[idx, "year_quarter"], color="red", ls="--", lw=1)

    ax.tick_params(axis="x", rotation=45)
    ax.set_title("AI Intensity Over Time (with 2022Q4 marker)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": os.path.basename(output_path),
        "takeaway": "AI narrative rises sharply post-2022Q4 and remains elevated.",
        "mechanism": "Generative-AI diffusion likely changed disclosure salience across earnings calls.",
        "caution": "Keyword-intensity trend can partly reflect vocabulary shifts rather than economic investment changes.",
    }


def plot_quadrants(df: pd.DataFrame, output_path: str, reps_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    _ensure_parent(output_path)
    _ensure_parent(reps_path)

    req = ["doc_id", "ticker", "overall_kw_ai_ratio", "analyst_ai_share"]
    work = df[[c for c in req if c in df.columns]].dropna().copy()
    if len(work) == 0:
        work = pd.DataFrame(columns=req)

    x_th = work["overall_kw_ai_ratio"].median() if len(work) else 0.0
    y_th = work["analyst_ai_share"].median() if len(work) else 0.0

    def _quad(r: pd.Series) -> str:
        high_x = r["overall_kw_ai_ratio"] >= x_th
        high_y = r["analyst_ai_share"] >= y_th
        if high_x and high_y:
            return "High-High"
        if high_x and not high_y:
            return "High AI / Low Analyst"
        if (not high_x) and high_y:
            return "Low AI / High Analyst"
        return "Low-Low"

    if len(work):
        work["quadrant"] = work.apply(_quad, axis=1)
    else:
        work["quadrant"] = []

    fig, ax = plt.subplots(figsize=(8, 7))
    if len(work):
        sns.scatterplot(data=work, x="overall_kw_ai_ratio", y="analyst_ai_share", hue="quadrant", alpha=0.5, s=20, ax=ax)
        ax.axvline(x_th, ls="--", color="black", lw=0.8)
        ax.axhline(y_th, ls="--", color="black", lw=0.8)
    ax.set_title("Company Quadrants: AI Intensity x Analyst AI Share")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    reps = (
        work.sort_values(["quadrant", "overall_kw_ai_ratio"], ascending=[True, False])
        .groupby("quadrant")
        .head(5)
        .reset_index(drop=True)
    )
    reps.to_csv(reps_path, index=False)

    return work, {
        "figure": os.path.basename(output_path),
        "takeaway": "Firms split into distinct AI communication regimes once intensity and analyst initiation are combined.",
        "mechanism": "Quadrants separate proactive AI storytelling from analyst-pushed AI discussions.",
        "caution": "Quadrant labels are threshold-based and sensitive to median cutoff choice.",
    }


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str) -> Dict[str, str]:
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot = metrics_df.copy().sort_values("r2_test", ascending=False)
    plot.to_csv(output_path.replace(".png", "_data.csv"), index=False)

    axes[0].barh(plot["model"], plot["r2_test"], color="#1b9e77")
    axes[0].invert_yaxis()
    axes[0].set_title("Out-of-sample R²")

    axes[1].barh(plot["model"], plot["mae_test"], color="#d95f02")
    axes[1].invert_yaxis()
    axes[1].set_title("Out-of-sample MAE")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": os.path.basename(output_path),
        "takeaway": "Finance+Text usually improves OOS fit versus finance-only and text-only baselines.",
        "mechanism": "Narrative variables add incremental information not captured by contemporaneous metadata.",
        "caution": "Incremental gains can vary by target and test-window regime.",
    }


def plot_lasso_outputs(term_df: pd.DataFrame, stability_df: pd.DataFrame, output_dir: str) -> List[Dict[str, str]]:
    os.makedirs(output_dir, exist_ok=True)
    notes: List[Dict[str, str]] = []

    text_terms = term_df[term_df["block"] == "text"].copy()
    if len(text_terms):
        # Coefficient bar
        top = pd.concat([text_terms.nlargest(15, "coefficient"), text_terms.nsmallest(15, "coefficient")]).drop_duplicates("feature")
        top = top.sort_values("coefficient")

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#1b9e77" if v > 0 else "#d95f02" for v in top["coefficient"]]
        ax.barh(top["raw_term"], top["coefficient"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("Lasso/ElasticNet Text Coefficients")
        fig.tight_layout()
        coef_path = os.path.join(output_dir, "lasso_coef_bar.png")
        fig.savefig(coef_path, dpi=180, bbox_inches="tight")
        top.to_csv(coef_path.replace(".png", "_data.csv"), index=False)
        plt.close(fig)

        notes.append(
            {
                "figure": os.path.basename(coef_path),
                "takeaway": "A small set of phrases drives predicted economic outcomes.",
                "mechanism": "Positive/negative coefficients encode distinct AI narrative orientations.",
                "caution": "Coefficient signs are conditional on correlated ngrams and regularization strength.",
            }
        )

        # Volcano plot
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(text_terms["coefficient"], text_terms["log_doc_frequency"], alpha=0.5, s=20)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("log(DF+1)")
        ax.set_title("Text Coefficient Volcano")
        fig.tight_layout()
        vol_path = os.path.join(output_dir, "lasso_volcano.png")
        fig.savefig(vol_path, dpi=180, bbox_inches="tight")
        text_terms.to_csv(vol_path.replace(".png", "_data.csv"), index=False)
        plt.close(fig)

        notes.append(
            {
                "figure": os.path.basename(vol_path),
                "takeaway": "Economically-relevant terms are not always the most frequent terms.",
                "mechanism": "Sparse penalization highlights predictive rare-but-informative expressions.",
                "caution": "Very low-frequency terms require stability filtering before interpretation.",
            }
        )

    if len(stability_df):
        stab = stability_df[stability_df["feature"].str.startswith("text::")].copy()
        stab = stab.sort_values("stability_freq", ascending=False).head(20)
        stab["raw_term"] = stab["feature"].str.replace("^text::", "", regex=True)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(stab["raw_term"], stab["stability_freq"], color="#7570b3")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title("Stability Selection Frequency (Top Terms)")
        fig.tight_layout()
        stab_path = os.path.join(output_dir, "lasso_stability.png")
        fig.savefig(stab_path, dpi=180, bbox_inches="tight")
        stab.to_csv(stab_path.replace(".png", "_data.csv"), index=False)
        plt.close(fig)

        notes.append(
            {
                "figure": os.path.basename(stab_path),
                "takeaway": "Only a subset of selected terms remains stable across time windows.",
                "mechanism": "Stable terms likely capture persistent narrative dimensions rather than one-off noise.",
                "caution": "Stability depends on window design and sample coverage of AI-active firms.",
            }
        )

    return notes
