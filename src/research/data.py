from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from src.utils.doc_id import attach_doc_keys as attach_doc_keys_shared
from src.utils.doc_id import parse_doc_id as parse_doc_id_shared


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: pd.DataFrame
    data_dictionary: pd.DataFrame


def parse_doc_id(doc_id: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    parsed = parse_doc_id_shared(
        doc_id,
        allow_ticker_without_q=True,
        allow_ticker_on_invalid=True,
    )
    return parsed.ticker, parsed.year, parsed.quarter


def attach_doc_keys(df: pd.DataFrame, doc_id_col: str = "doc_id") -> pd.DataFrame:
    return attach_doc_keys_shared(
        df,
        doc_id_col=doc_id_col,
        ticker_col="ticker",
        year_col="year",
        quarter_col="quarter",
        yearq_col="",
        keep_existing=False,
        allow_ticker_without_q=True,
        allow_ticker_on_invalid=True,
    )


def make_quarter_index(df: pd.DataFrame, year_col: str = "year", quarter_col: str = "quarter") -> pd.Series:
    year = pd.to_numeric(df[year_col], errors="coerce")
    qtr = pd.to_numeric(df[quarter_col], errors="coerce")
    return year * 4 + qtr


def compute_structural_features(sentences_kw: pd.DataFrame) -> pd.DataFrame:
    cols = ["doc_id", "section", "role", "turn_idx", "kw_is_ai"]
    keep = [c for c in cols if c in sentences_kw.columns]
    s = sentences_kw[keep].copy()
    s["kw_is_ai"] = s["kw_is_ai"].fillna(False).astype(bool)
    s["role"] = s.get("role", "unknown").fillna("unknown").astype(str).str.lower()
    s["section"] = s["section"].fillna("unknown").astype(str).str.lower()

    section_counts = (
        s.groupby(["doc_id", "section"], as_index=False)
        .agg(total_sentences=("kw_is_ai", "size"), ai_sentences=("kw_is_ai", "sum"))
    )

    pivot_total = section_counts.pivot(index="doc_id", columns="section", values="total_sentences").fillna(0)
    pivot_ai = section_counts.pivot(index="doc_id", columns="section", values="ai_sentences").fillna(0)

    out = pd.DataFrame({"doc_id": sorted(s["doc_id"].dropna().unique())})
    out = out.merge(pivot_total.reset_index(), on="doc_id", how="left", suffixes=("", "_total"))

    out["speech_sentences"] = out.get("speech", 0.0)
    out["qa_sentences"] = out.get("qa", 0.0)
    out = out.drop(columns=[c for c in ["speech", "qa", "unknown"] if c in out.columns], errors="ignore")

    out = out.merge(
        pivot_ai.reset_index().rename(columns={"speech": "speech_ai_sentences", "qa": "qa_ai_sentences"}),
        on="doc_id",
        how="left",
    )
    for c in ["speech_ai_sentences", "qa_ai_sentences"]:
        if c not in out.columns:
            out[c] = 0.0

    out["total_sentences"] = out[["speech_sentences", "qa_sentences"]].sum(axis=1)
    out["qa_sentence_share"] = np.where(out["total_sentences"] > 0, out["qa_sentences"] / out["total_sentences"], np.nan)
    out["speech_ai_share"] = np.where(out["speech_sentences"] > 0, out["speech_ai_sentences"] / out["speech_sentences"], np.nan)
    out["qa_ai_share"] = np.where(out["qa_sentences"] > 0, out["qa_ai_sentences"] / out["qa_sentences"], np.nan)

    qa = s[s["section"] == "qa"].copy()
    qa_ai = qa[qa["kw_is_ai"]].copy()

    role_ai = (
        qa_ai.assign(
            is_analyst=lambda x: (x["role"] == "analyst").astype(int),
            is_management=lambda x: (x["role"] == "management").astype(int),
        )
        .groupby("doc_id", as_index=False)
        .agg(
            analyst_ai_sentences=("is_analyst", "sum"),
            management_ai_sentences=("is_management", "sum"),
            qa_ai_sentences=("kw_is_ai", "size"),
        )
    )

    out = out.merge(role_ai, on="doc_id", how="left", suffixes=("", "_role"))
    for c in ["analyst_ai_sentences", "management_ai_sentences", "qa_ai_sentences_role"]:
        if c not in out.columns:
            out[c] = 0.0
    if "qa_ai_sentences_role" in out.columns:
        out["qa_ai_sentences"] = out[["qa_ai_sentences", "qa_ai_sentences_role"]].max(axis=1)
        out = out.drop(columns=["qa_ai_sentences_role"])

    out["analyst_ai_share"] = np.where(out["qa_ai_sentences"] > 0, out["analyst_ai_sentences"] / out["qa_ai_sentences"], np.nan)
    out["management_ai_share"] = np.where(out["qa_ai_sentences"] > 0, out["management_ai_sentences"] / out["qa_ai_sentences"], np.nan)

    if "turn_idx" in qa.columns:
        qa["turn_idx"] = pd.to_numeric(qa["turn_idx"], errors="coerce")
        max_turn = qa.groupby("doc_id", as_index=False)["turn_idx"].max().rename(columns={"turn_idx": "max_qa_turn_idx"})

        first_ai = (
            qa_ai.sort_values(["doc_id", "turn_idx"])
            .groupby("doc_id", as_index=False)
            .first()[["doc_id", "turn_idx", "role"]]
            .rename(columns={"turn_idx": "first_ai_turn_idx", "role": "first_ai_role"})
        )

        out = out.merge(max_turn, on="doc_id", how="left")
        out = out.merge(first_ai, on="doc_id", how="left")
        out["first_ai_turn_position"] = np.where(
            (out["max_qa_turn_idx"].notna()) & (out["max_qa_turn_idx"] >= 0) & (out["first_ai_turn_idx"].notna()),
            (out["first_ai_turn_idx"] + 1.0) / (out["max_qa_turn_idx"] + 1.0),
            np.nan,
        )
        out["first_ai_by_analyst"] = (out["first_ai_role"] == "analyst").astype(float)
        out["first_ai_by_management"] = (out["first_ai_role"] == "management").astype(float)

    numeric_fill_zero = [
        "speech_sentences",
        "qa_sentences",
        "speech_ai_sentences",
        "qa_ai_sentences",
        "analyst_ai_sentences",
        "management_ai_sentences",
    ]
    for c in numeric_fill_zero:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    return out


def prepare_wrds_features(wrds_path: str) -> pd.DataFrame:
    wrds = pd.read_csv(wrds_path, low_memory=False).rename(columns={"tic": "ticker"})

    if "datadate" in wrds.columns:
        wrds["datadate"] = pd.to_datetime(wrds["datadate"], errors="coerce")

    qtr_col = "datacqtr" if "datacqtr" in wrds.columns else ("datafqtr" if "datafqtr" in wrds.columns else None)
    if qtr_col:
        dq = wrds[qtr_col].astype(str)
        wrds["year"] = pd.to_numeric(dq.str[:4], errors="coerce")
        wrds["quarter"] = pd.to_numeric(dq.str[-1], errors="coerce")

    wrds = wrds.dropna(subset=["ticker", "year", "quarter"]).copy()
    wrds["year"] = wrds["year"].astype(int)
    wrds["quarter"] = wrds["quarter"].astype(int)

    sort_cols = ["ticker", "year", "quarter"] + (["datadate"] if "datadate" in wrds.columns else [])
    wrds = wrds.sort_values(sort_cols)
    wrds = wrds.drop_duplicates(["ticker", "year", "quarter"], keep="last")

    wrds["rd_intensity"] = wrds["xrdq"] / wrds["mkvaltq"].replace(0, np.nan)
    wrds["log_mktcap"] = np.log(wrds["mkvaltq"].replace(0, np.nan))
    wrds["ln_price"] = np.log(wrds["prccq"].replace(0, np.nan))
    wrds["eps_positive"] = (wrds["epspxq"] > 0).astype(float)

    wrds = wrds.sort_values(["ticker", "year", "quarter"])
    grp = wrds.groupby("ticker", sort=False)

    wrds["eps_growth_yoy"] = grp["epspxq"].pct_change(4, fill_method=None)
    wrds["price_growth_yoy"] = grp["prccq"].pct_change(4, fill_method=None)
    wrds["mktcap_growth_qoq"] = grp["mkvaltq"].pct_change(1, fill_method=None)
    wrds["rd_intensity_change_qoq"] = grp["rd_intensity"].diff(1)

    wrds["rd_intensity_lead1"] = grp["rd_intensity"].shift(-1)
    wrds["mkvaltq_lead1"] = grp["mkvaltq"].shift(-1)
    wrds["eps_growth_yoy_lead1"] = grp["eps_growth_yoy"].shift(-1)

    wrds["y_next_rd_intensity_change"] = wrds["rd_intensity_lead1"] - wrds["rd_intensity"]
    wrds["y_next_mktcap_growth"] = np.where(
        wrds["mkvaltq"].notna() & (wrds["mkvaltq"] != 0),
        wrds["mkvaltq_lead1"] / wrds["mkvaltq"] - 1.0,
        np.nan,
    )
    wrds["y_next_eps_growth_yoy"] = wrds["eps_growth_yoy_lead1"]

    keep = [
        "ticker",
        "year",
        "quarter",
        "gsector",
        "gsubind",
        "sic",
        "mkvaltq",
        "xrdq",
        "prccq",
        "epspxq",
        "cshoq",
        "rd_intensity",
        "log_mktcap",
        "ln_price",
        "eps_positive",
        "eps_growth_yoy",
        "price_growth_yoy",
        "mktcap_growth_qoq",
        "rd_intensity_change_qoq",
        "y_next_rd_intensity_change",
        "y_next_mktcap_growth",
        "y_next_eps_growth_yoy",
    ]
    keep = [c for c in keep if c in wrds.columns]
    return wrds[keep].copy()


def build_research_dataset(
    document_metrics: pd.DataFrame,
    initiation_scores: pd.DataFrame,
    sentences_with_keywords: pd.DataFrame,
    parsed_transcripts: Optional[pd.DataFrame],
    final_dataset: Optional[pd.DataFrame],
    wrds_features: pd.DataFrame,
) -> DatasetBuildResult:
    docs = attach_doc_keys(document_metrics)

    init = attach_doc_keys(initiation_scores) if len(initiation_scores) else initiation_scores.copy()
    merge_cols = [c for c in init.columns if c not in {"ticker", "year", "quarter"}]
    docs = docs.merge(init[merge_cols], on="doc_id", how="left")

    structural = compute_structural_features(sentences_with_keywords)
    docs = docs.merge(structural, on="doc_id", how="left")

    if parsed_transcripts is not None and len(parsed_transcripts):
        parsed = parsed_transcripts.copy()
        parsed["doc_id"] = (
            parsed["ticker"].astype(str)
            + "_"
            + parsed["year"].astype(int).astype(str)
            + "Q"
            + parsed["quarter"].astype(int).astype(str)
        )
        parsed_keep = [
            "doc_id",
            "speech_word_count",
            "qa_word_count",
            "num_qa_exchanges",
            "date",
        ]
        parsed_keep = [c for c in parsed_keep if c in parsed.columns]
        docs = docs.merge(parsed[parsed_keep], on="doc_id", how="left")

    if final_dataset is not None and len(final_dataset):
        fcols = [
            "ticker",
            "year",
            "quarter",
            "sector",
            "industry",
            "industry_name",
            "gsector",
            "gsubind",
            "date",
        ]
        fcols = [c for c in fcols if c in final_dataset.columns]
        fmeta = final_dataset[fcols].drop_duplicates(["ticker", "year", "quarter"])
        docs = docs.merge(fmeta, on=["ticker", "year", "quarter"], how="left", suffixes=("", "_call"))

    docs = docs.merge(wrds_features, on=["ticker", "year", "quarter"], how="left", suffixes=("", "_wrds"))

    docs["year_quarter"] = docs["year"].astype("Int64").astype(str) + "Q" + docs["quarter"].astype("Int64").astype(str)
    docs["quarter_index"] = make_quarter_index(docs)

    # Robust fallback when initiation rows are absent.
    for col, fill in {
        "total_ai_exchanges": 0.0,
        "analyst_initiated_ratio": 0.0,
        "management_pivot_ratio": 0.0,
        "ai_initiation_score": 0.5,
    }.items():
        if col in docs.columns:
            docs[col] = docs[col].fillna(fill)

    docs = docs.sort_values(["ticker", "year", "quarter"]).reset_index(drop=True)

    descriptions: Dict[str, str] = {
        "doc_id": "Call-level identifier: ticker_yearQquarter",
        "overall_kw_ai_ratio": "Share of AI-keyword sentences in whole call",
        "speech_kw_ai_ratio": "Share of AI-keyword sentences in prepared remarks",
        "qa_kw_ai_ratio": "Share of AI-keyword sentences in Q&A",
        "ai_initiation_score": "Management-pivot share among AI exchanges (higher=more management initiated)",
        "qa_sentence_share": "Q&A sentence share of total call sentences",
        "analyst_ai_share": "Share of AI Q&A sentences spoken by analysts",
        "management_ai_share": "Share of AI Q&A sentences spoken by management",
        "first_ai_turn_position": "Position of first AI mention in Q&A turn order (0-1)",
        "rd_intensity": "R&D over market value",
        "log_mktcap": "Log market value",
        "eps_growth_yoy": "Quarterly EPS year-over-year growth",
        "y_next_rd_intensity_change": "Next-quarter change in R&D intensity",
        "y_next_mktcap_growth": "Next-quarter market-value growth",
        "y_next_eps_growth_yoy": "Next-quarter EPS YoY growth",
    }

    data_dict = pd.DataFrame(
        {
            "variable": docs.columns,
            "dtype": [str(docs[c].dtype) for c in docs.columns],
            "non_missing_pct": [float(docs[c].notna().mean()) for c in docs.columns],
            "description": [descriptions.get(c, "") for c in docs.columns],
        }
    )

    return DatasetBuildResult(dataset=docs, data_dictionary=data_dict)


def run_basic_sanity_checks(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("Research dataset is empty")

    required = ["doc_id", "ticker", "year", "quarter", "quarter_index"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[["doc_id", "ticker", "year", "quarter"]].isna().any().any():
        raise ValueError("Key identifiers contain missing values")

    dup = df["doc_id"].duplicated().sum()
    if int(dup) > 0:
        raise ValueError(f"Duplicate doc_id rows found: {int(dup)}")

    # Quarter monotonicity within ticker
    check = df.sort_values(["ticker", "year", "quarter"]).groupby("ticker")["quarter_index"].diff().dropna()
    if (check < 0).any():
        raise ValueError("Quarter index is not monotonic within at least one ticker")
