from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


ALLOWED_WRDS_FIELDS: List[str] = [
    "costat",
    "curcdq",
    "datafmt",
    "indfmt",
    "consol",
    "tic",
    "datadate",
    "gvkey",
    "conm",
    "ggroup",
    "gind",
    "gsector",
    "gsubind",
    "naics",
    "sic",
    "spcindcd",
    "datafqtr",
    "fqtr",
    "fyearq",
    "cshoq",
    "epspxq",
    "xrdq",
    "capxy",
    "mkvaltq",
    "prccq",
]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def _parse_yearq(df: pd.DataFrame) -> pd.Series:
    yearq = pd.Series(index=df.index, dtype="object")
    if "datafqtr" in df.columns:
        yq = df["datafqtr"].astype(str)
        valid = yq.str.match(r"^\d{4}Q[1-4]$")
        yearq.loc[valid] = yq.loc[valid]

    missing = yearq.isna()
    if {"fyearq", "fqtr"}.issubset(df.columns):
        fy = pd.to_numeric(df["fyearq"], errors="coerce")
        fq = pd.to_numeric(df["fqtr"], errors="coerce")
        valid = missing & fy.notna() & fq.isin([1, 2, 3, 4])
        yearq.loc[valid] = fy.loc[valid].astype(int).astype(str) + "Q" + fq.loc[valid].astype(int).astype(str)

    missing = yearq.isna()
    if "datadate" in df.columns:
        dt = pd.to_datetime(df["datadate"], errors="coerce")
        valid = missing & dt.notna()
        yearq.loc[valid] = dt.loc[valid].dt.year.astype(str) + "Q" + dt.loc[valid].dt.quarter.astype(str)
    return yearq


def build_wrds_feature_store(
    wrds_path: str,
    output_dir: str = "outputs/features",
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    wrds = pd.read_csv(wrds_path, low_memory=False)
    keep_cols = [c for c in ALLOWED_WRDS_FIELDS if c in wrds.columns]
    wrds = wrds[keep_cols].copy()

    wrds["ticker"] = wrds["tic"].astype(str)
    wrds["datadate"] = pd.to_datetime(wrds["datadate"], errors="coerce")
    wrds["yearq"] = _parse_yearq(wrds)
    wrds = wrds.dropna(subset=["ticker", "yearq"]).copy()

    wrds["year"] = pd.to_numeric(wrds["yearq"].str.slice(0, 4), errors="coerce")
    wrds["quarter"] = pd.to_numeric(wrds["yearq"].str.extract(r"Q([1-4])", expand=False), errors="coerce")
    wrds = wrds.dropna(subset=["year", "quarter"]).copy()
    wrds["year"] = wrds["year"].astype(int)
    wrds["quarter"] = wrds["quarter"].astype(int)
    wrds["quarter_index"] = wrds["year"] * 4 + wrds["quarter"]

    wrds = wrds.sort_values(["ticker", "quarter_index", "datadate"])
    wrds = wrds.drop_duplicates(["ticker", "yearq"], keep="last").copy()
    if int(wrds.duplicated(["ticker", "yearq"]).sum()) > 0:
        raise ValueError("WRDS feature store has duplicate ticker+yearq keys after dedup.")

    wrds["sector_code"] = wrds.get("gsector")
    wrds["industry_code"] = wrds.get("gind")
    if "gsubind" in wrds.columns:
        wrds["industry_code"] = wrds["industry_code"].where(wrds["industry_code"].notna(), wrds["gsubind"])
    wrds["is_missing_industry"] = wrds["industry_code"].isna().astype(int)

    wrds["mkcap"] = pd.to_numeric(wrds.get("mkvaltq"), errors="coerce")
    wrds["log_mkcap"] = np.where(wrds["mkcap"] >= 0, np.log1p(wrds["mkcap"]), np.nan)
    wrds["price"] = pd.to_numeric(wrds.get("prccq"), errors="coerce")
    wrds["shares"] = pd.to_numeric(wrds.get("cshoq"), errors="coerce")
    wrds["eps"] = pd.to_numeric(wrds.get("epspxq"), errors="coerce")

    wrds["rd"] = pd.to_numeric(wrds.get("xrdq"), errors="coerce")
    wrds["rd_is_zero"] = wrds["rd"].fillna(np.nan).eq(0).astype(int)
    wrds["rd_is_missing"] = wrds["rd"].isna().astype(int)
    wrds["rd_per_share"] = _safe_divide(wrds["rd"], wrds["shares"])
    wrds["rd_intensity_mkcap"] = _safe_divide(wrds["rd"], wrds["mkcap"])

    wrds["capex_raw"] = pd.to_numeric(wrds.get("capxy"), errors="coerce")
    wrds["capex_intensity_mkcap"] = _safe_divide(wrds["capex_raw"], wrds["mkcap"])

    wrds = wrds.sort_values(["ticker", "quarter_index"]).reset_index(drop=True)
    grp = wrds.groupby("ticker", sort=False)
    prev_price = grp["price"].shift(1)
    wrds["ret_q"] = np.where(prev_price > 0, wrds["price"] / prev_price - 1.0, np.nan)
    wrds["d_log_mkcap"] = wrds["log_mkcap"] - grp["log_mkcap"].shift(1)
    wrds["d_eps"] = wrds["eps"] - grp["eps"].shift(1)
    wrds["d_rd"] = wrds["rd"] - grp["rd"].shift(1)
    wrds["d_rd_intensity"] = wrds["rd_intensity_mkcap"] - grp["rd_intensity_mkcap"].shift(1)
    wrds["lead_d_rd"] = grp["d_rd"].shift(-1)

    next_ticker = wrds["ticker"].shift(-1)
    cross = wrds["lead_d_rd"].notna() & (next_ticker != wrds["ticker"])
    if bool(cross.any()):
        raise ValueError("lead_d_rd cross-firm leakage detected.")

    feature_store_path = os.path.join(output_dir, "wrds_feature_store.parquet")
    wrds.to_parquet(feature_store_path, index=False)

    return {
        "feature_store": wrds,
        "feature_store_path": feature_store_path,
    }
