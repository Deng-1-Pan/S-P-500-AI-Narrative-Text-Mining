import os
import datetime
import hashlib
import json
import platform
import sys
import subprocess
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
import datasets as hf_datasets

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def prepare_dataset(
    output_dir: str = "data",
    wrds_meta_path: str = "data/wrds.csv",
    dataset_id: str = "kurry/sp500_earnings_transcripts",
    strict_repro: bool = False
) -> str:
    """
    Downloads raw S&P 500 earnings transcripts from HuggingFace, merges with WRDS metadata,
    and saves the final dataset to the specified output directory.
    
    Returns the path to the generated parquet dataset.
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("HF_TOKEN loaded from environment.")
    else:
        print("WARNING: HF_TOKEN not found. Checking if dataset is public or cached.")
        
    dataset_revision = os.getenv("HF_DATASET_REVISION")
    
    if strict_repro and not dataset_revision:
        raise RuntimeError("STRICT_REPRO=1 but HF_DATASET_REVISION is not set.")
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset: {dataset_id}")
    kwargs = {}
    if hf_token: kwargs["token"] = hf_token
    if dataset_revision: kwargs["revision"] = dataset_revision
    
    ds = load_dataset(dataset_id, **kwargs)
    train_fp = getattr(ds.get("train", None), "_fingerprint", None)
    print("Dataset loaded successfully.")
    print(f"Dataset revision: {dataset_revision}")
    print(f"Train fingerprint: {train_fp}")
    
    # Process transcripts
    required_cols = ["date", "symbol", "year", "quarter", "company_name", "structured_content"]
    split_df = ds["train"].to_pandas()[required_cols].copy()
    split_df["date"] = pd.to_datetime(split_df["date"], errors="coerce")
    split_df["year"] = pd.to_numeric(split_df["year"], errors="coerce").astype("Int64")
    split_df["quarter"] = pd.to_numeric(split_df["quarter"], errors="coerce").astype("Int64")
    
    if split_df["year"].isna().any() or split_df["quarter"].isna().any():
        raise RuntimeError("Found missing year/quarter in transcript dataset.")
        
    split_df["year"] = split_df["year"].astype(int)
    split_df["quarter"] = split_df["quarter"].astype(int)
    
    split_df = split_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    
    start_date = pd.Timestamp("2020-05-01")
    end_date_exclusive = pd.Timestamp("2025-06-01")
    split_df = split_df[(split_df["date"] >= start_date) & (split_df["date"] < end_date_exclusive)].copy()
    
    split_df["ticker"] = split_df["symbol"].astype("string").str.strip().str.upper()
    split_df = split_df.dropna(subset=["ticker"]).copy()
    
    rows_after_window = len(split_df)
    tickers_after_window = split_df["ticker"].nunique(dropna=True)
    
    required_years = list(range(2020, 2026))
    split_df["calendar_year"] = split_df["date"].dt.year
    years_by_ticker = split_df.groupby("ticker")["calendar_year"].agg(lambda s: set(int(x) for x in s.dropna().astype(int).tolist()))
    full_year_tickers = years_by_ticker.index[years_by_ticker.apply(lambda ys: set(required_years).issubset(ys))].tolist()
    split_df = split_df[split_df["ticker"].isin(full_year_tickers)].copy()
    
    rows_after_full_year = len(split_df)
    tickers_after_full_year = split_df["ticker"].nunique(dropna=True)
    
    print(f"Loading WRDS metadata from: {wrds_meta_path}")
    wrds_tickers = (
        pd.read_csv(wrds_meta_path, usecols=["tic"], dtype={"tic": "string"})["tic"]
        .astype("string").str.strip().str.upper().dropna().unique().tolist()
    )
    
    split_df = split_df[split_df["ticker"].isin(set(wrds_tickers))].copy()
    rows_after_wrds = len(split_df)
    tickers_after_wrds = split_df["ticker"].nunique(dropna=True)
    
    split_df = split_df[["date", "symbol", "ticker", "year", "quarter", "company_name", "structured_content"]].copy()
    
    # WRDS processing
    GICS_SECTOR_MAP = {
        10: "Energy", 15: "Materials", 20: "Industrials", 25: "Consumer Discretionary",
        30: "Consumer Staples", 35: "Health Care", 40: "Financials", 45: "Information Technology",
        50: "Communication Services", 55: "Utilities", 60: "Real Estate"
    }
    
    # We will use the simplified approach for industry_name or load it completely
    # Loading WRDS
    wrds_df = pd.read_csv(
        wrds_meta_path,
        dtype={"tic": "string", "conm": "string", "gsector": "Int64", "gsubind": "Int64"},
        parse_dates=["datadate"]
    )
    wrds_df["tic"] = wrds_df["tic"].astype("string").str.strip().str.upper()
    wrds_df["gsector"] = pd.to_numeric(wrds_df["gsector"], errors="coerce").astype("Int64")
    wrds_df["gsubind"] = pd.to_numeric(wrds_df["gsubind"], errors="coerce").astype("Int64")
    wrds_df["sector"] = wrds_df["gsector"].map(GICS_SECTOR_MAP)
    
    if wrds_df["sector"].isna().any():
        raise RuntimeError("Unexpected missing sector mapping from gsector.")
        
    wrds_df["industry_code"] = wrds_df["gsubind"].astype("Int64").astype("string").str.zfill(8)
    wrds_df["industry"] = "GICS_" + wrds_df["industry_code"]
    wrds_df["industry_name"] = wrds_df["industry"] # simplified for pipeline
    
    wrds_time_df = (
        wrds_df[["tic", "conm", "datadate", "gsector", "gsubind", "industry_code", "sector", "industry", "industry_name"]]
        .rename(columns={"tic": "ticker", "conm": "company"})
        .dropna(subset=["ticker", "datadate"])
        .sort_values(["datadate", "ticker"])
        .reset_index(drop=True)
    )
    
    # Merging
    valid_mask = split_df["ticker"].notna() & split_df["date"].notna()
    valid_df = split_df.loc[valid_mask].copy()
    valid_df = valid_df.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    
    merged_valid = pd.merge_asof(
        valid_df,
        wrds_time_df,
        left_on="date",
        right_on="datadate",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    
    no_asof_match = merged_valid["datadate"].isna()
    if bool(no_asof_match.any()):
        raise RuntimeError("merge_asof produced rows with no <=datadate match.")
        
    merged_df = merged_valid.copy()
    merged_df["Company_name"] = merged_df["company_name"].replace(r"^\s*$", pd.NA, regex=True).combine_first(merged_df["company"])
    merged_df.loc[merged_df["Company_name"].isna(), "Company_name"] = merged_df.loc[merged_df["Company_name"].isna(), "ticker"]
    
    final_dataset_df = merged_df[[
        "ticker", "Company_name", "sector", "industry", "industry_name", 
        "gsector", "gsubind", "industry_code", "datadate", 
        "year", "quarter", "date", "structured_content"
    ]].rename(columns={"datadate": "wrds_datadate"})
    
    final_dataset_df["classification_source"] = "wrds_official_codes"
    final_dataset_df["classification_confidence"] = "high"
    
    final_dataset_df = final_dataset_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    leakage_mask = final_dataset_df["wrds_datadate"].notna() & (final_dataset_df["wrds_datadate"] > final_dataset_df["date"])
    if bool(leakage_mask.any()):
        raise RuntimeError("Found wrds_datadate > transcript date (time leakage).")
        
    # Validation
    key_missing = final_dataset_df[["ticker", "sector", "industry", "year", "quarter", "date", "structured_content", "gsector", "gsubind", "wrds_datadate"]].isna().sum()
    if int(key_missing.sum()) != 0:
        raise RuntimeError("Final dataset contains missing values in required columns.")
        
    # Save files
    output_csv = os.path.join(output_dir, "final_dataset.csv")
    output_parquet = os.path.join(output_dir, "final_dataset.parquet")
    manifest_path = os.path.join(output_dir, "final_dataset_manifest.json")
    
    final_dataset_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    final_dataset_df.to_parquet(output_parquet, index=False)
    
    print(f"Saved total {len(final_dataset_df)} rows to: {output_parquet}")
    
    # Save manifest
    git_head = None
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
        
    manifest = {
        "run_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "git_head": git_head,
        "dataset": {
            "id": dataset_id,
            "revision": dataset_revision,
            "train_fingerprint": train_fp,
        },
        "filters": {
            "start_date": str(start_date),
            "end_date_exclusive": str(end_date_exclusive),
            "rows_after_wrds": rows_after_wrds,
        },
        "wrds_meta": {
            "path": wrds_meta_path,
        },
        "output": {
            "rows": int(len(final_dataset_df)),
            "unique_tickers": int(final_dataset_df["ticker"].nunique(dropna=True)),
            "parquet": {"path": output_parquet, "sha256": _sha256_file(output_parquet)},
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    return output_parquet
