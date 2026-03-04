"""Utilities for parsing and attaching doc_id-derived keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ParsedDocId:
    ticker: Optional[str]
    year: Optional[int]
    quarter: Optional[int]
    yearq: Optional[str]


def parse_doc_id(
    doc_id: str,
    allow_ticker_without_q: bool = False,
    allow_ticker_on_invalid: bool = False,
    raise_on_invalid: bool = False,
) -> ParsedDocId:
    text = str(doc_id)
    parts = text.rsplit("_", 1)
    if len(parts) != 2:
        return ParsedDocId(None, None, None, None)

    ticker, yq = parts[0], parts[1]
    if "Q" not in yq:
        if allow_ticker_without_q:
            return ParsedDocId(ticker, None, None, None)
        return ParsedDocId(None, None, None, None)

    try:
        year = int(yq.split("Q")[0])
        quarter = int(yq.split("Q")[1])
    except Exception:
        if raise_on_invalid:
            raise
        if allow_ticker_on_invalid:
            return ParsedDocId(ticker, None, None, None)
        return ParsedDocId(None, None, None, None)

    return ParsedDocId(ticker, year, quarter, f"{year}Q{quarter}")


def attach_doc_keys(
    df: pd.DataFrame,
    doc_id_col: str = "doc_id",
    ticker_col: str = "ticker",
    year_col: str = "year",
    quarter_col: str = "quarter",
    yearq_col: str = "yearq",
    keep_existing: bool = False,
    **parse_kwargs,
) -> pd.DataFrame:
    out = df.copy()
    parsed = out[doc_id_col].apply(lambda x: parse_doc_id(x, **parse_kwargs))

    ticker_vals = [x.ticker for x in parsed]
    year_vals = [x.year for x in parsed]
    quarter_vals = [x.quarter for x in parsed]
    yearq_vals = [x.yearq for x in parsed]

    if keep_existing and ticker_col in out.columns:
        out[ticker_col] = out[ticker_col].where(out[ticker_col].notna(), ticker_vals)
    else:
        out[ticker_col] = ticker_vals

    if keep_existing and year_col in out.columns:
        out[year_col] = out[year_col].where(out[year_col].notna(), year_vals)
    else:
        out[year_col] = year_vals

    if keep_existing and quarter_col in out.columns:
        out[quarter_col] = out[quarter_col].where(out[quarter_col].notna(), quarter_vals)
    else:
        out[quarter_col] = quarter_vals

    if yearq_col:
        if keep_existing and yearq_col in out.columns:
            out[yearq_col] = out[yearq_col].where(out[yearq_col].notna(), yearq_vals)
        else:
            out[yearq_col] = yearq_vals

    return out
