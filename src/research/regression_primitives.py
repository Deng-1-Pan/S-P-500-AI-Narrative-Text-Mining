"""Shared regression primitives for Stage 13 and Stage 15."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def fit_stage16_regression(
    panel: pd.DataFrame,
    y: str,
    x_list: List[str],
    fe_sector: str = "gsector",
    fe_time: str = "yearq",
    cluster_col: str = "ticker",
    min_obs: int = 20,
) -> Tuple[pd.DataFrame, str]:
    use_cols = [y] + x_list + [fe_sector, fe_time, cluster_col]
    use_cols = [c for c in use_cols if c in panel.columns]
    work = panel[use_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < min_obs:
        return (
            pd.DataFrame(columns=["model", "y", "x", "coef", "se", "p_value", "n_obs", "r2", "se_type"]),
            "insufficient_data",
        )

    work = work.copy()
    work[fe_sector] = work[fe_sector].astype(str)
    work[fe_time] = work[fe_time].astype(str)
    work[cluster_col] = work[cluster_col].astype(str)

    formula = f"{y} ~ {' + '.join(x_list)} + C({fe_sector}) + C({fe_time})"
    se_type = "cluster_by_firm"
    try:
        model = smf.ols(formula=formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work[cluster_col]})
    except Exception:
        model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")
        se_type = "HC1_fallback"

    rows = []
    for x in x_list:
        rows.append(
            {
                "model": f"{y}_on_metadata",
                "y": y,
                "x": x,
                "coef": float(model.params.get(x, np.nan)),
                "se": float(model.bse.get(x, np.nan)),
                "p_value": float(model.pvalues.get(x, np.nan)),
                "n_obs": int(model.nobs),
                "r2": float(getattr(model, "rsquared", np.nan)),
                "se_type": se_type,
            }
        )
    return pd.DataFrame(rows), formula


def fit_path_regression(
    df: pd.DataFrame,
    mechanism_name: str,
    x_var: str,
    y_var: str,
    controls: List[str],
    fe_var: str,
    min_obs: int = 8,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    needed = [x_var, y_var] + [c for c in controls if c in df.columns]
    work = df[needed + [fe_var, "year_quarter"]].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[x_var, y_var])

    if len(work) < min_obs:
        return (
            {
                "mechanism": mechanism_name,
                "dependent_var": y_var,
                "feature": x_var,
                "coef": np.nan,
                "std_err": np.nan,
                "t_value": np.nan,
                "p_value": np.nan,
                "r_squared": np.nan,
                "n_obs": int(len(work)),
                "formula": "insufficient_data",
            },
            work,
        )

    rhs = [x_var] + [c for c in controls if c in work.columns]
    formula = f"{y_var} ~ {' + '.join(rhs)} + C({fe_var}) + C(year_quarter)"

    try:
        model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")
    except Exception:
        formula = f"{y_var} ~ {' + '.join(rhs)}"
        model = smf.ols(formula=formula, data=work).fit(cov_type="HC1")

    row = {
        "mechanism": mechanism_name,
        "dependent_var": y_var,
        "feature": x_var,
        "coef": float(model.params.get(x_var, np.nan)),
        "std_err": float(model.bse.get(x_var, np.nan)),
        "t_value": float(model.tvalues.get(x_var, np.nan)),
        "p_value": float(model.pvalues.get(x_var, np.nan)),
        "r_squared": float(getattr(model, "rsquared", np.nan)),
        "n_obs": int(model.nobs),
        "formula": formula,
    }
    return row, work
