from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import sparse
from scipy.stats import kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from src.utils.ml_helpers import aggregate_doc_text


@dataclass
class RegressionResult:
    name: str
    target: str
    n_obs: int
    r2: float
    coef_table: pd.DataFrame


@dataclass
class ModelComparisonResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame


def _safe_one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tau, _ = kendalltau(y_true, y_pred)
    if tau is None or np.isnan(tau):
        return 0.0
    return float(tau)


def run_fe_regressions(
    df: pd.DataFrame,
    output_dir: str,
    add_firm_fe: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    model_specs = [
        ("fe_rd_next", "y_next_rd_intensity_change"),
        ("fe_mktcap_next", "y_next_mktcap_growth"),
        ("fe_eps_next", "y_next_eps_growth_yoy"),
    ]
    x_core = [
        "overall_kw_ai_ratio",
        "qa_kw_ai_ratio",
        "speech_kw_ai_ratio",
        "ai_initiation_score",
        "analyst_ai_share",
        "management_ai_share",
        "first_ai_turn_position",
    ]
    controls = ["log_mktcap", "rd_intensity", "eps_positive", "ln_price", "eps_growth_yoy"]

    rows = []
    detail_frames: List[pd.DataFrame] = []

    for model_name, target in model_specs:
        use_cols = [target] + x_core + controls + ["gsector", "year_quarter", "ticker"]
        use_cols = [c for c in use_cols if c in df.columns]
        work = df[use_cols].copy()
        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=[target, "gsector", "year_quarter"])
        work["gsector"] = work["gsector"].astype(str)
        work["year_quarter"] = work["year_quarter"].astype(str)
        if "ticker" in work.columns:
            work["ticker"] = work["ticker"].astype(str)
        if len(work) < 400:
            continue

        for col in x_core + controls:
            if col in work.columns:
                work[col] = winsorize_series(work[col])
        work[target] = winsorize_series(work[target])

        rhs_terms = [c for c in x_core + controls if c in work.columns]
        rhs = " + ".join(rhs_terms + ["C(gsector)", "C(year_quarter)"])
        if add_firm_fe and "ticker" in work.columns and work["ticker"].nunique() < 600:
            rhs += " + C(ticker)"
        formula = f"{target} ~ {rhs}"

        fitted = smf.ols(formula=formula, data=work).fit(cov_type="HC1")

        rows.append(
            {
                "model": model_name,
                "target": target,
                "n_obs": int(fitted.nobs),
                "r2": float(fitted.rsquared),
                "adj_r2": float(fitted.rsquared_adj),
                "spec": formula,
            }
        )

        coef_df = pd.DataFrame(
            {
                "term": fitted.params.index,
                "coef": fitted.params.values,
                "std_err": fitted.bse.values,
                "p_value": fitted.pvalues.values,
                "model": model_name,
                "target": target,
            }
        )
        detail_frames.append(coef_df)

    summary_df = pd.DataFrame(rows)
    coef_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    summary_df.to_csv(os.path.join(output_dir, "fe_regression_summary.csv"), index=False)
    coef_all.to_csv(os.path.join(output_dir, "fe_regression_coefficients.csv"), index=False)

    # Count-data robustness on AI exchanges.
    if {"total_ai_exchanges", "gsector", "year_quarter"}.issubset(df.columns):
        poi_cols = [
            "total_ai_exchanges",
            "overall_kw_ai_ratio",
            "qa_kw_ai_ratio",
            "speech_kw_ai_ratio",
            "log_mktcap",
            "rd_intensity",
            "eps_positive",
            "gsector",
            "year_quarter",
        ]
        poi_cols = [c for c in poi_cols if c in df.columns]
        poi = df[poi_cols].dropna(subset=["total_ai_exchanges", "gsector", "year_quarter"]).copy()
        poi = poi.replace([np.inf, -np.inf], np.nan).dropna()
        poi["gsector"] = poi["gsector"].astype(str)
        poi["year_quarter"] = poi["year_quarter"].astype(str)
        if len(poi) >= 500:
            for col in ["overall_kw_ai_ratio", "qa_kw_ai_ratio", "speech_kw_ai_ratio", "log_mktcap", "rd_intensity"]:
                if col in poi.columns:
                    poi[col] = winsorize_series(poi[col])

            rhs = [c for c in ["overall_kw_ai_ratio", "qa_kw_ai_ratio", "speech_kw_ai_ratio", "log_mktcap", "rd_intensity", "eps_positive"] if c in poi.columns]
            formula = "total_ai_exchanges ~ " + " + ".join(rhs + ["C(gsector)", "C(year_quarter)"])
            glm = smf.poisson(formula=formula, data=poi).fit(cov_type="HC1", disp=0)
            poi_df = pd.DataFrame(
                {
                    "term": glm.params.index,
                    "coef": glm.params.values,
                    "std_err": glm.bse.values,
                    "p_value": glm.pvalues.values,
                }
            )
            poi_df.to_csv(os.path.join(output_dir, "poisson_robustness_coefficients.csv"), index=False)

    return summary_df


def _temporal_train_test_split(df: pd.DataFrame, test_quarters: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q = sorted(df["quarter_index"].dropna().unique())
    if len(q) <= test_quarters:
        raise ValueError("Not enough quarters for requested temporal split")
    cutoff = q[-test_quarters]
    train = df[df["quarter_index"] < cutoff].copy()
    test = df[df["quarter_index"] >= cutoff].copy()
    return train, test


def _build_tabular_pipeline(num_cols: Sequence[str], cat_cols: Sequence[str]) -> Pipeline:
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(num_cols),
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _safe_one_hot())]
                ),
                list(cat_cols),
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    return Pipeline(steps=[("pre", pre), ("model", model)])


def run_model_comparison(
    df: pd.DataFrame,
    target: str,
    output_dir: str,
    test_quarters: int = 4,
) -> ModelComparisonResult:
    os.makedirs(output_dir, exist_ok=True)

    base_cols = [
        target,
        "quarter_index",
        "ticker",
        "year_quarter",
        "gsector",
        "log_mktcap",
        "rd_intensity",
        "eps_positive",
        "ln_price",
        "eps_growth_yoy",
        "overall_kw_ai_ratio",
        "qa_kw_ai_ratio",
        "speech_kw_ai_ratio",
        "ai_initiation_score",
        "analyst_ai_share",
        "management_ai_share",
        "first_ai_turn_position",
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    work = df[base_cols].dropna(subset=[target, "quarter_index", "year_quarter", "gsector"]).copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    if "gsector" in work.columns:
        work["gsector"] = work["gsector"].astype(str)
    if "year_quarter" in work.columns:
        work["year_quarter"] = work["year_quarter"].astype(str)
    if len(work) < 800:
        raise ValueError(f"Not enough rows for model comparison on {target}: {len(work)}")

    # Winsorize heavy-tail targets/features for stability.
    work[target] = winsorize_series(work[target])
    for c in ["log_mktcap", "rd_intensity", "ln_price", "eps_growth_yoy"]:
        if c in work.columns:
            work[c] = winsorize_series(work[c])

    train, test = _temporal_train_test_split(work, test_quarters=test_quarters)

    finance_features = [c for c in ["log_mktcap", "rd_intensity", "eps_positive", "ln_price", "eps_growth_yoy"] if c in work.columns]
    text_features = [c for c in ["overall_kw_ai_ratio", "qa_kw_ai_ratio", "speech_kw_ai_ratio", "ai_initiation_score", "analyst_ai_share", "management_ai_share", "first_ai_turn_position"] if c in work.columns]
    cat_cols = ["gsector"] # Removed year_quarter because it'll entirely be unseen OOS categories

    model_defs = {
        "Finance-only": finance_features + cat_cols,
        "Text-only": text_features + cat_cols,
        "Finance+Text": finance_features + text_features + cat_cols,
    }

    pred_rows = []
    metric_rows = []
    for name, feats in model_defs.items():
        feats = [c for c in feats if c in work.columns]
        num_cols = [c for c in feats if c not in cat_cols]
        cats = [c for c in feats if c in cat_cols]

        pipe = _build_tabular_pipeline(num_cols=num_cols, cat_cols=cats)
        pipe.fit(train[feats], train[target])
        pred = pipe.predict(test[feats])

        metric_rows.append(
            {
                "model": name,
                "target": target,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "r2_test": float(r2_score(test[target], pred)),
                "mae_test": float(mean_absolute_error(test[target], pred)),
                "rmse_test": float(math.sqrt(mean_squared_error(test[target], pred))),
                "kendall_tau_test": _kendall_tau(test[target].to_numpy(), pred),
            }
        )

        p = test[["ticker", "year_quarter", target]].copy()
        p["pred"] = pred
        p["model"] = name
        pred_rows.append(p)

    # Speech vs Q&A vs Analyst block comparison (text-only blocks).
    block_defs = {
        "Speech-block": ["speech_kw_ai_ratio", "gsector"],
        "Q&A-block": ["qa_kw_ai_ratio", "gsector"],
        "Analyst-block": ["analyst_ai_share", "management_ai_share", "first_ai_turn_position", "gsector"],
    }

    for name, feats in block_defs.items():
        feats = [c for c in feats if c in work.columns]
        if len(feats) < 2:
            continue
        num_cols = [c for c in feats if c not in cat_cols]
        cats = [c for c in feats if c in cat_cols]
        pipe = _build_tabular_pipeline(num_cols=num_cols, cat_cols=cats)
        pipe.fit(train[feats], train[target])
        pred = pipe.predict(test[feats])
        metric_rows.append(
            {
                "model": name,
                "target": target,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "r2_test": float(r2_score(test[target], pred)),
                "mae_test": float(mean_absolute_error(test[target], pred)),
                "rmse_test": float(math.sqrt(mean_squared_error(test[target], pred))),
                "kendall_tau_test": _kendall_tau(test[target].to_numpy(), pred),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("r2_test", ascending=False).reset_index(drop=True)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()

    metrics_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    preds_df.to_csv(os.path.join(output_dir, "model_predictions.csv"), index=False)

    return ModelComparisonResult(summary=metrics_df, predictions=preds_df)


def _aggregate_doc_text(sentences_df: pd.DataFrame, section: Optional[str] = None) -> pd.DataFrame:
    return aggregate_doc_text(
        sentences_df,
        section=section,
        ai_only=False,
        mask_non_ai=True,
        output_col="text",
        sort=True,
    )


def _build_example_sentences(
    sentences_df: pd.DataFrame,
    terms: Sequence[str],
    max_examples: int = 1,
    max_source_rows: int = 200_000,
) -> pd.DataFrame:
    rows = []
    keep_cols = [c for c in ["doc_id", "text", "section", "role", "kw_is_ai"] if c in sentences_df.columns]
    source = sentences_df[keep_cols].copy()
    if "kw_is_ai" in source.columns:
        ai_only = source[source["kw_is_ai"].fillna(False).astype(bool)]
        if len(ai_only) > 0:
            source = ai_only
    if len(source) > max_source_rows:
        source = source.sample(max_source_rows, random_state=42)
    source["text_lower"] = source["text"].fillna("").astype(str).str.lower()

    for term in terms:
        term_l = str(term).lower().strip()
        if not term_l:
            continue
        hits = source[source["text_lower"].str.contains(term_l, regex=False, na=False)].head(max_examples)
        if hits.empty:
            rows.append({"feature": term, "example_doc_id": "", "example_sentence": ""})
            continue
        for _, row in hits.iterrows():
            rows.append(
                {
                    "feature": term,
                    "example_doc_id": row["doc_id"],
                    "example_sentence": row["text"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "example_doc_id", "example_sentence"])
    return pd.DataFrame(rows)


def run_interpretable_lasso(
    dataset: pd.DataFrame,
    sentences_df: pd.DataFrame,
    output_dir: str,
    target: str = "y_next_mktcap_growth",
    section: str = "qa",
    max_features: int = 900,
    min_df: int = 40,
    ngram_range: Tuple[int, int] = (1, 2),
    test_quarters: int = 4,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    text_df = _aggregate_doc_text(sentences_df, section=section)
    base_cols = [
        "doc_id",
        "quarter_index",
        "ticker",
        target,
        "log_mktcap",
        "rd_intensity",
        "eps_positive",
        "ln_price",
        "eps_growth_yoy",
    ]
    base_cols = [c for c in base_cols if c in dataset.columns]
    base = dataset[base_cols].merge(text_df, on="doc_id", how="inner")
    base = base.dropna(subset=[target, "quarter_index"]).copy()
    if len(base) > 5000:
        qn = max(1, int(base["quarter_index"].nunique()))
        per_q = max(80, 5000 // qn)
        base = (
            base.sort_values(["quarter_index", "doc_id"])
            .groupby("quarter_index", group_keys=False)
            .head(per_q)
            .reset_index(drop=True)
        )

    finance_cols = [c for c in ["log_mktcap", "rd_intensity", "eps_positive", "ln_price", "eps_growth_yoy"] if c in base.columns]
    for c in finance_cols + [target]:
        base[c] = winsorize_series(base[c])

    train, test = _temporal_train_test_split(base, test_quarters=test_quarters)

    train = train.sort_values("quarter_index").reset_index(drop=True)
    test = test.sort_values("quarter_index").reset_index(drop=True)

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )

    train_texts = train["text"].fillna("").astype(str).tolist()
    test_texts = test["text"].fillna("").astype(str).tolist()
    try:
        X_text_train = vec.fit_transform(train_texts)
    except ValueError:
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=max(5, min_df // 4),
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )
        X_text_train = vec.fit_transform(train_texts)
    X_text_test = vec.transform(test_texts)

    scaler = StandardScaler()
    if finance_cols:
        fin_train = train[finance_cols].copy()
        fin_medians = fin_train.median(numeric_only=True)
        fin_train = fin_train.fillna(fin_medians)
        fin_test = test[finance_cols].copy().fillna(fin_medians)
        X_fin_train_arr = scaler.fit_transform(fin_train)
        X_fin_test_arr = scaler.transform(fin_test)
    else:
        fin_medians = pd.Series(dtype=float)
        X_fin_train_arr = np.zeros((len(train), 0))
        X_fin_test_arr = np.zeros((len(test), 0))

    X_fin_train = sparse.csr_matrix(X_fin_train_arr)
    X_fin_test = sparse.csr_matrix(X_fin_test_arr)

    X_train = sparse.hstack([X_fin_train, X_text_train], format="csr")
    X_test = sparse.hstack([X_fin_test, X_text_test], format="csr")

    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)

    # Lightweight time-aware alpha tuning
    tscv = TimeSeriesSplit(n_splits=2)
    alpha_grid = np.logspace(-5, -2, 4) # [1e-5, 1e-4, 1e-3, 1e-2] to avoid over-shrinking TFIDF
    l1_ratio = 0.9
    best_alpha = float(alpha_grid[0])
    best_mae = float("inf")
    for alpha in alpha_grid:
        fold_err = []
        for tr_idx, va_idx in tscv.split(np.arange(X_train.shape[0])):
            m = ElasticNet(alpha=float(alpha), l1_ratio=l1_ratio, max_iter=6000)
            m.fit(X_train[tr_idx], y_train[tr_idx])
            pred_va = m.predict(X_train[va_idx])
            fold_err.append(mean_absolute_error(y_train[va_idx], pred_va))
        avg = float(np.mean(fold_err)) if fold_err else float("inf")
        if avg < best_mae:
            best_mae = avg
            best_alpha = float(alpha)

    model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=6000)
    model.fit(X_train, y_train)
    # Ensure interpretability output is non-degenerate when CV picks an overly sparse alpha.
    if np.count_nonzero(model.coef_[len(finance_cols):]) == 0 and X_train.shape[1] > len(finance_cols):
        alpha_try = best_alpha
        for _ in range(5):
            alpha_try = alpha_try * 0.3
            relaxed = ElasticNet(alpha=alpha_try, l1_ratio=l1_ratio, max_iter=6000)
            relaxed.fit(X_train, y_train)
            if np.count_nonzero(relaxed.coef_[len(finance_cols):]) > 0:
                model = relaxed
                best_alpha = alpha_try
                break

    y_pred = model.predict(X_test)
    metrics = pd.DataFrame(
        [
            {
                "target": target,
                "section": section,
                "n_train": len(train),
                "n_test": len(test),
                "r2_test": float(r2_score(y_test, y_pred)),
                "mae_test": float(mean_absolute_error(y_test, y_pred)),
                "rmse_test": float(math.sqrt(mean_squared_error(y_test, y_pred))),
                "kendall_tau_test": _kendall_tau(y_test, y_pred),
                "alpha": float(best_alpha),
                "l1_ratio": float(l1_ratio),
            }
        ]
    )

    feature_names = finance_cols + [f"text::{t}" for t in vec.get_feature_names_out()]
    coefs = model.coef_

    term_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "block": ["finance"] * len(finance_cols) + ["text"] * (len(feature_names) - len(finance_cols)),
        }
    )
    term_df = term_df[term_df["coefficient"] != 0].copy()

    # Add text document frequency and raw token name for text features.
    doc_freq = np.array([])
    if X_text_train.shape[1] > 0:
        doc_freq = np.asarray((X_text_train > 0).sum(axis=0)).reshape(-1)
        df_map = {f"text::{name}": int(doc_freq[idx]) for idx, name in enumerate(vec.get_feature_names_out())}
        term_df["doc_frequency"] = term_df["feature"].map(df_map).fillna(0).astype(int)
    else:
        term_df["doc_frequency"] = 0
    term_df["log_doc_frequency"] = np.log1p(term_df["doc_frequency"])

    # Stability selection across rolling windows.
    windows = []
    q_vals = sorted(train["quarter_index"].dropna().unique())
    if len(q_vals) >= 10:
        cutoffs = q_vals[8::6][:3]
        for cutoff in cutoffs:
            sub = train[train["quarter_index"] <= cutoff].copy()
            if len(sub) < 500:
                continue
            X_text_sub = vec.transform(sub["text"].fillna("").astype(str).tolist())
            if finance_cols:
                X_fin_sub_arr = scaler.transform(sub[finance_cols].fillna(fin_medians))
            else:
                X_fin_sub_arr = np.zeros((len(sub), 0))
            X_fin_sub = sparse.csr_matrix(X_fin_sub_arr)
            X_sub = sparse.hstack([X_fin_sub, X_text_sub], format="csr")
            y_sub = sub[target].to_numpy(dtype=float)

            stable_model = ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, max_iter=6000)
            stable_model.fit(X_sub, y_sub)
            
            sub_alpha = best_alpha
            if np.count_nonzero(stable_model.coef_[len(finance_cols):]) == 0:
                for _ in range(4):
                    sub_alpha = sub_alpha * 0.3
                    rel = ElasticNet(alpha=sub_alpha, l1_ratio=l1_ratio, max_iter=6000)
                    rel.fit(X_sub, y_sub)
                    if np.count_nonzero(rel.coef_[len(finance_cols):]) > 0:
                        stable_model = rel
                        break
            
            windows.append(stable_model.coef_)

    if windows:
        coef_matrix = np.vstack(windows)
        nonzero_freq = (coef_matrix != 0).mean(axis=0)
        sign_mean = np.sign(coef_matrix).mean(axis=0)
    else:
        nonzero_freq = np.zeros_like(coefs)
        sign_mean = np.zeros_like(coefs)

    stability_df = pd.DataFrame(
        {
            "feature": feature_names,
            "stability_freq": nonzero_freq,
            "avg_sign": sign_mean,
        }
    )

    term_df = term_df.merge(stability_df, on="feature", how="left")
    term_df["raw_term"] = term_df["feature"].str.replace("^text::", "", regex=True)

    # Fallback: if regularization yields no text terms, use signed univariate scores for interpretability table.
    if term_df[term_df["block"] == "text"].empty and X_text_train.shape[1] > 0:
        y_center = y_train - float(np.mean(y_train))
        score = np.asarray(X_text_train.T.dot(y_center)).reshape(-1)
        denom = np.sqrt(np.asarray(X_text_train.power(2).sum(axis=0)).reshape(-1)) + 1e-9
        score = score / denom
        if len(score) > 0:
            pos_idx = np.argsort(score)[-20:]
            neg_idx = np.argsort(score)[:20]
            idx = np.unique(np.concatenate([pos_idx, neg_idx]))
            fallback = pd.DataFrame(
                {
                    "feature": [f"text::{vec.get_feature_names_out()[i]}" for i in idx],
                    "coefficient": [float(score[i]) for i in idx],
                    "block": "text",
                    "doc_frequency": [int(doc_freq[i]) if len(doc_freq) > i else 0 for i in idx],
                    "log_doc_frequency": [float(np.log1p(doc_freq[i])) if len(doc_freq) > i else 0.0 for i in idx],
                    "stability_freq": 0.0,
                    "avg_sign": [float(np.sign(score[i])) for i in idx],
                }
            )
            fallback["raw_term"] = fallback["feature"].str.replace("^text::", "", regex=True)
            term_df = pd.concat([term_df, fallback], ignore_index=True)

    top_text = term_df[term_df["block"] == "text"].copy()
    top_pos = top_text.nlargest(20, "coefficient")
    top_neg = top_text.nsmallest(20, "coefficient")
    top_terms = pd.concat([top_pos, top_neg], ignore_index=True)

    examples = _build_example_sentences(
        sentences_df[sentences_df["section"].astype(str).str.lower() == section.lower()].copy(),
        terms=top_terms["raw_term"].tolist(),
        max_examples=1,
    )
    top_terms = top_terms.merge(examples, left_on="raw_term", right_on="feature", how="left", suffixes=("", "_example"))

    preds = test[["doc_id", "ticker", "quarter_index", target]].copy()
    preds["prediction"] = y_pred
    preds["residual"] = preds[target] - preds["prediction"]

    metrics.to_csv(os.path.join(output_dir, "lasso_metrics.csv"), index=False)
    term_df.sort_values("coefficient", ascending=False).to_csv(os.path.join(output_dir, "lasso_terms_full.csv"), index=False)
    top_terms.to_csv(os.path.join(output_dir, "lasso_top_terms_with_examples.csv"), index=False)
    preds.to_csv(os.path.join(output_dir, "lasso_predictions.csv"), index=False)
    stability_df.to_csv(os.path.join(output_dir, "lasso_stability.csv"), index=False)

    return {
        "metrics": metrics,
        "terms": term_df,
        "top_terms": top_terms,
        "predictions": preds,
        "stability": stability_df,
    }


def build_deep_dive_cases(
    dataset: pd.DataFrame,
    lasso_predictions: pd.DataFrame,
    sentences_df: pd.DataFrame,
    output_path: str,
    n_cases: int = 2,
) -> pd.DataFrame:
    if lasso_predictions.empty:
        case_df = pd.DataFrame(columns=["doc_id", "ticker", "case_type"])
        case_df.to_csv(output_path, index=False)
        return case_df

    pred = lasso_predictions.copy()
    enrich_cols = [c for c in ["doc_id", "ticker", "overall_kw_ai_ratio", "qa_kw_ai_ratio"] if c in dataset.columns]
    pred = pred.merge(dataset[enrich_cols], on=["doc_id", "ticker"], how="left")
    active = pred[
        (pred.get("overall_kw_ai_ratio", 0).fillna(0) > 0) | (pred.get("qa_kw_ai_ratio", 0).fillna(0) > 0)
    ].copy()
    candidate = active if len(active) >= 2 else pred

    picks = []
    high = candidate.nlargest(1, "prediction")
    low = candidate.nsmallest(1, "prediction")
    for case_type, frame in [("high_predicted_growth", high), ("low_predicted_growth", low)]:
        if frame.empty:
            continue
        row = frame.iloc[0]
        picks.append({"doc_id": row["doc_id"], "ticker": row.get("ticker", ""), "case_type": case_type})

    picks_df = pd.DataFrame(picks).drop_duplicates("doc_id")
    if len(picks_df) > n_cases:
        picks_df = picks_df.head(n_cases)

    merged = picks_df.merge(
        dataset[
            [
                "doc_id",
                "ticker",
                "year_quarter",
                "overall_kw_ai_ratio",
                "qa_kw_ai_ratio",
                "speech_kw_ai_ratio",
                "ai_initiation_score",
                "analyst_ai_share",
                "management_ai_share",
                "rd_intensity",
                "log_mktcap",
                "eps_growth_yoy",
                "y_next_mktcap_growth",
            ]
        ],
        on=["doc_id", "ticker"],
        how="left",
    )

    # Add representative AI snippets for each case.
    src = sentences_df.copy()
    src["section"] = src["section"].astype(str).str.lower()
    src["kw_is_ai"] = src["kw_is_ai"].fillna(False).astype(bool)
    src = src[src["kw_is_ai"]].copy()

    snippets = []
    for doc in merged["doc_id"].tolist():
        sub = src[src["doc_id"] == doc].sort_values(["section", "turn_idx", "sentence_idx"]).head(3)
        if sub.empty:
            sub = (
                sentences_df[sentences_df["doc_id"] == doc]
                .sort_values(["section", "turn_idx", "sentence_idx"])
                .head(3)
            )
        text = " || ".join(sub["text"].astype(str).tolist())
        snippets.append({"doc_id": doc, "ai_snippets": text})
    merged = merged.merge(pd.DataFrame(snippets), on="doc_id", how="left")

    merged.to_csv(output_path, index=False)
    return merged
