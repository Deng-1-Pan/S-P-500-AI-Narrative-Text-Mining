"""
Export manual-annotation samples and CSV templates for validation slides.

Outputs (CSV templates):
1. ai_sentence_audit.csv        - sentence-level AI keyword validation
2. role_audit_qa_turns.csv      - Q&A turn role validation
3. qa_boundary_audit_docs.csv   - document-level Q&A boundary/pairing spot-check
4. initiation_audit_exchanges.csv - Q&A exchange initiation label validation

Designed to work with the project's existing outputs in outputs/features/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Allow running as `python scripts/export_annotation_samples.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics.initiation_score import extract_qa_exchanges


@dataclass
class ExportConfig:
    features_dir: str
    output_dir: str
    seed: int = 42
    ai_pos_n: int = 60
    ai_neg_n: int = 60
    role_n: int = 80
    boundary_n: int = 30
    initiation_n: int = 50


def _resolve_feature_input(features_dir: Path, filename: str, stage_candidates: Iterable[int]) -> Path:
    candidates = [features_dir / f"stage{int(stage):02d}" / filename for stage in stage_candidates]
    candidates.append(features_dir / filename)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _sample_evenly(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Random sample of up to n rows."""
    if n <= 0 or df.empty:
        return df.head(0).copy()
    return df.sample(n=min(n, len(df)), random_state=seed).copy()


def _sample_stratified(
    df: pd.DataFrame,
    n: int,
    by: str,
    seed: int,
    preferred_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Roughly balanced sampling by categorical column with deterministic randomness.

    Falls back gracefully if some strata are small.
    """
    if n <= 0 or df.empty:
        return df.head(0).copy()
    if by not in df.columns:
        return _sample_evenly(df, n=n, seed=seed)

    work = df.copy()
    work[by] = work[by].fillna("NA").astype(str)

    groups = {k: g.copy() for k, g in work.groupby(by, dropna=False)}
    if preferred_order:
        keys = [k for k in preferred_order if k in groups] + [
            k for k in groups.keys() if k not in set(preferred_order)
        ]
    else:
        keys = sorted(groups.keys())

    if not keys:
        return work.head(0).copy()

    base = n // len(keys)
    rem = n % len(keys)
    sampled_parts: List[pd.DataFrame] = []
    remaining_groups = {k: groups[k] for k in keys}
    rng = np.random.default_rng(seed)

    # First pass: target roughly equal counts
    for idx, key in enumerate(keys):
        target = base + (1 if idx < rem else 0)
        g = remaining_groups[key]
        take = min(target, len(g))
        if take > 0:
            # derive deterministic per-stratum seed
            part = g.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
            sampled_parts.append(part)
            remaining_groups[key] = g.drop(part.index)

    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else work.head(0).copy()

    # Fill remainder from all unsampled rows if strata were too small
    shortfall = n - len(sampled)
    if shortfall > 0:
        residual = pd.concat(
            [g for g in remaining_groups.values() if not g.empty],
            ignore_index=False,
        ) if any(not g.empty for g in remaining_groups.values()) else work.head(0).copy()
        if not residual.empty:
            extra = residual.sample(n=min(shortfall, len(residual)), random_state=seed + 7)
            sampled = pd.concat([sampled, extra], ignore_index=False)

    # Final cap (can happen if n > len(df))
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed + 11)

    return sampled.copy()


def _ensure_doc_id_from_parsed(parsed_df: pd.DataFrame) -> pd.DataFrame:
    parsed_df = parsed_df.copy()
    parsed_df["doc_id"] = (
        parsed_df["ticker"].astype(str)
        + "_"
        + parsed_df["year"].astype(int).astype(str)
        + "Q"
        + parsed_df["quarter"].astype(int).astype(str)
    )
    return parsed_df


def _to_turn_list(value) -> List[dict]:
    """Coerce nested parquet cell into list[dict]."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        try:
            obj = json.loads(v)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    try:
        import numpy as np  # local import to avoid hard dependency if not needed
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return []


def _truncate(text: str, n: int = 220) -> str:
    s = str(text or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _join_turn_preview(turns: Iterable[dict], head: bool = True, n_turns: int = 2) -> str:
    turns = list(turns)
    if not turns:
        return ""
    chosen = turns[:n_turns] if head else turns[-n_turns:]
    out = []
    for t in chosen:
        speaker = str(t.get("speaker", "Unknown"))
        text = _truncate(str(t.get("text", "")), n=180)
        out.append(f"{speaker}: {text}")
    return " || ".join(out)


def _pred_initiation_type(question_is_ai: bool, answer_is_ai: bool) -> str:
    if question_is_ai and answer_is_ai:
        return "analyst_initiated"
    if (not question_is_ai) and answer_is_ai:
        return "management_pivot"
    if question_is_ai and (not answer_is_ai):
        return "analyst_only"
    return "non_ai"


def _build_merged_turns(speech_turns: List[dict], qa_turns: List[dict]) -> List[dict]:
    """Build a single chronological turn list from speech + QA turns.

    Uses a global absolute index so that speech turns always precede QA
    turns, regardless of the section-relative ``turn_idx`` stored inside
    each turn dict.
    """
    merged: List[dict] = []
    global_idx = 0
    for section_name, turns in [("speech", speech_turns), ("qa", qa_turns)]:
        for _local_idx, t in enumerate(turns):
            item = dict(t) if isinstance(t, dict) else {"text": str(t)}
            # Overwrite idx with a globally unique, monotonically increasing
            # value so that sort order = chronological order.
            item["idx"] = global_idx
            item["section_local_idx"] = _local_idx
            item.setdefault("speaker", item.get("speaker", ""))
            item.setdefault("text", item.get("text", ""))
            item["section"] = section_name
            merged.append(item)
            global_idx += 1
    # Already in correct order, but sort explicitly for safety.
    return sorted(merged, key=lambda x: x["idx"])


def export_full_call_context_sidecar(
    parsed_path: Path,
    output_dir: Path,
    sample_csv_path: Path,
    out_name: str,
) -> Optional[Path]:
    """
    Export full earnings-call script context (speech_turns + qa_turns + merged_turns)
    for the doc_ids referenced by a sample CSV.

    This keeps template CSV columns unchanged while providing a webapp-friendly context sidecar.
    """
    if not sample_csv_path.exists():
        return None
    sample_df = pd.read_csv(sample_csv_path, usecols=["doc_id"])
    doc_ids = sample_df["doc_id"].dropna().astype(str).drop_duplicates().tolist()
    if not doc_ids:
        return None

    parsed = pd.read_parquet(parsed_path)
    parsed = _ensure_doc_id_from_parsed(parsed)
    subset = parsed[parsed["doc_id"].astype(str).isin(set(doc_ids))].copy()
    if subset.empty:
        return None

    records = []
    for _, r in subset.iterrows():
        speech_turns = _to_turn_list(r.get("speech_turns"))
        qa_turns = _to_turn_list(r.get("qa_turns"))
        records.append(
            {
                "doc_id": str(r.get("doc_id", "")),
                "ticker": r.get("ticker", ""),
                "year": r.get("year", ""),
                "quarter": r.get("quarter", ""),
                "speech_turns": speech_turns,
                "qa_turns": qa_turns,
                "merged_turns": _build_merged_turns(speech_turns, qa_turns),
                "num_speech_turns": len(speech_turns),
                "num_qa_turns": len(qa_turns),
            }
        )

    sidecar_path = output_dir / out_name
    with open(sidecar_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sidecar_path


def export_ai_sentence_audit(
    sentences_kw_path: Path,
    output_dir: Path,
    pos_n: int,
    neg_n: int,
    seed: int,
) -> Path:
    cols = ["doc_id", "section", "text", "kw_is_ai"]
    df = pd.read_parquet(sentences_kw_path, columns=cols)
    df["kw_is_ai"] = df["kw_is_ai"].fillna(False).astype(bool)
    df["section"] = df["section"].fillna("NA").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    # Exported rows only expose doc_id/section/text for annotators. If the source
    # parquet contains multiple identical sentence texts within the same doc/section
    # (common for short phrases like "Thank you."), row-level sampling can pick
    # duplicates that become indistinguishable in the annotation batch.
    df = df.drop_duplicates(subset=["doc_id", "section", "text"], keep="first").copy()

    pos = _sample_stratified(df[df["kw_is_ai"]], n=pos_n, by="section", seed=seed, preferred_order=["speech", "qa"])
    neg = _sample_stratified(df[~df["kw_is_ai"]], n=neg_n, by="section", seed=seed + 1, preferred_order=["speech", "qa"])
    sample = pd.concat([pos, neg], ignore_index=True)
    sample = sample.sample(frac=1, random_state=seed + 2).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "sample_id": [f"AI_SENT_{i+1:04d}" for i in range(len(sample))],
            "doc_id": sample["doc_id"],
            "section": sample["section"],
            "text": sample["text"],
            "kw_is_ai_pred": sample["kw_is_ai"].astype(int),
            "annotator_a_is_ai_true": "",
            "annotator_b_is_ai_true": "",
            "adjudicated_is_ai_true": "",
            "false_positive_type": "",
            "notes": "",
        }
    )

    path = output_dir / "ai_sentence_audit.csv"
    out.to_csv(path, index=False)
    return path


def _build_qa_turns_from_sentences(sentences_kw_path: Path) -> pd.DataFrame:
    cols = ["doc_id", "section", "turn_idx", "sentence_idx", "speaker", "role", "text", "kw_is_ai"]
    df = pd.read_parquet(sentences_kw_path, columns=cols)
    df = df[df["section"] == "qa"].copy()
    if df.empty:
        return df
    if "sentence_idx" in df.columns:
        df = df.sort_values(["doc_id", "turn_idx", "sentence_idx"])
    else:
        df = df.sort_values(["doc_id", "turn_idx"])
    turns = (
        df.groupby(["doc_id", "turn_idx"], sort=False)
        .agg(
            speaker=("speaker", "first"),
            role=("role", "first"),
            text=("text", lambda s: " ".join(s.astype(str))),
            kw_is_ai=("kw_is_ai", "any"),
            n_sentences=("text", "size"),
        )
        .reset_index()
    )
    turns["speaker"] = turns["speaker"].fillna("").astype(str)
    turns["role"] = turns["role"].fillna("unknown").astype(str)
    return turns


def export_role_audit(
    qa_turns_df: pd.DataFrame,
    output_dir: Path,
    role_n: int,
    seed: int,
) -> Path:
    sample = _sample_stratified(
        qa_turns_df,
        n=role_n,
        by="role",
        seed=seed,
        preferred_order=["analyst", "management", "operator", "unknown"],
    ).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "sample_id": [f"ROLE_{i+1:04d}" for i in range(len(sample))],
            "doc_id": sample["doc_id"],
            "turn_idx": sample["turn_idx"].astype("Int64"),
            "speaker": sample["speaker"],
            "text": sample["text"],
            "role_pred": sample["role"],
            "turn_kw_is_ai_pred": sample["kw_is_ai"].astype(int),
            "n_sentences_in_turn": sample["n_sentences"].astype("Int64"),
            "annotator_a_role_true": "",
            "annotator_b_role_true": "",
            "adjudicated_role_true": "",
            "notes": "",
        }
    )

    path = output_dir / "role_audit_qa_turns.csv"
    out.to_csv(path, index=False)
    return path


def export_boundary_audit(
    parsed_path: Path,
    doc_metrics_path: Path,
    output_dir: Path,
    boundary_n: int,
    seed: int,
) -> Path:
    parsed = pd.read_parquet(parsed_path)
    parsed = _ensure_doc_id_from_parsed(parsed)
    doc_metrics = pd.read_parquet(doc_metrics_path, columns=["doc_id", "overall_kw_ai_ratio", "speech_kw_ai_ratio", "qa_kw_ai_ratio"])

    merged = parsed.merge(doc_metrics, on="doc_id", how="left")
    merged["overall_kw_ai_ratio"] = merged["overall_kw_ai_ratio"].fillna(0.0)
    merged = merged.sort_values("overall_kw_ai_ratio").reset_index(drop=True)
    if merged.empty:
        out = pd.DataFrame(columns=["sample_id", "doc_id"])
        path = output_dir / "qa_boundary_audit_docs.csv"
        out.to_csv(path, index=False)
        return path

    n = min(boundary_n, len(merged))
    low_n = min(n // 3, len(merged))
    high_n = min(n // 3, len(merged))
    mid_n = n - low_n - high_n

    low = _sample_evenly(merged.head(max(low_n * 3, low_n)), low_n, seed=seed)
    high = _sample_evenly(merged.tail(max(high_n * 3, high_n)), high_n, seed=seed + 1)

    excluded_idx = set(low.index).union(set(high.index))
    remaining = merged.drop(index=list(excluded_idx), errors="ignore")
    mid_band = remaining.copy()
    if not remaining.empty:
        q1 = remaining["overall_kw_ai_ratio"].quantile(0.25)
        q3 = remaining["overall_kw_ai_ratio"].quantile(0.75)
        mid_band = remaining[(remaining["overall_kw_ai_ratio"] >= q1) & (remaining["overall_kw_ai_ratio"] <= q3)]
        if mid_band.empty:
            mid_band = remaining
    mid = _sample_evenly(mid_band, mid_n, seed=seed + 2)

    sample = pd.concat([low, mid, high], ignore_index=True)
    if len(sample) < n:
        more = merged.drop_duplicates(subset=["doc_id"])
        sample = pd.concat([sample, _sample_evenly(more[~more["doc_id"].isin(sample["doc_id"])], n - len(sample), seed=seed + 3)], ignore_index=True)
    sample = sample.drop_duplicates(subset=["doc_id"]).sample(frac=1, random_state=seed + 4).reset_index(drop=True)

    rows = []
    for _, r in sample.iterrows():
        speech_turns = _to_turn_list(r.get("speech_turns"))
        qa_turns = _to_turn_list(r.get("qa_turns"))
        rows.append(
            {
                "sample_id": None,  # fill later
                "doc_id": r["doc_id"],
                "ticker": r.get("ticker", ""),
                "year": r.get("year", ""),
                "quarter": r.get("quarter", ""),
                "overall_kw_ai_ratio": r.get("overall_kw_ai_ratio", np.nan),
                "speech_kw_ai_ratio": r.get("speech_kw_ai_ratio", np.nan),
                "qa_kw_ai_ratio": r.get("qa_kw_ai_ratio", np.nan),
                "speech_turn_count_pred": len(speech_turns),
                "qa_turn_count_pred": len(qa_turns),
                "num_qa_exchanges_pred_parser": r.get("num_qa_exchanges", np.nan),
                "speech_tail_preview": _join_turn_preview(speech_turns, head=False, n_turns=2),
                "qa_head_preview": _join_turn_preview(qa_turns, head=True, n_turns=2),
                "annotator_a_boundary_correct": "",
                "annotator_b_boundary_correct": "",
                "adjudicated_boundary_correct": "",
                "annotator_a_pairing_quality": "",
                "annotator_b_pairing_quality": "",
                "adjudicated_pairing_quality": "",
                "notes": "",
            }
        )
    out = pd.DataFrame(rows)
    out["sample_id"] = [f"BOUND_{i+1:04d}" for i in range(len(out))]

    path = output_dir / "qa_boundary_audit_docs.csv"
    out.to_csv(path, index=False)
    return path


def export_initiation_audit(
    sentences_kw_path: Path,
    output_dir: Path,
    initiation_n: int,
    seed: int,
) -> Path:
    cols = ["doc_id", "section", "role", "turn_idx", "sentence_idx", "text", "kw_is_ai", "speaker"]
    df = pd.read_parquet(sentences_kw_path, columns=cols)
    exchanges = extract_qa_exchanges(df)

    rows = []
    for ex in exchanges:
        q_ai = bool(ex.question_is_ai)
        a_ai = bool(ex.answer_is_ai)
        rows.append(
            {
                "doc_id": ex.doc_id,
                "exchange_idx": ex.exchange_idx,
                "questioner": ex.questioner,
                "answerer": ex.answerer,
                "question_text": ex.question_text,
                "answer_text": ex.answer_text,
                "question_is_ai_pred": int(q_ai),
                "answer_is_ai_pred": int(a_ai),
                "initiation_type_pred": _pred_initiation_type(q_ai, a_ai),
            }
        )
    ex_df = pd.DataFrame(rows)
    if ex_df.empty:
        out = pd.DataFrame(columns=["sample_id", "doc_id", "exchange_idx"])
        path = output_dir / "initiation_audit_exchanges.csv"
        out.to_csv(path, index=False)
        return path

    ai_related = ex_df[(ex_df["question_is_ai_pred"] == 1) | (ex_df["answer_is_ai_pred"] == 1)].copy()
    if ai_related.empty:
        sample = _sample_evenly(ex_df, n=initiation_n, seed=seed)
    else:
        # Balance across predicted initiation types
        sample = _sample_stratified(
            ai_related,
            n=initiation_n,
            by="initiation_type_pred",
            seed=seed,
            preferred_order=["management_pivot", "analyst_initiated", "analyst_only", "non_ai"],
        )
    sample = sample.reset_index(drop=True)

    out = pd.DataFrame(
        {
            "sample_id": [f"INIT_{i+1:04d}" for i in range(len(sample))],
            "doc_id": sample["doc_id"],
            "exchange_idx": sample["exchange_idx"].astype("Int64"),
            "questioner": sample["questioner"],
            "answerer": sample["answerer"],
            "question_text": sample["question_text"],
            "answer_text": sample["answer_text"],
            "question_is_ai_pred": sample["question_is_ai_pred"].astype(int),
            "answer_is_ai_pred": sample["answer_is_ai_pred"].astype(int),
            "initiation_type_pred": sample["initiation_type_pred"],
            "annotator_a_question_is_ai_true": "",
            "annotator_b_question_is_ai_true": "",
            "adjudicated_question_is_ai_true": "",
            "annotator_a_answer_is_ai_true": "",
            "annotator_b_answer_is_ai_true": "",
            "adjudicated_answer_is_ai_true": "",
            "annotator_a_initiation_type_true": "",
            "annotator_b_initiation_type_true": "",
            "adjudicated_initiation_type_true": "",
            "notes": "",
        }
    )

    path = output_dir / "initiation_audit_exchanges.csv"
    out.to_csv(path, index=False)
    return path


def run_export(cfg: ExportConfig) -> List[Path]:
    features_dir = Path(cfg.features_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences_kw_path = _resolve_feature_input(features_dir, "sentences_with_keywords.parquet", stage_candidates=(3,))
    parsed_path = _resolve_feature_input(features_dir, "parsed_transcripts.parquet", stage_candidates=(1,))
    doc_metrics_path = _resolve_feature_input(features_dir, "document_metrics.parquet", stage_candidates=(5,))

    missing = [str(p) for p in [sentences_kw_path, parsed_path, doc_metrics_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required input files not found: {missing}")

    paths: List[Path] = []

    print("Exporting AI sentence audit sample...")
    paths.append(
        export_ai_sentence_audit(
            sentences_kw_path=sentences_kw_path,
            output_dir=output_dir,
            pos_n=cfg.ai_pos_n,
            neg_n=cfg.ai_neg_n,
            seed=cfg.seed,
        )
    )

    print("Building QA turns (for role audit)...")
    qa_turns = _build_qa_turns_from_sentences(sentences_kw_path)
    print(f"  QA turns built: {len(qa_turns)}")
    print("Exporting role audit sample...")
    role_path = export_role_audit(qa_turns_df=qa_turns, output_dir=output_dir, role_n=cfg.role_n, seed=cfg.seed + 10)
    paths.append(role_path)

    print("Exporting Q&A boundary audit sample...")
    boundary_path = export_boundary_audit(
        parsed_path=parsed_path,
        doc_metrics_path=doc_metrics_path,
        output_dir=output_dir,
        boundary_n=cfg.boundary_n,
        seed=cfg.seed + 20,
    )
    paths.append(boundary_path)

    print("Exporting initiation exchange audit sample...")
    paths.append(
        export_initiation_audit(
            sentences_kw_path=sentences_kw_path,
            output_dir=output_dir,
            initiation_n=cfg.initiation_n,
            seed=cfg.seed + 30,
        )
    )

    print("Exporting full earnings-call script context sidecars (role / boundary)...")
    role_ctx_path = export_full_call_context_sidecar(
        parsed_path=parsed_path,
        output_dir=output_dir,
        sample_csv_path=role_path,
        out_name="role_audit_qa_turns_full_call_contexts.jsonl",
    )
    boundary_ctx_path = export_full_call_context_sidecar(
        parsed_path=parsed_path,
        output_dir=output_dir,
        sample_csv_path=boundary_path,
        out_name="qa_boundary_audit_docs_full_call_contexts.jsonl",
    )
    for extra in [role_ctx_path, boundary_ctx_path]:
        if extra is not None:
            paths.append(extra)

    manifest = {
        "config": asdict(cfg),
        "outputs": [str(p) for p in paths],
    }
    with open(output_dir / "annotation_export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return paths


def parse_args() -> ExportConfig:
    p = argparse.ArgumentParser(description="Export manual annotation samples/templates")
    p.add_argument("--features-dir", default="outputs/features")
    p.add_argument("--output-dir", default="outputs/annotation_samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ai-pos-n", type=int, default=60, help="AI-positive sentence samples")
    p.add_argument("--ai-neg-n", type=int, default=60, help="AI-negative sentence samples")
    p.add_argument("--role-n", type=int, default=80, help="Q&A turn role audit samples")
    p.add_argument("--boundary-n", type=int, default=30, help="Q&A boundary audit docs")
    p.add_argument("--initiation-n", type=int, default=50, help="Initiation exchange audit samples")
    a = p.parse_args()
    return ExportConfig(
        features_dir=a.features_dir,
        output_dir=a.output_dir,
        seed=a.seed,
        ai_pos_n=a.ai_pos_n,
        ai_neg_n=a.ai_neg_n,
        role_n=a.role_n,
        boundary_n=a.boundary_n,
        initiation_n=a.initiation_n,
    )


if __name__ == "__main__":
    cfg = parse_args()
    out_paths = run_export(cfg)
    print("\nGenerated annotation templates:")
    for p in out_paths:
        print(f"  - {p}")
