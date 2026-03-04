"""Manual validation audit metrics and visualizations for heuristic defense."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

# Force a writable Matplotlib cache/config dir inside the project.
_PROJECT_ROOT_HINT = Path(__file__).resolve().parents[2]
_MPLCONFIGDIR = _PROJECT_ROOT_HINT / "outputs" / ".mplconfig"
_XDG_CACHE_HOME = _PROJECT_ROOT_HINT / "outputs" / ".cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "human_annotation"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "validation"

AI_FILE = "ai_sentence_audit__double.csv"
ROLE_FILE = "role_audit_qa_turns__double.csv"
BOUNDARY_FILE = "qa_boundary_audit_docs__double.csv"
INITIATION_FILE = "initiation_audit_exchanges__double.csv"

INITIATION_CLASS_ORDER = [
    "analyst_initiated",
    "management_pivot",
    "analyst_only",
    "non_ai",
]

ROLE_TOP3 = ["analyst", "management", "operator"]


def _load_csv(file_name: str) -> pd.DataFrame:
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"{dataset_name} is missing required columns: {missing_str}")

import sys
import datetime

class Tee:
    """Redirects stdout to both the original stdout and a log file."""
    def __init__(self, log_dir: Path, base_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{base_name}_{timestamp}.txt"
        self.file = open(self.log_file, "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        print(f"Logging output to: {self.log_file}")

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def _get_log_path(self):
        return self.log_file


def _normalize_text(series: pd.Series) -> pd.Series:
    out = series.copy()
    out = out.where(~out.isna(), np.nan)
    return out.astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})


def _normalize_binary(series: pd.Series) -> pd.Series:
    out = series.copy()
    numeric = pd.to_numeric(out, errors="coerce")
    mapped = _normalize_text(out).map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "1": 1,
            "0": 0,
        }
    )
    result = numeric.where(~numeric.isna(), mapped)
    return result.where(~result.isna(), np.nan).astype("float64")


def _normalize_pairing_quality(series: pd.Series) -> pd.Series:
    out = _normalize_text(series)
    return out.map(
        {
            "good": "good",
            "poor": "poor",
            "minor_issue": "poor",
            "major_issue": "poor",
            "unusable": "unusable",
        }
    )


def _paired_non_null(left: pd.Series, right: pd.Series) -> Tuple[pd.Series, pd.Series]:
    paired = pd.DataFrame({"left": left, "right": right}).dropna(subset=["left", "right"])
    return paired["left"], paired["right"]


def _compute_agreement(
    left: pd.Series,
    right: pd.Series,
    normalizer: Callable[[pd.Series], pd.Series],
) -> Dict[str, float]:
    l_norm = normalizer(left)
    r_norm = normalizer(right)
    l_final, r_final = _paired_non_null(l_norm, r_norm)
    n = len(l_final)

    if n == 0:
        return {"n": 0, "raw_agreement_pct": np.nan, "kappa": np.nan}

    raw_agreement_pct = float((l_final == r_final).mean() * 100.0)
    kappa = float(cohen_kappa_score(l_final, r_final))
    if np.isnan(kappa):
        kappa = 1.0 if (l_final == r_final).all() else 0.0

    return {"n": n, "raw_agreement_pct": raw_agreement_pct, "kappa": kappa}


def compute_ai_keyword_metrics(df: pd.DataFrame) -> Dict[str, float]:
    required = ["kw_is_ai_pred", "adjudicated_is_ai_true"]
    _require_columns(df, required, AI_FILE)

    y_pred = _normalize_binary(df["kw_is_ai_pred"])
    y_true = _normalize_binary(df["adjudicated_is_ai_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    return {
        "n": len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_role_metrics(df: pd.DataFrame) -> Dict[str, float]:
    required = ["role_pred", "adjudicated_role_true"]
    _require_columns(df, required, ROLE_FILE)

    y_pred = _normalize_text(df["role_pred"])
    y_true = _normalize_text(df["adjudicated_role_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)

    analyst_p, analyst_r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["analyst"], zero_division=0
    )
    mgmt_p, mgmt_r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["management"], zero_division=0
    )

    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "analyst_precision": float(analyst_p[0]),
        "analyst_recall": float(analyst_r[0]),
        "management_precision": float(mgmt_p[0]),
        "management_recall": float(mgmt_r[0]),
    }


def compute_boundary_parser_metrics(df: pd.DataFrame) -> Dict[str, float]:
    required = ["adjudicated_boundary_correct", "adjudicated_pairing_quality"]
    _require_columns(df, required, BOUNDARY_FILE)

    boundary = _normalize_binary(df["adjudicated_boundary_correct"])
    boundary = boundary.dropna().astype(int)
    boundary_success_pct = float((boundary == 1).mean() * 100.0) if len(boundary) else np.nan

    quality = _normalize_pairing_quality(df["adjudicated_pairing_quality"])
    quality = quality.dropna()
    quality_dist = quality.value_counts(normalize=True)

    return {
        "n": int(max(len(boundary), len(quality))),
        "boundary_success_pct": boundary_success_pct,
        "quality_good_pct": float(quality_dist.get("good", 0.0) * 100.0),
        "quality_poor_pct": float(quality_dist.get("poor", 0.0) * 100.0),
        "quality_unusable_pct": float(quality_dist.get("unusable", 0.0) * 100.0),
    }


def compute_initiation_metrics(df: pd.DataFrame) -> Dict[str, float]:
    required = ["initiation_type_pred", "adjudicated_initiation_type_true"]
    _require_columns(df, required, INITIATION_FILE)

    y_pred = _normalize_text(df["initiation_type_pred"])
    y_true = _normalize_text(df["adjudicated_initiation_type_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)

    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=INITIATION_CLASS_ORDER,
            average="macro",
            zero_division=0,
        ),
    }


def build_agreement_table(
    ai_df: pd.DataFrame,
    role_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    initiation_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(ai_df, ["annotator_a_is_ai_true", "annotator_b_is_ai_true"], AI_FILE)
    _require_columns(role_df, ["annotator_a_role_true", "annotator_b_role_true"], ROLE_FILE)
    _require_columns(
        boundary_df,
        ["annotator_a_boundary_correct", "annotator_b_boundary_correct"],
        BOUNDARY_FILE,
    )
    _require_columns(
        initiation_df,
        ["annotator_a_initiation_type_true", "annotator_b_initiation_type_true"],
        INITIATION_FILE,
    )

    specs = [
        (
            "AI sentence audit",
            ai_df["annotator_a_is_ai_true"],
            ai_df["annotator_b_is_ai_true"],
            _normalize_binary,
        ),
        (
            "Role QA turns audit",
            role_df["annotator_a_role_true"],
            role_df["annotator_b_role_true"],
            _normalize_text,
        ),
        (
            "QA boundary docs audit",
            boundary_df["annotator_a_boundary_correct"],
            boundary_df["annotator_b_boundary_correct"],
            _normalize_binary,
        ),
        (
            "Initiation exchanges audit",
            initiation_df["annotator_a_initiation_type_true"],
            initiation_df["annotator_b_initiation_type_true"],
            _normalize_text,
        ),
    ]

    rows = []
    for audit_name, ann_a, ann_b, norm in specs:
        agreement = _compute_agreement(ann_a, ann_b, normalizer=norm)
        rows.append(
            {
                "Audit Dataset": audit_name,
                "N (paired)": int(agreement["n"]),
                "Raw Agreement (%)": agreement["raw_agreement_pct"],
                "Cohen's Kappa": agreement["kappa"],
            }
        )
    return pd.DataFrame(rows)


def plot_initiation_confusion_matrix(df: pd.DataFrame, output_path: Path) -> None:
    required = ["initiation_type_pred", "adjudicated_initiation_type_true"]
    _require_columns(df, required, INITIATION_FILE)

    y_pred = _normalize_text(df["initiation_type_pred"])
    y_true = _normalize_text(df["adjudicated_initiation_type_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=INITIATION_CLASS_ORDER)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annotations = np.array(
        [[f"{count}\n({pct:.0%})" for count, pct in zip(row, pct_row)] for row, pct_row in zip(cm, cm_pct)]
    )

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=1.0,
        linecolor="#F2F2F2",
        annot_kws={"fontsize": 13, "fontweight": "semibold"},
        ax=ax,
    )

    pretty_labels = [label.replace("_", " ").title() for label in INITIATION_CLASS_ORDER]
    ax.set_title("Initiation Type: Predicted vs Adjudicated", fontsize=17, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=10)
    ax.set_ylabel("Adjudicated Label", fontsize=14, labelpad=10)
    ax.set_xticklabels(pretty_labels, rotation=25, ha="right", fontsize=12)
    ax.set_yticklabels(pretty_labels, rotation=0, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ai_keyword_confusion_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Plot a 2x2 confusion matrix for the AI Keyword Detector.

    Rows = Adjudicated True label; Columns = Keyword Detector Prediction.
    Each cell shows the raw count and the row-normalised percentage.
    """
    required = ["kw_is_ai_pred", "adjudicated_is_ai_true"]
    _require_columns(df, required, AI_FILE)

    y_pred = _normalize_binary(df["kw_is_ai_pred"])
    y_true = _normalize_binary(df["adjudicated_is_ai_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    labels = [1, 0]                    # Positive first so TP sits top-left
    label_names = ["Positive (AI)", "Negative (Non-AI)"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annotations = np.array(
        [
            [f"{count}\n({pct:.0%})" for count, pct in zip(row, pct_row)]
            for row, pct_row in zip(cm, cm_pct)
        ]
    )

    # Compute summary metrics for the subtitle
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    subtitle  = f"Precision = {precision:.3f}  |  Recall = {recall:.3f}  |  F1 = {f1:.3f}"

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=1.0,
        linecolor="#F2F2F2",
        annot_kws={"fontsize": 14, "fontweight": "semibold"},
        ax=ax,
    )

    ax.set_title("AI Keyword Detector: Predicted vs Adjudicated", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("Adjudicated Label", fontsize=13, labelpad=10)
    ax.set_xticklabels(label_names, rotation=15, ha="right", fontsize=12)
    ax.set_yticklabels(label_names, rotation=0, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5, -0.08, subtitle,
        ha="center", va="top",
        fontsize=12, color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_role_performance_bars(df: pd.DataFrame, output_path: Path) -> None:
    required = ["role_pred", "adjudicated_role_true"]
    _require_columns(df, required, ROLE_FILE)

    y_pred = _normalize_text(df["role_pred"])
    y_true = _normalize_text(df["adjudicated_role_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)

    rows = []
    for role in ROLE_TOP3:
        true_bin = (y_true == role).astype(int)
        pred_bin = (y_pred == role).astype(int)
        rows.append(
            {
                "Role": role.title(),
                "Accuracy": accuracy_score(true_bin, pred_bin),
                "F1": f1_score(true_bin, pred_bin, zero_division=0),
            }
        )

    perf_df = pd.DataFrame(rows)
    plot_df = perf_df.melt(id_vars="Role", var_name="Metric", value_name="Score")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"Accuracy": "#4C78A8", "F1": "#8FB3D9"}
    sns.barplot(
        data=plot_df,
        x="Role",
        y="Score",
        hue="Metric",
        palette=palette,
        edgecolor="#D9D9D9",
        ax=ax,
    )

    ax.set_title("Role Classification Performance (Top 3 Roles)", fontsize=16, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(title="", frameon=False, fontsize=12)
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_role_confusion_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Plot a 3×3 confusion matrix for the Speaker Role Classifier.

    Rows = Adjudicated True label; Columns = Predicted label.
    Class order: analyst, management, operator.
    """
    required = ["role_pred", "adjudicated_role_true"]
    _require_columns(df, required, ROLE_FILE)

    y_pred = _normalize_text(df["role_pred"])
    y_true = _normalize_text(df["adjudicated_role_true"])
    y_true, y_pred = _paired_non_null(y_true, y_pred)

    label_order = ["analyst", "management", "operator"]
    label_names  = ["Analyst", "Management", "Operator"]

    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annotations = np.array(
        [
            [f"{count}\n({pct:.0%})" for count, pct in zip(row, pct_row)]
            for row, pct_row in zip(cm, cm_pct)
        ]
    )

    acc = float(accuracy_score(y_true, y_pred))
    mac_f1 = float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0))
    subtitle = f"Accuracy = {acc:.3f}  |  Macro-F1 = {mac_f1:.3f}"

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=1.0,
        linecolor="#F2F2F2",
        annot_kws={"fontsize": 13, "fontweight": "semibold"},
        ax=ax,
    )

    ax.set_title("Speaker Role Classifier: Predicted vs Adjudicated", fontsize=16, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("Adjudicated Label", fontsize=13, labelpad=10)
    ax.set_xticklabels(label_names, rotation=15, ha="right", fontsize=12)
    ax.set_yticklabels(label_names, rotation=0, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5, -0.04, subtitle,
        ha="center", va="top",
        fontsize=12, color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_confusion_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Plot a 2×2 confusion matrix for the QA Boundary Parser.

    Predicted label = majority vote of annotator_a and annotator_b on
    `boundary_correct` (ties go to 1).
    True label      = `adjudicated_boundary_correct`.
    Labels: 1 = Correct, 0 = Incorrect.
    """
    required = [
        "adjudicated_boundary_correct",
        "annotator_a_boundary_correct",
        "annotator_b_boundary_correct",
    ]
    _require_columns(df, required, BOUNDARY_FILE)

    y_true = _normalize_binary(df["adjudicated_boundary_correct"])
    ann_a  = _normalize_binary(df["annotator_a_boundary_correct"])
    ann_b  = _normalize_binary(df["annotator_b_boundary_correct"])

    # Build majority-vote prediction (tie → 1, i.e. lean toward "correct")
    votes  = pd.DataFrame({"a": ann_a, "b": ann_b, "true": y_true}).dropna()
    y_true = votes["true"].astype(int)
    y_pred = ((votes["a"] + votes["b"]) >= 1).astype(int)   # majority ≥1 of 2

    labels     = [1, 0]
    label_names = ["Correct (1)", "Incorrect (0)"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annotations = np.array(
        [
            [f"{count}\n({pct:.0%})" for count, pct in zip(row, pct_row)]
            for row, pct_row in zip(cm, cm_pct)
        ]
    )

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    subtitle  = f"Precision = {precision:.3f}  |  Recall = {recall:.3f}  |  F1 = {f1:.3f}"

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=1.0,
        linecolor="#F2F2F2",
        annot_kws={"fontsize": 14, "fontweight": "semibold"},
        ax=ax,
    )

    ax.set_title("QA Boundary Parser: Annotator Vote vs Adjudicated", fontsize=15, pad=14)
    ax.set_xlabel("Annotator Majority Vote", fontsize=13, labelpad=10)
    ax.set_ylabel("Adjudicated Label", fontsize=13, labelpad=10)
    ax.set_xticklabels(label_names, rotation=15, ha="right", fontsize=12)
    ax.set_yticklabels(label_names, rotation=0, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5, -0.08, subtitle,
        ha="center", va="top",
        fontsize=12, color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_num(value: float, digits: int = 3) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def print_pipeline_summary(
    ai_metrics: Dict[str, float],
    role_metrics: Dict[str, float],
    boundary_metrics: Dict[str, float],
    initiation_metrics: Dict[str, float],
) -> None:
    print("\n=== Manual Validation Defense: Pipeline Performance ===")
    print(
        "AI Keyword Detector "
        f"(n={ai_metrics['n']}): "
        f"Precision={_format_num(ai_metrics['precision'])} | "
        f"Recall={_format_num(ai_metrics['recall'])} | "
        f"F1={_format_num(ai_metrics['f1'])}"
    )
    print(
        "Speaker Role Classifier "
        f"(n={role_metrics['n']}): "
        f"Accuracy={_format_num(role_metrics['accuracy'])} | "
        f"Analyst P/R={_format_num(role_metrics['analyst_precision'])}/"
        f"{_format_num(role_metrics['analyst_recall'])} | "
        f"Management P/R={_format_num(role_metrics['management_precision'])}/"
        f"{_format_num(role_metrics['management_recall'])}"
    )
    print(
        "QA Document Parser "
        f"(n={boundary_metrics['n']}): "
        f"Boundary Success={_format_num(boundary_metrics['boundary_success_pct'], 2)}% | "
        f"Pairing Quality (Good/Poor/Unusable)="
        f"{_format_num(boundary_metrics['quality_good_pct'], 2)}%/"
        f"{_format_num(boundary_metrics['quality_poor_pct'], 2)}%/"
        f"{_format_num(boundary_metrics['quality_unusable_pct'], 2)}%"
    )
    print(
        "Initiation Type Logic "
        f"(n={initiation_metrics['n']}): "
        f"Accuracy={_format_num(initiation_metrics['accuracy'])} | "
        f"Macro-F1={_format_num(initiation_metrics['macro_f1'])}"
    )


def print_agreement_table(agreement_df: pd.DataFrame) -> None:
    display_df = agreement_df.copy()
    display_df["Raw Agreement (%)"] = display_df["Raw Agreement (%)"].map(lambda x: _format_num(x, 2))
    display_df["Cohen's Kappa"] = display_df["Cohen's Kappa"].map(lambda x: _format_num(x, 3))
    print("\n=== Inter-Annotator Agreement (A vs B) ===")
    print(display_df.to_string(index=False))


def main() -> None:
    log_dir = PROJECT_ROOT / "outputs" / "logs" / "inspect"
    tee = Tee(log_dir, "manual_validation")
    try:
        ai_df = _load_csv(AI_FILE)
        role_df = _load_csv(ROLE_FILE)
        boundary_df = _load_csv(BOUNDARY_FILE)
        initiation_df = _load_csv(INITIATION_FILE)

        ai_metrics = compute_ai_keyword_metrics(ai_df)
        role_metrics = compute_role_metrics(role_df)
        boundary_metrics = compute_boundary_parser_metrics(boundary_df)
        initiation_metrics = compute_initiation_metrics(initiation_df)

        print_pipeline_summary(ai_metrics, role_metrics, boundary_metrics, initiation_metrics)

        agreement_df = build_agreement_table(ai_df, role_df, boundary_df, initiation_df)
        print_agreement_table(agreement_df)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        initiation_cm_path   = FIG_DIR / "initiation_confusion_matrix.png"
        role_perf_path       = FIG_DIR / "role_performance_bars.png"
        ai_kw_cm_path        = FIG_DIR / "ai_keyword_confusion_matrix.png"
        role_cm_path         = FIG_DIR / "role_confusion_matrix.png"
        boundary_cm_path     = FIG_DIR / "boundary_confusion_matrix.png"

        plot_initiation_confusion_matrix(initiation_df, initiation_cm_path)
        plot_role_performance_bars(role_df, role_perf_path)
        plot_ai_keyword_confusion_matrix(ai_df, ai_kw_cm_path)
        plot_role_confusion_matrix(role_df, role_cm_path)
        plot_boundary_confusion_matrix(boundary_df, boundary_cm_path)

        print("\n=== Figures Saved ===")
        print(f"- {initiation_cm_path}")
        print(f"- {role_perf_path}")
        print(f"- {ai_kw_cm_path}")
        print(f"- {role_cm_path}")
        print(f"- {boundary_cm_path}")
    finally:
        sys.stdout = tee.stdout


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"[manual_validation] {exc}")
