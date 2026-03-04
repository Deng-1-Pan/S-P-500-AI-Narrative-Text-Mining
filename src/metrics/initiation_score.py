"""
AI Initiation Score Module

Computes who initiates AI discussions in Q&A sessions using strict burden-of-proof
heuristics:
- `analyst_initiated`: analyst question clearly raises AI, management answer also AI
- `management_pivot`: analyst question is non-AI, management introduces AI
- `analyst_only`: analyst raises AI, management does not continue AI
- `non_ai`: neither side clearly raises AI
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from src.baselines.keyword_detector import AIKeywordDetector
from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark-minimal")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


@dataclass
class QAExchange:
    """Represents a Q&A exchange (question + answer pair)."""
    doc_id: str
    exchange_idx: int
    question_text: str
    answer_text: str
    questioner: str
    answerer: str
    question_is_ai: bool
    answer_is_ai: bool
    initiation_type: str = "non_ai"
    question_strong_count: int = 0
    question_weak_count: int = 0
    answer_strong_count: int = 0
    answer_weak_count: int = 0


_GLOBAL_DETECTOR = AIKeywordDetector()


def _is_question_ai_trigger(profile: Dict[str, int | bool]) -> bool:
    """
    Strict analyst-side burden of proof.

    Analyst gets AI-initiation credit only with clear evidence:
    - >=1 strong signal, OR
    - clear weak combo (>=2 distinct non-excluded weak-term families)
    """
    strong = int(profile.get("strong_count", 0))
    weak_nonexcluded_unique = int(profile.get("weak_nonexcluded_unique", 0))
    return strong >= 1 or weak_nonexcluded_unique >= AIKeywordDetector.MIN_DISTINCT_WEAK_FOR_AI


def _classify_initiation_type(
    question_profile: Dict[str, int | bool],
    answer_profile: Dict[str, int | bool],
) -> tuple[str, bool, bool]:
    """
    Classify initiation type for a macro question-answer exchange.

    Returns:
        (initiation_type, question_is_ai, answer_is_ai)
    """
    question_is_ai = bool(question_profile.get("is_ai", False))
    answer_is_ai = bool(answer_profile.get("is_ai", False))

    # Analyst label uses stricter trigger than generic AI detection.
    question_is_ai = question_is_ai and _is_question_ai_trigger(question_profile)

    if question_is_ai and answer_is_ai:
        return "analyst_initiated", True, True
    if (not question_is_ai) and answer_is_ai:
        return "management_pivot", False, True
    if question_is_ai and (not answer_is_ai):
        return "analyst_only", True, False
    return "non_ai", False, False


def extract_qa_exchanges(
    sentences_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    section_col: str = 'section',
    role_col: str = 'role',
    turn_idx_col: str = 'turn_idx',
    text_col: str = 'text',
    kw_pred_col: str = 'kw_is_ai',
    detector: Optional[AIKeywordDetector] = None,
) -> List[QAExchange]:
    """
    Extract Q&A exchanges from sentence data using Macro-Question /
    Macro-Answer fusion and strict AI signal rescoring.

    An exchange is one or more consecutive analyst turns (Macro-Question)
    followed by one or more consecutive management/unknown turns
    (Macro-Answer).  Operator turns act as explicit exchange boundaries
    (they introduce the next analyst).

    Consecutive same-role merging rules:
      - Multiple analyst turns in a row (follow-up questions, clarifications)
        are fused into a single Macro-Question.
      - Multiple management/unknown turns in a row (CEO→CFO relay answers)
        are fused into a single Macro-Answer.
      - An operator turn always ends the current exchange and starts a new
        boundary context.

    Args:
        sentences_df: Sentence-level data

    Returns:
        List of QAExchange objects
    """
    exchanges: List[QAExchange] = []

    # Filter to Q&A section only
    qa_df = sentences_df[sentences_df[section_col] == 'qa'].copy()

    if len(qa_df) == 0:
        return exchanges

    # Ensure AI flag column exists (legacy fallback only)
    if kw_pred_col not in qa_df.columns:
        qa_df[kw_pred_col] = False

    if detector is None:
        detector = _GLOBAL_DETECTOR

    # Preserve sentence order within turns when available
    sort_cols = [doc_id_col, turn_idx_col]
    if "sentence_idx" in qa_df.columns:
        sort_cols.append("sentence_idx")
    qa_df = qa_df.sort_values(sort_cols)

    # Aggregate per turn (much faster than repeated filtering)
    agg_map = {
        text_col: lambda s: ' '.join(s.astype(str)),
        role_col: 'first',
        kw_pred_col: 'any'
    }
    if 'speaker' in qa_df.columns:
        agg_map['speaker'] = 'first'

    turns_df = (
        qa_df.groupby([doc_id_col, turn_idx_col], sort=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={
            text_col: 'text',
            role_col: 'role',
            kw_pred_col: 'is_ai'
        })
    )
    if 'speaker' not in turns_df.columns:
        turns_df['speaker'] = ''
    turns_df['role'] = turns_df['role'].fillna('unknown')

    # ------------------------------------------------------------------
    # Macro-Question / Macro-Answer pairing per document
    # ------------------------------------------------------------------
    n_docs = turns_df[doc_id_col].nunique()
    doc_groups = turns_df.groupby(doc_id_col, sort=False)
    for doc_id, doc_turns in tqdm(doc_groups, total=n_docs, desc="Q&A exchanges"):
        turns = doc_turns.sort_values(turn_idx_col).to_dict('records')

        # --- Phase 1: build run-length-encoded segments ---
        # Each segment is a group of consecutive turns with the same
        # effective role: 'analyst', 'management', or 'operator'.
        # 'unknown' is treated as 'management' (most unknowns in Q&A
        # are management speakers whose titles were not parsed).
        segments: List[dict] = []  # {role, texts[], speakers[], is_ai_flags[]}
        for turn in turns:
            raw_role = turn['role']
            # Normalise: unknown → management for pairing purposes
            eff_role = 'management' if raw_role in ('management', 'unknown') else raw_role

            if segments and segments[-1]['role'] == eff_role:
                # Extend current segment
                segments[-1]['texts'].append(turn['text'])
                segments[-1]['speakers'].append(turn.get('speaker', ''))
                segments[-1]['is_ai_flags'].append(turn['is_ai'])
            else:
                # New segment
                segments.append({
                    'role': eff_role,
                    'texts': [turn['text']],
                    'speakers': [turn.get('speaker', '')],
                    'is_ai_flags': [turn['is_ai']],
                })

        # --- Phase 2: pair adjacent analyst→management segments ---
        exchange_idx = 0
        i = 0
        while i < len(segments):
            seg = segments[i]

            # Skip operator segments (they are boundaries, not content)
            if seg['role'] == 'operator':
                i += 1
                continue

            if seg['role'] == 'analyst':
                # Macro-Question: this entire analyst segment
                macro_q_text = ' '.join(seg['texts'])
                macro_q_speakers = seg['speakers']
                macro_q_is_ai = any(seg['is_ai_flags'])

                # Look ahead for Macro-Answer
                j = i + 1
                # Skip any intervening operator segment (operator may
                # appear between question and answer in some transcripts)
                while j < len(segments) and segments[j]['role'] == 'operator':
                    j += 1

                if j < len(segments) and segments[j]['role'] == 'management':
                    ans_seg = segments[j]
                    macro_a_text = ' '.join(ans_seg['texts'])
                    macro_a_speakers = ans_seg['speakers']
                    # Legacy flags as fallback when detector scoring fails.
                    legacy_q_is_ai = any(seg['is_ai_flags'])
                    legacy_a_is_ai = any(ans_seg['is_ai_flags'])

                    try:
                        q_profile = detector.get_signal_profile(macro_q_text)
                        a_profile = detector.get_signal_profile(macro_a_text)
                        initiation_type, macro_q_is_ai, macro_a_is_ai = _classify_initiation_type(
                            q_profile, a_profile
                        )
                        q_strong = int(q_profile.get("strong_count", 0))
                        q_weak = int(q_profile.get("weak_nonexcluded_count", 0))
                        a_strong = int(a_profile.get("strong_count", 0))
                        a_weak = int(a_profile.get("weak_nonexcluded_count", 0))
                    except Exception:
                        macro_q_is_ai = legacy_q_is_ai
                        macro_a_is_ai = legacy_a_is_ai
                        if macro_q_is_ai and macro_a_is_ai:
                            initiation_type = "analyst_initiated"
                        elif (not macro_q_is_ai) and macro_a_is_ai:
                            initiation_type = "management_pivot"
                        elif macro_q_is_ai and (not macro_a_is_ai):
                            initiation_type = "analyst_only"
                        else:
                            initiation_type = "non_ai"
                        q_strong = q_weak = a_strong = a_weak = 0

                    exchanges.append(QAExchange(
                        doc_id=doc_id,
                        exchange_idx=exchange_idx,
                        question_text=macro_q_text,
                        answer_text=macro_a_text,
                        questioner=macro_q_speakers[0],
                        answerer=macro_a_speakers[0],
                        question_is_ai=macro_q_is_ai,
                        answer_is_ai=macro_a_is_ai,
                        initiation_type=initiation_type,
                        question_strong_count=q_strong,
                        question_weak_count=q_weak,
                        answer_strong_count=a_strong,
                        answer_weak_count=a_weak,
                    ))
                    exchange_idx += 1
                    i = j + 1
                    continue

            # If we reach here the segment is management without a
            # preceding analyst question (e.g. management opens Q&A
            # with a preamble).  Skip it.
            i += 1

    return exchanges


def compute_initiation_scores(
    exchanges: List[QAExchange]
) -> pd.DataFrame:
    """
    Compute AI initiation scores per document.
    
    Metrics:
    - analyst_initiated_ratio: % of AI exchanges started by analyst
    - management_pivot_ratio: % of AI exchanges introduced by management
    - total_ai_exchanges: # exchanges where initiation_type != non_ai
    
    Args:
        exchanges: List of QAExchange objects
        
    Returns:
        DataFrame with per-document initiation scores
    """
    if not exchanges:
        return pd.DataFrame()
    
    # Convert to DataFrame
    exchange_df = pd.DataFrame([{
        'doc_id': e.doc_id,
        'exchange_idx': e.exchange_idx,
        'question_is_ai': e.question_is_ai,
        'answer_is_ai': e.answer_is_ai,
        'initiation_type': getattr(e, "initiation_type", None),
    } for e in exchanges])

    if 'initiation_type' not in exchange_df.columns:
        exchange_df['initiation_type'] = None
    exchange_df['initiation_type'] = exchange_df['initiation_type'].where(
        exchange_df['initiation_type'].notna(),
        np.where(
            exchange_df['question_is_ai'] & exchange_df['answer_is_ai'],
            'analyst_initiated',
            np.where(
                (~exchange_df['question_is_ai']) & exchange_df['answer_is_ai'],
                'management_pivot',
                np.where(
                    exchange_df['question_is_ai'] & (~exchange_df['answer_is_ai']),
                    'analyst_only',
                    'non_ai'
                )
            )
        )
    )
    
    results = []
    
    for doc_id in exchange_df['doc_id'].unique():
        doc_df = exchange_df[exchange_df['doc_id'] == doc_id]
        
        total_exchanges = len(doc_df)
        
        # AI-related exchanges = all non-non_ai labels
        ai_exchanges = doc_df[doc_df['initiation_type'] != 'non_ai']
        total_ai_exchanges = len(ai_exchanges)

        if total_ai_exchanges == 0:
            results.append({
                'doc_id': doc_id,
                'total_exchanges': total_exchanges,
                'total_ai_exchanges': 0,
                'analyst_initiated_count': 0,
                'management_pivot_count': 0,
                'analyst_only_count': 0,
                'non_ai_count': total_exchanges,
                'mutual_ai_count': 0,
                'analyst_initiated_ratio': 0.0,
                'management_pivot_ratio': 0.0,
                'ai_initiation_score': 0.5  # Neutral
            })
            continue

        analyst_initiated_count = int((doc_df['initiation_type'] == 'analyst_initiated').sum())
        management_pivot_count = int((doc_df['initiation_type'] == 'management_pivot').sum())
        analyst_only_count = int((doc_df['initiation_type'] == 'analyst_only').sum())
        non_ai_count = int((doc_df['initiation_type'] == 'non_ai').sum())

        # AI Initiation Score: Higher = more management-driven
        # Score = management_pivot / (analyst_initiated + management_pivot)
        denom = analyst_initiated_count + management_pivot_count
        if denom > 0:
            ai_initiation_score = management_pivot_count / denom
        else:
            ai_initiation_score = 0.5
        
        results.append({
            'doc_id': doc_id,
            'total_exchanges': total_exchanges,
            'total_ai_exchanges': total_ai_exchanges,
            'analyst_initiated_count': analyst_initiated_count,
            'management_pivot_count': management_pivot_count,
            'analyst_only_count': analyst_only_count,
            'non_ai_count': non_ai_count,
            'analyst_initiated_ratio': analyst_initiated_count / total_ai_exchanges if total_ai_exchanges > 0 else 0.0,
            'management_pivot_ratio': management_pivot_count / total_ai_exchanges if total_ai_exchanges > 0 else 0.0,
            'ai_initiation_score': ai_initiation_score
        })
    
    return pd.DataFrame(results)


def compute_all_initiation_metrics(
    sentences_df: pd.DataFrame,
    output_dir: str = "outputs/features",
    figures_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Full pipeline to compute initiation scores.
    
    Args:
        sentences_df: Sentence-level data with keyword flags
        output_dir: Output directory
        
    Returns:
        DataFrame with initiation scores
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(output_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Extracting Q&A exchanges...")
    exchanges = extract_qa_exchanges(sentences_df)
    print(f"Found {len(exchanges)} Q&A exchanges")
    
    print("Computing initiation scores...")
    scores_df = compute_initiation_scores(exchanges)
    
    # Save
    scores_df.to_parquet(f"{output_dir}/initiation_scores.parquet", index=False)

    print(f"\n=== Initiation Score Summary ===")
    if len(scores_df) == 0 or 'total_ai_exchanges' not in scores_df.columns:
        print("No Q&A exchanges found in the data.")
    else:
        print(f"Documents with AI exchanges: {(scores_df['total_ai_exchanges'] > 0).sum()}")
        print(f"Avg AI exchanges per doc: {scores_df['total_ai_exchanges'].mean():.1f}")
        print(f"Avg analyst-initiated ratio: {scores_df['analyst_initiated_ratio'].mean():.3f}")
        print(f"Avg management-pivot ratio: {scores_df['management_pivot_ratio'].mean():.3f}")
        print(f"Avg AI initiation score: {scores_df['ai_initiation_score'].mean():.3f}")
        print("  (Higher = more management-driven)")

    # Visualizations
    if len(scores_df) > 0:
        try:
            plot_initiation_distributions(scores_df, figures_dir)
            plot_initiation_ratios(scores_df, figures_dir)
            plot_initiation_scatter(scores_df, figures_dir)
        except Exception as e:
            print(f"Warning: failed to generate initiation score plots: {e}")
    
    return scores_df


def plot_initiation_distributions(
    scores_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot distribution of AI initiation scores.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    sns.histplot(scores_df["ai_initiation_score"], bins=30, kde=True, ax=ax, color=SPOTIFY_COLORS.get("accent", "#1DB954"))
    ax.axvline(0.5, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title("Distribution of AI Initiation Scores")
    ax.set_xlabel("AI Initiation Score (Higher = Management-Driven)")
    ax.set_ylabel("Count")
    style_axes(ax, grid_axis="y", grid_alpha=0.08)

    output_path = os.path.join(output_dir, "ai_initiation_distribution.png")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved initiation score distribution plot to {output_path}")

    if "total_ai_exchanges" in scores_df.columns:
        active = scores_df[scores_df["total_ai_exchanges"].fillna(0) > 0].copy()
        if len(active) > 0:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            fig2.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
            sns.histplot(active["ai_initiation_score"], bins=30, kde=True, ax=ax2, color=SPOTIFY_COLORS.get("blue", "#4EA1FF"))
            ax2.axvline(0.5, color=SPOTIFY_COLORS.get("muted", "#B3B3B3"), linestyle="--", linewidth=1, alpha=0.8)
            ax2.set_title("AI Initiation Scores (Active AI Exchanges Only)")
            ax2.set_xlabel("AI Initiation Score")
            ax2.set_ylabel("Count")
            style_axes(ax2, grid_axis="y", grid_alpha=0.08)
            active_out = os.path.join(output_dir, "ai_initiation_distribution_active_only.png")
            fig2.tight_layout()
            save_figure(fig2, active_out, dpi=180)
            print(f"Saved active-only initiation score distribution plot to {active_out}")


def plot_initiation_ratios(
    scores_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot average initiation ratios.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    df = scores_df.copy()
    df = df[df["total_ai_exchanges"] > 0].copy()
    if len(df) == 0:
        print("No AI exchanges found for initiation ratio plot.")
        return

    if "analyst_only_count" not in df.columns:
        df["analyst_only_count"] = 0

    df["analyst_only_ratio"] = df["analyst_only_count"] / df["total_ai_exchanges"].replace(0, np.nan)

    ratios = {
        "Analyst Initiated": df["analyst_initiated_ratio"].mean(),
        "Management Pivot": df["management_pivot_ratio"].mean(),
        "Analyst Only": df["analyst_only_ratio"].mean(),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    ax.bar(
        list(ratios.keys()),
        list(ratios.values()),
        color=[SPOTIFY_COLORS.get("blue", "#4EA1FF"), SPOTIFY_COLORS.get("accent", "#1DB954"), SPOTIFY_COLORS.get("muted", "#B3B3B3")],
        alpha=0.9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Ratio")
    ax.set_title("Average AI Initiation Composition")
    style_axes(ax, grid_axis="y", grid_alpha=0.08)

    output_path = os.path.join(output_dir, "ai_initiation_ratios.png")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved initiation ratio plot to {output_path}")


def plot_initiation_scatter(
    scores_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Plot AI initiation score vs total AI exchanges.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))
    ax.scatter(
        scores_df["total_ai_exchanges"],
        scores_df["ai_initiation_score"],
        alpha=0.6,
        color=SPOTIFY_COLORS.get("accent", "#1DB954"),
        s=40
    )
    ax.set_xlabel("Total AI Exchanges (per document)")
    ax.set_ylabel("AI Initiation Score")
    ax.set_title("AI Initiation Score vs AI Exchange Volume")
    style_axes(ax, grid_axis="both", grid_alpha=0.08)

    output_path = os.path.join(output_dir, "ai_initiation_scatter.png")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    print(f"Saved initiation scatter plot to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute AI initiation scores")
    parser.add_argument("--input", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    
    args = parser.parse_args()
    
    sentences_df = pd.read_parquet(args.input)
    compute_all_initiation_metrics(sentences_df, args.output_dir)
