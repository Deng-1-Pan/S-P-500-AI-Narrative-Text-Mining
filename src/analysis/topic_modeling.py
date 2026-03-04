"""
Topic Modeling Module (Quarterly)

Uses LDA to extract topics per quarter. Designed for manual topic naming.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, PCA

from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
save_figure = _STYLE.save_figure


def _parse_doc_id(doc_id: str) -> Tuple[Optional[int], Optional[int]]:
    parts = str(doc_id).rsplit("_", 1)
    if len(parts) != 2:
        return None, None
    yq = parts[1]
    if "Q" not in yq:
        return None, None
    try:
        year = int(yq.split("Q")[0])
        quarter = int(yq.split("Q")[1])
        return year, quarter
    except Exception:
        return None, None


def _build_stopwords() -> List[str]:
    custom = {
        "company", "companies", "quarter", "year", "management", "analyst",
        "call", "calls", "question", "questions", "answer", "answers",
        "thank", "thanks", "good", "morning", "afternoon", "evening",
        "said", "say", "says", "will", "would", "could", "should",
        "also", "one", "two", "three", "four", "five",
        "customers", "customer", "business", "businesses"
    }
    # CountVectorizer expects 'english', list, or None (not a set).
    return sorted(set(ENGLISH_STOP_WORDS).union(custom))


def _prepare_docs(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    doc_text = df.groupby("doc_id")["text"].apply(lambda x: " ".join(x.astype(str))).reset_index()
    return doc_text["doc_id"].tolist(), doc_text["text"].tolist()


def _extract_topics(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n_words: int = 12
) -> List[Dict]:
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_idx = np.argsort(topic_weights)[::-1][:top_n_words]
        top_terms = [feature_names[i] for i in top_idx]
        top_weights = [float(topic_weights[i]) for i in top_idx]
        topics.append({
            "topic_id": int(topic_idx),
            "topic_label": "",
            "top_terms": " | ".join(top_terms),
            "top_weights": " | ".join([f"{w:.4f}" for w in top_weights])
        })
    return topics


def _plot_topic_cluster(
    doc_topic: np.ndarray,
    topics: List[Dict],
    output_path: str,
    title: str,
) -> None:
    """Plot a PCA cluster view of document-topic mixtures for a quarter."""
    if doc_topic is None or len(doc_topic) < 3 or doc_topic.shape[1] < 2:
        print(f"Skipping topic cluster plot (insufficient data): {title}")
        return

    try:
        coords = PCA(n_components=2, random_state=42).fit_transform(doc_topic)
    except Exception as e:
        print(f"Skipping topic cluster plot ({title}): {e}")
        return

    dominant_topic = np.argmax(doc_topic, axis=1)
    confidence = np.max(doc_topic, axis=1)

    apply_spotify_theme()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(SPOTIFY_COLORS.get("background", "#121212"))

    cmap = plt.cm.get_cmap("tab20", int(np.max(dominant_topic)) + 1)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=dominant_topic,
        cmap=cmap,
        s=22 + 60 * confidence,
        alpha=0.72,
        linewidths=0,
    )

    for topic_id in sorted(set(int(x) for x in dominant_topic)):
        mask = dominant_topic == topic_id
        if mask.sum() == 0:
            continue
        cx = float(coords[mask, 0].mean())
        cy = float(coords[mask, 1].mean())
        topic_row = next((t for t in topics if int(t.get("topic_id", -1)) == topic_id), None)
        label = f"T{topic_id}"
        if topic_row and topic_row.get("top_terms"):
            terms = [s.strip() for s in str(topic_row["top_terms"]).split("|")[:2]]
            label = " / ".join([t for t in terms if t]) or label
        ax.text(
            cx,
            cy,
            label,
            fontsize=8,
            ha="center",
            va="center",
            color=SPOTIFY_COLORS.get("fg", "#F5F5F5"),
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": SPOTIFY_COLORS.get("background", "#121212"),
                "ec": SPOTIFY_COLORS.get("grid", "#2A2A2A"),
                "alpha": 0.85,
            },
        )

    ax.set_title(title)
    ax.set_xlabel("PCA 1 (topic mixture space)")
    ax.set_ylabel("PCA 2 (topic mixture space)")
    style_axes(ax, grid_axis="both", grid_alpha=0.08)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Dominant Topic")
    cbar.ax.tick_params(colors=SPOTIFY_COLORS.get("muted", "#B3B3B3"))
    cbar.ax.yaxis.label.set_color(SPOTIFY_COLORS.get("fg", "#F5F5F5"))

    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)


def run_quarterly_topic_modeling(
    sentences_path: str,
    output_dir: str = "outputs/features",
    start_year: int = 2020,
    end_year: int = 2025,
    n_topics: int = 20,
    top_n_words: int = 12,
    filter_ai: bool = True,
    min_docs: int = 10,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42,
    generate_cluster_plots: bool = True,
) -> pd.DataFrame:
    """
    Run LDA per quarter and save topic tables for manual naming.

    If filter_ai=True, uses kw_is_ai==1 sentences to focus on AI topics.
    """
    topics_dir = os.path.join(output_dir, "topics")
    os.makedirs(topics_dir, exist_ok=True)

    print("Loading sentences for topic modeling...")
    cols = ["text", "doc_id"]
    if filter_ai:
        cols.append("kw_is_ai")
    df = pd.read_parquet(sentences_path, columns=cols)
    if filter_ai:
        df = df[df["kw_is_ai"] == 1].copy()

    if len(df) == 0:
        print("No sentences available for topic modeling. Skipping.")
        return pd.DataFrame()

    df["year"], df["quarter"] = zip(*df["doc_id"].map(_parse_doc_id))
    df = df.dropna(subset=["year", "quarter"])
    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)

    stopwords = _build_stopwords()
    all_topics: List[Dict] = []

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            q_df = df[(df["year"] == year) & (df["quarter"] == quarter)]
            if len(q_df) == 0:
                continue

            doc_ids, docs = _prepare_docs(q_df)
            if len(docs) < min_docs:
                print(f"Skipping {year}Q{quarter}: not enough documents ({len(docs)}).")
                continue

            vec = CountVectorizer(
                stop_words=stopwords,
                max_df=0.9,
                min_df=2,
                max_features=max_features,
                ngram_range=ngram_range
            )
            X = vec.fit_transform(docs)

            if X.shape[1] == 0:
                print(f"Skipping {year}Q{quarter}: no features after vectorization.")
                continue

            n_components = min(n_topics, len(docs), X.shape[1])
            if n_components < 2:
                print(f"Skipping {year}Q{quarter}: insufficient components.")
                continue

            lda = LatentDirichletAllocation(
                n_components=n_components,
                random_state=random_state,
                learning_method="batch"
            )
            doc_topic = lda.fit_transform(X)

            topics = _extract_topics(lda, vec, top_n_words=top_n_words)
            for t in topics:
                t.update({
                    "year": year,
                    "quarter": quarter,
                    "n_docs": len(docs),
                    "n_terms": int(X.shape[1])
                })
                all_topics.append(t)

            # Save per-quarter topic table
            q_topics_df = pd.DataFrame(topics)
            q_topics_path = os.path.join(topics_dir, f"topics_{year}Q{quarter}.csv")
            q_topics_df.to_csv(q_topics_path, index=False)

            # Save doc-topic distributions for manual inspection
            doc_topic_df = pd.DataFrame(doc_topic, columns=[f"topic_{i}" for i in range(doc_topic.shape[1])])
            doc_topic_df.insert(0, "doc_id", doc_ids)
            doc_topic_df["dominant_topic"] = doc_topic_df.drop(columns=["doc_id"]).idxmax(axis=1)
            doc_topic_path = os.path.join(topics_dir, f"doc_topics_{year}Q{quarter}.parquet")
            doc_topic_df.to_parquet(doc_topic_path, index=False)
            if generate_cluster_plots:
                cluster_path = os.path.join(topics_dir, f"topic_cluster_{year}Q{quarter}.png")
                _plot_topic_cluster(
                    doc_topic=doc_topic,
                    topics=topics,
                    output_path=cluster_path,
                    title=f"Topic Clusters (PCA) — {year}Q{quarter}",
                )

            print(f"Saved topics for {year}Q{quarter}: {q_topics_path}")

    topics_df = pd.DataFrame(all_topics)
    summary_path = os.path.join(topics_dir, "topics_per_quarter.csv")
    topics_df.to_csv(summary_path, index=False)

    # Save manifest for manual naming workflow
    manifest = {
        "filter_ai": filter_ai,
        "n_topics": n_topics,
        "top_n_words": top_n_words,
        "min_docs": min_docs,
        "max_features": max_features,
        "ngram_range": ngram_range,
        "output_dir": topics_dir
    }
    with open(os.path.join(topics_dir, "topic_model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved topic summary to {summary_path}")
    return topics_df



def merge_topic_features(
    doc_metrics_path: str,
    topics_dir: str,
    output_path: Optional[str] = None,
    use_mixture: bool = False,
) -> pd.DataFrame:
    """
    Merge LDA topic features into a document-level DataFrame for regression.

    For each document, provides:
      - dominant_topic          : The single most probable topic (integer)
      - topic_<K>_dummy         : One-hot indicator for dominant topic K
      - topic_<K>_prop (optional): Proportion of topic K in the document mixture
                                   (only when use_mixture=True)

    These features can then be added as IVs in regression.py (Model 5),
    testing whether the *type* of AI discussion topic predicts initiation scores
    \u2014 an interpretation that directly connects LDA output to economic outcomes.

    Args:
        doc_metrics_path: Path to the document-level metrics parquet.
        topics_dir:       Directory containing doc_topics_<YEAR>Q<Q>.parquet files
                          (output of run_quarterly_topic_modeling).
        output_path:      If provided, save the merged DataFrame to this parquet path.
        use_mixture:      If True, include raw topic proportion columns.

    Returns:
        DataFrame with doc_id, year, quarter + topic feature columns.
    """
    import glob

    doc_metrics = pd.read_parquet(doc_metrics_path)

    # Collect all per-quarter doc-topic files
    pattern = os.path.join(topics_dir, "doc_topics_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[merge_topic_features] No doc_topics_*.parquet files found in {topics_dir}.")
        return doc_metrics

    frames = []
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            frames.append(df)
        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")

    if not frames:
        print("[merge_topic_features] No topic files could be loaded.")
        return doc_metrics

    doc_topic_df = pd.concat(frames, ignore_index=True)

    # Keep only doc_id + topic columns
    topic_cols = [c for c in doc_topic_df.columns if c.startswith("topic_") and c != "dominant_topic"]

    # Dominant topic as integer category
    doc_topic_df["dominant_topic_id"] = (
        doc_topic_df["dominant_topic"]
        .str.replace("topic_", "")
        .astype(int)
    )

    # One-hot dummies for dominant topic
    dummies = pd.get_dummies(doc_topic_df["dominant_topic_id"], prefix="dom_topic")
    doc_topic_df = pd.concat([doc_topic_df[["doc_id", "dominant_topic_id"]], dummies], axis=1)

    # Optionally attach raw mixture proportions
    if use_mixture:
        raw_props = pd.read_parquet(files[0])  # representative to get column names
        all_topic_frames = [pd.read_parquet(f)[["doc_id"] + topic_cols]
                            for f in files if pd.read_parquet(f).shape[1] > 2]
        # Re-read for safety
        prop_frames = []
        for fpath in files:
            try:
                df = pd.read_parquet(fpath)
                avail = [c for c in topic_cols if c in df.columns]
                if avail:
                    prop_frames.append(df[["doc_id"] + avail])
            except Exception:
                pass
        if prop_frames:
            prop_df = pd.concat(prop_frames, ignore_index=True)
            prop_df.columns = ["doc_id"] + [f"mix_{c}" for c in prop_df.columns[1:]]
            doc_topic_df = doc_topic_df.merge(prop_df, on="doc_id", how="left")

    # Merge with doc_metrics
    merged = doc_metrics.merge(doc_topic_df, on="doc_id", how="left")

    print(f"[merge_topic_features] Merged {len(doc_topic_df)} doc-topic rows with {len(doc_metrics)} doc-metrics rows.")
    print(f"  Topic feature columns added: {[c for c in merged.columns if c.startswith('dom_topic') or c.startswith('mix_')]}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        merged.to_parquet(output_path, index=False)
        print(f"  Saved merged features → {output_path}")

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quarterly topic modeling (LDA)")
    parser.add_argument("--sentences", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/features")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--n-topics", type=int, default=20)
    parser.add_argument("--top-words", type=int, default=12)
    parser.add_argument("--filter-ai", action="store_true", help="Use only kw_is_ai sentences")
    parser.add_argument(
        "--merge-features",
        action="store_true",
        help="After modeling, merge topic features into document metrics"
    )
    parser.add_argument("--doc-metrics", default="outputs/features/document_metrics.parquet")
    parser.add_argument("--no-topic-cluster-plots", action="store_true")

    args = parser.parse_args()

    run_quarterly_topic_modeling(
        args.sentences,
        args.output_dir,
        args.start_year,
        args.end_year,
        n_topics=args.n_topics,
        top_n_words=args.top_words,
        filter_ai=args.filter_ai,
        generate_cluster_plots=not args.no_topic_cluster_plots,
    )

    if args.merge_features:
        topics_dir = os.path.join(args.output_dir, "topics")
        output_path = os.path.join(args.output_dir, "doc_metrics_with_topics.parquet")
        merge_topic_features(
            doc_metrics_path=args.doc_metrics,
            topics_dir=topics_dir,
            output_path=output_path,
        )
