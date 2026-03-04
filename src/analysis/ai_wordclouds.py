"""
AI Word Cloud Visualizations

Creates word clouds from AI-related sentences for each year and overall.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import os
from collections import Counter

import pandas as pd


def _parse_doc_id_year(doc_id: str) -> Optional[int]:
    parts = str(doc_id).rsplit("_", 1)
    if len(parts) != 2:
        return None
    yq = parts[1]
    if "Q" not in yq:
        return None
    try:
        return int(yq.split("Q")[0])
    except Exception:
        return None


def _build_frequency_from_keywords(texts: List[str]) -> Counter:
    """
    Build frequency from dictionary keyword matches only.
    This ensures the wordcloud reflects AI-topic terms rather than general talk.
    """
    from src.baselines.keyword_detector import AIKeywordDetector

    detector = AIKeywordDetector()
    counter = Counter()
    for text in texts:
        matches = detector.detect(text)
        for m in matches:
            key = str(m.keyword).lower()
            counter[key] += 1
    return counter


def _plot_wordcloud(freq: Counter, output_path: str, title: str) -> None:
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    if not freq:
        print(f"No tokens for wordcloud: {output_path}")
        return

    wc = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_words=200,
        colormap="viridis"
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved wordcloud to {output_path}")


def run_ai_wordclouds(
    sentences_path: str,
    output_dir: str = "outputs/figures",
    start_year: int = 2020,
    end_year: int = 2025,
    sample_n: Optional[int] = None
) -> None:
    """
    Create AI-related wordclouds by year and overall (dictionary-based).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading sentence data for wordclouds...")
    df = pd.read_parquet(sentences_path, columns=["text", "doc_id", "kw_is_ai"])
    df = df[df["kw_is_ai"] == 1].copy()

    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=42)
        print(f"Using sample of {len(df)} AI sentences for wordclouds")

    if len(df) == 0:
        print("No AI-related sentences found. Skipping wordclouds.")
        return

    df["year"] = df["doc_id"].apply(_parse_doc_id_year)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Per-year wordclouds
    for year in range(start_year, end_year + 1):
        year_df = df[df["year"] == year]
        if len(year_df) == 0:
            print(f"No AI sentences for {year}. Skipping.")
            continue

        freq = _build_frequency_from_keywords(year_df["text"].astype(str).tolist())
        output_path = os.path.join(output_dir, f"ai_wordcloud_{year}.png")
        terms_path = os.path.join(output_dir, f"ai_wordcloud_terms_{year}.csv")
        title = f"AI-Related Wordcloud ({year}, Dictionary)"

        _plot_wordcloud(freq, output_path, title)

        # Save top terms for reference
        pd.DataFrame(freq.most_common(50), columns=["term", "count"]).to_csv(terms_path, index=False)

    # Overall wordcloud
    freq_all = _build_frequency_from_keywords(df["text"].astype(str).tolist())
    output_path = os.path.join(output_dir, "ai_wordcloud_all_years.png")
    terms_path = os.path.join(output_dir, "ai_wordcloud_terms_all_years.csv")
    title = "AI-Related Wordcloud (All Years, Dictionary)"

    _plot_wordcloud(freq_all, output_path, title)
    pd.DataFrame(freq_all.most_common(50), columns=["term", "count"]).to_csv(terms_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI wordclouds from AI-related sentences")
    parser.add_argument("--sentences", default="outputs/features/sentences_with_keywords.parquet")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--sample", type=int, default=None)

    args = parser.parse_args()

    run_ai_wordclouds(
        args.sentences,
        args.output_dir,
        args.start_year,
        args.end_year,
        args.sample
    )
