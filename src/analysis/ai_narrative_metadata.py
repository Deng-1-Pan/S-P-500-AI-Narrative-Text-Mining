"""
AI Narrative Metadata Analysis

Analyzes textual metadata (Speaker Role, Section, Language/Topics) of AI 
sentences distributed across the four Company Quadrants.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.plot_style_compat import load_plot_style

_STYLE = load_plot_style(profile="analysis-dark")
SPOTIFY_COLORS = _STYLE.colors
apply_spotify_theme = _STYLE.apply_theme
style_axes = _STYLE.style_axes
style_legend = _STYLE.style_legend
save_figure = _STYLE.save_figure

QUADRANT_ORDER = ['Aligned', 'Passive', 'Self-Promoting', 'Silent']

def analyze_metadata(
    sentences_path: str,
    quadrants_path: str,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    apply_spotify_theme()

    print(f"Loading data from {sentences_path} and {quadrants_path}...")
    try:
        sentences = pd.read_parquet(sentences_path)
        quadrants = pd.read_parquet(quadrants_path)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Merge to get quadrant label for each sentence
    df = pd.merge(sentences, quadrants[['doc_id', 'quadrant']], on='doc_id', how='inner')
    
    # Filter only AI sentences
    ai_df = df[df['kw_is_ai'] == True].copy()
    
    if len(ai_df) == 0:
        print("No AI sentences found to analyze.")
        return
        
    print(f"Analyzing {len(ai_df)} AI-related sentences.")

    # 1. Speakers by Quadrant
    print("Computing Speaker Role distribution...")
    ai_df['role_clean'] = ai_df['role'].fillna('Unknown')
    # Standardize roles a bit if needed (simplification)
    ai_df['role_simplified'] = ai_df['role_clean'].apply(
        lambda x: 'Executive' if 'exec' in str(x).lower() or 'ceo' in str(x).lower() or 'cfo' in str(x).lower() else
                  ('Analyst' if 'analyst' in str(x).lower() else 
                  ('Operator' if 'operator' in str(x).lower() else 'Other Management'))
    )
    
    role_dist = pd.crosstab(ai_df['quadrant'], ai_df['role_simplified'], normalize='index') * 100
    if len(role_dist) > 0:
        role_dist = role_dist.reindex(QUADRANT_ORDER).dropna(how='all')
        role_dist.to_csv(os.path.join(output_dir, "quadrant_role_dist.csv"))
        
        # Plot
        ax = role_dist.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
        plt.title('AI Speaker Role Distribution by Quadrant', fontsize=14, color=SPOTIFY_COLORS.get('fg', 'k'))
        plt.ylabel('Percentage of AI Sentences (%)')
        plt.xlabel('Quadrant')
        plt.xticks(rotation=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save_figure(plt.gcf(), os.path.join(output_dir, "quadrant_role_dist.png"))

    # 2. Section by Quadrant
    print("Computing Section (Speech vs Q&A) distribution...")
    ai_df['section_clean'] = ai_df['section'].fillna('Unknown')
    section_dist = pd.crosstab(ai_df['quadrant'], ai_df['section_clean'], normalize='index') * 100
    if len(section_dist) > 0:
        section_dist = section_dist.reindex(QUADRANT_ORDER).dropna(how='all')
        section_dist.to_csv(os.path.join(output_dir, "quadrant_section_dist.csv"))
        
        # Plot
        ax = section_dist.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
        plt.title('AI Conversation Context (Speech vs Q&A) by Quadrant', fontsize=14, color=SPOTIFY_COLORS.get('fg', 'k'))
        plt.ylabel('Percentage of AI Sentences (%)')
        plt.xlabel('Quadrant')
        plt.xticks(rotation=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        save_figure(plt.gcf(), os.path.join(output_dir, "quadrant_section_dist.png"))

    # 3. Language Subtopics by Quadrant
    print("Computing Language Subtopics distribution...")
    subtopic_cols = [
        col for col in ai_df.columns if col.startswith('kw_') and col.endswith('_count') and col not in ['kw_match_count', 'kw_strong_count', 'kw_weak_count', 'kw_weak_nonexcluded_count']
    ]
    
    if subtopic_cols:
        # Sum counts per quadrant, then normalize by total AI sentences in that quadrant
        subtopic_means = ai_df.groupby('quadrant')[subtopic_cols].mean()
        # Convert to percentages for readability roughly
        subtopic_means = subtopic_means * 100
        subtopic_means = subtopic_means.reindex(QUADRANT_ORDER).dropna(how='all')
        
        # Rename columns for plot clarity
        clean_names = {c: c.replace('kw_', '').replace('_count', '').replace('_', ' ').title() for c in subtopic_cols}
        subtopic_means = subtopic_means.rename(columns=clean_names)
        subtopic_means.to_csv(os.path.join(output_dir, "quadrant_ai_subtopics.csv"))
        
        # Heatmap plot
        plt.figure(figsize=(12, 6))
        sns.heatmap(subtopic_means.T, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Average Mentions per 100 AI Sentences'})
        plt.title('AI Subtopics Focus by Quadrant', fontsize=14, color=SPOTIFY_COLORS.get('fg', 'k'))
        plt.ylabel('Subtopic Category')
        plt.xlabel('Quadrant')
        save_figure(plt.gcf(), os.path.join(output_dir, "quadrant_ai_subtopics.png"))

    # 4. Top Distinctive Vocabulary by Quadrant (TF-IDF)
    print("Extracting distinctive vocabulary via TF-IDF...")
    try:
        # We group all AI text by quadrant
        corpus_df = ai_df.groupby('quadrant')['text'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        corpus_df = corpus_df.set_index('quadrant').reindex(QUADRANT_ORDER).dropna()
        
        if len(corpus_df) > 0:
            vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=0.1, max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(corpus_df['text'])
            feature_names = vectorizer.get_feature_names_out()
            
            top_words_per_quadrant = []
            for i, quad in enumerate(corpus_df.index):
                vec = tfidf_matrix[i].toarray()[0]
                top_indices = vec.argsort()[-10:][::-1]
                top_words = [(feature_names[j], vec[j]) for j in top_indices]
                words_only = [w[0] for w in top_words]
                top_words_per_quadrant.append({'Quadrant': quad, 'Top_10_Distinctive_Words': ', '.join(words_only)})
                
            tfidf_df = pd.DataFrame(top_words_per_quadrant)
            tfidf_df.to_csv(os.path.join(output_dir, "quadrant_top_distinctive_words.csv"), index=False)
            print("TF-IDF extraction complete.")
    except Exception as e:
        print(f"Could not compute TF-IDF vocabulary: {e}")

    print(f"Metadata analysis complete. Check the '{output_dir}' directory.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Narrative Metadata Analysis")
    parser.add_argument("--sentences", required=True, help="Path to sentences_with_keywords parquet")
    parser.add_argument("--quadrants", required=True, help="Path to documents_with_quadrants parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    analyze_metadata(args.sentences, args.quadrants, args.output_dir)
