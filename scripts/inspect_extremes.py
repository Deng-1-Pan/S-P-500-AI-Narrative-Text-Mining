import pandas as pd
import numpy as np
from pathlib import Path


def _resolve_feature_path(filename: str, stage_candidates: tuple[int, ...]) -> Path:
    features_dir = Path("outputs") / "features"
    candidates = [features_dir / f"stage{int(stage):02d}" / filename for stage in stage_candidates]
    candidates.append(features_dir / filename)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

def inspect_extremes():
    print("Loading metrics...")
    try:
        metrics_path = _resolve_feature_path("document_metrics.parquet", stage_candidates=(5,))
        df = pd.read_parquet(metrics_path)
    except FileNotFoundError:
        print("Error: content file not found.")
        return

    try:
        # Load dataset to get sector info
        ds = pd.read_parquet("data/final_dataset.parquet", columns=["ticker", "sector"]).drop_duplicates()
    except Exception as e:
        ds = None
        print(f"Warning: Could not load data/final_dataset.parquet for sector info: {e}")

    # Calculate overall means
    speech_mean = df['speech_kw_ai_ratio'].mean()
    qa_mean = df['qa_kw_ai_ratio'].mean()
    df['combined_ratio'] = df['speech_kw_ai_ratio'] + df['qa_kw_ai_ratio']

    def print_doc_table(title, top_df):
        print(f"\n{title}")
        print(f"{'Rank':<5} | {'Doc ID':<15} | {'Combined':<10} | {'Speech':<10} | {'Q&A':<10}")
        print("-" * 60)
        for i, (_, row) in enumerate(top_df.iterrows(), 1):
            print(f"{i:<5} | {row['doc_id']:<15} | {row['combined_ratio']:<10.4f} | {row['speech_kw_ai_ratio']:<10.4f} | {row['qa_kw_ai_ratio']:<10.4f}")

    # --- Document Level ---
    print("\n" + "="*70)
    print("DOCUMENT EXTREMES (from quadrant_scatter_documents.png)")
    print("="*70)
    
    # 1. AI Intensity QA and Speech combined highest
    print_doc_table("[1] Top 10 Documents by Combined AI Intensity (Speech + QA):", 
                    df.nlargest(10, 'combined_ratio'))
    
    # 2. QA highest
    print_doc_table("[2] Top 10 Documents by QA AI Intensity:", 
                    df.nlargest(10, 'qa_kw_ai_ratio'))
    
    # 3. Speech highest
    print_doc_table("[3] Top 10 Documents by Speech AI Intensity:", 
                    df.nlargest(10, 'speech_kw_ai_ratio'))
    
    # 4. Most Self Promoting (High Speech, Low QA)
    # Heuristic: Speech > Mean, QA < Mean, sorted by Speech descending
    self_promoting_docs = df[(df['speech_kw_ai_ratio'] > speech_mean) & (df['qa_kw_ai_ratio'] <= qa_mean)]
    print_doc_table("[4] Top 10 Most 'Self-Promoting' Documents (High Speech, Low QA):", 
                    self_promoting_docs.nlargest(10, 'speech_kw_ai_ratio'))
    
    # 5. Most Passive (Low Speech, High QA)
    # Heuristic: QA > Mean, Speech < Mean, sorted by QA descending
    passive_docs = df[(df['qa_kw_ai_ratio'] > qa_mean) & (df['speech_kw_ai_ratio'] <= speech_mean)]
    print_doc_table("[5] Top 10 Most 'Passive' Documents (Low Speech, High QA):", 
                    passive_docs.nlargest(10, 'qa_kw_ai_ratio'))


    # --- Company Level ---
    print("\n" + "="*85)
    print("COMPANY EXTREMES (from quadrant_scatter_companies.png)")
    print("="*85)

    # Aggregate to company level
    df['ticker'] = df['doc_id'].apply(lambda x: str(x).rsplit('_', 1)[0] if '_' in str(x) else x)
    comp_df = df.groupby('ticker').agg({
        'speech_kw_ai_ratio': 'mean',
        'qa_kw_ai_ratio': 'mean',
        'doc_id': 'count'
    }).reset_index()
    
    comp_df['combined_ratio'] = comp_df['speech_kw_ai_ratio'] + comp_df['qa_kw_ai_ratio']
    comp_speech_mean = comp_df['speech_kw_ai_ratio'].mean()
    comp_qa_mean = comp_df['qa_kw_ai_ratio'].mean()

    if ds is not None:
        comp_df = pd.merge(comp_df, ds, on="ticker", how="left")
        comp_df['sector'] = comp_df['sector'].fillna('Unknown')
    else:
        comp_df['sector'] = 'Unknown'

    def print_comp_table(title, top_df):
        print(f"\n{title}")
        print(f"{'Rank':<5} | {'Ticker':<8} | {'Sector':<25} | {'Combined':<10} | {'Speech':<10} | {'Q&A':<10} | {'Calls':<5}")
        print("-" * 85)
        for i, (_, row) in enumerate(top_df.iterrows(), 1):
            sec = str(row.get('sector', 'Unknown'))[:23]
            print(f"{i:<5} | {row['ticker']:<8} | {sec:<25} | {row['combined_ratio']:<10.4f} | {row['speech_kw_ai_ratio']:<10.4f} | {row['qa_kw_ai_ratio']:<10.4f} | {row['doc_id']:<5}")

    # Top-Right: Combined Highest
    print_comp_table("[1] Top 10 Most 'Aligned' Companies (High combined Speech + QA):", 
                     comp_df.nlargest(10, 'combined_ratio'))

    # Self-Promoting
    comp_self_promoting = comp_df[(comp_df['speech_kw_ai_ratio'] > comp_speech_mean) & (comp_df['qa_kw_ai_ratio'] <= comp_qa_mean)]
    print_comp_table("[2] Top 10 Most 'Self-Promoting' Companies (High Speech, Low QA):", 
                     comp_self_promoting.nlargest(10, 'speech_kw_ai_ratio'))

    # Passive
    comp_passive = comp_df[(comp_df['qa_kw_ai_ratio'] > comp_qa_mean) & (comp_df['speech_kw_ai_ratio'] <= comp_speech_mean)]
    print_comp_table("[3] Top 10 Most 'Passive' Companies (Low Speech, High QA):", 
                     comp_passive.nlargest(10, 'qa_kw_ai_ratio'))

    # Industry distribution in 4 quadrants
    print("\n" + "="*85)
    print("INDUSTRY DISTRIBUTION BY QUADRANT")
    print("="*85)
    
    q1 = comp_df[(comp_df['speech_kw_ai_ratio'] > comp_speech_mean) & (comp_df['qa_kw_ai_ratio'] > comp_qa_mean)]
    q2 = comp_df[(comp_df['speech_kw_ai_ratio'] <= comp_speech_mean) & (comp_df['qa_kw_ai_ratio'] > comp_qa_mean)]
    q3 = comp_df[(comp_df['speech_kw_ai_ratio'] <= comp_speech_mean) & (comp_df['qa_kw_ai_ratio'] <= comp_qa_mean)]
    q4 = comp_df[(comp_df['speech_kw_ai_ratio'] > comp_speech_mean) & (comp_df['qa_kw_ai_ratio'] <= comp_qa_mean)]

    def print_quadrant_sectors(name, q_df):
        print(f"\n{name} (Total Companies: {len(q_df)})")
        if q_df.empty:
            print("  None")
            return
        sector_counts = q_df['sector'].value_counts()
        for sec, count in sector_counts.items(): # display top few
            if count > 0:
                pct = count / len(q_df) * 100
                print(f"  - {sec}: {count} ({pct:.1f}%)")

    print_quadrant_sectors("Quadrant 1: Aligned (High Speech, High Q&A)", q1)
    print_quadrant_sectors("Quadrant 4: Self-Promoting (High Speech, Low Q&A)", q4)
    print_quadrant_sectors("Quadrant 2: Passive (Low Speech, High Q&A)", q2)
    print_quadrant_sectors("Quadrant 3: Avoiders (Low Speech, Low Q&A)", q3)

if __name__ == "__main__":
    import os
    import sys
    from datetime import datetime
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
                
    log_dir = "outputs/logs/inspect"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"extremes_{timestamp}.txt")
    
    f = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    
    try:
        inspect_extremes()
    finally:
        sys.stdout = original_stdout
        f.close()
        print(f"\n[INFO] Log saved to {log_path}")
