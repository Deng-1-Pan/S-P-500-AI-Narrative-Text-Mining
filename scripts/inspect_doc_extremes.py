import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path


def _resolve_feature_path(filename: str, stage_candidates: tuple[int, ...]) -> Path:
    features_dir = Path("outputs") / "features"
    candidates = [features_dir / f"stage{int(stage):02d}" / filename for stage in stage_candidates]
    candidates.append(features_dir / filename)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

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

def extract_context(sentences_df, doc_id, section, filter_col, top_n=3, window=2):
    # filter to doc and section
    doc_df = sentences_df[(sentences_df['doc_id'] == doc_id) & (sentences_df['section'] == section)].copy()
    if doc_df.empty:
        return f"No sentences found for {doc_id} in {section}\n"
    
    # Sort to ensure order
    sort_cols = [c for c in ['turn_idx', 'sentence_idx'] if c in doc_df.columns]
    if sort_cols:
        doc_df = doc_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        doc_df = doc_df.reset_index(drop=True)
    
    # Find top sentences
    # We will use kw_match_count to rank the AI intensity of sentences
    top_indices = doc_df.nlargest(top_n, filter_col).index.tolist()
    
    out = []
    out.append(f"--- Top {top_n} AI Sentences in {doc_id} ({section}) ---")
    
    shown_indices = set()
    for idx in top_indices:
        if doc_df.loc[idx, filter_col] == 0:
            continue # No AI keywords at all
            
        start_idx = max(0, idx - window)
        end_idx = min(len(doc_df) - 1, idx + window)
        
        # Determine if we should print (avoids fully re-printing overlapping contexts)
        if idx in shown_indices:
            continue
            
        out.append(f"\n[AI Keywords Match Count: {doc_df.loc[idx, filter_col]}] (Speaker: {doc_df.loc[idx, 'speaker']})")
        for i in range(start_idx, end_idx + 1):
            shown_indices.add(i)
            prefix = ">> " if i == idx else "   "
            row = doc_df.loc[i]
            out.append(f"{prefix}[{row['speaker']}] {row['text'].strip()}")
            
    if len(out) == 1:
        out.append("No AI keywords found in this section.")
        
    return "\n".join(out) + "\n"

def inspect_documents():
    print("Loading metrics and sentences...")
    metrics_path = _resolve_feature_path("document_metrics.parquet", stage_candidates=(5,))
    sentences_path = _resolve_feature_path("sentences_with_keywords.parquet", stage_candidates=(3,))
    df = pd.read_parquet(metrics_path)
    sentences_df = pd.read_parquet(sentences_path)
    rank_col = "kw_match_count" if "kw_match_count" in sentences_df.columns else "kw_is_ai"
    if rank_col == "kw_is_ai":
        sentences_df["kw_is_ai"] = sentences_df["kw_is_ai"].astype(int)
    
    speech_mean = df['speech_kw_ai_ratio'].mean()
    qa_mean = df['qa_kw_ai_ratio'].mean()
    df['combined_ratio'] = df['speech_kw_ai_ratio'] + df['qa_kw_ai_ratio']

    # Identify target extreme documents
    
    # 1. Combined Highest
    top_combined_doc = df.loc[df['combined_ratio'].idxmax(), 'doc_id']
    # 2. QA highest
    top_qa_doc = df.loc[df['qa_kw_ai_ratio'].idxmax(), 'doc_id']
    # 3. Speech highest
    top_speech_doc = df.loc[df['speech_kw_ai_ratio'].idxmax(), 'doc_id']
    
    # 4. Most Self Promoting (High Speech, Low QA)
    self_promoting_docs = df[(df['speech_kw_ai_ratio'] > speech_mean) & (df['qa_kw_ai_ratio'] <= qa_mean)]
    top_sp_doc = self_promoting_docs.loc[self_promoting_docs['speech_kw_ai_ratio'].idxmax(), 'doc_id'] if not self_promoting_docs.empty else None
    
    # 5. Most Passive (Low Speech, High QA)
    passive_docs = df[(df['qa_kw_ai_ratio'] > qa_mean) & (df['speech_kw_ai_ratio'] <= speech_mean)]
    top_passive_doc = passive_docs.loc[passive_docs['qa_kw_ai_ratio'].idxmax(), 'doc_id'] if not passive_docs.empty else None
    
    targets = [
        ("1. AI Intensity QA和Speech都是最高的文档 (Combined Highest)", top_combined_doc, ["Speech", "Q&A"]),
        ("2. QA最高的文档", top_qa_doc, ["Q&A"]),
        ("3. Speech最高的文档", top_speech_doc, ["Speech"]),
        ("4. 最Self Promoting的文档", top_sp_doc, ["Speech"]),
        ("5. 最Passive的文档", top_passive_doc, ["Q&A"])
    ]
    
    # Add manual requests explicitly in case the automatic identification missed them
    manual_requests = [
        ("Explicit: NVDA_2024Q3 (User requested QA)", "NVDA_2024Q3", ["Q&A"]),
        ("Explicit: NVDA_2024Q4 (User requested Speech)", "NVDA_2024Q4", ["Speech"]),
        ("Explicit: NVDA_2025Q4 (User requested Speech)", "NVDA_2025Q4", ["Speech"]),
        ("Explicit: GOOGL_2023Q2 (User requested QA)", "GOOGL_2023Q2", ["Q&A"])
    ]
    
    targets.extend(manual_requests)
    
    seen = set()
    for category, doc_id, target_sections in targets:
        if not doc_id:
            continue
            
        print(f"\n=======================================================")
        print(f"CATEGORY: {category} -> Doc ID: {doc_id}")
        print(f"=======================================================")
        
        # Check if doc exists in sentences_df
        available_sections = sentences_df[sentences_df['doc_id'] == doc_id]['section'].unique()
        if len(available_sections) == 0:
            print(f"Document {doc_id} not found in sentences dataset.")
            continue
            
        for tgt_sec in target_sections:
            sec = None
            if tgt_sec == "Speech":
                sec = "Speech" if "Speech" in available_sections else ("speech" if "speech" in available_sections else None)
            elif tgt_sec == "Q&A":
                sec = "Q&A" if "Q&A" in available_sections else ("qa" if "qa" in available_sections else None)
            
            if sec is None:
                print(f"Section {tgt_sec} not found in {doc_id}.")
                continue
                
            key = f"{doc_id}_{sec}"
            if key in seen:
                print(f"(Already output previously above)")
                continue
            seen.add(key)
            
            # Print top 5 AI sentences with context +/- 2 sentences
            print(extract_context(sentences_df, doc_id, sec, rank_col, top_n=5, window=2))

if __name__ == "__main__":
    log_dir = "outputs/logs/inspect"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"doc_extremes_{timestamp}.txt")
    
    f = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    
    try:
        inspect_documents()
    finally:
        sys.stdout = original_stdout
        f.close()
        print(f"\n[INFO] Log saved to {log_path}")
