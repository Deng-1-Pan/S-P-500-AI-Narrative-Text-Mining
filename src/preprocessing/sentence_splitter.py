"""
Sentence Splitter Module

Splits transcript text into sentences for model inference.
"""

import re
from typing import List, Dict, Tuple
import pandas as pd
from dataclasses import dataclass
import nltk
from tqdm import tqdm

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


@dataclass
class Sentence:
    """Represents a sentence with metadata."""
    text: str
    doc_id: str
    section: str  # 'speech' or 'qa'
    speaker: str = ""
    role: str = ""
    turn_idx: int = 0
    sentence_idx: int = 0
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'doc_id': self.doc_id,
            'section': self.section,
            'speaker': self.speaker,
            'role': self.role,
            'turn_idx': self.turn_idx,
            'sentence_idx': self.sentence_idx
        }


class SentenceSplitter:
    """
    Splits transcript sections into sentences for NLP model inference.
    """
    
    def __init__(self, min_length: int = 10, max_length: int = 512):
        """
        Args:
            min_length: Minimum sentence length (characters) to keep
            max_length: Maximum sentence length (characters) before truncation
        """
        self.min_length = min_length
        self.max_length = max_length
        self.sent_tokenizer = nltk.sent_tokenize
    
    def clean_text(self, text: str) -> str:
        """Clean text for sentence splitting."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # [inaudible], [crosstalk], etc.
        text = re.sub(r'\(.*?\)', '', text)  # (phonetic), etc.
        return text.strip()
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence strings
        """
        if not text or not text.strip():
            return []
        
        text = self.clean_text(text)
        sentences = self.sent_tokenizer(text)
        
        # Filter and clean sentences
        result = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= self.min_length:
                # Truncate if too long
                if len(sent) > self.max_length:
                    sent = sent[:self.max_length] + "..."
                result.append(sent)
        
        return result
    
    def split_turns(
        self, 
        turns: List[Dict],
        doc_id: str,
        section: str
    ) -> List[Sentence]:
        """
        Split turns from transcript into sentences.
        
        Args:
            turns: List of turn dictionaries with 'text', 'speaker', 'role'
            doc_id: Document identifier (e.g., ticker_YYYYQX)
            section: 'speech' or 'qa'
            
        Returns:
            List of Sentence objects
        """
        sentences = []
        global_sent_idx = 0
        
        for turn_idx, turn in enumerate(turns):
            text = turn.get('text', '')
            speaker = turn.get('speaker', '')
            role = turn.get('role', '')
            
            sent_texts = self.split_text(text)
            
            for sent_text in sent_texts:
                sentences.append(Sentence(
                    text=sent_text,
                    doc_id=doc_id,
                    section=section,
                    speaker=speaker,
                    role=role,
                    turn_idx=turn_idx,
                    sentence_idx=global_sent_idx
                ))
                global_sent_idx += 1
        
        return sentences
    
    def process_parsed_transcript(
        self,
        parsed_row: Dict,
        ticker_col: str = 'ticker',
        year_col: str = 'year',
        quarter_col: str = 'quarter'
    ) -> Tuple[List[Sentence], List[Sentence]]:
        """
        Process a single parsed transcript row.
        
        Args:
            parsed_row: Row from parsed transcripts DataFrame
            
        Returns:
            Tuple of (speech_sentences, qa_sentences)
        """
        # Create document ID
        ticker = parsed_row.get(ticker_col, 'UNK')
        year = parsed_row.get(year_col, 0)
        quarter = parsed_row.get(quarter_col, 0)
        doc_id = f"{ticker}_{year}Q{quarter}"
        
        # Process speech turns
        speech_turns = parsed_row.get('speech_turns', [])
        if isinstance(speech_turns, str):
            import json
            try:
                speech_turns = json.loads(speech_turns)
            except:
                speech_turns = []
        
        speech_sentences = self.split_turns(speech_turns, doc_id, 'speech')
        
        # Process Q&A turns
        qa_turns = parsed_row.get('qa_turns', [])
        if isinstance(qa_turns, str):
            import json
            try:
                qa_turns = json.loads(qa_turns)
            except:
                qa_turns = []
        
        qa_sentences = self.split_turns(qa_turns, doc_id, 'qa')
        
        return speech_sentences, qa_sentences
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Process all parsed transcripts into sentences.
        
        Args:
            df: DataFrame with parsed transcript data
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with one row per sentence
        """
        all_sentences = []
        
        iterator = tqdm(df.iterrows(), total=len(df), desc="Splitting sentences") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            try:
                speech_sents, qa_sents = self.process_parsed_transcript(row.to_dict())
                
                for sent in speech_sents + qa_sents:
                    sent_dict = sent.to_dict()
                    # Add original row info
                    sent_dict['original_idx'] = idx
                    all_sentences.append(sent_dict)
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        if not all_sentences:
            return pd.DataFrame(
                columns=[
                    "text",
                    "doc_id",
                    "section",
                    "speaker",
                    "role",
                    "turn_idx",
                    "sentence_idx",
                    "original_idx",
                ]
            )

        return pd.DataFrame(all_sentences)


def create_sentence_dataset(
    parsed_transcripts_path: str,
    output_path: str,
    sample_n: int = None
) -> pd.DataFrame:
    """
    Create sentence-level dataset from parsed transcripts.
    
    Args:
        parsed_transcripts_path: Path to parsed transcripts parquet
        output_path: Path to save sentence dataset
        sample_n: Number of documents to sample (for testing)
        
    Returns:
        DataFrame with sentence data
    """
    print(f"Loading parsed transcripts from {parsed_transcripts_path}...")
    df = pd.read_parquet(parsed_transcripts_path)
    
    if sample_n:
        df = df.head(sample_n)
        print(f"Processing sample of {sample_n} documents")
    
    print(f"Total documents: {len(df)}")
    
    splitter = SentenceSplitter()
    sentences_df = splitter.process_dataframe(df)
    
    print(f"\n=== Sentence Split Summary ===")
    print(f"Total sentences: {len(sentences_df)}")
    if len(sentences_df) == 0:
        print("Speech sentences: 0")
        print("Q&A sentences: 0")
        print("Avg sentence length: 0 chars")
    else:
        print(f"Speech sentences: {len(sentences_df[sentences_df['section'] == 'speech'])}")
        print(f"Q&A sentences: {len(sentences_df[sentences_df['section'] == 'qa'])}")
        print(f"Avg sentence length: {sentences_df['text'].str.len().mean():.0f} chars")
    
    print(f"\nSaving to {output_path}...")
    sentences_df.to_parquet(output_path, index=False)
    
    return sentences_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split transcripts into sentences")
    parser.add_argument("--input", default="outputs/features/parsed_transcripts.parquet")
    parser.add_argument("--output", default="outputs/features/sentences.parquet")
    parser.add_argument("--sample", type=int, default=None)
    
    args = parser.parse_args()
    
    create_sentence_dataset(args.input, args.output, args.sample)
