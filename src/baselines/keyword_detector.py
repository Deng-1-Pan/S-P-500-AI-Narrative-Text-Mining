"""
Keyword Detector Module

Dictionary-based detector for AI topic detection with strong/weak signal
heuristics tuned for earnings-call language.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


@dataclass
class KeywordMatch:
    """Represents a keyword match in text."""

    keyword: str
    category: str
    start: int
    end: int
    context: str


@dataclass(frozen=True)
class _KeywordSpec:
    keyword: str
    category: str
    signal: str  # "strong" or "weak"
    pattern: re.Pattern


class AIKeywordDetector:
    """
    Dictionary-based detector for AI-related content.

    Heuristic threshold policy (strict):
    - AI-positive if text contains >= 1 strong AI keyword.
    - Otherwise AI-positive only if text contains a clear weak-signal combo:
      at least 2 distinct non-excluded weak keywords (or >= 3 non-excluded
      weak matches when repeated wording dominates).
    - Weak-only matches in routine finance/operations contexts are excluded.
    """

    KEYWORD_DICT = {
        "core_ai": [
            "artificial intelligence",
            "ai",
            "machine learning",
            "ml",
            "deep learning",
            "neural network",
            "neural net",
            "ai model",
            "ai models",
            "language model",
        ],
        "generative_ai": [
            "generative ai",
            "gen ai",
            "genai",
            "chatgpt",
            "gpt",
            "gpt4",
            "gpt-4",
            "large language model",
            "llm",
            "llms",
            "foundation model",
            "transformer model",
            "retrieval augmented generation",
            "rag",
            "prompt engineering",
            "copilot",
            "co-pilot",
            "openai",
            "anthropic",
            "claude",
            "gemini",
            "bard",
            "midjourney",
            "dall-e",
            "stable diffusion",
            "deepseek",
        ],
        "ml_techniques": [
            "natural language processing",
            "nlp",
            "computer vision",
            "image recognition",
            "speech recognition",
            "voice recognition",
            "reinforcement learning",
            "supervised learning",
            "unsupervised learning",
            "anomaly detection",
            "predictive model",
            "classification model",
            "regression model",
            "recommendation engine",
            "recommendation system",
        ],
        "automation": [
            "automation",
            "automate",
            "automated",
            "automating",
            "robotic process automation",
            "rpa",
            "workflow automation",
            "process automation",
            "industrial automation",
            "grid automation",
            "factory automation",
        ],
        "data_analytics": [
            "data analytics",
            "advanced analytics",
            "predictive analytics",
            "big data",
            "data science",
            "data-driven",
            "algorithm",
            "algorithmic",
            "analytics",
        ],
        "ai_infrastructure": [
            "gpu",
            "gpus",
            "cuda",
            "tensor",
            "ai chip",
            "ai chips",
            "compute capacity",
            "training data",
            "inference",
            "inference workload",
            "data center",
            "data centers",
            "cloud computing",
            "edge computing",
        ],
        "ai_applications": [
            "conversational ai",
            "ai-powered",
            "ai-driven",
            "ai-enabled",
            "ai-based",
            "intelligent assistant",
            "virtual assistant",
            "chatbot",
            "cognitive computing",
            "machine intelligence",
        ],
    }

    STRONG_CATEGORIES = {
        "core_ai",
        "generative_ai",
        "ml_techniques",
        "ai_applications",
    }

    STRONG_AI_KEYWORDS = sorted(
        {
            kw
            for cat, kws in KEYWORD_DICT.items()
            if cat in {"core_ai", "generative_ai", "ml_techniques", "ai_applications"}
            for kw in kws
        }
    )

    WEAK_TECH_KEYWORDS = sorted(
        {
            kw
            for cat, kws in KEYWORD_DICT.items()
            if cat not in {"core_ai", "generative_ai", "ml_techniques", "ai_applications"}
            for kw in kws
        }
    )

    GENERIC_EXCLUSION_PATTERNS = [
        r"\bair\b",
        r"\baid\b",
        r"\baim\b",
        r"\bpaid\b",
        r"\b\d+(?:\.\d+)?\s*m[lL]\b",
        r"\bemail\b",
        r"\bretail\b",
        r"\bdetail\b",
        r"\bmaintain\b",
        r"\bcontainer\b",
    ]

    # Weak-signal anti-patterns for routine finance/operations context.
    WEAK_CONTEXT_EXCLUSION_PATTERNS = [
        r"\bdata\s+center(?:s)?\b.{0,50}\b(capex|load(?:\s+growth)?|mw|megawatt|power|electric|utility|grid|cooling|lease|land|construction|facility|occupancy|real\s+estate)\b",
        r"\b(capex|load(?:\s+growth)?|mw|megawatt|power|electric|utility|grid|cooling|lease|land|construction|facility|occupancy|real\s+estate)\b.{0,50}\bdata\s+center(?:s)?\b",
        r"\bcloud\b.{0,50}\b(spend|cost|pricing|consumption|migration|workload|optimization|infrastructure)\b",
        r"\b(spend|cost|pricing|consumption|migration|workload|optimization)\b.{0,50}\bcloud\b",
        r"\b(pricing|revenue|margin|eps|guidance|outlook|growth|cost|opex|capex|expense|profit|algorithm)\b.{0,50}\balgorithm(?:ic)?\b",
        r"\balgorithm(?:ic)?\b.{0,50}\b(pricing|revenue|margin|eps|guidance|outlook|growth|cost|opex|capex|expense|profit|search|ranking|index|allocation)\b",
        r"\bautomation\b.{0,50}\b(factory|manufacturing|plant|warehouse|workflow|process|utility|grid|industrial|distribution|fulfillment)\b",
        r"\b(factory|manufacturing|plant|warehouse|workflow|process|utility|grid|industrial|distribution|fulfillment)\b.{0,50}\bautomation\b",
        r"\bdata\s+analytics\b.{0,50}\b(marketing|customer|sales|pricing|risk|fraud|credit|portfolio|operations?)\b",
        r"\b(marketing|customer|sales|pricing|risk|fraud|credit|portfolio|operations?)\b.{0,50}\bdata\s+analytics\b",
        r"\blong[-\s]?term\s+algorithm\b",
        r"\bfinancial\s+algorithm\b",
    ]

    # Require at least two distinct weak keywords to avoid one-term echoes
    # (e.g., repeated "data center") from counting as AI.
    MIN_DISTINCT_WEAK_FOR_AI = 2

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        self._compile_patterns()

    @staticmethod
    def _keyword_pattern(keyword: str) -> str:
        short_terms = {
            "ai",
            "ml",
            "llm",
            "llms",
            "nlp",
            "rpa",
            "rag",
            "gpu",
            "gpus",
            "gpt",
            "gpt4",
            "gpt-4",
        }
        escaped = re.escape(keyword)
        if keyword in short_terms or len(keyword) <= 3:
            return rf"\b{escaped}\b"
        return rf"\b{escaped}(?:s)?\b"

    def _compile_patterns(self) -> None:
        flags = 0 if self.case_sensitive else re.IGNORECASE

        self._keyword_specs: List[_KeywordSpec] = []
        for category, keywords in self.KEYWORD_DICT.items():
            signal = "strong" if category in self.STRONG_CATEGORIES else "weak"
            for kw in keywords:
                self._keyword_specs.append(
                    _KeywordSpec(
                        keyword=kw,
                        category=category,
                        signal=signal,
                        pattern=re.compile(self._keyword_pattern(kw), flags),
                    )
                )

        # Backward-compatible structure used in some downstream code.
        self.patterns: Dict[str, List[re.Pattern]] = {cat: [] for cat in self.KEYWORD_DICT}
        for spec in self._keyword_specs:
            self.patterns[spec.category].append(spec.pattern)

        self.exclusions = [
            re.compile(p, re.IGNORECASE) for p in self.GENERIC_EXCLUSION_PATTERNS
        ]
        self.weak_context_exclusions = [
            re.compile(p, re.IGNORECASE) for p in self.WEAK_CONTEXT_EXCLUSION_PATTERNS
        ]

    @staticmethod
    def _weak_family(keyword: str) -> str:
        """Collapse weak variants into semantic families to avoid echo FPs."""
        kw = keyword.lower().strip()
        if "data center" in kw:
            return "data_center"
        if "automat" in kw or kw == "rpa" or "process automation" in kw:
            return "automation"
        if "algorithm" in kw:
            return "algorithm"
        if "analytic" in kw or kw == "data science" or kw == "big data":
            return "analytics"
        if "cloud" in kw:
            return "cloud"
        if "gpu" in kw or "cuda" in kw or "tensor" in kw or "chip" in kw:
            return "compute_hw"
        return kw

    def _is_generic_excluded(self, text: str, match_start: int, match_end: int) -> bool:
        context_start = max(0, match_start - 10)
        context_end = min(len(text), match_end + 10)
        context = text[context_start:context_end].lower()
        return any(excl.search(context) for excl in self.exclusions)

    def _is_weak_context_excluded(
        self, text: str, match_start: int, match_end: int
    ) -> bool:
        # Use a wider span for weak-context anti-patterns.
        window_start = max(0, match_start - 120)
        window_end = min(len(text), match_end + 120)
        window = text[window_start:window_end]
        return any(p.search(window) for p in self.weak_context_exclusions)

    def detect(self, text: str) -> List[KeywordMatch]:
        """Detect AI-related keywords in text (raw match extraction)."""
        if not text:
            return []

        matches: List[KeywordMatch] = []
        for spec in self._keyword_specs:
            for m in spec.pattern.finditer(text):
                if self._is_generic_excluded(text, m.start(), m.end()):
                    continue

                ctx_start = max(0, m.start() - 50)
                ctx_end = min(len(text), m.end() + 50)
                matches.append(
                    KeywordMatch(
                        keyword=m.group(),
                        category=spec.category,
                        start=m.start(),
                        end=m.end(),
                        context=text[ctx_start:ctx_end],
                    )
                )

        return matches

    def get_signal_profile(self, text: str) -> Dict[str, int | bool]:
        """
        Return signal counts used by strict AI gating and initiation logic.

        Returns dict keys:
        - strong_count, strong_unique
        - weak_count, weak_unique
        - weak_nonexcluded_count, weak_nonexcluded_unique
        - is_ai
        """
        counts = self.count_matches(text)

        strong_count = int(counts["strong_count"])
        weak_count = int(counts["weak_count"])
        weak_nonexcluded_count = int(counts["weak_nonexcluded_count"])
        strong_unique = int(counts["strong_unique"])
        weak_unique = int(counts["weak_unique"])
        weak_nonexcluded_unique = int(counts["weak_nonexcluded_unique"])

        weak_combo = (
            weak_nonexcluded_unique >= self.MIN_DISTINCT_WEAK_FOR_AI
        )
        is_ai = strong_count >= 1 or weak_combo

        return {
            "strong_count": strong_count,
            "strong_unique": strong_unique,
            "weak_count": weak_count,
            "weak_unique": weak_unique,
            "weak_nonexcluded_count": weak_nonexcluded_count,
            "weak_nonexcluded_unique": weak_nonexcluded_unique,
            "is_ai": is_ai,
        }

    def is_ai_related(self, text: str) -> bool:
        """
        Strict binary AI detection.

        Rule: 1 strong keyword OR clear weak combo (>=2 distinct non-excluded
        weak-term families).
        """
        return bool(self.get_signal_profile(text).get("is_ai", False))

    def count_matches(self, text: str) -> Dict[str, int]:
        """
        Count AI keyword matches by category and signal strength.

        Keeps backward-compatible category keys while adding strict-signal keys.
        """
        matches = self.detect(text)

        counts: Dict[str, int] = {cat: 0 for cat in self.KEYWORD_DICT.keys()}
        counts["total"] = 0
        counts["strong_count"] = 0
        counts["weak_count"] = 0
        counts["weak_nonexcluded_count"] = 0

        strong_terms = set()
        weak_terms = set()
        weak_nonexcluded_terms = set()

        # Lookup by lowercase keyword for signal type.
        signal_by_keyword = {
            spec.keyword.lower(): spec.signal for spec in self._keyword_specs
        }

        # Resolve canonical keyword for duplicate-cased matches.
        all_keywords = sorted(signal_by_keyword.keys(), key=len, reverse=True)

        for m in matches:
            counts[m.category] += 1
            counts["total"] += 1

            match_key = m.keyword.lower().strip()
            signal = signal_by_keyword.get(match_key)
            if signal is None:
                # Fallback for pluralized/variant matches.
                signal = "weak"
                for kw in all_keywords:
                    if match_key == kw or match_key.rstrip("s") == kw.rstrip("s"):
                        signal = signal_by_keyword[kw]
                        match_key = kw
                        break

            if signal == "strong":
                counts["strong_count"] += 1
                strong_terms.add(match_key)
            else:
                counts["weak_count"] += 1
                weak_terms.add(self._weak_family(match_key))
                if not self._is_weak_context_excluded(text, m.start, m.end):
                    counts["weak_nonexcluded_count"] += 1
                    weak_nonexcluded_terms.add(self._weak_family(match_key))

        counts["strong_unique"] = len(strong_terms)
        counts["weak_unique"] = len(weak_terms)
        counts["weak_nonexcluded_unique"] = len(weak_nonexcluded_terms)

        return counts

    def get_ai_score(self, text: str, normalize: bool = True) -> float:
        """
        Get AI intensity score for text.

        Uses strict AI signal count (`strong_count + weak_nonexcluded_count`).
        """
        counts = self.count_matches(text)
        count = int(counts["strong_count"]) + int(counts["weak_nonexcluded_count"])

        if normalize and text:
            word_count = len(text.split())
            if word_count > 0:
                return count / word_count * 100

        return float(count)


def _process_texts_chunk(texts: List[str]) -> List[Dict[str, int | float | bool]]:
    detector = AIKeywordDetector()
    results = []
    for text in texts:
        matches = detector.detect(text)
        counts = detector.count_matches(text)
        result = {
            "kw_is_ai": detector.is_ai_related(text),
            "kw_match_count": len(matches),
            "kw_ai_score": detector.get_ai_score(text),
            "kw_strong_count": counts["strong_count"],
            "kw_weak_count": counts["weak_count"],
            "kw_weak_nonexcluded_count": counts["weak_nonexcluded_count"],
            **{f"kw_{cat}_count": counts[cat] for cat in detector.KEYWORD_DICT.keys()},
        }
        results.append(result)
    return results


def compute_keyword_metrics(
    sentences_df: pd.DataFrame,
    text_col: str = "text",
    doc_id_col: str = "doc_id",
    section_col: str = "section",
    num_workers: Optional[int] = None,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """
    Compute keyword-based AI metrics for sentences.

    Keeps legacy output columns and adds strong/weak signal counts.
    """
    print("Detecting AI keywords in sentences...")

    texts = sentences_df[text_col].fillna("").astype(str).tolist()
    total = len(texts)
    if total == 0:
        result_df = pd.DataFrame(
            columns=[
                "kw_is_ai",
                "kw_match_count",
                "kw_ai_score",
                "kw_strong_count",
                "kw_weak_count",
                "kw_weak_nonexcluded_count",
            ]
        )
        return pd.concat([sentences_df.reset_index(drop=True), result_df], axis=1)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    if num_workers <= 1 or total < chunk_size:
        detector = AIKeywordDetector()
        results = []
        for text in tqdm(texts, total=total):
            matches = detector.detect(text)
            counts = detector.count_matches(text)
            result = {
                "kw_is_ai": detector.is_ai_related(text),
                "kw_match_count": len(matches),
                "kw_ai_score": detector.get_ai_score(text),
                "kw_strong_count": counts["strong_count"],
                "kw_weak_count": counts["weak_count"],
                "kw_weak_nonexcluded_count": counts["weak_nonexcluded_count"],
                **{f"kw_{cat}_count": counts[cat] for cat in detector.KEYWORD_DICT.keys()},
            }
            results.append(result)
    else:
        print(f"Using multiprocessing with {num_workers} workers (chunk_size={chunk_size})")
        chunks = [texts[i : i + chunk_size] for i in range(0, total, chunk_size)]
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for chunk_res in tqdm(ex.map(_process_texts_chunk, chunks), total=len(chunks)):
                results.extend(chunk_res)

    result_df = pd.DataFrame(results)
    return pd.concat([sentences_df.reset_index(drop=True), result_df], axis=1)


def compute_document_metrics(
    sentences_df: pd.DataFrame,
    doc_id_col: str = "doc_id",
    section_col: str = "section",
) -> pd.DataFrame:
    """
    Aggregate keyword metrics at document level.

    Args:
        sentences_df: DataFrame with sentence-level keyword metrics

    Returns:
        DataFrame with document-level metrics
    """
    agg_funcs = {
        "kw_is_ai": ["sum", "mean"],
        "kw_match_count": "sum",
        "kw_ai_score": "mean",
    }

    results = []

    for doc_id in sentences_df[doc_id_col].unique():
        doc_df = sentences_df[sentences_df[doc_id_col] == doc_id]

        doc_result = {"doc_id": doc_id}

        for section in ["speech", "qa"]:
            section_df = doc_df[doc_df[section_col] == section]

            if len(section_df) > 0:
                doc_result[f"{section}_total_sentences"] = len(section_df)
                doc_result[f"{section}_ai_sentences"] = section_df["kw_is_ai"].sum()
                doc_result[f"{section}_ai_ratio"] = section_df["kw_is_ai"].mean()
                doc_result[f"{section}_total_matches"] = section_df["kw_match_count"].sum()
                doc_result[f"{section}_avg_ai_score"] = section_df["kw_ai_score"].mean()
            else:
                doc_result[f"{section}_total_sentences"] = 0
                doc_result[f"{section}_ai_sentences"] = 0
                doc_result[f"{section}_ai_ratio"] = 0.0
                doc_result[f"{section}_total_matches"] = 0
                doc_result[f"{section}_avg_ai_score"] = 0.0

        results.append(doc_result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    detector = AIKeywordDetector()

    test_texts = [
        "We are investing heavily in artificial intelligence and machine learning capabilities.",
        "Our ChatGPT integration has been transformative for customer service.",
        "Revenue increased by 15% this quarter.",
        "The automation of our processes using AI has reduced costs significantly.",
        "Can you discuss data center capex and expected load growth?",
    ]

    print("=== Keyword Detection Demo ===\n")
    for text in test_texts:
        matches = detector.detect(text)
        is_ai = detector.is_ai_related(text)
        score = detector.get_ai_score(text)
        profile = detector.get_signal_profile(text)

        print(f"Text: {text[:70]}...")
        print(f"  AI-related: {is_ai}")
        print(f"  AI Score: {score:.2f}")
        print(
            "  Signal: "
            f"strong={profile['strong_count']} "
            f"weak_nonexcluded={profile['weak_nonexcluded_count']}"
        )
        print(f"  Matches: {[m.keyword for m in matches]}")
        print()
