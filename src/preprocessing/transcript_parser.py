"""
Transcript Parser Module

Parses earnings call transcripts to split into Speech (prepared remarks)
and Q&A sections, with turn-level extraction for Q&A.

Role classification uses session-aware inference:
  1. Parse operator introductions to extract analyst names & firms
  2. Word-boundary matching for management titles in speaker fields
  3. Structural heuristics (operator→analyst→management turn patterns)
  4. Role propagation: once identified, a speaker keeps that role
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm


@dataclass
class Turn:
    """Represents a single turn in the Q&A session."""
    speaker: str
    role: str  # 'analyst', 'management', 'operator', 'unknown'
    text: str
    is_question: bool = False

    def to_dict(self) -> dict:
        return {
            'speaker': self.speaker,
            'role': self.role,
            'text': self.text,
            'is_question': self.is_question
        }


@dataclass
class ParsedTranscript:
    """Container for a parsed earnings call transcript."""
    # Metadata
    ticker: str
    date: str
    quarter: int
    year: int

    # Split sections
    speech_text: str = ""
    qa_text: str = ""

    # Detailed turns
    speech_turns: List[Turn] = field(default_factory=list)
    qa_turns: List[Turn] = field(default_factory=list)

    # Statistics
    speech_word_count: int = 0
    qa_word_count: int = 0
    num_qa_exchanges: int = 0

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'date': self.date,
            'quarter': self.quarter,
            'year': self.year,
            'speech_text': self.speech_text,
            'qa_text': self.qa_text,
            'speech_turns': [t.to_dict() for t in self.speech_turns],
            'qa_turns': [t.to_dict() for t in self.qa_turns],
            'speech_word_count': self.speech_word_count,
            'qa_word_count': self.qa_word_count,
            'num_qa_exchanges': self.num_qa_exchanges
        }


class TranscriptParser:
    """
    Parser for S&P 500 earnings call transcripts.

    Handles the `structured_content` field which contains speaker-attributed
    text in JSON format.
    """

    # Patterns to identify Q&A section start
    QA_START_PATTERNS = [
        r"question.{0,10}answer",
        r"q\s*&\s*a\s+session",
        r"q\s*&\s*a\s+portion",
        r"open.*(?:for|to).*questions",
        r"take.*questions",
        r"we(?:'ll|\s+will)\s+now\s+take\s+(?:your|our|any)?\s*questions?",
        r"we(?:'re|\s+are)\s+ready\s+for\s+questions?",
        r"first question",
        r"over to.*(?:operator|questions)",
    ]

    OPERATOR_SPEAKER_PATTERNS = [
        r"\boperator\b",
        r"\bconference\s+operator\b",
        r"\bmoderator\b",
        r"\bcoordinator\b",
        r"\bquestion[-\s]?and[-\s]?answer\s+operator\b",
    ]

    # Analyst firm patterns — used with re.search on operator introduction text
    # and on speaker fields (when they contain affiliation).
    ANALYST_FIRM_PATTERNS = [
        # Bulge bracket / major sell-side
        r"\bgoldman\b", r"\bsachs\b", r"\bmorgan\s+stanley\b",
        r"\bjp\s*morgan\b", r"\bj\.?p\.?\s*morgan\b",
        r"\bbank\s+of\s+america\b", r"\bbofa\b",
        r"\bbarclays\b", r"\bcitigroup\b", r"\bciti\b",
        r"\bdeutsche\s+bank\b", r"\bdeutsche\b",
        r"\bubs\b", r"\bwells\s+fargo\b", r"\bcredit\s+suisse\b",
        r"\bnomura\b", r"\bhsbc\b", r"\bmacquarie\b", r"\bmizuho\b",
        # Mid-cap / boutique sell-side
        r"\bjefferies\b", r"\bpiper\s+sandler\b", r"\bstifel\b",
        r"\brbc\b", r"\bcowen\b", r"\bbaird\b", r"\bevercore\b",
        r"\bwolfe\s+research\b", r"\bbernstein\b", r"\bwilliam\s+blair\b",
        r"\btruist\b", r"\braymond\s+james\b", r"\bbmo\b", r"\bbtig\b",
        r"\boppenheimer\b", r"\bneedham\b", r"\bkbw\b",
        r"\bmoffett\s*nathanson\b", r"\bredburn\b", r"\bguggenheim\b",
        r"\bloop\s+capital\b", r"\bnew\s+street\b", r"\bnorthcoast\b",
        r"\bglenrock\b", r"\batlantic\s+equities\b", r"\bcanaccord\b",
        r"\bwedbush\b", r"\bkeybanc\b", r"\bstephens\b",
        r"\bleerink\b", r"\brosenblatt\b", r"\blake\s+street\b",
        r"\bd\.?\s*a\.?\s+davidson\b", r"\bjanney\b", r"\btelsey\b",
        r"\bwolfe\b", r"\bscotia\b",
        # Generic affiliation words (in speaker field or operator intro)
        r"\banalyst\b", r"\bresearch\b", r"\bsecurities\b",
        r"\badvisors?\b", r"\bassociates\b",
    ]

    # Management/executive title patterns — word-boundary to prevent
    # "coo" matching "Cook", "cto" matching "Spector", etc.
    MANAGEMENT_TITLE_PATTERNS = [
        r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcto\b", r"\bcio\b",
        r"\bpresident\b", r"\bchairman\b", r"\bchairwoman\b", r"\bchair\b",
        r"\bchief\b", r"\bexecutive\b", r"\bofficer\b",
        r"\bdirector\b", r"\bvice\s+president\b",
        r"\bhead\s+of\b", r"\bsenior\s+vice\b",
        r"\be\.?v\.?p\.?\b", r"\bs\.?v\.?p\.?\b",
        r"\btreasurer\b", r"\bcontroller\b", r"\bcomptroller\b",
        r"\bfounder\b", r"\bsecretary\b",
        r"\bgeneral\s+manager\b", r"\bgeneral\s+counsel\b",
        r"\bmanaging\s+director\b",
        r"\binvestor\s+relations\b", r"\b(?:head|vp|dir)\s+(?:of\s+)?ir\b",
    ]

    # Regex to extract analyst name and firm from operator introductions.
    # Matches patterns like:
    #   "next question comes from John Smith with Goldman Sachs"
    #   "next question from Jane Doe at JPMorgan"
    #   "our next caller is Bob Lee of Barclays"
    OPERATOR_INTRO_PATTERNS = [
        re.compile(
            r"(?:next\s+question|first\s+question|next\s+caller|"
            r"our\s+(?:next|first|final|last)\s+question)"
            r".*?(?:comes?\s+from|from|is)\s+"
            r"([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,3})"
            r"\s+(?:with|at|from|of)\s+"
            r"(.+?)(?:\.\s|\.?$|\.\s*(?:Please|Your|Go|One))",
            re.IGNORECASE
        ),
        re.compile(
            r"(?:question|caller).*?(?:line\s+of|from)\s+"
            r"([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,3})"
            r"\s+(?:with|at|from|of)\s+"
            r"(.+?)(?:\.\s|\.?$|\.\s*(?:Please|Your|Go|One))",
            re.IGNORECASE
        ),
        re.compile(
            r"(?:next\s+question|first\s+question|next\s+caller|our\s+(?:next|first|final|last)\s+question)"
            r".*?(?:comes?\s+from|from|is)\s+"
            r"([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,3})"
            r"(?:\.\s|\.?$|\s*(?:please|go\s+ahead))",
            re.IGNORECASE,
        ),
    ]

    def __init__(self):
        self.qa_pattern = re.compile(
            "|".join(self.QA_START_PATTERNS),
            re.IGNORECASE
        )
        self._analyst_firm_re = re.compile(
            "|".join(self.ANALYST_FIRM_PATTERNS), re.IGNORECASE
        )
        self._mgmt_title_re = re.compile(
            "|".join(self.MANAGEMENT_TITLE_PATTERNS), re.IGNORECASE
        )
        self._operator_speaker_re = re.compile(
            "|".join(self.OPERATOR_SPEAKER_PATTERNS), re.IGNORECASE
        )

    def parse_structured_content(self, content: str) -> List[Dict]:
        """
        Parse the structured_content field.

        Args:
            content: JSON string or list of speaker turns

        Returns:
            List of turn dictionaries with 'speaker' and 'text' keys
        """
        if isinstance(content, str):
            try:
                # Try parsing as JSON
                if content.startswith('['):
                    return json.loads(content)
                else:
                    # Single text block, no structure
                    return [{'speaker': 'Unknown', 'text': content}]
            except json.JSONDecodeError:
                return [{'speaker': 'Unknown', 'text': content}]
        elif isinstance(content, list):
            return content
        elif isinstance(content, tuple):
            return list(content)
        else:
            # PyArrow can materialize list columns as numpy arrays per cell
            try:
                import numpy as np

                if isinstance(content, np.ndarray):
                    return content.tolist()
            except Exception:
                pass
            return []

    # ------------------------------------------------------------------
    # Operator introduction parsing
    # ------------------------------------------------------------------

    def _extract_analyst_from_operator(self, text: str) -> Optional[str]:
        """Extract analyst name from operator introduction text.

        Returns the analyst's name (normalised) if found, else None.
        """
        for pat in self.OPERATOR_INTRO_PATTERNS:
            m = pat.search(text)
            if m:
                return self._normalise_name(m.group(1))
        return None

    @staticmethod
    def _normalise_name(name: str) -> str:
        """Lowercase, strip, collapse whitespace."""
        return re.sub(r"\s+", " ", name.strip()).lower()

    def _is_operator_speaker(self, speaker: str) -> bool:
        return bool(self._operator_speaker_re.search(speaker or ""))

    @staticmethod
    def _looks_like_generic_analyst_label(speaker_lower: str) -> bool:
        generic_analyst = [
            "unidentified analyst",
            "analyst",
            "questioner",
            "caller",
        ]
        return any(term in speaker_lower for term in generic_analyst)

    # ------------------------------------------------------------------
    # Role classification — single-turn (used as building block)
    # ------------------------------------------------------------------

    def classify_role(self, speaker: str, text: str = "") -> str:
        """
        Classify speaker role based on name/title field only.

        Uses word-boundary regex to avoid substring collisions
        (e.g. 'coo' matching 'Cook').

        Returns:
            'management', 'analyst', 'operator', 'unknown'
        """
        speaker_lower = speaker.lower()

        if self._is_operator_speaker(speaker_lower):
            return 'operator'

        # Analyst: firm name / role in speaker field (regex, not `in`)
        has_analyst_signal = bool(self._analyst_firm_re.search(speaker_lower))
        has_mgmt_signal = bool(self._mgmt_title_re.search(speaker_lower))

        if has_analyst_signal and (not has_mgmt_signal or self._looks_like_generic_analyst_label(speaker_lower)):
            return 'analyst'
        if has_mgmt_signal:
            return 'management'
        if has_analyst_signal:
            return 'analyst'
        if self._looks_like_generic_analyst_label(speaker_lower):
            return 'analyst'

        return 'unknown'

    # ------------------------------------------------------------------
    # Session-aware Q&A role inference
    # ------------------------------------------------------------------

    def classify_qa_roles(self, qa_turns: List[Dict],
                          speech_speakers: Set[str]) -> List[Tuple[str, str]]:
        """Infer roles for all Q&A turns using session-level context.

        Strategy (in priority order):
          1. Operator → 'operator'
          2. Speaker appeared in prepared remarks → 'management'
          3. Speaker field contains management title → 'management'
          4. Speaker introduced by operator → 'analyst'
          5. Speaker field contains analyst firm → 'analyst'
          6. Structural: first non-operator after operator intro → 'analyst'
          7. Structural: non-operator after analyst question → 'management'
          8. Role propagation: reuse previously assigned role for same speaker
          9. Fallback remains 'unknown'

        Returns:
            List of (role, speaker_normalised) per turn.
        """
        n = len(qa_turns)
        roles: List[Optional[str]] = [None] * n

        # Caches: speaker name → resolved role
        speaker_role_cache: Dict[str, str] = {}
        # Names mentioned by operator as analysts
        operator_announced: Set[str] = set()
        # Normalised speech speakers for comparison
        speech_norm = {self._normalise_name(s) for s in speech_speakers}

        # --- Pass 1: deterministic signals --------------------------------
        for i, turn in enumerate(qa_turns):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            sp_norm = self._normalise_name(speaker)

            # 1. Operator
            if self._is_operator_speaker(sp_norm):
                roles[i] = 'operator'
                # Extract analyst name from introduction
                announced = self._extract_analyst_from_operator(text)
                if announced:
                    operator_announced.add(announced)
                    speaker_role_cache[announced] = 'analyst'
                continue

            # 2. Speech speaker → management
            if sp_norm in speech_norm:
                roles[i] = 'management'
                speaker_role_cache[sp_norm] = 'management'
                continue

            # 3. Title in speaker field
            field_role = self.classify_role(speaker)
            if field_role in ('management', 'analyst'):
                roles[i] = field_role
                speaker_role_cache[sp_norm] = field_role
                continue

            # 4. Operator-announced analyst
            if sp_norm in operator_announced:
                roles[i] = 'analyst'
                speaker_role_cache[sp_norm] = 'analyst'
                continue

            # 5. Cache hit from earlier turn
            if sp_norm in speaker_role_cache:
                roles[i] = speaker_role_cache[sp_norm]
                continue

        # --- Pass 2: structural heuristics --------------------------------
        for i in range(n):
            if roles[i] is not None:
                continue

            speaker = qa_turns[i].get('speaker', '')
            sp_norm = self._normalise_name(speaker)

            # Cache hit (may have been populated during pass-2 itself)
            if sp_norm in speaker_role_cache:
                roles[i] = speaker_role_cache[sp_norm]
                continue

            # Find preceding operator or classified turn
            prev_role = None
            for j in range(i - 1, -1, -1):
                if roles[j] is not None:
                    prev_role = roles[j]
                    break

            # After operator intro → analyst
            if prev_role == 'operator':
                roles[i] = 'analyst'
                speaker_role_cache[sp_norm] = 'analyst'
                continue

            # After analyst → management (the responder)
            if prev_role == 'analyst':
                roles[i] = 'management'
                speaker_role_cache[sp_norm] = 'management'
                continue

            # Question-like turn by unknown speaker is typically analyst.
            text = qa_turns[i].get('text', '')
            if self.is_question(text):
                roles[i] = 'analyst'
                speaker_role_cache[sp_norm] = 'analyst'
                continue

            # Non-question follow-up after management usually remains management.
            if prev_role in ('management', 'unknown'):
                roles[i] = 'management'
                speaker_role_cache[sp_norm] = 'management'
                continue

        # --- Pass 3: propagate remaining unknowns -------------------------
        for i in range(n):
            if roles[i] is not None:
                continue
            speaker = qa_turns[i].get('speaker', '')
            sp_norm = self._normalise_name(speaker)
            if sp_norm in speaker_role_cache:
                roles[i] = speaker_role_cache[sp_norm]
            else:
                text = qa_turns[i].get('text', '')
                if self.is_question(text):
                    roles[i] = 'analyst'
                else:
                    roles[i] = 'management'

        return [(roles[i], self._normalise_name(qa_turns[i].get('speaker', '')))
                for i in range(n)]

    # --- Strong QA transition patterns (operator explicitly opening Q&A) ----
    _STRONG_QA_PATTERNS = [
        re.compile(
            r"(?:we\s+will\s+now\s+begin|let'?s\s+begin|"
            r"we(?:'re|\s+are)\s+now\s+(?:ready\s+to\s+)?(?:begin|open|start|move))"
            r".*?(?:question|q\s*&\s*a)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:open|opening)\s+(?:the\s+)?(?:floor|line|call)\s+(?:for|to)\s+questions",
            re.IGNORECASE,
        ),
        re.compile(
            r"operator\s+instructions",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:our\s+)?(?:first|next)\s+question\s+(?:comes?\s+from|is\s+from|is\s+coming|from\s+(?:the\s+)?line|today)",
            re.IGNORECASE,
        ),
        re.compile(
            r"we(?:'ll|\s+will)\s+take\s+(?:our\s+)?(?:first|next)\s+question",
            re.IGNORECASE,
        ),
        re.compile(
            r"we(?:'ll|\s+will)\s+now\s+take\s+(?:your|our|any)?\s*questions?",
            re.IGNORECASE,
        ),
    ]

    # Minimum number of non-operator turns (management / unknown speakers)
    # before patterns are considered valid.  Strong anchors (unambiguous
    # operator QA openers) need fewer turns; weak/generic patterns need more.
    _MIN_SPEECH_TURNS_STRONG = 2
    _MIN_SPEECH_TURNS_WEAK = 3

    # Minimum total management/unknown words before a QA signal required to
    # consider a transcript as having a real Speech section.  Below this
    # threshold the transcript is treated as "QA-only": entire content is Q&A.
    _MIN_SPEECH_WORD_THRESHOLD = 200

    # Phrases that mark an operator turn as a standard call-opening greeting
    # rather than an actual QA transition.
    _OPERATOR_WELCOME_PHRASES = [
        'welcome to',
        'welcome everyone',
        'listen-only mode',
        'listen only mode',
        'all participants are in',
        'conference call',
        'conference operator',
        'earnings call',
        'earnings conference',
        'i would like to welcome',
        'i would now like to turn',
        'you may begin',
        'please go ahead',
        'will follow the formal presentation',
    ]

    @staticmethod
    def _is_operator_welcome(text: str) -> bool:
        """Return True if the operator turn is a standard opening greeting.

        Operator greetings often include boilerplate like
        '[Operator Instructions]' as a template notice, not as an actual
        signal that Q&A has started.  We detect these by looking for
        characteristic welcome-ceremony phrases.
        """
        text_lower = text.lower()
        return any(p in text_lower for p in TranscriptParser._OPERATOR_WELCOME_PHRASES)

    def find_qa_start_index(self, turns: List[Dict]) -> int:
        """
        Find the index where Q&A section begins.

        Uses a multi-signal scoring approach:
          0. **QA-only early detection**: if the first strong QA signal from
             an operator turn appears before any substantial management speech
             (< _MIN_SPEECH_WORD_THRESHOLD words), the transcript has no real
             prepared-remarks section.  Return that operator index immediately.
          1. Warm-up buffer: at least _MIN_SPEECH_TURNS_STRONG non-operator
             turns must appear before any pattern match is allowed.
          2. Strong anchor: operator turns matching _STRONG_QA_PATTERNS get
             highest priority — these are unambiguous QA openers.
          3. Weak anchor: generic QA_START_PATTERNS in operator turns
             (after warm-up) are accepted as fallback.
          4. Non-operator speakers can NEVER trigger the cut via pattern
             matching alone — this prevents "Operator Pledge" false positives
             where a CEO mentions "questions" in prepared remarks.

        Returns:
            Index of first Q&A turn, or len(turns) if no Q&A found
        """
        n = len(turns)

        # ------------------------------------------------------------------
        # Pass 0 — QA-only transcript early detection
        #   Some transcripts have no prepared remarks at all: the operator's
        #   very first turn already opens Q&A.  Detect this by measuring
        #   total management speech content before the first strong QA signal.
        #   If < threshold words, accept the signal immediately.
        # ------------------------------------------------------------------
        mgmt_words_before_qa = 0
        for i, turn in enumerate(turns):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            is_operator = self._is_operator_speaker(speaker.lower())

            if is_operator:
                # Skip operator opening greetings — these often contain
                # boilerplate like "[Operator Instructions]" as part of the
                # standard welcome template, NOT as an actual QA start.
                if self._is_operator_welcome(text):
                    continue

                # Does this operator turn carry a strong QA signal?
                for pat in self._STRONG_QA_PATTERNS:
                    if pat.search(text):
                        if mgmt_words_before_qa < self._MIN_SPEECH_WORD_THRESHOLD:
                            # No real speech section — QA-only transcript
                            return i
                        # Substantial speech exists; fall through to
                        # warm-up-buffered passes below.
                        break
            else:
                role = self.classify_role(speaker, text)
                if role in ('management', 'unknown'):
                    mgmt_words_before_qa += len(text.split())

        # ------------------------------------------------------------------
        # Pass 1 — strong operator anchor (with warm-up buffer)
        # ------------------------------------------------------------------
        non_operator_count = 0
        for i, turn in enumerate(turns):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            is_operator = self._is_operator_speaker(speaker.lower())

            if not is_operator:
                role = self.classify_role(speaker, text)
                if role in ('management', 'unknown'):
                    non_operator_count += 1
                continue  # only operators can trigger in this pass

            # Before warm-up, skip
            if non_operator_count < self._MIN_SPEECH_TURNS_STRONG:
                continue

            # Strong anchor — unambiguous Q&A start
            for pat in self._STRONG_QA_PATTERNS:
                if pat.search(text):
                    return i

        # ------------------------------------------------------------------
        # Pass 2 — weak operator anchor (generic QA_START_PATTERNS)
        #   Only operator turns, only after warm-up
        # ------------------------------------------------------------------
        non_operator_count = 0
        for i, turn in enumerate(turns):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            is_operator = self._is_operator_speaker(speaker.lower())

            if not is_operator:
                role = self.classify_role(speaker, text)
                if role in ('management', 'unknown'):
                    non_operator_count += 1
                continue

            if non_operator_count < self._MIN_SPEECH_TURNS_WEAK:
                continue

            if self.qa_pattern.search(text):
                return i

        # ------------------------------------------------------------------
        # Pass 3 — IR / management transition phrase
        #   Some calls have an IR director saying "let's move to Q&A" right
        #   before the operator opens the line.  Accept this ONLY if the very
        #   next turn is an operator turn (confirming the transition).
        # ------------------------------------------------------------------
        non_operator_count = 0
        for i, turn in enumerate(turns):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            is_operator = self._is_operator_speaker(speaker.lower())

            if not is_operator:
                role = self.classify_role(speaker, text)
                if role in ('management', 'unknown'):
                    non_operator_count += 1

                if non_operator_count < self._MIN_SPEECH_TURNS_STRONG:
                    continue

                # Check for strong transition phrase by non-operator
                text_lower = text.lower()
                transition_phrases = [
                    'move over to q&a', 'move to q&a', 'move to questions',
                    'open it up for questions', 'open the line for questions',
                    'begin the q&a', 'start the q&a', 'ready to start the q&a',
                    'we will now begin the question', 'let\'s move to questions',
                ]
                found_transition = any(p in text_lower for p in transition_phrases)
                if found_transition:
                    # Confirm: next turn should be operator or analyst
                    if i + 1 < n:
                        next_sp = turns[i + 1].get('speaker', '').lower()
                        if 'operator' in next_sp:
                            return i  # cut at the transition turn
                    # Even without confirmation, if strong transition phrase
                    return i

        # ------------------------------------------------------------------
        # Pass 4 — structural fallback: operator turn with "first question"
        #   after at least one non-operator turn
        # ------------------------------------------------------------------
        seen_non_operator = False
        for i, turn in enumerate(turns):
            speaker = turn.get('speaker', '')
            if not self._is_operator_speaker(speaker.lower()):
                seen_non_operator = True
                continue
            if not seen_non_operator:
                continue
            text = turn.get('text', '').strip().lower()
            if 'first question' in text:
                return i

        # ------------------------------------------------------------------
        # Pass 5 — last resort: first analyst question (by role + text)
        # ------------------------------------------------------------------
        for i, turn in enumerate(turns):
            text = turn.get('text', '').strip()
            if '?' in text[:300]:
                role = self.classify_role(turn.get('speaker', ''), text)
                if role == 'analyst':
                    return max(0, i - 1)

        return n  # No Q&A found

    def is_question(self, text: str) -> bool:
        """Determine if text contains a question."""
        text_start = text[:500] if len(text) > 500 else text

        if '?' in text_start:
            return True

        first_words = text.strip().lower()[:50]
        question_starters = ['can ', 'could ', 'what ', 'how ', 'why ', 'when ',
                            'where ', 'is ', 'are ', 'do ', 'does ', 'would ',
                            'will ', 'should ', 'may ', 'might ']
        for starter in question_starters:
            if first_words.startswith(starter):
                return True

        return False

    def parse(
        self,
        structured_content,
        ticker: str,
        date: str,
        quarter: int,
        year: int
    ) -> ParsedTranscript:
        """
        Parse a single transcript into Speech and Q&A sections.

        Args:
            structured_content: The structured_content field (JSON or list)
            ticker: Stock ticker symbol
            date: Earnings call date
            quarter: Fiscal quarter
            year: Fiscal year

        Returns:
            ParsedTranscript object
        """
        result = ParsedTranscript(
            ticker=ticker,
            date=str(date),
            quarter=quarter,
            year=year
        )

        # Parse structured content
        turns = self.parse_structured_content(structured_content)
        if not turns:
            return result

        # Find Q&A boundary
        qa_start = self.find_qa_start_index(turns)

        # Split into speech and Q&A
        speech_turns_raw = turns[:qa_start]
        qa_turns_raw = turns[qa_start:]

        # Collect speech speakers (these are management by definition)
        speech_speakers: Set[str] = set()

        # Process speech turns
        speech_texts = []
        for turn in speech_turns_raw:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            role = self.classify_role(speaker, text)
            if role == 'unknown':
                role = 'management'

            speech_speakers.add(speaker)

            result.speech_turns.append(Turn(
                speaker=speaker,
                role=role,
                text=text,
                is_question=False
            ))
            speech_texts.append(text)

        result.speech_text = "\n\n".join(speech_texts)
        result.speech_word_count = len(result.speech_text.split())

        # Process Q&A turns with session-aware classification
        qa_roles = self.classify_qa_roles(qa_turns_raw, speech_speakers)

        qa_texts = []
        num_exchanges = 0
        for turn, (role, _) in zip(qa_turns_raw, qa_roles):
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            is_q = self.is_question(text) and role == 'analyst'

            if is_q:
                num_exchanges += 1

            result.qa_turns.append(Turn(
                speaker=speaker,
                role=role,
                text=text,
                is_question=is_q
            ))
            qa_texts.append(text)

        result.qa_text = "\n\n".join(qa_texts)
        result.qa_word_count = len(result.qa_text.split())
        result.num_qa_exchanges = num_exchanges

        return result
    
    def parse_dataframe(
        self, 
        df: pd.DataFrame,
        structured_content_col: str = 'structured_content',
        ticker_col: str = 'ticker',
        date_col: str = 'date',
        quarter_col: str = 'quarter',
        year_col: str = 'year',
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Parse all transcripts in a DataFrame.
        
        Args:
            df: DataFrame with transcript data
            structured_content_col: Column name for structured content
            ticker_col: Column name for ticker
            date_col: Column name for date
            quarter_col: Column name for quarter
            year_col: Column name for year
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with parsed transcript information
        """
        results = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing transcripts") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            try:
                parsed = self.parse(
                    structured_content=row[structured_content_col],
                    ticker=row.get(ticker_col, 'UNK'),
                    date=row.get(date_col, ''),
                    quarter=row.get(quarter_col, 0),
                    year=row.get(year_col, 0)
                )
                results.append(parsed.to_dict())
            except Exception as e:
                print(f"Error parsing row {idx}: {e}")
                results.append({
                    'ticker': row.get(ticker_col, 'UNK'),
                    'date': str(row.get(date_col, '')),
                    'quarter': row.get(quarter_col, 0),
                    'year': row.get(year_col, 0),
                    'error': str(e)
                })
        
        return pd.DataFrame(results)


def process_dataset(
    input_path: str,
    output_path: str,
    sample_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Process the full dataset and save parsed transcripts.
    
    Args:
        input_path: Path to input parquet/csv file
        output_path: Path to save output parquet file
        sample_n: If set, only process this many rows (for testing)
        
    Returns:
        DataFrame with parsed data
    """
    print(f"Loading data from {input_path}...")
    
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    if sample_n:
        df = df.head(sample_n)
        print(f"Processing sample of {sample_n} rows")
    
    print(f"Total rows to process: {len(df)}")
    
    # Initialize parser
    parser = TranscriptParser()
    
    # Parse all transcripts
    parsed_df = parser.parse_dataframe(df)
    
    # Save results
    print(f"Saving parsed data to {output_path}...")
    parsed_df.to_parquet(output_path, index=False)
    
    # Print summary
    print("\n=== Parsing Summary ===")
    print(f"Total transcripts: {len(parsed_df)}")
    print(f"Avg speech words: {parsed_df['speech_word_count'].mean():.0f}")
    print(f"Avg Q&A words: {parsed_df['qa_word_count'].mean():.0f}")
    print(f"Avg Q&A exchanges: {parsed_df['num_qa_exchanges'].mean():.1f}")
    
    return parsed_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse earnings call transcripts into speech and Q&A blocks.")
    parser.add_argument("--input", default="data/final_dataset.parquet", help="Input file path")
    parser.add_argument("--output", default="outputs/features/parsed_transcripts.parquet", help="Output file path")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples for testing")
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.sample)
