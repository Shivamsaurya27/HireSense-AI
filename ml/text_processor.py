"""
ml/text_processor.py
---------------------
Core NLP Text Processing Module for HireSense AI.

Takes the raw text produced by `resume_parser.py` and turns it into
clean, structured, analysis-ready data:
    - normalized/cleaned text
    - tokens (words, sentences)
    - stopword-free, lemmatized tokens
    - detected resume sections (Experience, Education, Skills, etc.)
    - extracted contact info (email, phone, links)
    - basic named-entity extraction (via spaCy, if available)
    - word frequency / n-gram helpers used later for keyword extraction

This module does NOT do skill extraction or scoring — see
`skill_extractor.py` and `ats_scorer.py` for that. Its job is to be the
shared NLP foundation everything else builds on.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

logger = logging.getLogger("hiresense.text_processor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# --- Optional NLP dependencies ------------------------------------------
# We degrade gracefully rather than crash if a library or its data files
# aren't installed — important since students often forget `nltk.download`
# or `python -m spacy download en_core_web_sm`.

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    import spacy

    _SPACY_LIB_AVAILABLE = True
except ImportError:
    _SPACY_LIB_AVAILABLE = False


def _ensure_nltk_resources() -> bool:
    """Make sure the NLTK corpora we need are available, downloading them
    on first use if missing. Returns True if everything needed is ready.
    """
    if not _NLTK_AVAILABLE:
        return False

    required = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    all_ok = True
    for resource_path, package_name in required:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not download NLTK resource '%s': %s. "
                    "Falling back to simpler text processing where needed.",
                    package_name,
                    exc,
                )
                all_ok = False
    return all_ok


_NLTK_READY = _ensure_nltk_resources()


def _load_spacy_model():
    """Try to load a small English spaCy model. Returns None if the
    library or the model isn't installed (NER features then degrade
    to regex-based heuristics instead of crashing).
    """
    if not _SPACY_LIB_AVAILABLE:
        return None
    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    logger.warning(
        "No spaCy English model found. Run "
        "`python -m spacy download en_core_web_sm` for full NER support. "
        "Falling back to regex-based heuristics for entity extraction."
    )
    return None


_SPACY_NLP = _load_spacy_model()
_SPACY_AVAILABLE = _SPACY_NLP is not None


# Minimal built-in English stopword list used ONLY if NLTK's stopword
# corpus is unavailable. This is intentionally small; NLTK's ~180-word
# list is preferred whenever possible.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "on", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "can", "could", "i", "you", "he", "she", "it", "we", "they",
    "this", "that", "these", "those", "as", "so", "than", "too", "very",
}


# --- Resume section detection --------------------------------------------
# Canonical section names mapped to the header phrases/synonyms that
# commonly introduce them in real-world resumes. Matching is done
# line-by-line against short, mostly-uppercase-or-titlecase lines that
# look like headers, so we don't accidentally match the word "education"
# appearing mid-sentence in a paragraph.
SECTION_HEADER_PATTERNS: dict[str, list[str]] = {
    "summary": [
        "summary", "professional summary", "profile", "objective",
        "career objective", "about me",
    ],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "career history",
    ],
    "education": [
        "education", "academic background", "academic qualifications",
        "educational qualifications",
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "key skills", "skill set", "areas of expertise",
    ],
    "projects": [
        "projects", "academic projects", "personal projects",
        "key projects",
    ],
    "certifications": [
        "certifications", "certificates", "licenses",
        "certifications and licenses",
    ],
    "achievements": [
        "achievements", "awards", "honors", "accomplishments",
    ],
    "publications": ["publications", "research papers"],
    "extracurricular": [
        "extracurricular", "activities", "volunteer experience",
        "volunteering",
    ],
    "contact": ["contact", "contact information", "personal details"],
}

# Pre-build a reverse lookup: normalized header phrase -> canonical section.
_HEADER_TO_SECTION: dict[str, str] = {}
for canonical_name, phrases in SECTION_HEADER_PATTERNS.items():
    for phrase in phrases:
        _HEADER_TO_SECTION[phrase.lower()] = canonical_name


# --- Contact info regex patterns ------------------------------------------
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3,4}[\s.\-]?\d{3,4}\b"
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s,;]+", re.IGNORECASE)
_LINKEDIN_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s,;]+", re.IGNORECASE)
_GITHUB_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[^\s,;]+", re.IGNORECASE)


@dataclass
class ProcessedText:
    """Structured NLP output for a piece of resume/job-description text."""

    original_text: str
    cleaned_text: str = ""
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    tokens_no_stopwords: list[str] = field(default_factory=list)
    lemmatized_tokens: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    email: Optional[str] = None
    phone: Optional[str] = None
    links: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "cleaned_text": self.cleaned_text,
            "sentence_count": len(self.sentences),
            "word_count": self.word_count,
            "sections": list(self.sections.keys()),
            "email": self.email,
            "phone": self.phone,
            "links": self.links,
            "entities": self.entities,
        }


class TextProcessor:
    """Cleans, tokenizes, and structurally analyzes resume/job-description
    text. Instantiate once and reuse — spaCy/NLTK setup happens at import
    time, not per-call.
    """

    def __init__(self) -> None:
        self.nltk_ready = _NLTK_READY
        self.spacy_ready = _SPACY_AVAILABLE

        if _NLTK_READY:
            try:
                self._stopwords = set(nltk_stopwords.words("english"))
            except Exception:  # noqa: BLE001
                self._stopwords = set(_FALLBACK_STOPWORDS)
            self._lemmatizer = WordNetLemmatizer()
        else:
            self._stopwords = set(_FALLBACK_STOPWORDS)
            self._lemmatizer = None

    # -- Public API ----------------------------------------------------

    def process(self, text: str) -> ProcessedText:
        """Run the full processing pipeline on a block of text and return
        a ProcessedText result. Safe to call on empty/None text.
        """
        result = ProcessedText(original_text=text or "")

        if not text or not text.strip():
            return result

        result.cleaned_text = self.clean_text(text)
        result.sentences = self.tokenize_sentences(result.cleaned_text)
        result.tokens = self.tokenize_words(result.cleaned_text)
        result.tokens_no_stopwords = self.remove_stopwords(result.tokens)
        result.lemmatized_tokens = self.lemmatize(result.tokens_no_stopwords)
        result.word_count = len(result.tokens)

        result.sections = self.detect_sections(text)

        result.email = self.extract_email(text)
        result.phone = self.extract_phone(text)
        result.links = self.extract_links(text)

        result.entities = self.extract_entities(text)

        return result

    # -- Cleaning & tokenization ----------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize text for downstream NLP: lowercase-safe cleanup that
        preserves word boundaries, strips odd bullet/control characters,
        and collapses whitespace. Does NOT lowercase by default since
        section detection and NER benefit from original casing; call
        `.lower()` on the result where case-insensitivity is needed.
        """
        if not text:
            return ""

        # Remove common bullet/control characters left over from PDF/DOCX
        # extraction (•, ●, ▪, arrows, non-breaking spaces, etc.).
        text = re.sub(r"[•●▪◦‣∙→➤➔\uf0b7\u2022]", " ", text)
        text = text.replace("\xa0", " ").replace("\u200b", "")

        # Remove stray control characters but keep newlines/tabs for now.
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)

        # Collapse repeated whitespace (but keep single newlines as-is;
        # ProcessedText.sections relies on line structure elsewhere).
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def tokenize_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK if available, otherwise a
        regex-based fallback splitter.
        """
        if not text:
            return []

        if self.nltk_ready:
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception as exc:  # noqa: BLE001
                logger.warning("NLTK sentence tokenization failed (%s); using fallback.", exc)

        # Fallback: split on sentence-ending punctuation followed by
        # whitespace and a capital letter/newline.
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+", text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def tokenize_words(self, text: str) -> list[str]:
        """Split text into word tokens using NLTK if available, otherwise
        a regex fallback. Keeps alphanumeric tokens (drops pure
        punctuation) but preserves things like 'c++' / 'node.js' loosely
        via a permissive pattern.
        """
        if not text:
            return []

        if self.nltk_ready:
            try:
                tokens = word_tokenize(text)
                return [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]
            except Exception as exc:  # noqa: BLE001
                logger.warning("NLTK word tokenization failed (%s); using fallback.", exc)

        # Fallback regex tokenizer: words, numbers, and common tech tokens
        # like "c++", "c#", "node.js".
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9+.#\-]*", text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Filter out stopwords and single-character noise tokens."""
        return [
            t for t in tokens
            if t.lower() not in self._stopwords and len(t) > 1
        ]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens to their base form (e.g. 'managing' -> 'manage').
        Falls back to returning lowercased tokens unchanged if NLTK's
        WordNet lemmatizer isn't available.
        """
        if not tokens:
            return []

        if self._lemmatizer is not None:
            try:
                return [self._lemmatizer.lemmatize(t.lower()) for t in tokens]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lemmatization failed (%s); returning lowercased tokens.", exc)

        return [t.lower() for t in tokens]

    # -- Section detection -------------------------------------------------

    def detect_sections(self, text: str) -> dict[str, str]:
        """Split resume text into canonical sections (summary, experience,
        education, skills, projects, certifications, etc.) by scanning
        for header-like lines.

        A line is treated as a potential header if it's short (<= 6
        words), matches a known section phrase, and doesn't end in
        typical sentence punctuation. Content between one header and the
        next is attributed to that section.
        """
        if not text:
            return {}

        lines = text.split("\n")
        sections: dict[str, list[str]] = {}
        current_section: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            matched_section = self._match_section_header(stripped)
            if matched_section:
                current_section = matched_section
                sections.setdefault(current_section, [])
                continue

            if current_section:
                sections[current_section].append(stripped)
            # Lines before the first detected header are effectively the
            # resume "header" (name/contact block) and are intentionally
            # not captured as a section here.

        return {name: "\n".join(content).strip() for name, content in sections.items()}

    @staticmethod
    def _match_section_header(line: str) -> Optional[str]:
        """Check whether a line looks like a section header and return
        the canonical section name if so, else None.
        """
        # Header lines are short — real resume headers are rarely more
        # than a few words (e.g. "Technical Skills", "Work Experience").
        word_count = len(line.split())
        if word_count > 6:
            return None

        # Strip common trailing punctuation/decorations (colons, dashes,
        # underscores used as visual separators).
        normalized = re.sub(r"[:\-_=]+$", "", line).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)

        return _HEADER_TO_SECTION.get(normalized)

    # -- Contact info extraction --------------------------------------------

    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """Return the first email address found in the text, if any."""
        if not text:
            return None
        match = _EMAIL_PATTERN.search(text)
        return match.group(0) if match else None

    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """Return the first plausible phone number found in the text.
        Uses a permissive pattern since resumes format numbers wildly
        differently across regions; length-filters obvious false
        positives (e.g. matching only 4 stray digits).
        """
        if not text:
            return None
        for match in _PHONE_PATTERN.finditer(text):
            candidate = match.group(0)
            digit_count = len(re.sub(r"\D", "", candidate))
            if 7 <= digit_count <= 15:
                return candidate.strip()
        return None

    @staticmethod
    def extract_links(text: str) -> list[str]:
        """Extract URLs, prioritizing LinkedIn/GitHub profile links, then
        any other URLs found. De-duplicates while preserving order.
        """
        if not text:
            return []

        found: list[str] = []
        for pattern in (_LINKEDIN_PATTERN, _GITHUB_PATTERN, _URL_PATTERN):
            for match in pattern.finditer(text):
                url = match.group(0).rstrip(").,;")
                if url not in found:
                    found.append(url)
        return found

    # -- Named entity extraction --------------------------------------------

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities (organizations, people, dates, locations)
        using spaCy if a model is loaded; otherwise returns an empty dict
        rather than guessing, since regex-based NER for names/orgs is too
        unreliable to be worth faking.
        """
        entities: dict[str, list[str]] = {
            "ORG": [], "PERSON": [], "DATE": [], "GPE": [],
        }

        if not text or not self.spacy_ready:
            return entities

        try:
            # Cap input length passed to spaCy for performance on very
            # large documents (resumes are short; this is a safety net).
            doc = _SPACY_NLP(text[:20000])
            for ent in doc.ents:
                if ent.label_ in entities:
                    value = ent.text.strip()
                    if value and value not in entities[ent.label_]:
                        entities[ent.label_].append(value)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy entity extraction failed: %s", exc)

        return entities

    # -- Frequency / n-gram helpers (used later for keyword extraction) ----

    @staticmethod
    def word_frequency(tokens: list[str], top_n: Optional[int] = None) -> list[tuple[str, int]]:
        """Return (token, count) pairs sorted by descending frequency."""
        counts = Counter(t.lower() for t in tokens)
        most_common = counts.most_common(top_n)
        return most_common

    @staticmethod
    def get_ngrams(tokens: list[str], n: int = 2) -> list[str]:
        """Generate n-grams (e.g. bigrams, trigrams) from a token list.
        Useful for catching multi-word skills/keywords like 'machine
        learning' or 'project management'.
        """
        if n < 1 or len(tokens) < n:
            return []
        return [
            " ".join(tokens[i:i + n]).lower()
            for i in range(len(tokens) - n + 1)
        ]


# --- Convenience module-level function --------------------------------------

def process_text(text: str) -> ProcessedText:
    """Convenience wrapper for one-off processing without instantiating
    the class explicitly.
    """
    return TextProcessor().process(text)


if __name__ == "__main__":
    # Simple manual smoke test when run directly:
    #   python ml/text_processor.py
    sample = """
    John Doe
    john.doe@email.com | +1 555-123-4567 | linkedin.com/in/johndoe

    SUMMARY
    Software engineer with 5 years of experience in Python and machine learning.

    SKILLS
    Python, Java, Machine Learning, SQL, Docker

    EXPERIENCE
    Senior Developer at TechCorp (2020-2024)
    Built scalable data pipelines using Python and AWS.

    EDUCATION
    B.Tech in Computer Science, IIT Delhi, 2019
    """
    processed = process_text(sample)
    print("NLTK ready:", processed and TextProcessor().nltk_ready)
    print("spaCy ready:", TextProcessor().spacy_ready)
    print("Email:", processed.email)
    print("Phone:", processed.phone)
    print("Links:", processed.links)
    print("Sections found:", list(processed.sections.keys()))
    print("Entities:", processed.entities)
    print("Word count:", processed.word_count)