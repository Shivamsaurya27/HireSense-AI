"""
ml/skill_extractor.py
-----------------------
Skill Extraction & Keyword Analysis Module for HireSense AI.

Responsible for:
    - Loading a skills taxonomy (data/skills.json) covering technical,
      soft, and domain skills, grouped by category, with common aliases.
    - Extracting skills mentioned in resume / job-description text using
      phrase matching over the taxonomy (handles multi-word skills like
      "machine learning" and aliases like "JS" -> "JavaScript").
    - Extracting general keywords from text using TF-IDF (real ML
      technique — not hardcoded), for job-description keyword analysis.
    - Matching a candidate's extracted skills against a job description's
      required skills to compute matched / missing skills.

This module depends only on `text_processor.py` for tokenization/cleaning
and on scikit-learn for TF-IDF. It does not do final ATS scoring — see
`ats_scorer.py` for that.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ml.text_processor import TextProcessor
except ImportError:
    # Allows running this file standalone (python ml/skill_extractor.py)
    # as well as via package-relative imports (from ml import ...).
    from text_processor import TextProcessor  # type: ignore

logger = logging.getLogger("hiresense.skill_extractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# Default location of the skills taxonomy relative to the project root.
DEFAULT_SKILLS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"

# A small embedded fallback taxonomy so this module still works even if
# data/skills.json hasn't been created/found yet. `ats_scorer` and the UI
# should prefer the full data/skills.json (generated separately) for
# real use — this is just a safety net.
_FALLBACK_SKILLS_DB: dict = {
    "programming_languages": {
        "skills": [
            "python", "java", "javascript", "typescript", "c++", "c#",
            "c", "go", "rust", "php", "ruby", "kotlin", "swift", "r",
            "sql", "scala", "matlab",
        ],
        "aliases": {"js": "javascript", "ts": "typescript", "golang": "go"},
    },
    "web_development": {
        "skills": [
            "html", "css", "react", "angular", "vue.js", "node.js",
            "django", "flask", "fastapi", "spring boot", "express.js",
            "next.js", "bootstrap", "tailwind css", "rest api",
        ],
        "aliases": {"vuejs": "vue.js", "nodejs": "node.js", "reactjs": "react"},
    },
    "data_science_ml": {
        "skills": [
            "machine learning", "deep learning", "natural language processing",
            "computer vision", "data analysis", "data visualization",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
            "keras", "opencv", "nltk", "spacy", "statistics",
        ],
        "aliases": {"ml": "machine learning", "dl": "deep learning", "nlp": "natural language processing", "cv": "computer vision"},
    },
    "databases": {
        "skills": [
            "mysql", "postgresql", "mongodb", "sqlite", "oracle",
            "redis", "firebase", "cassandra", "dynamodb",
        ],
        "aliases": {"postgres": "postgresql"},
    },
    "cloud_devops": {
        "skills": [
            "aws", "azure", "google cloud platform", "docker",
            "kubernetes", "jenkins", "ci/cd", "terraform", "git",
            "github", "gitlab", "linux",
        ],
        "aliases": {"gcp": "google cloud platform", "k8s": "kubernetes"},
    },
    "soft_skills": {
        "skills": [
            "communication", "leadership", "teamwork", "problem solving",
            "critical thinking", "time management", "adaptability",
            "collaboration", "project management",
        ],
        "aliases": {},
    },
}


@dataclass
class ExtractedSkill:
    """A single skill found in a piece of text."""
    name: str            # Canonical skill name (e.g. "machine learning")
    category: str        # Taxonomy category (e.g. "data_science_ml")
    matched_text: str    # The literal text that triggered the match

    def __hash__(self) -> int:
        return hash((self.name, self.category))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedSkill):
            return NotImplemented
        return self.name == other.name and self.category == other.category


@dataclass
class SkillExtractionResult:
    """Result of extracting skills from a document."""
    skills: list[ExtractedSkill] = field(default_factory=list)
    skills_by_category: dict[str, list[str]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)

    @property
    def skill_names(self) -> list[str]:
        return [s.name for s in self.skills]

    def to_dict(self) -> dict:
        return {
            "skills": self.skill_names,
            "skills_by_category": self.skills_by_category,
            "keywords": self.keywords,
        }


@dataclass
class SkillMatchResult:
    """Result of comparing candidate skills against required job skills."""
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)  # candidate has, JD doesn't ask for
    match_percentage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extra_skills": self.extra_skills,
            "match_percentage": round(self.match_percentage, 2),
        }


class SkillExtractor:
    """Extracts and matches skills using a JSON-defined skills taxonomy,
    and extracts general keywords using TF-IDF.

    Usage:
        extractor = SkillExtractor()  # loads data/skills.json if present
        result = extractor.extract_skills(resume_text)
        match = extractor.match_skills(candidate_skills, required_skills)
    """

    def __init__(self, skills_db_path: Optional[str] = None) -> None:
        self.skills_db_path = Path(skills_db_path) if skills_db_path else DEFAULT_SKILLS_DB_PATH
        self.skills_db: dict = self._load_skills_db()
        self.text_processor = TextProcessor()

        # Build a flat lookup: normalized skill/alias phrase -> (canonical_name, category)
        # and a set of canonical skill names sorted longest-first so that
        # multi-word skills are matched before their shorter substrings
        # (e.g. "machine learning" before "learning").
        self._phrase_to_skill: dict[str, tuple[str, str]] = {}
        self._build_lookup_table()

        # Precompute regex patterns per phrase for whole-word/phrase
        # matching (avoids matching "java" inside "javascript").
        self._compiled_patterns: dict[str, re.Pattern] = {
            phrase: re.compile(r"(?<![a-zA-Z0-9])" + re.escape(phrase) + r"(?![a-zA-Z0-9])")
            for phrase in self._phrase_to_skill
        }

    # -- Setup --------------------------------------------------------------

    def _load_skills_db(self) -> dict:
        """Load the skills taxonomy from JSON, falling back to a built-in
        default set if the file is missing or malformed.
        """
        if self.skills_db_path.exists():
            try:
                with open(self.skills_db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
                logger.warning(
                    "%s did not contain a valid skills taxonomy; using fallback.",
                    self.skills_db_path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load skills DB from %s (%s); using fallback.",
                    self.skills_db_path, exc,
                )
        else:
            logger.info(
                "Skills DB not found at %s; using built-in fallback taxonomy. "
                "Provide data/skills.json for the full taxonomy.",
                self.skills_db_path,
            )
        return _FALLBACK_SKILLS_DB

    def _build_lookup_table(self) -> None:
        """Flatten the (possibly nested category -> {skills, aliases})
        taxonomy into a single phrase -> (canonical_name, category) map.
        """
        for category, payload in self.skills_db.items():
            if not isinstance(payload, dict):
                continue
            skill_list = payload.get("skills", [])
            aliases = payload.get("aliases", {})

            for skill in skill_list:
                normalized = skill.lower().strip()
                if normalized:
                    self._phrase_to_skill[normalized] = (normalized, category)

            for alias, canonical in aliases.items():
                normalized_alias = alias.lower().strip()
                normalized_canonical = canonical.lower().strip()
                if normalized_alias and normalized_canonical:
                    self._phrase_to_skill[normalized_alias] = (normalized_canonical, category)

    # -- Skill extraction -----------------------------------------------

    def extract_skills(self, text: str) -> SkillExtractionResult:
        """Find all taxonomy skills (and their aliases) mentioned in the
        given text using word-boundary phrase matching. Returns skills
        deduplicated by canonical name, grouped by category, plus a set
        of general TF-IDF keywords.
        """
        result = SkillExtractionResult()
        if not text or not text.strip():
            return result

        normalized_text = self._normalize_for_matching(text)

        found: dict[str, ExtractedSkill] = {}
        for phrase, pattern in self._compiled_patterns.items():
            match = pattern.search(normalized_text)
            if match:
                canonical_name, category = self._phrase_to_skill[phrase]
                # Prefer the first/longest match found for a given
                # canonical skill; don't overwrite once found.
                if canonical_name not in found:
                    found[canonical_name] = ExtractedSkill(
                        name=canonical_name,
                        category=category,
                        matched_text=match.group(0),
                    )

        result.skills = sorted(found.values(), key=lambda s: (s.category, s.name))

        skills_by_category: dict[str, list[str]] = {}
        for skill in result.skills:
            skills_by_category.setdefault(skill.category, []).append(skill.name)
        result.skills_by_category = skills_by_category

        result.keywords = self.extract_keywords(text)

        return result

    @staticmethod
    def _normalize_for_matching(text: str) -> str:
        """Lowercase and lightly normalize text for reliable phrase
        matching (e.g. so 'Node.JS' matches 'node.js').
        """
        text = text.lower()
        # Collapse whitespace so multi-word phrases match across line
        # breaks (e.g. a skill listed with a line break before a comma).
        text = re.sub(r"\s+", " ", text)
        return text

    # -- Keyword extraction (TF-IDF) -----------------------------------

    def extract_keywords(
        self,
        text: str,
        top_n: int = 20,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> list[str]:
        """Extract the most important keywords/phrases from a single
        document using TF-IDF term weighting. Since TF-IDF traditionally
        needs a corpus, a single document is scored against itself by
        treating each sentence as a "sub-document" — this lets TF-IDF's
        inverse-document-frequency component meaningfully downweight
        terms that appear in almost every sentence (e.g. filler words)
        versus ones concentrated in specific, information-dense
        sentences.
        """
        if not text or not text.strip():
            return []

        processed = self.text_processor.process(text)
        sentences = processed.sentences if processed.sentences else [text]

        # Need at least 2 "documents" for meaningful IDF variation; if
        # there's only one sentence, fall back to simple term frequency.
        if len(sentences) < 2:
            return self._top_terms_by_frequency(text, top_n)

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=ngram_range,
                max_features=500,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]*\b",
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            # Aggregate TF-IDF scores across all sentences (sum per term)
            # to get a single document-level importance ranking.
            scores = tfidf_matrix.sum(axis=0).A1
            term_scores = list(zip(feature_names, scores))
            term_scores.sort(key=lambda pair: pair[1], reverse=True)

            return [term for term, score in term_scores[:top_n] if score > 0]
        except ValueError as exc:
            # Can happen if the vocabulary is empty after stopword
            # removal (e.g. very short or non-English text).
            logger.warning("TF-IDF keyword extraction failed (%s); using frequency fallback.", exc)
            return self._top_terms_by_frequency(text, top_n)

    def _top_terms_by_frequency(self, text: str, top_n: int) -> list[str]:
        """Simple frequency-based keyword fallback for edge cases where
        TF-IDF can't run (e.g. single-sentence input).
        """
        processed = self.text_processor.process(text)
        freq = self.text_processor.word_frequency(processed.lemmatized_tokens, top_n=top_n)
        return [term for term, _count in freq]

    # -- Skill matching (candidate vs job description) -------------------

    def match_skills(
        self,
        candidate_skills: list[str],
        required_skills: list[str],
    ) -> SkillMatchResult:
        """Compare a candidate's extracted skills against a job's required
        skills. All comparisons are done on normalized (lowercased,
        trimmed) skill names so matching is robust to casing differences.
        """
        result = SkillMatchResult()

        normalized_candidate = {s.lower().strip() for s in candidate_skills if s.strip()}
        normalized_required = {s.lower().strip() for s in required_skills if s.strip()}

        if not normalized_required:
            # No requirements specified — nothing to match against.
            result.extra_skills = sorted(normalized_candidate)
            return result

        matched = normalized_candidate & normalized_required
        missing = normalized_required - normalized_candidate
        extra = normalized_candidate - normalized_required

        result.matched_skills = sorted(matched)
        result.missing_skills = sorted(missing)
        result.extra_skills = sorted(extra)
        result.match_percentage = (
            (len(matched) / len(normalized_required)) * 100
            if normalized_required else 0.0
        )

        return result

    def extract_and_match(self, resume_text: str, job_description_text: str) -> dict:
        """Convenience method: extract skills from both resume and job
        description, then compute the match in one call. Returns a
        combined dict useful for the ATS scorer / report generator.
        """
        resume_skills_result = self.extract_skills(resume_text)
        jd_skills_result = self.extract_skills(job_description_text)

        match_result = self.match_skills(
            candidate_skills=resume_skills_result.skill_names,
            required_skills=jd_skills_result.skill_names,
        )

        return {
            "resume_skills": resume_skills_result.to_dict(),
            "job_description_skills": jd_skills_result.to_dict(),
            "match": match_result.to_dict(),
        }


# --- Convenience module-level functions ---------------------------------

def extract_skills(text: str, skills_db_path: Optional[str] = None) -> SkillExtractionResult:
    """Convenience wrapper for one-off skill extraction."""
    return SkillExtractor(skills_db_path=skills_db_path).extract_skills(text)


if __name__ == "__main__":
    # Simple manual smoke test when run directly:
    #   python ml/skill_extractor.py
    sample_resume = """
    Experienced software engineer skilled in Python, machine learning,
    and deep learning. Built REST APIs using Django and Flask. Worked
    with pandas, numpy, and scikit-learn for data analysis. Familiar
    with Docker, AWS, and Git. Strong communication and teamwork skills.
    """
    sample_jd = """
    We are looking for a candidate with strong Python and SQL skills,
    experience in machine learning, TensorFlow, React, and AWS. Good
    communication and leadership abilities are a plus.
    """

    extractor = SkillExtractor()
    combined = extractor.extract_and_match(sample_resume, sample_jd)

    print("Resume skills:", combined["resume_skills"]["skills"])
    print("JD skills:", combined["job_description_skills"]["skills"])
    print("Matched:", combined["match"]["matched_skills"])
    print("Missing:", combined["match"]["missing_skills"])
    print("Match %:", combined["match"]["match_percentage"])
    print("JD keywords (TF-IDF):", combined["job_description_skills"]["keywords"][:10])
    