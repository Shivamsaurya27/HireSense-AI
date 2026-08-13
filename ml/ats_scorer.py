"""
ml/ats_scorer.py
------------------
ATS-Style Compatibility Scoring Module for HireSense AI.

IMPORTANT / DISCLAIMER
-----------------------
This module produces a *project-generated compatibility score*, not an
official or industry-standard ATS (Applicant Tracking System) algorithm.
Real commercial ATS platforms (Workday, Taleo, Greenhouse, etc.) use
proprietary, undisclosed scoring logic. This score is a transparent,
explainable, rule-based + NLP approximation built for demonstration and
educational purposes as part of a college project. Every component and
weight is disclosed and configurable — see `ATSWeights` below.

Scoring components (default weights, all configurable):
    Skills Match              = 30%
    Job Description Similarity = 25%   (TF-IDF + Cosine Similarity)
    Experience                = 20%
    Education                 = 10%
    Projects                  = 10%
    Certifications            = 5%

Total = 100 (weights must sum to 100; validated at construction time).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ml.text_processor import TextProcessor
    from ml.skill_extractor import SkillExtractor
except ImportError:
    # Allow standalone execution (python ml/ats_scorer.py).
    from text_processor import TextProcessor  # type: ignore
    from skill_extractor import SkillExtractor  # type: ignore

logger = logging.getLogger("hiresense.ats_scorer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


DISCLAIMER = (
    "This is a project-generated compatibility score created for "
    "educational purposes. It is NOT an official ATS algorithm and does "
    "not reflect how any real Applicant Tracking System evaluates "
    "candidates. Weights and methodology are fully transparent and "
    "configurable — see ATSWeights."
)


# --- Configurable scoring weights -----------------------------------------

@dataclass
class ATSWeights:
    """Weights for each scoring component. Must sum to 100.

    Adjust these to experiment with different scoring philosophies
    (e.g. weight skills more heavily for technical roles, or education
    more heavily for research roles).
    """
    skills: float = 30.0
    similarity: float = 25.0
    experience: float = 20.0
    education: float = 10.0
    projects: float = 10.0
    certifications: float = 5.0

    def validate(self) -> None:
        total = (
            self.skills + self.similarity + self.experience
            + self.education + self.projects + self.certifications
        )
        if not abs(total - 100.0) < 0.01:
            raise ValueError(
                f"ATSWeights must sum to 100 (got {total}). "
                f"Current values: {self.to_dict()}"
            )

    def to_dict(self) -> dict:
        return {
            "skills": self.skills,
            "similarity": self.similarity,
            "experience": self.experience,
            "education": self.education,
            "projects": self.projects,
            "certifications": self.certifications,
        }


# --- Education level reference scale ---------------------------------------
# Ordered lowest -> highest. Used as a simple proxy scale, not a judgment
# of a person's worth — purely "how far along a common degree ladder".
_EDUCATION_LEVELS: list[tuple[str, list[str], float]] = [
    ("phd", ["phd", "ph.d", "doctorate", "doctoral"], 100.0),
    ("master", ["master", "m.tech", "mtech", "msc", "m.sc", "mba", "m.s.", " ms ", "post graduate", "postgraduate"], 85.0),
    ("bachelor", ["bachelor", "b.tech", "btech", "bsc", "b.sc", "b.e.", " be ", "b.a.", "undergraduate degree"], 70.0),
    ("diploma", ["diploma", "associate degree", "associate's degree"], 50.0),
    ("high_school", ["high school", "senior secondary", "12th grade", "higher secondary"], 30.0),
]

# Common certification signal keywords (used in addition to a detected
# "certifications" section) so certs mentioned inline are still counted.
_CERTIFICATION_KEYWORDS = [
    "certified", "certification", "certificate in", "licensed",
    "aws certified", "microsoft certified", "google certified",
    "pmp", "scrum master", "comptia", "cisco certified",
]


@dataclass
class ScoreComponent:
    """A single scoring component's raw score, weight, and contribution."""
    name: str
    raw_score: float          # 0-100, before weighting
    weight_percent: float     # e.g. 30.0
    weighted_contribution: float  # raw_score * weight_percent / 100
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "raw_score": round(self.raw_score, 2),
            "weight_percent": self.weight_percent,
            "weighted_contribution": round(self.weighted_contribution, 2),
            "details": self.details,
        }


@dataclass
class ATSScoreResult:
    """Full breakdown of an ATS compatibility score calculation."""
    total_score: float = 0.0
    components: dict[str, ScoreComponent] = field(default_factory=dict)
    weights_used: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 2),
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "weights_used": self.weights_used,
            "disclaimer": self.disclaimer,
        }


class ATSScorer:
    """Computes a transparent, configurable, project-generated ATS
    compatibility score between a resume and a job description.

    Usage:
        scorer = ATSScorer()  # default weights
        result = scorer.calculate_ats_score(resume_text, job_description_text)
        print(result.total_score, result.components)
    """

    def __init__(
        self,
        weights: Optional[ATSWeights] = None,
        skills_db_path: Optional[str] = None,
    ) -> None:
        self.weights = weights or ATSWeights()
        self.weights.validate()

        self.text_processor = TextProcessor()
        self.skill_extractor = SkillExtractor(skills_db_path=skills_db_path)

    # -- Main entry point -------------------------------------------------

    def calculate_ats_score(
        self,
        resume_text: str,
        job_description_text: str,
    ) -> ATSScoreResult:
        """Compute the full weighted ATS compatibility score.

        Each component is scored 0-100 independently, then combined
        according to `self.weights`. Handles empty/short input
        gracefully rather than raising.
        """
        result = ATSScoreResult(weights_used=self.weights.to_dict())

        resume_text = resume_text or ""
        job_description_text = job_description_text or ""

        if not resume_text.strip():
            logger.warning("Empty resume text provided to ATS scorer.")
        if not job_description_text.strip():
            logger.warning("Empty job description text provided to ATS scorer.")

        # --- Component 1: Skills Match ---
        skills_score, skills_details = self._score_skills(resume_text, job_description_text)
        result.components["skills"] = self._make_component(
            "skills", skills_score, self.weights.skills, skills_details
        )

        # --- Component 2: Job Description Similarity (TF-IDF + Cosine) ---
        similarity_score, similarity_details = self._score_similarity(
            resume_text, job_description_text
        )
        result.components["similarity"] = self._make_component(
            "similarity", similarity_score, self.weights.similarity, similarity_details
        )

        # --- Component 3: Experience ---
        experience_score, experience_details = self._score_experience(resume_text)
        result.components["experience"] = self._make_component(
            "experience", experience_score, self.weights.experience, experience_details
        )

        # --- Component 4: Education ---
        education_score, education_details = self._score_education(resume_text)
        result.components["education"] = self._make_component(
            "education", education_score, self.weights.education, education_details
        )

        # --- Component 5: Projects ---
        projects_score, projects_details = self._score_projects(resume_text)
        result.components["projects"] = self._make_component(
            "projects", projects_score, self.weights.projects, projects_details
        )

        # --- Component 6: Certifications ---
        certifications_score, certifications_details = self._score_certifications(resume_text)
        result.components["certifications"] = self._make_component(
            "certifications", certifications_score, self.weights.certifications, certifications_details
        )

        result.total_score = round(
            sum(c.weighted_contribution for c in result.components.values()), 2
        )
        # Clamp for safety (should already be 0-100 given each raw score is 0-100).
        result.total_score = max(0.0, min(100.0, result.total_score))

        return result

    @staticmethod
    def _make_component(
        name: str, raw_score: float, weight: float, details: dict
    ) -> ScoreComponent:
        raw_score = max(0.0, min(100.0, raw_score))
        return ScoreComponent(
            name=name,
            raw_score=raw_score,
            weight_percent=weight,
            weighted_contribution=raw_score * weight / 100.0,
            details=details,
        )

    # -- Component 1: Skills Match --------------------------------------

    def _score_skills(self, resume_text: str, jd_text: str) -> tuple[float, dict]:
        """Score = percentage of JD-required skills found in the resume
        (via `SkillExtractor.match_skills`), using real taxonomy-based
        skill extraction rather than naive keyword overlap.
        """
        combined = self.skill_extractor.extract_and_match(resume_text, jd_text)
        match = combined["match"]
        score = match["match_percentage"]
        details = {
            "matched_skills": match["matched_skills"],
            "missing_skills": match["missing_skills"],
            "resume_skill_count": len(combined["resume_skills"]["skills"]),
            "required_skill_count": len(combined["job_description_skills"]["skills"]),
        }
        return score, details

    # -- Component 2: JD Similarity (TF-IDF + Cosine Similarity) --------

    def _score_similarity(self, resume_text: str, jd_text: str) -> tuple[float, dict]:
        """Real ML technique: vectorize the resume and job description
        with TF-IDF, then compute cosine similarity between the two
        vectors. Score = similarity * 100.
        """
        if not resume_text.strip() or not jd_text.strip():
            return 0.0, {"note": "One or both documents were empty."}

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=1000,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]*\b",
            )
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score = float(similarity) * 100.0
            return score, {"cosine_similarity": round(float(similarity), 4)}
        except ValueError as exc:
            # Empty vocabulary after stopword removal, extremely short docs, etc.
            logger.warning("TF-IDF similarity scoring failed: %s", exc)
            return 0.0, {"note": f"Similarity could not be computed: {exc}"}

    # -- Component 3: Experience -------------------------------------------

    def _score_experience(self, resume_text: str) -> tuple[float, dict]:
        """Estimate years of professional experience two ways and take
        the stronger signal:
          (a) explicit phrases like "5 years of experience"
          (b) summing date ranges found in the detected "experience"
              section (e.g. "2020 - 2024", "Jan 2021 - Present")

        Score is then scaled against a configurable "max useful years"
        cap (10 years = 100 score) since more years shows diminishing
        differentiation for most entry/mid-level roles typical of a
        college project's candidate pool.
        """
        MAX_YEARS_FOR_FULL_SCORE = 10.0

        explicit_years = self._extract_explicit_years(resume_text)
        section_years = self._estimate_years_from_date_ranges(resume_text)

        estimated_years = max(explicit_years, section_years)
        score = min(100.0, (estimated_years / MAX_YEARS_FOR_FULL_SCORE) * 100.0)

        details = {
            "estimated_years": round(estimated_years, 1),
            "explicit_years_mentioned": round(explicit_years, 1),
            "years_from_date_ranges": round(section_years, 1),
        }
        return score, details

    @staticmethod
    def _extract_explicit_years(text: str) -> float:
        """Regex for phrases like '5 years of experience', '3+ yrs', etc.
        Returns the maximum such figure found (a resume may mention
        experience multiple times; the largest is the most representative).
        """
        if not text:
            return 0.0
        pattern = re.compile(
            r"(\d{1,2}(?:\.\d)?)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
            re.IGNORECASE,
        )
        matches = pattern.findall(text)
        if not matches:
            return 0.0
        try:
            return max(float(m) for m in matches)
        except ValueError:
            return 0.0

    def _estimate_years_from_date_ranges(self, text: str) -> float:
        """Detect the resume's 'experience' section and sum the spans of
        date ranges found within it (e.g. "2019 - 2022", "Mar 2020 -
        Present"). This complements explicit year mentions since many
        resumes list roles by date range without stating a total.
        """
        sections = self.text_processor.detect_sections(text)
        experience_text = sections.get("experience", "")
        if not experience_text:
            experience_text = text  # fall back to scanning the whole resume

        current_year = datetime.now().year

        # Matches things like "2019 - 2022", "2019-Present", "Jan 2020 – Dec 2021"
        range_pattern = re.compile(
            r"(?:[A-Za-z]{3,9}\s+)?(\d{4})\s*(?:-|–|—|to)\s*"
            r"(?:[A-Za-z]{3,9}\s+)?(\d{4}|present|current)",
            re.IGNORECASE,
        )

        total_months = 0
        for start_str, end_str in range_pattern.findall(experience_text):
            try:
                start_year = int(start_str)
            except ValueError:
                continue

            if end_str.lower() in ("present", "current"):
                end_year = current_year
            else:
                try:
                    end_year = int(end_str)
                except ValueError:
                    continue

            if end_year < start_year or (end_year - start_year) > 50:
                continue  # discard nonsensical/garbled ranges

            total_months += (end_year - start_year) * 12

        return round(total_months / 12.0, 1)

    # -- Component 4: Education --------------------------------------------

    def _score_education(self, resume_text: str) -> tuple[float, dict]:
        """Detect the highest education level mentioned using keyword
        matching against a common degree-name reference scale. This is a
        simple proxy, not a judgment of qualification quality.
        """
        if not resume_text:
            return 0.0, {"detected_level": None}

        lower_text = f" {resume_text.lower()} "

        highest_score = 0.0
        highest_level = None
        for level_name, keywords, level_score in _EDUCATION_LEVELS:
            if any(keyword in lower_text for keyword in keywords):
                if level_score > highest_score:
                    highest_score = level_score
                    highest_level = level_name

        details = {"detected_level": highest_level}
        return highest_score, details

    # -- Component 5: Projects ---------------------------------------------

    def _score_projects(self, resume_text: str) -> tuple[float, dict]:
        """Score based on the number of distinct project entries detected
        in the resume's "projects" section. Falls back to scanning the
        whole document for lines starting with typical project-entry
        markers (bullets, dashes) if no dedicated section is found.

        Scaling: 4+ distinct projects = full score (reasonable ceiling
        for a student/early-career resume).
        """
        MAX_PROJECTS_FOR_FULL_SCORE = 4

        sections = self.text_processor.detect_sections(resume_text)
        projects_text = sections.get("projects", "")

        if projects_text:
            entry_count = self._count_entries(projects_text)
        else:
            # No dedicated section — do a conservative whole-document
            # scan for the word "project" as a weak fallback signal.
            entry_count = len(re.findall(r"\bproject\b", resume_text, re.IGNORECASE))
            entry_count = min(entry_count, MAX_PROJECTS_FOR_FULL_SCORE)

        score = min(100.0, (entry_count / MAX_PROJECTS_FOR_FULL_SCORE) * 100.0)
        details = {"detected_project_entries": entry_count}
        return score, details

    # -- Component 6: Certifications ----------------------------------------

    def _score_certifications(self, resume_text: str) -> tuple[float, dict]:
        """Score based on the number of certifications detected, combining
        (a) entries in a dedicated "certifications" section and
        (b) known certification keyword mentions elsewhere in the resume.

        Scaling: 3+ certifications = full score.
        """
        MAX_CERTS_FOR_FULL_SCORE = 3

        if not resume_text:
            return 0.0, {"detected_certifications": 0}

        sections = self.text_processor.detect_sections(resume_text)
        cert_section_text = sections.get("certifications", "")

        section_entry_count = self._count_entries(cert_section_text) if cert_section_text else 0

        lower_text = resume_text.lower()
        keyword_hits = sum(1 for kw in _CERTIFICATION_KEYWORDS if kw in lower_text)

        # Combine signals conservatively: use the section entry count if
        # a dedicated section exists (most reliable), otherwise fall
        # back to keyword hit count.
        detected_count = section_entry_count if cert_section_text else keyword_hits

        score = min(100.0, (detected_count / MAX_CERTS_FOR_FULL_SCORE) * 100.0)
        details = {"detected_certifications": detected_count}
        return score, details

    # -- Shared helpers -------------------------------------------------

    @staticmethod
    def _count_entries(section_text: str) -> int:
        """Count distinct entries in a resume section by counting
        non-empty lines. This is a simple, transparent heuristic —
        each resume line within a section (bullet point or title line)
        is treated as one entry.
        """
        if not section_text:
            return 0
        lines = [line for line in section_text.split("\n") if line.strip()]
        return len(lines)


# --- Convenience module-level function ---------------------------------

def calculate_ats_score(
    resume_text: str,
    job_description_text: str,
    weights: Optional[ATSWeights] = None,
) -> ATSScoreResult:
    """Convenience wrapper for one-off ATS scoring without instantiating
    the class explicitly.
    """
    return ATSScorer(weights=weights).calculate_ats_score(resume_text, job_description_text)


if __name__ == "__main__":
    # Simple manual smoke test when run directly:
    #   python ml/ats_scorer.py
    sample_resume = """
    John Doe
    Software Engineer with 4 years of experience.

    EXPERIENCE
    Software Engineer at TechCorp (2021 - Present)
    Backend Developer at StartUpX (2019 - 2021)

    EDUCATION
    B.Tech in Computer Science, IIT Delhi, 2019

    SKILLS
    Python, Machine Learning, Django, AWS, Docker, SQL

    PROJECTS
    Resume Screening AI - built an NLP pipeline using Python and spaCy
    Chatbot Assistant - built a chatbot using NLTK and Flask
    Stock Price Predictor - regression model using scikit-learn

    CERTIFICATIONS
    AWS Certified Cloud Practitioner
    """

    sample_jd = """
    Looking for a Software Engineer with 3+ years of experience in
    Python, Machine Learning, Django, and AWS. Bachelor's degree in
    Computer Science required. Strong problem solving skills.
    """

    scorer = ATSScorer()
    score_result = scorer.calculate_ats_score(sample_resume, sample_jd)

    print(f"TOTAL ATS COMPATIBILITY SCORE: {score_result.total_score}/100")
    print(f"Disclaimer: {score_result.disclaimer}\n")
    for component_name, component in score_result.components.items():
        print(
            f"- {component_name}: raw={component.raw_score:.1f} "
            f"weight={component.weight_percent}% "
            f"contribution={component.weighted_contribution:.2f}"
        )
        print(f"  details: {component.details}")