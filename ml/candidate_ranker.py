"""
ml/candidate_ranker.py
------------------------
Candidate Ranking & Comparison Module for HireSense AI.

Takes a pool of candidate resumes and a single job description, scores
each candidate using `ats_scorer.py`, and produces:
    - a ranked list of candidates (highest compatibility first)
    - a recommendation tier per candidate (Strong / Good / Moderate / Weak)
    - head-to-head comparison between any two candidates

This module does NOT do resume parsing or text cleaning itself — callers
are expected to supply already-extracted resume text (e.g. from
`resume_parser.py`). It orchestrates `ats_scorer.py` across multiple
candidates and adds ranking/comparison logic on top.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    from ml.ats_scorer import ATSScorer, ATSScoreResult, ATSWeights
except ImportError:
    # Allow standalone execution (python ml/candidate_ranker.py).
    from ats_scorer import ATSScorer, ATSScoreResult, ATSWeights  # type: ignore

logger = logging.getLogger("hiresense.candidate_ranker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# Score thresholds (out of 100) used to assign a human-readable
# recommendation tier. Configurable via `RankingConfig` below so a user
# of this module can tune sensitivity without editing code.
DEFAULT_TIER_THRESHOLDS: dict[str, float] = {
    "Strong Match": 75.0,
    "Good Match": 55.0,
    "Moderate Match": 35.0,
    "Weak Match": 0.0,
}


@dataclass
class CandidateInput:
    """A single candidate to be scored/ranked."""
    candidate_id: str
    name: str
    resume_text: str
    metadata: dict = field(default_factory=dict)  # e.g. {"source_file": "john.pdf"}


@dataclass
class RankedCandidate:
    """A candidate after scoring, with rank and recommendation tier."""
    candidate_id: str
    name: str
    rank: int
    score_result: ATSScoreResult
    tier: str
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None  # populated if scoring failed for this candidate

    @property
    def total_score(self) -> float:
        return self.score_result.total_score if self.score_result else 0.0

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "rank": self.rank,
            "total_score": round(self.total_score, 2),
            "tier": self.tier,
            "score_breakdown": self.score_result.to_dict() if self.score_result else None,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class ComparisonResult:
    """Head-to-head comparison of two candidates against the same job."""
    candidate_a: RankedCandidate
    candidate_b: RankedCandidate
    winner_id: Optional[str]  # None if tied
    score_difference: float
    common_matched_skills: list[str] = field(default_factory=list)
    unique_to_a: list[str] = field(default_factory=list)
    unique_to_b: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_a": self.candidate_a.to_dict(),
            "candidate_b": self.candidate_b.to_dict(),
            "winner_id": self.winner_id,
            "score_difference": round(self.score_difference, 2),
            "common_matched_skills": self.common_matched_skills,
            "unique_to_a": self.unique_to_a,
            "unique_to_b": self.unique_to_b,
        }


class CandidateRanker:
    """Scores and ranks multiple candidates against a job description,
    and supports direct candidate-to-candidate comparison.

    Usage:
        ranker = CandidateRanker()
        candidates = [
            CandidateInput(candidate_id="1", name="Alice", resume_text="..."),
            CandidateInput(candidate_id="2", name="Bob", resume_text="..."),
        ]
        ranked = ranker.rank_candidates(candidates, job_description_text)
        top_5 = ranker.get_top_candidates(ranked, top_n=5)
    """

    def __init__(
        self,
        weights: Optional[ATSWeights] = None,
        tier_thresholds: Optional[dict[str, float]] = None,
        skills_db_path: Optional[str] = None,
    ) -> None:
        self.scorer = ATSScorer(weights=weights, skills_db_path=skills_db_path)
        self.tier_thresholds = tier_thresholds or dict(DEFAULT_TIER_THRESHOLDS)
        self._validate_tier_thresholds()

    def _validate_tier_thresholds(self) -> None:
        """Ensure thresholds are sorted descending so tier assignment
        logic (first match wins) behaves correctly.
        """
        values = list(self.tier_thresholds.values())
        if values != sorted(values, reverse=True):
            raise ValueError(
                "tier_thresholds values must be in descending order, "
                f"got: {self.tier_thresholds}"
            )

    # -- Ranking ----------------------------------------------------------

    def rank_candidates(
        self,
        candidates: list[CandidateInput],
        job_description_text: str,
    ) -> list[RankedCandidate]:
        """Score every candidate against the job description and return
        them sorted by total score (descending). A candidate whose
        scoring fails (e.g. empty resume text) is still included in the
        output with `error` populated and ranked last, rather than
        silently dropped or crashing the whole batch.
        """
        if not candidates:
            return []

        scored: list[RankedCandidate] = []

        for candidate in candidates:
            try:
                if not candidate.resume_text or not candidate.resume_text.strip():
                    raise ValueError("Resume text is empty.")

                score_result = self.scorer.calculate_ats_score(
                    resume_text=candidate.resume_text,
                    job_description_text=job_description_text,
                )
                tier = self._assign_tier(score_result.total_score)

                scored.append(
                    RankedCandidate(
                        candidate_id=candidate.candidate_id,
                        name=candidate.name,
                        rank=0,  # assigned after sorting
                        score_result=score_result,
                        tier=tier,
                        metadata=candidate.metadata,
                        error=None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to score candidate %s (%s): %s",
                    candidate.candidate_id, candidate.name, exc,
                )
                empty_result = ATSScoreResult(total_score=0.0)
                scored.append(
                    RankedCandidate(
                        candidate_id=candidate.candidate_id,
                        name=candidate.name,
                        rank=0,
                        score_result=empty_result,
                        tier="Unscored",
                        metadata=candidate.metadata,
                        error=str(exc),
                    )
                )

        # Sort: highest total score first; on ties, prefer higher skills
        # match, then higher similarity, as tie-breakers (both are
        # available on the score breakdown).
        def sort_key(rc: RankedCandidate) -> tuple[float, float, float]:
            if rc.error or not rc.score_result:
                return (-1.0, -1.0, -1.0)
            skills_raw = rc.score_result.components.get("skills")
            similarity_raw = rc.score_result.components.get("similarity")
            return (
                rc.total_score,
                skills_raw.raw_score if skills_raw else 0.0,
                similarity_raw.raw_score if similarity_raw else 0.0,
            )

        scored.sort(key=sort_key, reverse=True)

        for index, ranked_candidate in enumerate(scored, start=1):
            ranked_candidate.rank = index

        return scored

    def _assign_tier(self, total_score: float) -> str:
        """Map a total score to a recommendation tier using the
        configured (descending) thresholds — first threshold the score
        meets or exceeds wins.
        """
        for tier_name, threshold in self.tier_thresholds.items():
            if total_score >= threshold:
                return tier_name
        return "Weak Match"  # safety net if thresholds don't cover 0

    # -- Recommendations ----------------------------------------------------

    @staticmethod
    def get_top_candidates(
        ranked_candidates: list[RankedCandidate],
        top_n: int = 5,
        min_score: float = 0.0,
    ) -> list[RankedCandidate]:
        """Return the top N ranked candidates, optionally filtering out
        anyone below `min_score`. Candidates with scoring errors are
        always excluded from recommendations.
        """
        eligible = [
            rc for rc in ranked_candidates
            if rc.error is None and rc.total_score >= min_score
        ]
        return eligible[:top_n]

    @staticmethod
    def filter_by_tier(
        ranked_candidates: list[RankedCandidate], tier: str
    ) -> list[RankedCandidate]:
        """Return all candidates matching a specific recommendation tier
        (e.g. 'Strong Match').
        """
        return [rc for rc in ranked_candidates if rc.tier == tier]

    # -- Comparison -----------------------------------------------------

    def compare_candidates(
        self,
        candidate_a: CandidateInput,
        candidate_b: CandidateInput,
        job_description_text: str,
    ) -> ComparisonResult:
        """Score two candidates against the same job description and
        produce a direct head-to-head comparison, including which
        matched skills they share versus which are unique to each.
        """
        ranked = self.rank_candidates([candidate_a, candidate_b], job_description_text)

        # Preserve original A/B identity regardless of ranking order.
        ranked_by_id = {rc.candidate_id: rc for rc in ranked}
        result_a = ranked_by_id[candidate_a.candidate_id]
        result_b = ranked_by_id[candidate_b.candidate_id]

        matched_a = set(
            result_a.score_result.components.get("skills").details.get("matched_skills", [])
            if not result_a.error else []
        )
        matched_b = set(
            result_b.score_result.components.get("skills").details.get("matched_skills", [])
            if not result_b.error else []
        )

        common = sorted(matched_a & matched_b)
        unique_to_a = sorted(matched_a - matched_b)
        unique_to_b = sorted(matched_b - matched_a)

        score_diff = round(result_a.total_score - result_b.total_score, 2)
        if score_diff > 0:
            winner_id = candidate_a.candidate_id
        elif score_diff < 0:
            winner_id = candidate_b.candidate_id
        else:
            winner_id = None

        return ComparisonResult(
            candidate_a=result_a,
            candidate_b=result_b,
            winner_id=winner_id,
            score_difference=abs(score_diff),
            common_matched_skills=common,
            unique_to_a=unique_to_a,
            unique_to_b=unique_to_b,
        )

    # -- Batch summary -------------------------------------------------

    @staticmethod
    def summarize_ranking(ranked_candidates: list[RankedCandidate]) -> dict:
        """Produce a quick aggregate summary of a ranked candidate pool —
        useful for a dashboard/report header (counts per tier, average
        score, etc.).
        """
        scorable = [rc for rc in ranked_candidates if rc.error is None]
        if not scorable:
            return {
                "total_candidates": len(ranked_candidates),
                "scored_candidates": 0,
                "average_score": 0.0,
                "tier_counts": {},
            }

        tier_counts: dict[str, int] = {}
        for rc in scorable:
            tier_counts[rc.tier] = tier_counts.get(rc.tier, 0) + 1

        average_score = sum(rc.total_score for rc in scorable) / len(scorable)

        return {
            "total_candidates": len(ranked_candidates),
            "scored_candidates": len(scorable),
            "unscored_candidates": len(ranked_candidates) - len(scorable),
            "average_score": round(average_score, 2),
            "top_score": round(scorable[0].total_score, 2) if scorable else 0.0,
            "tier_counts": tier_counts,
        }


# --- Convenience module-level function ---------------------------------

def rank_candidates(
    candidates: list[CandidateInput],
    job_description_text: str,
    weights: Optional[ATSWeights] = None,
) -> list[RankedCandidate]:
    """Convenience wrapper for one-off ranking without instantiating the
    class explicitly.
    """
    return CandidateRanker(weights=weights).rank_candidates(candidates, job_description_text)


if __name__ == "__main__":
    # Simple manual smoke test when run directly:
    #   python ml/candidate_ranker.py
    job_description = """
    Looking for a Software Engineer with 3+ years of experience in
    Python, Machine Learning, Django, and AWS. Bachelor's degree in
    Computer Science required. Strong problem solving skills.
    """

    resume_alice = """
    Alice Smith - Software Engineer, 5 years experience.
    EXPERIENCE
    Software Engineer at TechCorp (2019 - Present)
    EDUCATION
    B.Tech Computer Science, 2018
    SKILLS
    Python, Machine Learning, Django, AWS, Docker
    PROJECTS
    Resume Screener - NLP pipeline
    Fraud Detection Model - ML classifier
    CERTIFICATIONS
    AWS Certified Solutions Architect
    """

    resume_bob = """
    Bob Lee - Junior Developer.
    EXPERIENCE
    Intern at StartUpX (2023 - 2024)
    EDUCATION
    Diploma in Information Technology, 2023
    SKILLS
    HTML, CSS, JavaScript
    PROJECTS
    Portfolio website
    """

    candidates = [
        CandidateInput(candidate_id="c1", name="Alice Smith", resume_text=resume_alice),
        CandidateInput(candidate_id="c2", name="Bob Lee", resume_text=resume_bob),
        CandidateInput(candidate_id="c3", name="Empty Resume", resume_text=""),
    ]

    ranker = CandidateRanker()
    ranked = ranker.rank_candidates(candidates, job_description)

    print("=== RANKING ===")
    for rc in ranked:
        print(f"#{rc.rank} {rc.name}: {rc.total_score}/100 [{rc.tier}]" + (f" ERROR: {rc.error}" if rc.error else ""))

    print("\n=== SUMMARY ===")
    print(ranker.summarize_ranking(ranked))

    print("\n=== HEAD-TO-HEAD: Alice vs Bob ===")
    comparison = ranker.compare_candidates(candidates[0], candidates[1], job_description)
    print(f"Winner: {comparison.winner_id} (diff: {comparison.score_difference})")
    print(f"Common matched skills: {comparison.common_matched_skills}")
    print(f"Unique to Alice: {comparison.unique_to_a}")
    print(f"Unique to Bob: {comparison.unique_to_b}")