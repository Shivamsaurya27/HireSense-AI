"""
HireSense AI - Candidate Details Page
==========================================================================
Deep-dive profile view for a single candidate: score breakdown, matched
vs missing skills, experience, education, projects, certifications,
strengths, areas for improvement, and final recommendation.

MOCK DATA NOTICE
------------------------------------------------------------------------
`get_candidate_detail()` is placeholder data for UI development.
Real integration point (future):

    from ml.ats_scorer import get_candidate_detail as _real_detail
    return _real_detail(candidate_id)

Rendering code below only depends on the dict shape returned by this
function, so the swap will not require changes elsewhere in this file.

CANDIDATE SOURCE NOTE
------------------------------------------------------------------------
Two separate candidate pools exist during UI development:
  1. The fixed mock "ranked candidates" pool (ui.ranking.get_ranked_candidates()).
  2. Whatever the user actually uploaded on the Resume Screening page
     (st.session_state.uploaded_resumes).
A candidate clicked via "View ->" on the Screening page usually only
exists in pool #2. This file merges both pools so the picker and the
detail lookup both recognize an uploaded resume's candidate, instead
of silently falling back to whichever mock candidate happens to be
first in pool #1.
"""

from __future__ import annotations

import random

import streamlit as st

from ui.styles import COLORS
from ui.components import (
    render_section_header,
    render_label,
    render_badge,
    render_status_badge,
    render_empty_state,
    render_progress_with_label,
    render_divider,
    get_score_tier,
)
from ui.ranking import get_ranked_candidates, _recommendation_for_score, _RECOMMENDATION_TIER


# ======================================================================
# MOCK DATA  (placeholder only — see module docstring)
# ======================================================================

_ALL_SKILLS = [
    "Python", "SQL", "TensorFlow", "PyTorch", "AWS", "Docker", "NLP",
    "Pandas", "Power BI", "Java", "React", "MLOps", "Spark", "GCP", "Kubernetes",
]

_STRENGTHS_POOL = [
    "Strong hands-on experience with production ML pipelines",
    "Clear, well-structured project history with measurable outcomes",
    "Broad toolset spanning modeling, data engineering, and deployment",
    "Consistent career progression with increasing scope",
    "Relevant certifications aligned with the role's tech stack",
]

_IMPROVEMENT_POOL = [
    "Limited exposure to large-scale distributed systems",
    "No direct experience with the team's primary cloud provider",
    "Gap in leadership or mentorship experience for a senior role",
    "Resume lacks quantified impact metrics for key projects",
    "Missing one or two core skills listed as required in the JD",
]

_PROJECTS_POOL = [
    ("Customer Churn Prediction", "Built an XGBoost model improving retention forecasting accuracy."),
    ("Real-time Fraud Detection", "Deployed a streaming pipeline for transaction anomaly detection."),
    ("Resume Ranking Engine", "Developed an NLP-based scoring system for internal recruiting."),
    ("Sales Forecasting Dashboard", "Created an interactive BI dashboard for regional sales trends."),
    ("Chatbot for Support Tickets", "Fine-tuned a transformer model to auto-triage support requests."),
]

_CERTS_POOL = [
    "AWS Certified Machine Learning – Specialty",
    "Google Data Analytics Professional Certificate",
    "TensorFlow Developer Certificate",
    "Microsoft Certified: Azure Data Scientist Associate",
    "Certified Scrum Master (CSM)",
]

_EDUCATION_POOL = [
    "B.Tech in Computer Science, IIT Delhi",
    "M.Sc. in Data Science, University of Edinburgh",
    "B.E. in Information Technology, VJTI Mumbai",
    "M.Tech in Artificial Intelligence, IIIT Hyderabad",
    "B.Sc. in Statistics, Delhi University",
]


def _status_from_score(score: float) -> str:
    """Same thresholds used on the Resume Screening results table."""
    if score >= 80:
        return "shortlisted"
    if score >= 50:
        return "in review"
    return "rejected"


def _uploaded_resume_candidates() -> list[dict]:
    """
    Normalize st.session_state.uploaded_resumes (from Resume Screening)
    into the same shape as get_ranked_candidates(), so a candidate the
    user actually uploaded can be found and rendered here even if they
    don't exist in the separate mock ranking pool.
    """
    uploaded = st.session_state.get("uploaded_resumes", [])
    normalized = []
    for r in uploaded:
        score = r["score"]
        normalized.append(
            {
                "name": r["candidate_name"],
                "role": "Not specified",
                "experience": "Not specified",
                "score": score,
                "status": _status_from_score(score),
                "matched_skills": r["matched_skills"],
                "skills_match": score,
                "recommendation": _recommendation_for_score(score),
            }
        )
    return normalized


def _all_known_candidates() -> list[dict]:
    """
    Merge the mock ranking pool with whatever the user has actually
    uploaded. Uploaded resumes take precedence on name collisions.
    """
    merged: dict[str, dict] = {c["name"]: c for c in get_ranked_candidates()}
    for c in _uploaded_resume_candidates():
        merged[c["name"]] = c
    return list(merged.values())


def get_candidate_detail(name: str) -> dict | None:
    """
    Build a full mock profile for the given candidate name, layered on
    top of the ranking summary (or an uploaded resume, if that's where
    the candidate actually came from) so score/status/role stay
    consistent with wherever the candidate was selected from.
    """
    base = next((c for c in _all_known_candidates() if c["name"] == name), None)
    if base is None:
        return None

    rng = random.Random(name)
    all_missing_pool = [s for s in _ALL_SKILLS if s not in base["matched_skills"]]

    return {
        **base,
        "missing_skills": rng.sample(all_missing_pool, k=min(3, len(all_missing_pool))),
        "experience_match": round(min(base["score"] + rng.uniform(-8, 6), 100), 1),
        "education_match": round(min(base["score"] + rng.uniform(-10, 8), 100), 1),
        "education": rng.choice(_EDUCATION_POOL),
        "projects": rng.sample(_PROJECTS_POOL, k=2),
        "certifications": rng.sample(_CERTS_POOL, k=rng.randint(1, 3)),
        "strengths": rng.sample(_STRENGTHS_POOL, k=3),
        "improvements": rng.sample(_IMPROVEMENT_POOL, k=2),
    }


# ======================================================================
# PAGE RENDER
# ======================================================================

def render() -> None:
    selected_name = st.session_state.get("selected_candidate")

    all_candidates = _all_known_candidates()

    # ------------------------------------------------------------
    # CANDIDATE PICKER (works even if navigated here without a selection)
    # ------------------------------------------------------------
    picker_col, _ = st.columns([2, 3])
    with picker_col:
        names = [c["name"] for c in all_candidates]
        default_idx = names.index(selected_name) if selected_name in names else 0
        chosen = st.selectbox("Select candidate", options=names, index=default_idx if names else 0)
        st.session_state.selected_candidate = chosen

    if not chosen:
        render_empty_state(
            "No candidate selected",
            "Choose a candidate from the ranking page or the dropdown above.",
            icon="👤",
        )
        return

    detail = get_candidate_detail(chosen)
    if detail is None:
        render_empty_state("Candidate not found", "This candidate may have been removed.", icon="❓")
        return

    render_divider()

    # ------------------------------------------------------------
    # PROFILE HEADER
    # ------------------------------------------------------------
    initials = "".join([p[0] for p in detail["name"].split()[:2]]).upper()
    rec_tier = _RECOMMENDATION_TIER[detail["recommendation"]]
    score_tier = get_score_tier(detail["score"])

    header_col1, header_col2 = st.columns([3, 1.3])
    with header_col1:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; align-items:center; gap:1.25rem;">
                <div style="width:64px; height:64px; border-radius:16px; flex-shrink:0;
                            background:linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
                            display:flex; align-items:center; justify-content:center;
                            font-weight:800; font-size:1.3rem; color:white;">
                    {initials}
                </div>
                <div>
                    <div style="font-size:1.3rem; font-weight:800; color:{COLORS['text_primary']};">{detail['name']}</div>
                    <div style="font-size:0.88rem; color:{COLORS['text_muted']}; margin-top:0.1rem;">{detail['role']} · {detail['experience']} experience</div>
                    <div style="margin-top:0.6rem; display:flex; gap:0.4rem;">
                        <span class="badge badge-{rec_tier}">{detail['recommendation']}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_col2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;">
                <div class="section-subtitle" style="font-weight:700;">Compatibility Score</div>
                <div style="font-size:2.4rem; font-weight:800; color:{COLORS[score_tier]}; margin-top:0.3rem;">
                    {detail['score']:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_status_badge(detail["status"])

    render_divider()

    # ------------------------------------------------------------
    # SCORE BREAKDOWN
    # ------------------------------------------------------------
    render_section_header("Score Breakdown", icon="📊")
    b1, b2, b3 = st.columns(3)
    with b1:
        render_progress_with_label("Skills Match", detail["skills_match"])
    with b2:
        render_progress_with_label("Experience Match", detail["experience_match"])
    with b3:
        render_progress_with_label("Education Match", detail["education_match"])

    render_divider()

    # ------------------------------------------------------------
    # SKILLS (matched vs missing)
    # ------------------------------------------------------------
    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        render_section_header("Matched Skills", icon="✅")
        chips = "".join(f'<span class="badge badge-success" style="margin:0.2rem;">{s}</span>' for s in detail["matched_skills"])
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
    with skill_col2:
        render_section_header("Missing Skills", icon="⚠️")
        if detail["missing_skills"]:
            chips = "".join(f'<span class="badge badge-danger" style="margin:0.2rem;">{s}</span>' for s in detail["missing_skills"])
            st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{COLORS['text_muted']}; font-size:0.85rem;'>No significant skill gaps identified.</div>", unsafe_allow_html=True)

    render_divider()

    # ------------------------------------------------------------
    # EDUCATION, PROJECTS, CERTIFICATIONS
    # ------------------------------------------------------------
    left_col, right_col = st.columns([1, 1])

    with left_col:
        render_section_header("Education", icon="🎓")
        st.markdown(
            f"""<div class="glass-card">{detail['education']}</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        render_section_header("Certifications", icon="📜")
        for cert in detail["certifications"]:
            st.markdown(
                f"""<div class="glass-card" style="padding:0.75rem 1rem; margin-bottom:0.5rem;">🏅 {cert}</div>""",
                unsafe_allow_html=True,
            )

    with right_col:
        render_section_header("Projects", icon="🛠️")
        for title, desc in detail["projects"]:
            st.markdown(
                f"""
                <div class="glass-card" style="padding:0.9rem 1.1rem; margin-bottom:0.6rem;">
                    <div style="font-weight:700; color:{COLORS['text_primary']}; font-size:0.9rem;">{title}</div>
                    <div style="font-size:0.8rem; color:{COLORS['text_muted']}; margin-top:0.25rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_divider()

    # ------------------------------------------------------------
    # STRENGTHS / IMPROVEMENTS / RECOMMENDATION
    # ------------------------------------------------------------
    str_col, imp_col = st.columns(2)
    with str_col:
        render_section_header("Strengths", icon="💪")
        for s in detail["strengths"]:
            st.markdown(
                f"""<div style="display:flex; gap:0.5rem; margin-bottom:0.5rem; font-size:0.85rem; color:{COLORS['text_secondary']};">
                        <span style="color:{COLORS['success']};">●</span> {s}
                    </div>""",
                unsafe_allow_html=True,
            )
    with imp_col:
        render_section_header("Areas for Improvement", icon="🎯")
        for imp in detail["improvements"]:
            st.markdown(
                f"""<div style="display:flex; gap:0.5rem; margin-bottom:0.5rem; font-size:0.85rem; color:{COLORS['text_secondary']};">
                        <span style="color:{COLORS['warning']};">●</span> {imp}
                    </div>""",
                unsafe_allow_html=True,
            )

    render_divider()

    st.markdown(
        f"""
        <div class="glass-card" style="border-color: rgba(99,102,241,0.35);">
            <div class="section-subtitle" style="font-weight:700; margin-bottom:0.4rem;">Final Recommendation</div>
            <div>
                <span class="badge badge-{rec_tier}" style="font-size:0.85rem; padding:0.4rem 0.9rem;">{detail['recommendation']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    back_col, _sp = st.columns([1, 3])
    with back_col:
        if st.button("← Back to Ranking", use_container_width=True):
            st.session_state.active_page = "ranking"
            st.rerun()

    st.caption(
        "Profile details shown are from a mock dataset for UI development. "
        "Real candidate details will be powered by the ml/ modules once integrated."
    )
