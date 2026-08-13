"""
HireSense AI - Candidate Ranking Page
==========================================================================
Full sortable/filterable/searchable ranking table across all screened
candidates, with medal treatment for the top 3.

MOCK DATA NOTICE
------------------------------------------------------------------------
`get_ranked_candidates()` returns placeholder data for UI development.
Real integration point (future):

    from ml.candidate_ranker import get_ranked_candidates as _real_ranker
    return _real_ranker(job_description=..., resumes=...)

The rendering/filtering/sorting logic below operates purely on the list
of dicts returned by this function, so swapping the data source later
requires no changes elsewhere in this file.
"""

from __future__ import annotations

import random

import streamlit as st

from ui.styles import COLORS
from ui.components import (
    render_section_header,
    render_label,
    render_status_badge,
    render_score_badge,
    render_empty_state,
    render_divider,
    get_rank_medal,
    get_score_tier,
)


# ======================================================================
# MOCK DATA  (placeholder only — see module docstring)
# ======================================================================

_MOCK_CANDIDATES_RAW = [
    ("Ananya Sharma", "Senior Data Scientist", 6, 94.0, ["Python", "PyTorch", "SQL", "NLP", "AWS"], "shortlisted"),
    ("Rahul Verma", "ML Engineer", 4, 87.0, ["Python", "TensorFlow", "Docker", "MLOps"], "shortlisted"),
    ("Devika Rao", "Data Scientist", 5, 85.5, ["Python", "SQL", "Pandas", "AWS", "NLP"], "shortlisted"),
    ("Aditya Kulkarni", "ML Engineer", 3, 81.0, ["Python", "TensorFlow", "Docker"], "shortlisted"),
    ("Priya Nair", "Data Analyst", 3, 76.0, ["SQL", "Power BI", "Excel"], "in review"),
    ("Vikram Singh", "Data Engineer", 4, 72.5, ["Python", "SQL", "Spark", "AWS"], "in review"),
    ("Neha Gupta", "Data Scientist", 2, 68.0, ["Python", "Pandas", "SQL"], "in review"),
    ("Meera Joshi", "BI Analyst", 3, 63.5, ["Power BI", "SQL", "Excel"], "in review"),
    ("Karan Mehta", "Backend Engineer", 5, 58.0, ["Java", "Spring Boot"], "in review"),
    ("Sara Iqbal", "Data Scientist", 2, 41.0, ["Python", "Pandas"], "rejected"),
    ("Farhan Ali", "Software Engineer", 1, 34.5, ["Java", "SQL"], "rejected"),
    ("Ishita Bose", "Data Analyst", 1, 29.0, ["Excel"], "rejected"),
]


def _recommendation_for_score(score: float) -> str:
    if score >= 85:
        return "Strong Fit"
    if score >= 65:
        return "Good Fit"
    if score >= 45:
        return "Consider"
    return "Not a Fit"


def get_ranked_candidates() -> list[dict]:
    """Returns the full ranked candidate list, sorted by score descending."""
    candidates = []
    for name, role, exp_years, score, skills, status in _MOCK_CANDIDATES_RAW:
        rng = random.Random(name)
        candidates.append(
            {
                "name": name,
                "role": role,
                "experience_years": exp_years,
                "experience": f"{exp_years} yrs",
                "score": score,
                "skills_match": round(min(score + rng.uniform(-6, 4), 100), 1),
                "matched_skills": skills,
                "status": status,
                "recommendation": _recommendation_for_score(score),
            }
        )
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


_RECOMMENDATION_TIER = {
    "Strong Fit": "success",
    "Good Fit": "primary",
    "Consider": "warning",
    "Not a Fit": "danger",
}


# ======================================================================
# PAGE RENDER
# ======================================================================

def render() -> None:
    render_section_header(
        "Candidate Ranking",
        "All screened candidates ranked by compatibility with the job description.",
        icon="🏆",
    )

    all_candidates = get_ranked_candidates()

    if not all_candidates:
        render_empty_state(
            "No candidates ranked yet",
            "Screen resumes from the Resume Screening page to populate the ranking.",
            icon="🏆",
        )
        return

    # ------------------------------------------------------------
    # SEARCH + FILTERS + SORT
    # ------------------------------------------------------------
    search_col, status_col, sort_col, order_col = st.columns([2.2, 1.6, 1.4, 1])

    with search_col:
        search_query = st.text_input(
            "Search",
            placeholder="🔍  Search by candidate name or role…",
            label_visibility="collapsed",
        )
    with status_col:
        status_filter = st.multiselect(
            "Status",
            options=["shortlisted", "in review", "rejected"],
            default=[],
            placeholder="Filter by status",
            label_visibility="collapsed",
        )
    with sort_col:
        sort_field = st.selectbox(
            "Sort by",
            options=["Score", "Name", "Experience"],
            label_visibility="collapsed",
        )
    with order_col:
        sort_desc = st.selectbox(
            "Order",
            options=["High → Low", "Low → High"],
            label_visibility="collapsed",
        ) == "High → Low"

    min_score, max_score = st.slider(
        "Compatibility score range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        help="Filter candidates by compatibility score range",
    )

    # ------------------------------------------------------------
    # APPLY FILTERS
    # ------------------------------------------------------------
    filtered = all_candidates

    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [c for c in filtered if q in c["name"].lower() or q in c["role"].lower()]

    if status_filter:
        filtered = [c for c in filtered if c["status"] in status_filter]

    filtered = [c for c in filtered if min_score <= c["score"] <= max_score]

    sort_key_map = {
        "Score": lambda c: c["score"],
        "Name": lambda c: c["name"].lower(),
        "Experience": lambda c: c["experience_years"],
    }
    filtered = sorted(filtered, key=sort_key_map[sort_field], reverse=sort_desc)

    render_divider()
    render_label(f"{len(filtered)} OF {len(all_candidates)} CANDIDATES")

    if not filtered:
        render_empty_state(
            "No candidates match your filters",
            "Try widening the score range or clearing the search/status filters.",
            icon="🔎",
        )
        return

    # ------------------------------------------------------------
    # TABLE HEADER
    # ------------------------------------------------------------
    col_widths = [0.5, 2.4, 1.6, 1.6, 1, 1.5, 1.3, 1]
    header_cols = st.columns(col_widths)
    headers = ["Rank", "Candidate", "Score", "Skills Match", "Experience", "Recommendation", "Status", ""]
    for col, label in zip(header_cols, headers):
        col.markdown(f"<div class='section-subtitle' style='font-weight:700;'>{label}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='border-bottom:1px solid {COLORS['border']}; margin: 0.4rem 0 0.6rem 0;'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # TABLE ROWS
    # ------------------------------------------------------------
    for rank, c in enumerate(filtered, start=1):
        row = st.columns(col_widths)

        with row[0]:
            st.markdown(f"<div class='rank-medal' style='padding-top:0.4rem;'>{get_rank_medal(rank)}</div>", unsafe_allow_html=True)

        with row[1]:
            st.markdown(
                f"""
                <div style="padding-top:0.25rem;">
                    <div style="font-weight:700; color:{COLORS['text_primary']}; font-size:0.9rem;">{c['name']}</div>
                    <div style="font-size:0.76rem; color:{COLORS['text_muted']};">{c['role']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row[2]:
            st.markdown("<div style='padding-top:0.3rem;'>", unsafe_allow_html=True)
            render_score_badge(c["score"])
            st.markdown("</div>", unsafe_allow_html=True)

        with row[3]:
            tier = get_score_tier(c["skills_match"])
            st.markdown(
                f"<div style='padding-top:0.35rem; font-size:0.85rem; font-weight:600; color:{COLORS[tier if tier != 'danger' else 'danger']};'>{c['skills_match']:.0f}%</div>",
                unsafe_allow_html=True,
            )

        with row[4]:
            st.markdown(f"<div style='padding-top:0.35rem; font-size:0.85rem; color:{COLORS['text_secondary']};'>{c['experience']}</div>", unsafe_allow_html=True)

        with row[5]:
            rec_tier = _RECOMMENDATION_TIER[c["recommendation"]]
            st.markdown(
                f"<div style='padding-top:0.3rem;'><span class='badge badge-{rec_tier}'>{c['recommendation']}</span></div>",
                unsafe_allow_html=True,
            )

        with row[6]:
            st.markdown("<div style='padding-top:0.3rem;'>", unsafe_allow_html=True)
            render_status_badge(c["status"])
            st.markdown("</div>", unsafe_allow_html=True)

        with row[7]:
            if st.button("View →", key=f"rank_view_{c['name']}", use_container_width=True):
                st.session_state.selected_candidate = c["name"]
                st.session_state.active_page = "candidate_details"
                st.rerun()

    render_divider()
    st.caption(
        "Rankings shown are from a mock dataset for UI development. "
        "Real ranking will be powered by ml/candidate_ranker.py once integrated."
    )