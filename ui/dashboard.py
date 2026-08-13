"""
HireSense AI - Dashboard Page
==========================================================================
Premium recruiter overview: key metrics, score distribution chart,
recent candidates, and quick actions.

MOCK DATA NOTICE
------------------------------------------------------------------------
Everything under the "MOCK DATA" section below is placeholder data for
UI development only. Once ml/candidate_ranker.py and friends exist,
`get_dashboard_data()` should be replaced with a call into that real
pipeline — the rest of this file (rendering) does not need to change,
since it only consumes the dict shape returned by that function.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from ui.styles import COLORS
from ui.components import (
    render_metric_row,
    render_section_header,
    render_label,
    render_candidate_card,
    render_status_badge,
    render_score_badge,
    render_empty_state,
    render_divider,
    get_rank_medal,
)


# ======================================================================
# MOCK DATA  (placeholder only — swap for real ML pipeline output)
# ======================================================================

def get_dashboard_data() -> dict:
    """
    Returns the data shape the dashboard needs. Currently mocked.

    Real integration point (future):
        from ml.candidate_ranker import get_dashboard_summary
        return get_dashboard_summary()
    """
    return {
        "total_candidates": 128,
        "total_candidates_delta": "+18 this week",
        "avg_score": 71.4,
        "avg_score_delta": "+3.2 vs last batch",
        "top_candidate": {"name": "Ananya Sharma", "score": 94.0},
        "shortlisted_count": 22,
        "shortlisted_delta": "+5 this week",
        "score_distribution": {
            "buckets": ["0-20", "21-40", "41-60", "61-80", "81-100"],
            "counts": [4, 11, 38, 52, 23],
        },
        "recent_candidates": [
            {
                "name": "Ananya Sharma",
                "role": "Senior Data Scientist",
                "experience": "6 yrs",
                "score": 94.0,
                "status": "shortlisted",
                "matched_skills": ["Python", "PyTorch", "SQL", "NLP", "AWS"],
            },
            {
                "name": "Rahul Verma",
                "role": "ML Engineer",
                "experience": "4 yrs",
                "score": 87.0,
                "status": "shortlisted",
                "matched_skills": ["Python", "TensorFlow", "Docker", "MLOps"],
            },
            {
                "name": "Priya Nair",
                "role": "Data Analyst",
                "experience": "3 yrs",
                "score": 76.0,
                "status": "in review",
                "matched_skills": ["SQL", "Power BI", "Excel"],
            },
            {
                "name": "Karan Mehta",
                "role": "Backend Engineer",
                "experience": "5 yrs",
                "score": 58.0,
                "status": "in review",
                "matched_skills": ["Java", "Spring Boot"],
            },
            {
                "name": "Sara Iqbal",
                "role": "Data Scientist",
                "experience": "2 yrs",
                "score": 41.0,
                "status": "rejected",
                "matched_skills": ["Python", "Pandas"],
            },
        ],
    }


# ======================================================================
# CHART BUILDERS
# ======================================================================

def _build_score_distribution_chart(buckets: list[str], counts: list[int]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=buckets,
                y=counts,
                marker=dict(
                    color=counts,
                    colorscale=[[0, COLORS["accent"]], [1, COLORS["primary"]]],
                    line=dict(width=0),
                ),
                text=counts,
                textposition="outside",
                textfont=dict(color=COLORS["text_secondary"]),
                hovertemplate="Score %{x}: <b>%{y}</b> candidates<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        font=dict(family="Inter, sans-serif", color=COLORS["text_secondary"], size=12),
        xaxis=dict(showgrid=False, title="Compatibility score range"),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], title="Candidates"),
        showlegend=False,
    )
    return fig


# ======================================================================
# PAGE RENDER
# ======================================================================

def render() -> None:
    data = get_dashboard_data()

    # ------------------------------------------------------------
    # KEY METRICS
    # ------------------------------------------------------------
    render_metric_row(
        [
            {
                "label": "Total Candidates",
                "value": str(data["total_candidates"]),
                "delta": data["total_candidates_delta"],
            },
            {
                "label": "Average Compatibility",
                "value": f"{data['avg_score']:.1f}%",
                "delta": data["avg_score_delta"],
            },
            {
                "label": "Top Candidate",
                "value": data["top_candidate"]["name"],
                "delta": f"{data['top_candidate']['score']:.0f}% match",
            },
            {
                "label": "Shortlisted",
                "value": str(data["shortlisted_count"]),
                "delta": data["shortlisted_delta"],
            },
        ]
    )

    render_divider()

    # ------------------------------------------------------------
    # CHART + QUICK ACTIONS ROW
    # ------------------------------------------------------------
    chart_col, actions_col = st.columns([2, 1])

    with chart_col:
        render_section_header("Candidate Score Distribution", "Compatibility scores across all screened resumes")
        fig = _build_score_distribution_chart(
            data["score_distribution"]["buckets"],
            data["score_distribution"]["counts"],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with actions_col:
        render_section_header("Quick Actions", icon="⚡")

        if st.button("📤  Upload New Resumes", use_container_width=True, type="primary", key="qa_upload"):
            st.session_state.active_page = "screening"
            st.rerun()

        if st.button("📝  Edit Job Description", use_container_width=True, key="qa_jd"):
            st.session_state.active_page = "job_description"
            st.rerun()

        if st.button("🏆  View Full Ranking", use_container_width=True, key="qa_ranking"):
            st.session_state.active_page = "ranking"
            st.rerun()

        if st.button("🧾  Generate Report", use_container_width=True, key="qa_report"):
            st.session_state.active_page = "reports"
            st.rerun()

        st.markdown(
            f"""
            <div class="glass-card" style="margin-top: 0.75rem; padding: 1rem 1.1rem;">
                <div class="section-subtitle" style="margin-bottom: 0.4rem;">🎯 Today's Focus</div>
                <div style="font-size: 0.85rem; color: {COLORS['text_secondary']};">
                    {data['shortlisted_count']} candidates are shortlisted and ready for interview scheduling.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_divider()

    # ------------------------------------------------------------
    # RECENT CANDIDATES
    # ------------------------------------------------------------
    header_col, link_col = st.columns([3, 1])
    with header_col:
        render_section_header("Recent Candidates", "Latest resumes processed by the screening engine")
    with link_col:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("View all →", key="qa_view_all", use_container_width=True):
            st.session_state.active_page = "ranking"
            st.rerun()

    recent = data.get("recent_candidates", [])

    if not recent:
        render_empty_state(
            "No candidates yet",
            "Upload resumes from the Resume Screening page to see them appear here.",
            icon="📭",
        )
        return

    for idx, candidate in enumerate(recent, start=1):
        row = st.columns([0.5, 4.5, 1.5, 1.5])
        with row[0]:
            st.markdown(
                f"<div class='rank-medal' style='padding-top: 0.6rem;'>{get_rank_medal(idx)}</div>",
                unsafe_allow_html=True,
            )
        with row[1]:
            st.markdown(
                f"""
                <div style="padding-top: 0.35rem;">
                    <div style="font-weight:700; color:{COLORS['text_primary']}; font-size:0.92rem;">{candidate['name']}</div>
                    <div style="font-size:0.78rem; color:{COLORS['text_muted']};">{candidate['role']} · {candidate['experience']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row[2]:
            st.markdown("<div style='padding-top: 0.55rem;'>", unsafe_allow_html=True)
            render_score_badge(candidate["score"])
            st.markdown("</div>", unsafe_allow_html=True)
        with row[3]:
            st.markdown("<div style='padding-top: 0.55rem;'>", unsafe_allow_html=True)
            render_status_badge(candidate["status"])
            st.markdown("</div>", unsafe_allow_html=True)