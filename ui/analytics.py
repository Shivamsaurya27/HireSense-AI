"""
HireSense AI - Analytics Page
==========================================================================
Aggregate insights across all screened candidates: score distribution,
skill distribution, average score, experience distribution, shortlisted
breakdown, and skill-gap analysis against the job description.

MOCK DATA NOTICE
------------------------------------------------------------------------
This page derives its charts from `ui.ranking.get_ranked_candidates()`,
which is itself mock data (see that module's docstring). Once the real
ML pipeline is wired in, that single source will start returning real
candidates and every chart on this page updates automatically — no
changes needed here.
"""

from __future__ import annotations

from collections import Counter

import streamlit as st
import plotly.graph_objects as go

from ui.styles import COLORS
from ui.components import (
    render_section_header,
    render_label,
    render_metric_row,
    render_divider,
    render_empty_state,
)
from ui.ranking import get_ranked_candidates
from ui.screening import _MOCK_SKILL_POOL


# ======================================================================
# CHART THEME HELPERS
# ======================================================================

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text_secondary"], size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    height=300,
)


def _score_distribution_chart(candidates: list[dict]) -> go.Figure:
    buckets = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    counts = [0] * 5
    for c in candidates:
        s = c["score"]
        idx = min(int(s // 20), 4)
        counts[idx] += 1

    fig = go.Figure(
        data=[
            go.Bar(
                x=buckets,
                y=counts,
                marker=dict(color=counts, colorscale=[[0, COLORS["accent"]], [1, COLORS["primary"]]]),
                text=counts,
                textposition="outside",
                textfont=dict(color=COLORS["text_secondary"]),
                hovertemplate="Score %{x}: <b>%{y}</b><extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(showgrid=False, title="Score range"),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], title="Candidates"),
        showlegend=False,
    )
    return fig


def _skill_distribution_chart(candidates: list[dict]) -> go.Figure:
    counter = Counter()
    for c in candidates:
        counter.update(c["matched_skills"])
    top = counter.most_common(8)
    skills = [s for s, _ in top][::-1]
    counts = [n for _, n in top][::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts,
                y=skills,
                orientation="h",
                marker=dict(color=COLORS["secondary"]),
                text=counts,
                textposition="outside",
                textfont=dict(color=COLORS["text_secondary"]),
                hovertemplate="%{y}: <b>%{x}</b> candidates<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(showgrid=True, gridcolor=COLORS["border"], title="Candidates with skill"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def _experience_distribution_chart(candidates: list[dict]) -> go.Figure:
    buckets = {"0-2 yrs": 0, "3-5 yrs": 0, "6+ yrs": 0}
    for c in candidates:
        yrs = c["experience_years"]
        if yrs <= 2:
            buckets["0-2 yrs"] += 1
        elif yrs <= 5:
            buckets["3-5 yrs"] += 1
        else:
            buckets["6+ yrs"] += 1

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(buckets.keys()),
                values=list(buckets.values()),
                hole=0.55,
                marker=dict(colors=[COLORS["primary"], COLORS["accent"], COLORS["secondary"]]),
                textfont=dict(color="white", size=12),
                hovertemplate="%{label}: <b>%{value}</b> candidates<extra></extra>",
            )
        ]
    )
    fig.update_layout(**_BASE_LAYOUT, showlegend=True, legend=dict(orientation="h", y=-0.1, font=dict(color=COLORS["text_secondary"])))
    return fig


def _status_breakdown_chart(candidates: list[dict]) -> go.Figure:
    counter = Counter(c["status"] for c in candidates)
    order = ["shortlisted", "in review", "rejected"]
    labels = [s.title() for s in order]
    values = [counter.get(s, 0) for s in order]
    colors = [COLORS["success"], COLORS["warning"], COLORS["danger"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textfont=dict(color="white", size=12),
                hovertemplate="%{label}: <b>%{value}</b> candidates<extra></extra>",
            )
        ]
    )
    fig.update_layout(**_BASE_LAYOUT, showlegend=True, legend=dict(orientation="h", y=-0.1, font=dict(color=COLORS["text_secondary"])))
    return fig


def _skill_gap_chart(candidates: list[dict], required_skills: list[str]) -> go.Figure:
    total = len(candidates) or 1
    coverage = []
    for skill in required_skills:
        have = sum(1 for c in candidates if skill in c["matched_skills"])
        coverage.append(round(have / total * 100, 1))

    colors = [COLORS["danger"] if v < 40 else COLORS["warning"] if v < 70 else COLORS["success"] for v in coverage]

    fig = go.Figure(
        data=[
            go.Bar(
                x=required_skills,
                y=coverage,
                marker=dict(color=colors),
                text=[f"{v:.0f}%" for v in coverage],
                textposition="outside",
                textfont=dict(color=COLORS["text_secondary"]),
                hovertemplate="%{x}: <b>%{y}</b>%% candidate coverage<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(showgrid=False, title="Required skill"),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], title="% candidates covering", range=[0, 110]),
        showlegend=False,
    )
    return fig


# ======================================================================
# PAGE RENDER
# ======================================================================

def render() -> None:
    render_section_header(
        "Analytics",
        "Aggregate insights across every screened candidate.",
        icon="📈",
    )

    candidates = get_ranked_candidates()

    if not candidates:
        render_empty_state(
            "No data to analyze yet",
            "Screen resumes from the Resume Screening page to populate analytics.",
            icon="📉",
        )
        return

    # ------------------------------------------------------------
    # TOP-LINE METRICS
    # ------------------------------------------------------------
    avg_score = sum(c["score"] for c in candidates) / len(candidates)
    shortlisted = [c for c in candidates if c["status"] == "shortlisted"]
    unique_skills = len({s for c in candidates for s in c["matched_skills"]})

    render_metric_row(
        [
            {"label": "Total Candidates", "value": str(len(candidates))},
            {"label": "Average Score", "value": f"{avg_score:.1f}%"},
            {"label": "Shortlisted", "value": str(len(shortlisted)), "delta": f"{len(shortlisted) / len(candidates) * 100:.0f}% of pool"},
            {"label": "Unique Skills Seen", "value": str(unique_skills)},
        ]
    )

    render_divider()

    # ------------------------------------------------------------
    # SCORE + SKILL DISTRIBUTION
    # ------------------------------------------------------------
    col1, col2 = st.columns(2)
    with col1:
        render_section_header("Score Distribution", icon="📊")
        st.plotly_chart(_score_distribution_chart(candidates), use_container_width=True, config={"displayModeBar": False})
    with col2:
        render_section_header("Top Skills Across Candidates", icon="🧩")
        st.plotly_chart(_skill_distribution_chart(candidates), use_container_width=True, config={"displayModeBar": False})

    render_divider()

    # ------------------------------------------------------------
    # EXPERIENCE + STATUS BREAKDOWN
    # ------------------------------------------------------------
    col3, col4 = st.columns(2)
    with col3:
        render_section_header("Experience Distribution", icon="🧑‍💼")
        st.plotly_chart(_experience_distribution_chart(candidates), use_container_width=True, config={"displayModeBar": False})
    with col4:
        render_section_header("Shortlisted vs. Others", icon="✅")
        st.plotly_chart(_status_breakdown_chart(candidates), use_container_width=True, config={"displayModeBar": False})

    render_divider()

    # ------------------------------------------------------------
    # SKILL GAP ANALYSIS
    # ------------------------------------------------------------
    render_section_header(
        "Skill Gap Analysis",
        "Percentage of the candidate pool covering each required skill from the job description.",
        icon="🎯",
    )

    jd_fields = st.session_state.get("jd_fields")
    if jd_fields and jd_fields.get("required_skills"):
        required_skills = jd_fields["required_skills"]
        source_note = "Based on the currently analyzed job description."
    else:
        # Fallback sample of required skills when no JD has been analyzed yet
        required_skills = _MOCK_SKILL_POOL[:6]
        source_note = "No job description analyzed yet — showing a sample skill set. Analyze a JD for accurate results."

    st.plotly_chart(_skill_gap_chart(candidates, required_skills), use_container_width=True, config={"displayModeBar": False})
    st.caption(source_note)

    render_divider()
    st.caption(
        "Analytics shown are computed from a mock candidate dataset for UI development. "
        "Figures will reflect real screening results once the ml/ modules are integrated."
    )