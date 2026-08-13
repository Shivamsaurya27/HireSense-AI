"""
HireSense AI - Reusable UI Components
==========================================================================
A small component library of render_* helpers built on top of the CSS
classes defined in ui/styles.py. Page modules (dashboard, screening,
ranking, etc.) import from here to stay visually consistent instead of
re-writing HTML/CSS snippets inline.

IMPORTANT: This module contains NO mock data and NO ML logic. Every
function here is a pure "given data, render it" helper — callers pass
in real (or, for now, mock) values.
"""

from __future__ import annotations

import streamlit as st
from ui.styles import COLORS


# ======================================================================
# SCORE / STATUS HELPERS
# ======================================================================

def get_score_tier(score: float) -> str:
    """Map a 0-100 compatibility score to a semantic tier."""
    if score >= 80:
        return "success"
    if score >= 60:
        return "warning"
    return "danger"


def get_rank_medal(rank: int) -> str:
    """Return a medal emoji for the top 3 ranks, otherwise the rank number."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    return medals.get(rank, f"#{rank}")


# ======================================================================
# SECTION / PAGE HEADERS
# ======================================================================

def render_section_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    """Render a consistent section heading used at the top of a page block."""
    icon_html = f"{icon}&nbsp;" if icon else ""
    subtitle_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h3 class="section-header-title">{icon_html}{title}</h3>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_label(text: str) -> None:
    """Small uppercase eyebrow label (e.g. 'RECENT CANDIDATES')."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# ======================================================================
# BADGES / PILLS
# ======================================================================

def badge_html(text: str, tier: str = "primary", icon: str | None = None) -> str:
    """Return the raw HTML string for a badge (for embedding inside other HTML)."""
    icon_html = f"{icon} " if icon else ""
    return f'<span class="badge badge-{tier}">{icon_html}{text}</span>'


def render_badge(text: str, tier: str = "primary", icon: str | None = None) -> None:
    st.markdown(badge_html(text, tier, icon), unsafe_allow_html=True)


def render_score_badge(score: float) -> None:
    """Compatibility-score badge, auto-colored by tier."""
    tier = get_score_tier(score)
    st.markdown(badge_html(f"{score:.0f}%", tier), unsafe_allow_html=True)


def render_status_badge(status: str) -> None:
    """Status badge for candidate pipeline states."""
    status_map = {
        "shortlisted": ("success", "✓"),
        "in review": ("warning", "⏳"),
        "rejected": ("danger", "✕"),
        "new": ("primary", "●"),
        "processing": ("accent", "⟳"),
    }
    tier, icon = status_map.get(status.lower(), ("primary", "●"))
    render_badge(status.title(), tier, icon)


# ======================================================================
# METRIC / GLASS CARDS
# ======================================================================

def render_glass_metric(label: str, value: str, delta: str | None = None, icon: str | None = None) -> None:
    """
    Custom glassmorphic metric card (alternative to st.metric for spots
    where more visual control over icon/delta styling is wanted).
    """
    delta_html = f'<div class="glass-metric-delta">{delta}</div>' if delta else ""
    icon_html = f'<div class="glass-metric-icon">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="glass-card glass-metric-card">
            {icon_html}
            <div class="glass-metric-label">{label}</div>
            <div class="glass-metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics: list[dict]) -> None:
    """
    Render a row of native st.metric cards (styled via styles.py) inside
    equal-width columns.

    Each dict: {"label": str, "value": str, "delta": Optional[str]}
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(label=m["label"], value=m["value"], delta=m.get("delta"))


# ======================================================================
# CANDIDATE CARD
# ======================================================================

def render_candidate_card(candidate: dict) -> None:
    """
    Render a compact candidate summary card.

    Expected keys: name, role, score, matched_skills (list), status,
    experience (str)
    """
    tier = get_score_tier(candidate.get("score", 0))
    skills = candidate.get("matched_skills", [])
    skills_html = "".join(
        f'<span class="skill-chip">{s}</span>' for s in skills[:4]
    )
    if len(skills) > 4:
        skills_html += f'<span class="skill-chip skill-chip-muted">+{len(skills) - 4}</span>'

    st.markdown(
        f"""
        <div class="glass-card candidate-card">
            <div class="candidate-card-top">
                <div>
                    <div class="candidate-card-name">{candidate.get('name', 'Unknown')}</div>
                    <div class="candidate-card-role">{candidate.get('role', '—')} · {candidate.get('experience', '—')}</div>
                </div>
                <div class="candidate-card-score badge badge-{tier}">{candidate.get('score', 0):.0f}%</div>
            </div>
            <div class="candidate-card-skills">{skills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# STATE HELPERS: empty / loading / success / error
# ======================================================================

def render_empty_state(title: str, subtitle: str = "", icon: str = "📭") -> None:
    st.markdown(
        f"""
        <div class="empty-state-card">
            <div class="empty-state-icon">{icon}</div>
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_state(text: str = "Processing…") -> None:
    st.markdown(
        f"""
        <div class="loading-state">
            <div class="loading-spinner"></div>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success(message: str) -> None:
    st.success(message, icon="✅")


def render_error(message: str) -> None:
    st.error(message, icon="⚠️")


def render_info(message: str) -> None:
    st.info(message, icon="ℹ️")


# ======================================================================
# MISC LAYOUT HELPERS
# ======================================================================

def render_divider() -> None:
    st.markdown('<div class="content-divider"></div>', unsafe_allow_html=True)


def render_progress_with_label(label: str, value: float, suffix: str = "%") -> None:
    """A labeled progress bar, e.g. for skill-match breakdowns."""
    st.markdown(
        f"""
        <div class="progress-label-row">
            <span>{label}</span>
            <span class="progress-label-value">{value:.0f}{suffix}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(value / 100, 0.0), 1.0))


# ======================================================================
# COMPONENT-SPECIFIC CSS
# ======================================================================
# Small additive CSS block for the component classes introduced in this
# file (skill chips, candidate card layout, glass metric card, section
# header) that aren't already defined in ui/styles.py. Injected once via
# inject_component_styles(), which app.py / pages can call after
# apply_custom_css().
# ======================================================================

def inject_component_styles() -> None:
    st.markdown(
        f"""
        <style>
        .section-header {{ margin-bottom: 1rem; }}
        .section-header-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            margin-bottom: 0.15rem;
        }}
        .section-subtitle {{
            font-size: 0.82rem;
            color: {COLORS['text_muted']};
            margin: 0;
        }}

        .glass-metric-card {{ position: relative; }}
        .glass-metric-icon {{ font-size: 1.4rem; margin-bottom: 0.5rem; }}
        .glass-metric-label {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: {COLORS['text_muted']};
        }}
        .glass-metric-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: {COLORS['text_primary']};
            margin-top: 0.25rem;
        }}
        .glass-metric-delta {{
            font-size: 0.78rem;
            color: {COLORS['success']};
            margin-top: 0.3rem;
        }}

        .candidate-card {{ padding: 1.1rem 1.25rem; }}
        .candidate-card-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .candidate-card-name {{
            font-size: 1rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
        }}
        .candidate-card-role {{
            font-size: 0.8rem;
            color: {COLORS['text_muted']};
            margin-top: 0.1rem;
        }}
        .candidate-card-score {{ font-size: 0.85rem; }}
        .candidate-card-skills {{
            margin-top: 0.85rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }}

        .skill-chip {{
            background: {COLORS['bg_elevated']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['text_secondary']};
            font-size: 0.72rem;
            font-weight: 500;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
        }}
        .skill-chip-muted {{ color: {COLORS['text_muted']}; }}

        .progress-label-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: {COLORS['text_secondary']};
            margin-bottom: 0.25rem;
        }}
        .progress-label-value {{
            font-weight: 700;
            color: {COLORS['text_primary']};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )