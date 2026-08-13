"""
HireSense AI - Intelligent Resume Screening & Candidate Ranking Platform
==========================================================================
Main application shell.

This file is responsible ONLY for:
    - Page configuration
    - Global styling import
    - Sidebar navigation
    - Application branding
    - Routing between page modules

Page modules (ui/dashboard.py, ui/screening.py, etc.) will be generated
in subsequent steps and are imported lazily/defensively below so this
shell runs standalone even before they exist.
"""

import streamlit as st

# --------------------------------------------------------------------
# PAGE CONFIGURATION (must be the first Streamlit command)
# --------------------------------------------------------------------
st.set_page_config(
    page_title="HireSense AI | Intelligent Resume Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------
# GLOBAL STYLING
# --------------------------------------------------------------------
# ui/styles.py will expose `apply_custom_css()` which injects the dark,
# glassmorphic, SaaS-style CSS system (colors, typography, cards, etc.)
try:
    from ui.styles import apply_custom_css
    apply_custom_css()
    from ui.components import inject_component_styles
    inject_component_styles()
except ImportError:
    # Styling module not generated yet — fall back to minimal inline CSS
    # so the shell remains usable while the rest of the app is built out.
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #09090B;
                color: #FAFAFA;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------
# APPLICATION CONSTANTS
# --------------------------------------------------------------------
APP_NAME = "HireSense AI"
APP_TAGLINE = "Find the right talent faster with AI."
APP_ICON = "🧠"

NAV_ITEMS = [
    {"key": "dashboard", "label": "Dashboard", "icon": "📊"},
    {"key": "screening", "label": "Resume Screening", "icon": "📄"},
    {"key": "job_description", "label": "Job Description", "icon": "📝"},
    {"key": "ranking", "label": "Candidate Ranking", "icon": "🏆"},
    {"key": "candidate_details", "label": "Candidate Details", "icon": "👤"},
    {"key": "analytics", "label": "Analytics", "icon": "📈"},
    {"key": "reports", "label": "Reports", "icon": "🧾"},
]

# --------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "dashboard"


def set_active_page(page_key: str) -> None:
    """Callback to update the currently active page in session state."""
    st.session_state.active_page = page_key


# --------------------------------------------------------------------
# SIDEBAR — BRANDING + NAVIGATION
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="app-brand">
            <div class="app-brand-icon">{APP_ICON}</div>
            <div class="app-brand-text">
                <div class="app-brand-name">{APP_NAME}</div>
                <div class="app-brand-tagline">{APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">NAVIGATION</div>', unsafe_allow_html=True)

    for item in NAV_ITEMS:
        is_active = st.session_state.active_page == item["key"]
        button_type = "primary" if is_active else "secondary"
        st.button(
            f"{item['icon']}  {item['label']}",
            key=f"nav_{item['key']}",
            use_container_width=True,
            type=button_type,
            on_click=set_active_page,
            args=(item["key"],),
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-footer">
            <span class="status-dot"></span> System Online
            <br/>
            <span class="sidebar-footer-muted">v0.1.0 · Local Build</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------
# TOP HEADER
# --------------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    active_label = next(
        (item["label"] for item in NAV_ITEMS if item["key"] == st.session_state.active_page),
        "Dashboard",
    )
    st.markdown(
        f"""
        <div class="top-header">
            <h1 class="top-header-title">{active_label}</h1>
            <p class="top-header-subtitle">{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        """
        <div class="top-header-actions">
            <div class="header-badge">🟢 Recruiter Mode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="content-divider"></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------
# ROUTING — render the active page module
# --------------------------------------------------------------------
PAGE_RENDERERS = {}

try:
    from ui.dashboard import render as render_dashboard
    PAGE_RENDERERS["dashboard"] = render_dashboard
except ImportError:
    pass

try:
    from ui.screening import render as render_screening
    PAGE_RENDERERS["screening"] = render_screening
except ImportError:
    pass

try:
    from ui.ranking import render as render_ranking
    PAGE_RENDERERS["ranking"] = render_ranking
except ImportError:
    pass

try:
    from ui.candidate_details import render as render_candidate_details
    PAGE_RENDERERS["candidate_details"] = render_candidate_details
except ImportError:
    pass

try:
    from ui.analytics import render as render_analytics
    PAGE_RENDERERS["analytics"] = render_analytics
except ImportError:
    pass

try:
    from ui.reports import render as render_reports
    PAGE_RENDERERS["reports"] = render_reports
except ImportError:
    pass

# Job description page currently lives inside the screening module's
# workflow, but is routed independently for direct sidebar access.
try:
    from ui.screening import render_job_description as render_job_description
    PAGE_RENDERERS["job_description"] = render_job_description
except ImportError:
    pass


active_page_key = st.session_state.active_page

if active_page_key in PAGE_RENDERERS:
    PAGE_RENDERERS[active_page_key]()
else:
    # Empty state shown until the corresponding ui/ modulhas been generated.
    st.markdown(
        f"""
        <div class="empty-state-card">
            <div class="empty-state-icon">🚧</div>
            <div class="empty-state-title">"{active_page_key.replace('_', ' ').title()}" module not built yet</div>
            <div class="empty-state-subtitle">
                This section will be wired up once <code>ui/{active_page_key}.py</code> is generated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )