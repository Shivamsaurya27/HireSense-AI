"""
HireSense AI - Global Styling
==========================================================================
Injects the dark, glassmorphic, gradient-accented "premium SaaS" design
system used across every page of the app (Linear / Vercel / Stripe /
Notion / Framer inspired).

Usage:
    from ui.styles import apply_custom_css
    apply_custom_css()

Also exposes small style constants (COLORS) so other ui/ modules can
build inline HTML snippets (badges, cards, etc.) that stay consistent
with this palette.
"""

import streamlit as st

# --------------------------------------------------------------------
# COLOR SYSTEM — single source of truth, reused by other ui/ modules
# --------------------------------------------------------------------
COLORS = {
    "bg": "#09090B",
    "bg_elevated": "#0F0F12",
    "card": "#18181B",
    "card_hover": "#1F1F23",
    "border": "#27272A",
    "border_hover": "#3F3F46",
    "text_primary": "#FAFAFA",
    "text_secondary": "#A1A1AA",
    "text_muted": "#71717A",
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "accent": "#06B6D4",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}


def apply_custom_css() -> None:
    """Inject the global CSS design system into the Streamlit app."""

    st.markdown(
        f"""
        <style>

        /* ============================================================
           FONTS
        ============================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        code, pre, .mono {{
            font-family: 'JetBrains Mono', monospace;
        }}

        /* ============================================================
           BASE APP / LAYOUT
        ============================================================ */
        .stApp {{
            background: {COLORS['bg']};
            color: {COLORS['text_primary']};
        }}

        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 35%),
                radial-gradient(circle at 85% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 35%),
                {COLORS['bg']};
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        #MainMenu, footer {{ visibility: hidden; }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['border_hover']};
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLORS['primary']}; }}

        /* ============================================================
           TYPOGRAPHY
        ============================================================ */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: {COLORS['text_primary']};
        }}

        p, span, label, div {{
            color: {COLORS['text_secondary']};
        }}

        /* ============================================================
           SIDEBAR
        ============================================================ */
        [data-testid="stSidebar"] {{
            background: {COLORS['bg_elevated']};
            border-right: 1px solid {COLORS['border']};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.25rem;
        }}

        .app-brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0.25rem 1rem 0.25rem;
        }}

        .app-brand-icon {{
            font-size: 1.9rem;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
        }}

        .app-brand-name {{
            font-size: 1.15rem;
            font-weight: 800;
            color: {COLORS['text_primary']};
            letter-spacing: -0.02em;
            line-height: 1.2;
        }}

        .app-brand-tagline {{
            font-size: 0.72rem;
            color: {COLORS['text_muted']};
            line-height: 1.3;
        }}

        .sidebar-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, {COLORS['border']} 20%, {COLORS['border']} 80%, transparent);
            margin: 0.75rem 0;
        }}

        .sidebar-section-label {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: {COLORS['text_muted']};
            padding: 0 0.25rem 0.5rem 0.25rem;
        }}

        .sidebar-footer {{
            font-size: 0.75rem;
            color: {COLORS['text_secondary']};
            padding: 0.5rem 0.25rem;
            line-height: 1.8;
        }}

        .sidebar-footer-muted {{
            color: {COLORS['text_muted']};
            font-size: 0.68rem;
        }}

        .status-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {COLORS['success']};
            box-shadow: 0 0 8px {COLORS['success']};
            margin-right: 4px;
        }}

        /* Sidebar nav buttons */
        [data-testid="stSidebar"] .stButton > button {{
            background: transparent;
            border: 1px solid transparent;
            color: {COLORS['text_secondary']};
            font-weight: 500;
            font-size: 0.88rem;
            text-align: left;
            justify-content: flex-start;
            padding: 0.55rem 0.85rem;
            border-radius: 10px;
            transition: all 0.15s ease;
            box-shadow: none;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {COLORS['card_hover']};
            border-color: {COLORS['border']};
            color: {COLORS['text_primary']};
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.18) 0%, rgba(139, 92, 246, 0.14) 100%);
            border: 1px solid rgba(99, 102, 241, 0.4);
            color: {COLORS['text_primary']};
            font-weight: 600;
        }}

        /* ============================================================
           TOP HEADER
        ============================================================ */
        .top-header-title {{
            font-size: 1.7rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            background: linear-gradient(135deg, {COLORS['text_primary']} 30%, {COLORS['text_secondary']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .top-header-subtitle {{
            font-size: 0.9rem;
            color: {COLORS['text_muted']};
            margin-top: 0;
        }}

        .top-header-actions {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            height: 100%;
            padding-top: 0.5rem;
        }}

        .header-badge {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            color: {COLORS['text_secondary']};
            font-size: 0.78rem;
            font-weight: 500;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
        }}

        .content-divider {{
            height: 1px;
            background: linear-gradient(90deg, {COLORS['border']}, transparent 90%);
            margin: 0.5rem 0 1.75rem 0;
        }}

        /* ============================================================
           GLASS CARDS (generic reusable container)
        ============================================================ */
        .glass-card {{
            background: rgba(24, 24, 27, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {COLORS['border']};
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.2s ease;
        }}

        .glass-card:hover {{
            border-color: {COLORS['border_hover']};
            transform: translateY(-2px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        }}

        /* ============================================================
           METRIC CARDS (st.metric override)
        ============================================================ */
        [data-testid="stMetric"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            transition: all 0.2s ease;
        }}

        [data-testid="stMetric"]:hover {{
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.12);
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_muted']} !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLORS['text_primary']} !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }}

        /* ============================================================
           BUTTONS (main content area)
        ============================================================ */
        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.88rem;
            padding: 0.5rem 1.2rem;
            border: 1px solid {COLORS['border']};
            transition: all 0.15s ease;
        }}

        div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"],
        .main .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            border: none;
            color: white;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }}

        .main .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 6px 22px rgba(99, 102, 241, 0.45);
            transform: translateY(-1px);
        }}

        .main .stButton > button[kind="secondary"] {{
            background: {COLORS['card']};
            color: {COLORS['text_secondary']};
        }}

        .main .stButton > button[kind="secondary"]:hover {{
            border-color: {COLORS['border_hover']};
            color: {COLORS['text_primary']};
        }}

        /* ============================================================
           TABS
        ============================================================ */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {COLORS['card']};
            border-radius: 12px;
            padding: 4px;
            border: 1px solid {COLORS['border']};
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            color: {COLORS['text_muted']};
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
            color: white !important;
        }}

        /* ============================================================
           EXPANDER
        ============================================================ */
        [data-testid="stExpander"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ============================================================
           PROGRESS BAR
        ============================================================ */
        [data-testid="stProgress"] > div > div {{
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
            border-radius: 8px;
        }}

        [data-testid="stProgress"] {{
            background: {COLORS['card']};
            border-radius: 8px;
        }}

        /* ============================================================
           DATAFRAME / TABLE
        ============================================================ */
        [data-testid="stDataFrame"] {{
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ============================================================
           INPUTS
        ============================================================ */
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div {{
            background: {COLORS['card']} !important;
            border: 1px solid {COLORS['border']} !important;
            border-radius: 10px !important;
            color: {COLORS['text_primary']} !important;
        }}

        .stTextInput input:focus,
        .stTextArea textarea:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        }}

        /* ============================================================
           FILE UPLOADER
        ============================================================ */
        [data-testid="stFileUploader"] {{
            background: {COLORS['card']};
            border: 1.5px dashed {COLORS['border_hover']};
            border-radius: 14px;
            padding: 0.75rem;
        }}

        [data-testid="stFileUploader"]:hover {{
            border-color: {COLORS['primary']};
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background: transparent;
        }}

        /* ============================================================
           ALERTS / MESSAGES
        ============================================================ */
        .stAlert {{
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
        }}

        /* ============================================================
           CUSTOM COMPONENT CLASSES (used by ui/components.py etc.)
        ============================================================ */

        /* Generic badge */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            letter-spacing: 0.02em;
        }}

        .badge-success {{ background: rgba(34, 197, 94, 0.14); color: {COLORS['success']}; border: 1px solid rgba(34, 197, 94, 0.3); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.14); color: {COLORS['warning']}; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-danger  {{ background: rgba(239, 68, 68, 0.14); color: {COLORS['danger']}; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-primary {{ background: rgba(99, 102, 241, 0.14); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.3); }}
        .badge-accent  {{ background: rgba(6, 182, 212, 0.14); color: {COLORS['accent']}; border: 1px solid rgba(6, 182, 212, 0.3); }}

        /* Empty state */
        .empty-state-card {{
            background: {COLORS['card']};
            border: 1px dashed {COLORS['border_hover']};
            border-radius: 16px;
            padding: 3.5rem 2rem;
            text-align: center;
            margin-top: 1rem;
        }}

        .empty-state-icon {{
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
        }}

        .empty-state-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            margin-bottom: 0.35rem;
        }}

        .empty-state-subtitle {{
            font-size: 0.85rem;
            color: {COLORS['text_muted']};
        }}

        .empty-state-subtitle code {{
            background: {COLORS['bg_elevated']};
            color: {COLORS['accent']};
            padding: 0.1rem 0.4rem;
            border-radius: 6px;
            font-size: 0.8rem;
        }}

        /* Loading state */
        .loading-state {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: {COLORS['text_muted']};
            font-size: 0.85rem;
            padding: 1rem 0;
        }}

        .loading-spinner {{
            width: 14px;
            height: 14px;
            border: 2px solid {COLORS['border']};
            border-top-color: {COLORS['primary']};
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Candidate rank medals */
        .rank-medal {{
            font-size: 1.15rem;
        }}

        /* Section label used inside pages */
        .section-label {{
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {COLORS['text_muted']};
            margin-bottom: 0.5rem;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )
    