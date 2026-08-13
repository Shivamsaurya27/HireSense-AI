"""
HireSense AI - Resume Screening & Job Description Pages
==========================================================================
Two page renderers live in this file:

    render()                  -> Resume Screening (multi-file upload)
    render_job_description()  -> Job Description input & extracted fields

MOCK DATA / MOCK PROCESSING NOTICE
------------------------------------------------------------------------
`_mock_process_resume()` and `_mock_extract_jd_fields()` are placeholder
stand-ins for the real ML pipeline. They exist ONLY so the UI has
something to render during development. Real integration points:

    ml.resume_parser.parse_resume(file)          -> candidate name, raw text
    ml.text_processor.clean_text(raw_text)        -> normalized text
    ml.skill_extractor.extract_skills(text)        -> matched skills
    ml.ats_scorer.score_candidate(resume, jd)      -> compatibility score
    ml.skill_extractor.extract_jd_requirements(jd) -> required skills / keywords /
                                                        experience / education

None of the rendering code below needs to change once those are wired
in — only the two `_mock_*` functions get swapped out.
"""
from __future__ import annotations

from pathlib import Path
import re

import random
import time

import streamlit as st

from ui.styles import COLORS
from ui.components import (
    render_section_header,
    render_label,
    render_badge,
    render_status_badge,
    render_score_badge,
    render_empty_state,
    render_loading_state,
    render_success,
    render_error,
    render_divider,
)


# ======================================================================
# SESSION STATE HELPERS
# ======================================================================

def _init_state() -> None:
    if "uploaded_resumes" not in st.session_state:
        st.session_state.uploaded_resumes = []  # list[dict]
    if "job_description_text" not in st.session_state:
        st.session_state.job_description_text = ""
    if "jd_fields" not in st.session_state:
        st.session_state.jd_fields = None  # dict once "analyzed"


# ======================================================================
# MOCK PROCESSING  (placeholder only — see module docstring)
# ======================================================================

_MOCK_NAMES = [
    "Ananya Sharma", "Rahul Verma", "Priya Nair", "Karan Mehta", "Sara Iqbal",
    "Devika Rao", "Aditya Kulkarni", "Meera Joshi", "Vikram Singh", "Neha Gupta",
]

_MOCK_SKILL_POOL = [
    "Python", "SQL", "TensorFlow", "PyTorch", "AWS", "Docker",
    "NLP", "Pandas", "Power BI", "Java", "React", "MLOps",
]

def _extract_name_from_filename(filename: str) -> str:
    """
    Generate a readable candidate name from the uploaded filename.

    Examples:
    Shivam_Kumar_Resume2.pdf -> Shivam Kumar
    Rahul-Verma-CV.docx -> Rahul Verma
    """

    name = Path(filename).stem

    # Remove common resume-related words
    name = re.sub(
        r"(resume|cv|final|latest|updated|version|v\d+|\d+)",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Replace separators
    name = name.replace("_", " ")
    name = name.replace("-", " ")

    # Remove extra spaces
    name = " ".join(name.split())

    if not name:
        return "Unknown Candidate"

    return name.title()


def _mock_process_resume(filename: str) -> dict:
    """Temporary mock processing until ML pipeline is integrated."""

    rng = random.Random(filename)

    return {
        "filename": filename,
        "candidate_name": _extract_name_from_filename(filename),
        "status": "Completed",
        "score": round(rng.uniform(35, 96), 1),
        "matched_skills": rng.sample(_MOCK_SKILL_POOL, k=rng.randint(3, 6)),
    }


def _mock_extract_jd_fields(jd_text: str) -> dict:
    """Fake keyword / requirement extraction from a pasted job description."""
    rng = random.Random(len(jd_text))
    return {
        "required_skills": rng.sample(_MOCK_SKILL_POOL, k=5),
        "keywords": rng.sample(
            ["scalable", "cross-functional", "data-driven", "agile", "ownership", "stakeholder"],
            k=4,
        ),
        "experience": f"{rng.choice([2, 3, 4, 5])}+ years",
        "education": rng.choice(
            ["Bachelor's in Computer Science or related field", "Master's preferred, Bachelor's required"]
        ),
    }


# ======================================================================
# PAGE: RESUME SCREENING
# ======================================================================

def render() -> None:
    _init_state()

    render_section_header(
        "Resume Upload & Screening",
        "Upload candidate resumes — HireSense AI will parse, score, and rank them automatically.",
        icon="📄",
    )

    if not st.session_state.job_description_text:
        st.markdown(
            f"""
            <div class="glass-card" style="border-color: rgba(245, 158, 11, 0.35); margin-bottom: 1rem;">
                <span style="color:{COLORS['warning']}; font-weight:700;">⚠ No job description set.</span>
                <span style="color:{COLORS['text_secondary']};"> Add one on the
                <b>Job Description</b> page for more accurate compatibility scoring — screening will still work without it using general scoring.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_files = st.file_uploader(
        "Drop resumes here or click to browse",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT",
    )

    upload_col, clear_col = st.columns([1, 1])
    with upload_col:
        process_clicked = st.button(
            "🚀  Screen Resumes", type="primary", use_container_width=True,
            disabled=not uploaded_files,
        )
    with clear_col:
        if st.button("🗑️  Clear All", use_container_width=True):
            st.session_state.uploaded_resumes = []
            st.rerun()

    if process_clicked and uploaded_files:
        progress_bar = st.progress(0, text="Starting screening pipeline…")
        processed = []
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files, start=1):
            progress_bar.progress(
                i / total,
                text=f"Parsing & scoring '{f.name}' ({i}/{total})…",
            )
            time.sleep(0.15)  # simulated processing delay for UX feedback
            processed.append(_mock_process_resume(f.name))

        progress_bar.empty()
        st.session_state.uploaded_resumes = processed
        render_success(f"Successfully screened {total} resume(s).")

    render_divider()

    # ------------------------------------------------------------
    # RESULTS TABLE
    # ------------------------------------------------------------
    render_label("SCREENING RESULTS")

    results = st.session_state.uploaded_resumes

    if not results:
        render_empty_state(
            "No resumes screened yet",
            "Upload one or more resumes above and click 'Screen Resumes' to get started.",
            icon="📤",
        )
        return

    header_cols = st.columns([2.5, 2, 1.5, 1.5, 1.5])
    for col, label in zip(header_cols, ["File", "Candidate", "Status", "Score", ""]):
        col.markdown(f"<div class='section-subtitle' style='font-weight:700;'>{label}</div>", unsafe_allow_html=True)

    for r in results:
        row = st.columns([2.5, 2, 1.5, 1.5, 1.5])
        with row[0]:
            st.markdown(
                f"<div style='font-size:0.85rem; color:{COLORS['text_secondary']};'>📎 {r['filename']}</div>",
                unsafe_allow_html=True,
            )
        with row[1]:
            st.markdown(
                f"<div style='font-size:0.88rem; font-weight:600; color:{COLORS['text_primary']};'>{r['candidate_name']}</div>",
                unsafe_allow_html=True,
            )
        with row[2]:
            render_status_badge("shortlisted" if r["score"] >= 80 else "in review" if r["score"] >= 50 else "rejected")
        with row[3]:
            render_score_badge(r["score"])
        with row[4]:
            if st.button("View →", key=f"view_{r['filename']}", use_container_width=True):
                st.session_state.selected_candidate = r["candidate_name"]
                st.session_state.active_page = "candidate_details"
                st.rerun()

    render_divider()
    st.caption(
        "Scores shown are from a mock pipeline for UI development. "
        "Real parsing/scoring will be powered by the ml/ modules once integrated."
    )


# ======================================================================
# PAGE: JOB DESCRIPTION
# ======================================================================

def render_job_description() -> None:
    _init_state()

    render_section_header(
        "Job Description",
        "Paste the role's job description — HireSense AI extracts the key requirements used for scoring.",
        icon="📝",
    )

    jd_text = st.text_area(
        "Job description",
        value=st.session_state.job_description_text,
        height=280,
        placeholder=(
            "Paste the full job description here…\n\n"
            "e.g. We are looking for a Senior Data Scientist with 5+ years of experience "
            "in Python, machine learning, and cloud platforms (AWS/GCP)…"
        ),
        label_visibility="collapsed",
    )

    action_col, clear_col = st.columns([1, 1])
    with action_col:
        analyze_clicked = st.button(
            "🔍  Analyze Job Description", type="primary", use_container_width=True,
            disabled=not jd_text.strip(),
        )
    with clear_col:
        if st.button("🗑️  Clear", use_container_width=True):
            st.session_state.job_description_text = ""
            st.session_state.jd_fields = None
            st.rerun()

    if analyze_clicked:
        st.session_state.job_description_text = jd_text
        with st.spinner("Extracting requirements…"):
            time.sleep(0.4)  # simulated processing delay for UX feedback
            st.session_state.jd_fields = _mock_extract_jd_fields(jd_text)
        render_success("Job description analyzed.")
    elif jd_text != st.session_state.job_description_text:
        # Keep typed text in sync even if user hasn't clicked Analyze yet
        st.session_state.job_description_text = jd_text

    render_divider()

    fields = st.session_state.jd_fields

    if not fields:
        render_empty_state(
            "No analysis yet",
            "Paste a job description above and click 'Analyze Job Description' to extract requirements.",
            icon="🧾",
        )
        return

    render_label("EXTRACTED REQUIREMENTS")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div class='section-subtitle' style='font-weight:700; margin-bottom:0.5rem;'>Required Skills</div>", unsafe_allow_html=True)
        skills_html = "".join(f'<span class="skill-chip" style="margin:0.2rem;">{s}</span>' for s in fields["required_skills"])
        st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-subtitle' style='font-weight:700; margin-bottom:0.5rem;'>Keywords</div>", unsafe_allow_html=True)
        kw_html = "".join(f'<span class="badge badge-accent" style="margin:0.2rem;">{k}</span>' for k in fields["keywords"])
        st.markdown(f"<div>{kw_html}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom: 0.9rem;">
                <div class="section-subtitle" style="font-weight:700; margin-bottom:0.4rem;">💼 Experience Required</div>
                <div style="font-size:1.1rem; font-weight:700; color:{COLORS['text_primary']};">{fields['experience']}</div>
            </div>
            <div class="glass-card">
                <div class="section-subtitle" style="font-weight:700; margin-bottom:0.4rem;">🎓 Education Required</div>
                <div style="font-size:0.92rem; font-weight:600; color:{COLORS['text_primary']};">{fields['education']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_divider()
    st.caption(
        "Requirements shown are from a mock extraction pipeline for UI development. "
        "Real extraction will be powered by ml/skill_extractor.py once integrated."
    )