"""
HireSense AI - Resume Screening & Job Description Pages
==========================================================================
Two page renderers live in this file:

    render()                  -> Resume Screening (multi-file upload)
    render_job_description()  -> Job Description input & extracted fields

REAL PIPELINE INTEGRATION
------------------------------------------------------------------------
This file wires the real ML pipeline instead of the previous mock data:

    ml.resume_parser.ResumeParser      -> extract raw text from PDF/DOCX/TXT
    ml.text_processor.TextProcessor    -> clean text, detect resume sections
                                            (education, skills, projects,
                                            certifications, experience, etc.)
    ml.skill_extractor.SkillExtractor  -> taxonomy-based skill extraction,
                                            matched/missing skills, TF-IDF
                                            keywords
    ml.ats_scorer.ATSScorer            -> weighted compatibility score
                                            (skills, similarity, experience,
                                            education, projects, certifications)

Parser/scorer instances are expensive to build (skills taxonomy load +
regex compilation, NLTK/spaCy setup), so they're created once via
st.cache_resource rather than per-file / per-rerun.

DEPENDENCIES NEEDED FOR THIS TO RUN
------------------------------------------------------------------------
scikit-learn (TF-IDF + cosine similarity), and optionally pdfplumber
or PyPDF2 (PDF parsing), python-docx (DOCX parsing), nltk + its
punkt/stopwords/wordnet data, and a spaCy English model — all of these
degrade gracefully to simpler fallbacks per their own module docstrings
if missing, EXCEPT scikit-learn, which is a hard dependency of
skill_extractor.py and ats_scorer.py.

CANDIDATE NAME NOTE
------------------------------------------------------------------------
Candidate name is now extracted from the resume's actual content when
possible — see `_extract_name_from_resume_text()` — using spaCy PERSON
entities (if a spaCy model is installed) restricted to the resume's
"header block" (the lines before the first detected section, where a
name almost always lives), with a plain-text heuristic as a second
attempt if spaCy isn't available or found nothing there. Filename-based
naming (`_extract_name_from_filename()`) is kept only as the last-resort
fallback when neither of those finds anything plausible.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

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

from ml.resume_parser import ResumeParser
from ml.text_processor import TextProcessor
from ml.skill_extractor import SkillExtractor
from ml.ats_scorer import ATSScorer


# ======================================================================
# CACHED PIPELINE COMPONENTS (built once, reused across reruns)
# ======================================================================

@st.cache_resource
def _get_resume_parser() -> ResumeParser:
    return ResumeParser()


@st.cache_resource
def _get_text_processor() -> TextProcessor:
    return TextProcessor()


@st.cache_resource
def _get_skill_extractor() -> SkillExtractor:
    return SkillExtractor()


@st.cache_resource
def _get_ats_scorer() -> ATSScorer:
    return ATSScorer()


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


def _extract_name_from_filename(filename: str) -> str:
    """
    Generate a readable candidate name from the uploaded filename.

    Examples:
    Shivam_Kumar_Resume2.pdf -> Shivam Kumar
    Rahul-Verma-CV.docx -> Rahul Verma
    """
    name = Path(filename).stem

    name = re.sub(
        r"(resume|cv|final|latest|updated|version|v\d+|\d+)",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split())

    if not name:
        return "Unknown Candidate"
    return name.title()


_BULLET_PREFIX_RE = re.compile(r"^[\-•*‣▪◦●]\s*")


def _parse_entries_with_bullets(section_text: str) -> list[tuple[str, str]]:
    """
    Group a resume section's lines into (title, description) entries.

    detect_sections() only gives us raw lines — it doesn't know that a
    bullet line like '- Implemented data structures...' is a DETAIL of
    the project title above it, not a separate project. This groups
    bullet-prefixed lines into the description of the most recent
    non-bullet line instead of treating every line as its own entry.
    """
    entries: list[list] = []  # list of [title, [desc_lines]]
    for raw_line in section_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _BULLET_PREFIX_RE.match(line) and entries:
            desc_line = _BULLET_PREFIX_RE.sub("", line).strip()
            entries[-1][1].append(desc_line)
        else:
            entries.append([_BULLET_PREFIX_RE.sub("", line).strip(), []])
    return [(title, " ".join(desc_lines)) for title, desc_lines in entries]


def _generate_strengths(
    skill_count: int,
    certifications: list[str],
    projects: list[tuple[str, str]],
    experience_match: float,
    education_match: float,
    scored_against_jd: bool,
) -> list[str]:
    """
    Derive strengths from real extracted signals — no random pool. Every
    line here traces back to something actually detected on the resume
    (or, for the JD-relative lines, a real component score).
    """
    strengths: list[str] = []

    if skill_count >= 5:
        strengths.append(f"Strong technical skill set — {skill_count} relevant skills identified")
    elif skill_count >= 1:
        strengths.append(f"{skill_count} relevant skill{'s' if skill_count != 1 else ''} identified on the resume")

    if certifications:
        strengths.append(f"Holds {len(certifications)} certification{'s' if len(certifications) != 1 else ''}")

    if len(projects) >= 2:
        strengths.append(f"Hands-on experience shown across {len(projects)} projects")
    elif len(projects) == 1:
        strengths.append("At least one project demonstrating applied experience")

    if scored_against_jd and experience_match >= 60:
        strengths.append("Experience level aligns well with the job description")
    if scored_against_jd and education_match >= 60:
        strengths.append("Education background meets the role's stated requirement")

    if not strengths:
        strengths.append("No standout strengths could be determined from the extracted resume data")

    return strengths[:4]


def _generate_improvements(
    missing_skills: list[str],
    certifications: list[str],
    projects: list[tuple[str, str]],
    experience_match: float,
    education_match: float,
    scored_against_jd: bool,
) -> list[str]:
    """Derive improvement areas from real gaps — no random pool."""
    improvements: list[str] = []

    if scored_against_jd and missing_skills:
        shown = ", ".join(missing_skills[:5])
        improvements.append(f"Missing skills required by the job description: {shown}")
    if not certifications:
        improvements.append("No certifications detected on the resume")
    if not projects:
        improvements.append("No project section detected on the resume")
    if scored_against_jd and experience_match < 40:
        improvements.append("Experience level appears below what the job description expects")
    if scored_against_jd and education_match < 40:
        improvements.append("Education background appears below what the job description expects")

    if not improvements:
        improvements.append("No notable gaps identified from the extracted resume data")

    return improvements[:4]


def _status_from_score(score: float) -> str:
    """Same thresholds used across the app's status badges."""
    if score >= 80:
        return "shortlisted"
    if score >= 50:
        return "in review"
    return "rejected"


_NAME_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){1,3}$")


def _looks_like_a_name(line: str) -> bool:
    """
    Heuristic check for 'is this line plausibly a person's name' — used
    only as a fallback when spaCy isn't available. Deliberately narrow:
    2-4 words, letters only (plus . ' -), no digits, no email/URL. This
    will still occasionally misfire on a job-title line (e.g. 'Data
    Analyst') sitting right under the name — that's a known limitation
    of a regex heuristic without real NER.
    """
    line = line.strip()
    if "@" in line or "http" in line.lower() or "www." in line.lower():
        return False
    if any(ch.isdigit() for ch in line):
        return False
    return bool(_NAME_LINE_RE.match(line))


def _extract_name_from_resume_text(raw_text: str, entities: dict) -> Optional[str]:
    """
    Try to find the candidate's actual name from the resume content
    itself, rather than the filename:

      1. Build the resume's "header block" — the lines before the first
         detected section header (SUMMARY/EXPERIENCE/EDUCATION/etc.),
         which is where a name + contact info normally lives.
      2. Prefer a spaCy PERSON entity that appears inside that header
         block (most reliable — real NER, not a regex guess).
      3. Otherwise, fall back to checking whether the very first
         header-block line looks like a plain name.

    Returns None (letting the caller fall back to the filename) if
    nothing plausible is found — e.g. spaCy isn't installed AND the
    first line doesn't look name-like.
    """
    if not raw_text:
        return None

    header_lines: list[str] = []
    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if TextProcessor._match_section_header(stripped):
            break
        header_lines.append(stripped)
        if len(header_lines) >= 6:  # header block is always short
            break

    if not header_lines:
        return None

    header_block_text = "\n".join(header_lines)
    persons = (entities or {}).get("PERSON", [])
    for person in persons:
        if person in header_block_text:
            return person.title() if person.isupper() else person

    if _looks_like_a_name(header_lines[0]):
        candidate = header_lines[0]
        return candidate.title() if candidate.isupper() else candidate

    return None


# ======================================================================
# REAL RESUME PROCESSING (replaces _mock_process_resume)
# ======================================================================

def _process_resume(uploaded_file) -> dict:
    """
    Run the real pipeline on one uploaded file:
      resume_parser -> text_processor -> skill_extractor (via ats_scorer)

    Returns a dict consumed by the results table AND by the Candidate
    Details page. Never raises — parsing/scoring failures are captured
    in 'status' / 'errors' instead of crashing the whole batch.
    """
    parser = _get_resume_parser()
    text_processor = _get_text_processor()
    scorer = _get_ats_scorer()

    filename = uploaded_file.name
    candidate_name = _extract_name_from_filename(filename)
    jd_text = st.session_state.job_description_text or ""

    parsed = parser.parse_bytes(uploaded_file.getvalue(), filename)

    if not parsed.success:
        return {
            "filename": filename,
            "candidate_name": candidate_name,
            "status": "Failed",
            "score": 0.0,
            "matched_skills": [],
            "missing_skills": [],
            "detected_skills": [],
            "scored_against_jd": False,
            "skills_match": 0.0,
            "experience_match": 0.0,
            "education_match": 0.0,
            "education": None,
            "certifications": [],
            "projects": [],
            "strengths": [],
            "improvements": [],
            "errors": parsed.errors,
        }

    processed = text_processor.process(parsed.raw_text)
    score_result = scorer.calculate_ats_score(parsed.raw_text, jd_text)

    # Prefer a name extracted from the resume content; fall back to the
    # filename-based guess only if nothing plausible was found in the text.
    content_name = _extract_name_from_resume_text(parsed.raw_text, processed.entities)
    if content_name:
        candidate_name = content_name

    skills_component = score_result.components.get("skills")
    experience_component = score_result.components.get("experience")
    education_component = score_result.components.get("education")

    matched_skills = [
        s.title() for s in (skills_component.details.get("matched_skills", []) if skills_component else [])
    ]
    missing_skills = [
        s.title() for s in (skills_component.details.get("missing_skills", []) if skills_component else [])
    ]

    # matched_skills is only meaningful when there's a JD to match against —
    # with no JD, match_skills() intersects against an empty required set and
    # always returns []. Separately extract the resume's OWN skills (no JD
    # needed) so we still have something real to show in that case.
    scored_against_jd = bool(jd_text.strip())
    skill_extractor = _get_skill_extractor()
    detected_skills = [
        s.title() for s in skill_extractor.extract_skills(parsed.raw_text).skill_names
    ]

    education_text = processed.sections.get("education", "").strip() or None
    certifications = [
        line.strip()
        for line in processed.sections.get("certifications", "").split("\n")
        if line.strip()
    ]
    # Bullet-aware grouping — a '- detail line' under a project title is
    # that project's description, not a separate project entry.
    projects = _parse_entries_with_bullets(processed.sections.get("projects", ""))

    strengths = _generate_strengths(
        skill_count=len(detected_skills),
        certifications=certifications,
        projects=projects,
        experience_match=round(experience_component.raw_score, 1) if experience_component else 0.0,
        education_match=round(education_component.raw_score, 1) if education_component else 0.0,
        scored_against_jd=scored_against_jd,
    )
    improvements = _generate_improvements(
        missing_skills=missing_skills,
        certifications=certifications,
        projects=projects,
        experience_match=round(experience_component.raw_score, 1) if experience_component else 0.0,
        education_match=round(education_component.raw_score, 1) if education_component else 0.0,
        scored_against_jd=scored_against_jd,
    )

    return {
        "filename": filename,
        "candidate_name": candidate_name,
        "status": "Completed",
        "score": score_result.total_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "detected_skills": detected_skills,
        "scored_against_jd": scored_against_jd,
        "skills_match": round(skills_component.raw_score, 1) if skills_component else 0.0,
        "experience_match": round(experience_component.raw_score, 1) if experience_component else 0.0,
        "education_match": round(education_component.raw_score, 1) if education_component else 0.0,
        "education": education_text,
        "certifications": certifications,
        "projects": projects,
        "strengths": strengths,
        "improvements": improvements,
        "errors": parsed.warnings,  # non-fatal — parsing succeeded despite these
    }


# ======================================================================
# REAL JOB DESCRIPTION EXTRACTION (replaces _mock_extract_jd_fields)
# ======================================================================

_JD_EXPERIENCE_PATTERN = re.compile(
    r"\d{1,2}\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience", re.IGNORECASE
)

_JD_EDUCATION_KEYWORDS = [
    "phd", "doctorate", "master", "m.tech", "mtech", "msc", "mba",
    "bachelor", "b.tech", "btech", "bsc", "b.e.", "diploma",
]


def _extract_jd_fields(jd_text: str) -> dict:
    """
    Real extraction — no random sampling:
      - required_skills / keywords -> ml.skill_extractor (taxonomy match + TF-IDF)
      - experience  -> first explicit '<n> years of experience' phrase found
      - education   -> first sentence mentioning a recognized degree keyword

    Falls back to 'Not specified' (rather than a random guess) when
    nothing is found — an honest empty state beats a fabricated one.
    """
    extractor = _get_skill_extractor()
    text_processor = _get_text_processor()

    skills_result = extractor.extract_skills(jd_text)

    experience_match = _JD_EXPERIENCE_PATTERN.search(jd_text)
    experience = experience_match.group(0) if experience_match else "Not specified"

    education = "Not specified"
    for sentence in text_processor.tokenize_sentences(jd_text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in _JD_EDUCATION_KEYWORDS):
            education = sentence.strip()
            break

    return {
        "required_skills": [s.title() for s in skills_result.skill_names],
        "keywords": skills_result.keywords[:6],
        "experience": experience,
        "education": education,
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
            processed.append(_process_resume(f))

        progress_bar.empty()
        st.session_state.uploaded_resumes = processed

        failed = [r for r in processed if r["status"] == "Failed"]
        succeeded = total - len(failed)
        if failed:
            render_error(
                f"{len(failed)} of {total} resume(s) could not be parsed "
                f"({', '.join(r['filename'] for r in failed)}). See details below."
            )
        if succeeded:
            render_success(f"Successfully screened {succeeded} resume(s).")

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
            if r["status"] == "Failed":
                render_status_badge("rejected")
            else:
                render_status_badge(_status_from_score(r["score"]))
        with row[3]:
            render_score_badge(r["score"])
        with row[4]:
            if r["status"] != "Failed":
                if st.button("View →", key=f"view_{r['filename']}", use_container_width=True):
                    st.session_state.selected_candidate = r["candidate_name"]
                    st.session_state.active_page = "candidate_details"
                    st.rerun()

        if r["status"] == "Failed" and r.get("errors"):
            st.markdown(
                f"<div style='font-size:0.78rem; color:{COLORS['danger']}; margin:-0.4rem 0 0.6rem 0;'>"
                f"⚠ {'; '.join(r['errors'])}</div>",
                unsafe_allow_html=True,
            )

    render_divider()
    st.caption(
        "Scores and skill matches are computed by the real ml/ pipeline "
        "(resume_parser → text_processor → skill_extractor → ats_scorer)."
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
            st.session_state.jd_fields = _extract_jd_fields(jd_text)
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
        if fields["required_skills"]:
            skills_html = "".join(f'<span class="skill-chip" style="margin:0.2rem;">{s}</span>' for s in fields["required_skills"])
            st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{COLORS['text_muted']}; font-size:0.85rem;'>No recognized skills found in this job description.</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-subtitle' style='font-weight:700; margin-bottom:0.5rem;'>Keywords</div>", unsafe_allow_html=True)
        if fields["keywords"]:
            kw_html = "".join(f'<span class="badge badge-accent" style="margin:0.2rem;">{k}</span>' for k in fields["keywords"])
            st.markdown(f"<div>{kw_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{COLORS['text_muted']}; font-size:0.85rem;'>No standout keywords detected.</div>", unsafe_allow_html=True)

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
        "Requirements are extracted by ml/skill_extractor.py (taxonomy skill "
        "matching + TF-IDF keywords) plus lightweight pattern matching for "
        "experience/education phrasing."
    )
