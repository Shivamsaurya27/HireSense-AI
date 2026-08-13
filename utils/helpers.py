"""
utils/helpers.py
------------------
Shared Utility Functions for HireSense AI.

Generic, dependency-light helpers used across the ml/ modules and (later)
the report generator / UI layer: file validation, safe JSON I/O,
dataclass serialization, text formatting, batching, logging setup, and
small performance/debugging utilities.

Nothing in this file is specific to resumes/ATS logic — that logic lives
in the ml/ modules. This is the shared plumbing.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
import functools
from dataclasses import is_dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, TypeVar


# ============================================================
# Logging setup
# ============================================================

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Create (or fetch) a configured logger. Safe to call multiple times
    for the same name — won't duplicate handlers.

    Every ml/ module can call this instead of repeating boilerplate
    `logging.basicConfig` calls.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ============================================================
# File validation
# ============================================================

# Default limits — callers can override per-call.
DEFAULT_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
DEFAULT_MAX_FILE_SIZE_MB = 10


def validate_file(
    file_path: str,
    allowed_extensions: Optional[set[str]] = None,
    max_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
) -> tuple[bool, Optional[str]]:
    """Validate that a file exists, has an allowed extension, is non-empty,
    and is under the size limit. Returns (is_valid, error_message).

    This is intentionally separate from `resume_parser.py`'s internal
    checks so it can also be used as an early, fast pre-flight check
    (e.g. right after a file upload) before attempting full parsing.
    """
    allowed_extensions = allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Path is not a file: {file_path}"

    extension = path.suffix.lower()
    if extension not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        return False, f"Unsupported file type '{extension}'. Allowed types: {allowed_str}"

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return False, "File is empty (0 bytes)."

    max_size_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_size_bytes:
        actual_mb = size_bytes / (1024 * 1024)
        return False, f"File too large ({actual_mb:.1f}MB). Maximum allowed: {max_size_mb}MB."

    return True, None


def get_file_extension(file_path: str) -> str:
    """Return the lowercased file extension including the leading dot
    (e.g. '.pdf'). Returns '' if there is no extension.
    """
    return Path(file_path).suffix.lower()


def human_readable_size(size_bytes: int) -> str:
    """Convert a byte count into a human-readable string (e.g. '1.4 MB')."""
    if size_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Strip characters that are unsafe/ambiguous in filenames across
    operating systems, and collapse whitespace. Useful before saving
    generated reports (e.g. "John Doe's Resume?.pdf" -> "John_Does_Resume.pdf").
    """
    if not filename:
        return "unnamed_file"

    name = Path(filename).stem
    extension = Path(filename).suffix

    # Remove characters invalid on Windows/Mac/Linux filesystems.
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", name)
    name = re.sub(r"\s+", replacement, name.strip())
    name = name.strip(f"{replacement}.")

    if not name:
        name = "unnamed_file"

    return f"{name}{extension}"


# ============================================================
# ID generation
# ============================================================

def generate_id(prefix: str = "") -> str:
    """Generate a short, unique identifier (e.g. for candidate_id /
    report_id fields). Uses UUID4 truncated for readability — collision
    risk is negligible at the scale of a college project / single
    session usage.
    """
    short_uuid = uuid.uuid4().hex[:10]
    return f"{prefix}_{short_uuid}" if prefix else short_uuid


def generate_candidate_id(name: str, index: int) -> str:
    """Generate a readable, stable-ish candidate ID from a name and
    positional index (e.g. "john-doe-003"). Preferred over a raw UUID
    when human-readability in logs/reports matters.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "candidate").lower()).strip("-")
    slug = slug or "candidate"
    return f"{slug}-{index:03d}"


# ============================================================
# JSON / serialization helpers
# ============================================================

class _EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that knows how to serialize dataclasses, datetimes,
    sets, and Path objects — the types most likely to show up in this
    project's result objects (ParsedResume, ATSScoreResult, etc.).
    """

    def default(self, o: Any) -> Any:  # noqa: D102
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, set):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses (and nested dataclasses/lists/
    dicts) into plain dicts/lists — handy when a module's own `to_dict()`
    isn't available or when composing multiple result objects together
    before serialization.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """Serialize an object (including dataclasses, dates, sets) to a
    JSON string without raising on the common "not JSON serializable"
    errors that plague ML pipeline objects.
    """
    try:
        return json.dumps(obj, cls=_EnhancedJSONEncoder, indent=indent, ensure_ascii=False)
    except TypeError as exc:
        logging.getLogger("hiresense.helpers").warning(
            "safe_json_dumps fallback triggered: %s", exc
        )
        return json.dumps(str(obj), indent=indent)


def read_json_file(file_path: str) -> tuple[Optional[dict], Optional[str]]:
    """Read and parse a JSON file. Returns (data, error_message) — data
    is None if reading/parsing failed, with the reason in error_message.
    """
    path = Path(file_path)
    if not path.exists():
        return None, f"File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in {file_path}: {exc}"
    except OSError as exc:
        return None, f"Could not read {file_path}: {exc}"


def write_json_file(file_path: str, data: Any, indent: int = 2) -> tuple[bool, Optional[str]]:
    """Write data to a JSON file, creating parent directories if needed.
    Returns (success, error_message).
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(safe_json_dumps(data, indent=indent))
        return True, None
    except OSError as exc:
        return False, f"Could not write {file_path}: {exc}"


# ============================================================
# Text formatting helpers
# ============================================================

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to a maximum length, cutting at the nearest word
    boundary where possible so truncated previews (e.g. in a ranking
    table) don't end mid-word.
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(" ", 1)[0]
    return f"{truncated}{suffix}" if truncated else f"{text[:max_length]}{suffix}"


def format_score(score: float, decimals: int = 1) -> str:
    """Format a 0-100 score consistently for display (e.g. '84.5/100')."""
    clamped = max(0.0, min(100.0, score))
    return f"{clamped:.{decimals}f}/100"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a 0-100 value as a percentage string (e.g. '72.3%')."""
    clamped = max(0.0, min(100.0, value))
    return f"{clamped:.{decimals}f}%"


def calculate_percentage(part: float, whole: float) -> float:
    """Safely compute (part / whole) * 100, returning 0.0 if whole is 0
    instead of raising a ZeroDivisionError.
    """
    if not whole:
        return 0.0
    return (part / whole) * 100.0


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return '1 skill' vs '3 skills' style strings for report/UI text."""
    plural = plural or f"{singular}s"
    word = singular if count == 1 else plural
    return f"{count} {word}"


# ============================================================
# Collection helpers
# ============================================================

T = TypeVar("T")


def chunk_list(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Yield successive chunks of a list — useful for batch-processing
    large candidate pools (e.g. to avoid loading hundreds of resumes'
    parsed text into memory at once).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    """Remove duplicates from a list while preserving original order
    (Python sets don't guarantee order; this does).
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_get(dictionary: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse a nested dict without raising KeyError:
    safe_get(data, "components", "skills", "raw_score", default=0)
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ============================================================
# Performance / debugging helpers
# ============================================================

def timer(func: Callable) -> Callable:
    """Decorator that logs how long a function took to run. Useful during
    development to spot slow parsing/scoring steps (e.g. large PDFs,
    big candidate batches) without adding manual timing code everywhere.
    """
    logger = logging.getLogger("hiresense.timer")

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s took %.3fs", func.__qualname__, elapsed)
        return result

    return wrapper


def get_timestamp() -> str:
    """Return a filesystem-safe timestamp string for naming generated
    files (e.g. reports): '2026-08-13_14-32-05'.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    # Simple manual smoke test when run directly:
    #   python utils/helpers.py
    print("format_score:", format_score(84.567))
    print("format_percentage:", format_percentage(72.3))
    print("calculate_percentage(3, 7):", calculate_percentage(3, 7))
    print("truncate_text:", truncate_text("This is a fairly long resume summary sentence.", 20))
    print("sanitize_filename:", sanitize_filename("John Doe's Resume?.pdf"))
    print("generate_candidate_id:", generate_candidate_id("Alice Smith", 1))
    print("human_readable_size:", human_readable_size(2_500_000))
    print("pluralize:", pluralize(1, "skill"), "|", pluralize(4, "skill"))
    print("chunked:", list(chunk_list([1, 2, 3, 4, 5], 2)))
    print("dedup:", deduplicate_preserve_order(["python", "sql", "python", "java"]))
    print("safe_get:", safe_get({"a": {"b": 5}}, "a", "b"), "|", safe_get({"a": {}}, "a", "c", default="N/A"))
    print("safe_json_dumps:", safe_json_dumps({"score": 88.5, "when": datetime.now()}))

    @timer
    def slow_function():
        time.sleep(0.05)
        return "done"

    print("timer decorator result:", slow_function())