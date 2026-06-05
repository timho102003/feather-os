"""PDF text extraction helpers used by tools and attachment indexing."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_PDF_TIMEOUT_SECONDS = 90
_PDF_BACKEND_EXCEPTIONS = (
    ValueError,
    OSError,
    subprocess.SubprocessError,
    KeyError,
    IndexError,
)


def extract_pdf_text(path: Path, *, mode: str = "auto", max_chars: int = 12000) -> str:
    """Extract text from a PDF using the best available local extractor.

    Args:
        path: Local PDF path.
        mode: `auto`, `text`, or `opendataloader_hybrid`.
        max_chars: Maximum returned characters.

    Returns:
        Extracted text.

    Raises:
        ValueError: If no extraction backend is available.
    """

    if path.suffix.lower() != ".pdf":
        raise ValueError("read_pdf only supports .pdf files")
    normalized = (mode or "auto").strip().lower()
    if normalized not in {"auto", "text", "opendataloader_hybrid"}:
        raise ValueError("mode must be one of: auto, text, opendataloader_hybrid")

    errors: list[str] = []
    if normalized == "opendataloader_hybrid":
        try:
            return _run_opendataloader_python(
                path,
                max_chars=max_chars,
                hybrid=True,
            )
        except _PDF_BACKEND_EXCEPTIONS as exc:
            detail = _backend_error("OpenDataLoader hybrid Python backend", exc)
            errors.append(detail)
        try:
            return _run_opendataloader_command(path, max_chars=max_chars)
        except _PDF_BACKEND_EXCEPTIONS as exc:
            detail = _backend_error("OpenDataLoader hybrid command", exc)
            errors.append(detail)
        raise ValueError("No hybrid PDF extractor available. " + " ".join(errors))

    try:
        return _run_pypdf(path, max_chars=max_chars)
    except _PDF_BACKEND_EXCEPTIONS as exc:
        errors.append(_backend_error("pypdf", exc))

    try:
        return _run_pdftotext(path, max_chars=max_chars)
    except _PDF_BACKEND_EXCEPTIONS as exc:
        errors.append(_backend_error("pdftotext", exc))

    raise ValueError(
        "No PDF text extractor available. "
        + " ".join(errors)
        + " For scanned or layout-heavy PDFs, retry with mode='opendataloader_hybrid' "
        "after installing its runtime dependencies."
    )


def _run_opendataloader_python(path: Path, *, max_chars: int, hybrid: bool) -> str:
    try:
        from opendataloader_pdf import convert
    except ImportError as exc:
        raise ValueError(
            "Install dependency `opendataloader-pdf[hybrid]` to enable "
            "OpenDataLoader PDF extraction."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="feather-pdf-") as tmp:
        output_dir = Path(tmp)
        kwargs: dict[str, Any] = {
            "input_path": [str(path)],
            "output_dir": str(output_dir),
            "format": "markdown",
        }
        if hybrid:
            kwargs["hybrid"] = "docling-fast"
        convert(**kwargs)
        text = _read_opendataloader_output(output_dir)
    if not text.strip():
        raise ValueError("OpenDataLoader returned no text.")
    return _truncate(text.strip(), max_chars)


def _read_opendataloader_output(output_dir: Path) -> str:
    chunks: list[str] = []
    for suffix in ("*.md", "*.markdown", "*.txt"):
        for output in sorted(output_dir.rglob(suffix)):
            if output.is_file():
                chunks.append(output.read_text(encoding="utf-8", errors="replace"))
    if not chunks:
        raise ValueError("OpenDataLoader did not write markdown or text output.")
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _run_opendataloader_command(path: Path, *, max_chars: int) -> str:
    template = os.getenv("OPENDATALOADER_PDF_COMMAND", "").strip()
    if not template:
        raise ValueError(
            "Set OPENDATALOADER_PDF_COMMAND to override OpenDataLoader hybrid PDF extraction."
        )
    command = [
        part.format(path=str(path), mode="hybrid")
        for part in shlex.split(template)
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=_PDF_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise ValueError(f"OpenDataLoader PDF command failed: {detail}")
    text = result.stdout.strip()
    if not text:
        raise ValueError("OpenDataLoader PDF command returned no text.")
    return _truncate(text, max_chars)


def _run_pypdf(path: Path, *, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError
    except ImportError as exc:
        raise ValueError("Install dependency `pypdf` to enable fallback PDF text extraction.") from exc

    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for index, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"--- page {index} ---\n{text.strip()}")
    except PyPdfError as exc:
        raise ValueError(str(exc) or type(exc).__name__) from exc
    if not parts:
        raise ValueError("pypdf returned no text.")
    return _truncate("\n\n".join(parts), max_chars)


def _run_pdftotext(path: Path, *, max_chars: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=_PDF_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise ValueError(f"pdftotext failed: {detail}")
    text = result.stdout.strip()
    if not text:
        raise ValueError("pdftotext returned no text.")
    return _truncate(text, max_chars)


def _truncate(text: str, max_chars: int) -> str:
    limit = max(1000, int(max_chars))
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n... [truncated]"


def _backend_error(label: str, exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    return f"{label} failed: {message}"
