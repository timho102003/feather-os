"""Tests for PDF extraction helpers and tool wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather import pdf as pdf_module
from feather.models import ToolExecutionContext
from feather.pdf import extract_pdf_text
from feather.tools import pdf_tool
from feather.tools.pdf_tool import ReadPdfTool


async def test_read_pdf_tool_calls_extractor_with_workspace_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read_pdf should validate paths and return extractor output."""

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    calls: list[dict[str, object]] = []

    def fake_extract(path: Path, *, mode: str, max_chars: int) -> str:
        calls.append({"path": path, "mode": mode, "max_chars": max_chars})
        return "PDF text"

    monkeypatch.setattr(pdf_tool, "extract_pdf_text", fake_extract)

    result = await ReadPdfTool(tmp_path).execute(
        {"path": "paper.pdf", "mode": "auto", "max_chars": 5000},
        ToolExecutionContext(session_id="session", agent_name="Lead"),
    )

    assert result.output == "file: paper.pdf\nmode: auto\nPDF text"
    assert calls == [{"path": pdf, "mode": "auto", "max_chars": 5000}]


async def test_read_pdf_tool_rejects_non_pdf(tmp_path: Path) -> None:
    """The PDF tool should not become a generic binary file reader."""

    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="only supports .pdf"):
        await ReadPdfTool(tmp_path).execute(
            {"path": "notes.txt", "mode": "auto", "max_chars": 5000},
            ToolExecutionContext(session_id="session", agent_name="Lead"),
        )


def test_extract_pdf_text_hybrid_requires_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid mode should explain missing OpenDataLoader backends."""

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        pdf_module,
        "_run_opendataloader_python",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing python")),
    )
    monkeypatch.setattr(
        pdf_module,
        "_run_opendataloader_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing command")),
    )

    with pytest.raises(ValueError, match="No hybrid PDF extractor available"):
        extract_pdf_text(pdf, mode="opendataloader_hybrid")


def test_extract_pdf_text_auto_uses_pypdf_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should prefer lightweight pypdf for normal text PDFs."""

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(pdf_module, "_run_pypdf", lambda *args, **kwargs: "pypdf text")

    def fail_opendataloader(*args: object, **kwargs: object) -> str:
        raise AssertionError("OpenDataLoader should not be called")

    def fail_pdftotext(*args: object, **kwargs: object) -> str:
        raise AssertionError("pdftotext should not be called")

    monkeypatch.setattr(pdf_module, "_run_opendataloader_python", fail_opendataloader)
    monkeypatch.setattr(pdf_module, "_run_pdftotext", fail_pdftotext)

    assert extract_pdf_text(pdf, mode="auto") == "pypdf text"


def test_extract_pdf_text_auto_does_not_start_opendataloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto mode should stay lightweight and leave OCR/hybrid work explicit."""

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        pdf_module,
        "_run_pypdf",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no text")),
    )
    monkeypatch.setattr(
        pdf_module,
        "_run_pdftotext",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing binary")),
    )

    def fail_opendataloader(*args: object, **kwargs: object) -> str:
        raise AssertionError("OpenDataLoader should require explicit hybrid mode")

    monkeypatch.setattr(pdf_module, "_run_opendataloader_python", fail_opendataloader)

    with pytest.raises(ValueError, match="No PDF text extractor available") as exc:
        extract_pdf_text(pdf, mode="auto")

    assert "mode='opendataloader_hybrid'" in str(exc.value)


def test_extract_pdf_text_auto_falls_back_after_pypdf_parser_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pypdf parser errors should not prevent the next text backend."""

    from pypdf.errors import PdfStreamError

    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    def broken_reader(*args: object, **kwargs: object) -> object:
        raise PdfStreamError("broken stream")

    monkeypatch.setattr("pypdf.PdfReader", broken_reader)
    monkeypatch.setattr(pdf_module, "_run_pdftotext", lambda *args, **kwargs: "pdftotext")

    assert extract_pdf_text(pdf, mode="auto") == "pdftotext"
