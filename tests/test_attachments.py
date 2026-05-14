"""Tests for attachment parsing and provider block generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from feather.attachments import (
    build_attachment_content_blocks,
    parse_attachment_drops,
    render_attachment_message,
    validate_pending_attachments,
)
from feather.models import AttachmentKind, AttachmentRecord


def test_parse_attachment_drops_extracts_files_and_renders_markers(
    tmp_path: Path,
) -> None:
    """Local file paths should become ordered image/file placeholders."""

    image = tmp_path / "chart.png"
    document = tmp_path / "notes.txt"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    document.write_text("hello", encoding="utf-8")

    draft = parse_attachment_drops(
        f"please inspect {image} and {document}",
        root=tmp_path,
    )

    assert draft.text == "please inspect and"
    assert [item.kind for item in draft.attachments] == [
        AttachmentKind.IMAGE,
        AttachmentKind.FILE,
    ]
    assert render_attachment_message(draft.text, draft.attachments) == (
        "please inspect and\n[image #1] [File #1]"
    )


def test_parse_attachment_drops_accepts_file_uri(tmp_path: Path) -> None:
    """Dragged file:// URIs should be treated the same as paths."""

    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4")

    draft = parse_attachment_drops(f"read {document.as_uri()}", root=tmp_path)

    assert draft.text == "read"
    assert len(draft.attachments) == 1
    assert draft.attachments[0].mime_type == "application/pdf"


def test_parse_attachment_drops_requires_file_uri_outside_workspace(
    tmp_path: Path,
) -> None:
    """Plain absolute paths outside the workspace should stay as text."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    plain = parse_attachment_drops(f"read {outside}", root=workspace)
    explicit = parse_attachment_drops(f"read {outside.as_uri()}", root=workspace)

    assert plain.text == f"read {outside}"
    assert plain.attachments == ()
    assert explicit.text == "read"
    assert len(explicit.attachments) == 1


def test_parse_attachment_drops_recovers_duplicate_path_without_separator(
    tmp_path: Path,
) -> None:
    """A repeated terminal drop path should still become one attachment."""

    pdf = tmp_path / "label.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    draft = parse_attachment_drops(f"{pdf}{pdf} read this", root=tmp_path)

    assert draft.text == "read this"
    assert len(draft.attachments) == 1
    assert draft.attachments[0].original_name == "label.pdf"


def test_build_attachment_content_blocks_uses_data_urls(tmp_path: Path) -> None:
    """Saved attachments should become Responses-compatible content blocks."""

    image = tmp_path / "image.png"
    text_file = tmp_path / "notes.txt"
    image.write_bytes(b"png")
    text_file.write_text("hello", encoding="utf-8")
    records = [
        AttachmentRecord(
            id="image-id",
            session_id="session",
            message_id="message",
            kind=AttachmentKind.IMAGE,
            mime_type="image/png",
            original_name="image.png",
            filepath="image.png",
            size_bytes=image.stat().st_size,
            created_at="now",
        ),
        AttachmentRecord(
            id="file-id",
            session_id="session",
            message_id="message",
            kind=AttachmentKind.FILE,
            mime_type="text/plain",
            original_name="notes.txt",
            filepath="notes.txt",
            size_bytes=text_file.stat().st_size,
            created_at="now",
        ),
    ]

    blocks = build_attachment_content_blocks(root=tmp_path, attachments=records)

    assert blocks[0]["type"] == "input_image"
    assert blocks[0]["image_url"].startswith("data:image/png;base64,")
    assert blocks[1]["type"] == "input_text"
    assert "[File: notes.txt; mime=text/plain; path=notes.txt]" in blocks[1]["text"]
    assert "hello" in blocks[1]["text"]


def test_build_attachment_content_blocks_rejects_path_escape(
    tmp_path: Path,
) -> None:
    """Persisted attachment paths must stay inside the workspace."""

    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    record = AttachmentRecord(
        id="file-id",
        session_id="session",
        message_id="message",
        kind=AttachmentKind.FILE,
        mime_type="text/plain",
        original_name="outside.txt",
        filepath="../outside.txt",
        size_bytes=outside.stat().st_size,
        created_at="now",
    )

    try:
        build_attachment_content_blocks(root=tmp_path, attachments=[record])
    except ValueError as exc:
        assert "escapes workspace" in str(exc)
    else:
        raise AssertionError("path escape was accepted")


def test_parse_attachment_drops_does_not_swallow_relative_paths(
    tmp_path: Path,
) -> None:
    """Workspace path mentions should remain normal text unless explicitly dropped."""

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')", encoding="utf-8")

    draft = parse_attachment_drops("edit src/app.py", root=tmp_path)

    assert draft.text == "edit src/app.py"
    assert draft.attachments == ()


def test_parse_attachment_drops_survives_unresolvable_home_when_token_starts_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Path.expanduser`` raises ``RuntimeError("Could not determine home
    directory.")`` when neither ``$HOME`` nor ``pwd.getpwuid`` can supply
    the home dir (Python stdlib documented behaviour). The attachment
    parser runs on EVERY inbound user message — so any sub-agent
    subprocess launched in an environment where both fail would otherwise
    crash deterministically on any message containing a ``~`` token,
    which is exactly the regression observed in
    ``.feather/logs/feather.log`` after spawn_agent failed to propagate
    HOME.

    The parser must treat the failure the same as any other unresolvable
    token: ``~/whatever`` stays in the cleaned text, no attachment is
    extracted, and no exception escapes.

    Monkeypatch ``Path.expanduser`` to raise directly so the test
    reproduces the failure on any host — on Linux with a populated
    ``/etc/passwd`` the env-only path falls back to ``pwd``, which would
    make a pure ``delenv("HOME")`` flaky.
    """

    def _raise(_self: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", _raise)

    # Sanity: the patch fires on the call the parser makes.
    with pytest.raises(RuntimeError):
        Path("~/x").expanduser()

    draft = parse_attachment_drops("please open ~/Desktop/notes.pdf", root=tmp_path)

    # No crash, the tilde token survives in the cleaned text, no attachment.
    assert "~/Desktop/notes.pdf" in draft.text
    assert draft.attachments == ()


def test_parse_attachment_drops_blocks_sensitive_paths(tmp_path: Path) -> None:
    """Sensitive absolute paths should not be auto-attached."""

    outside = tmp_path / "outside" / ".ssh"
    outside.mkdir(parents=True)
    secret = outside / "id_rsa"
    secret.write_text("secret", encoding="utf-8")
    root = tmp_path / "workspace"
    root.mkdir()

    draft = parse_attachment_drops(f"inspect {secret}", root=root)

    assert draft.text == f"inspect {secret}"
    assert draft.attachments == ()


def test_validate_pending_attachments_rejects_unsupported_files(
    tmp_path: Path,
) -> None:
    """Binary formats that providers cannot read directly should fail early."""

    archive = tmp_path / "payload.zip"
    archive.write_bytes(b"PK\x03\x04")
    draft = parse_attachment_drops(f"inspect {archive}", root=tmp_path)

    try:
        validate_pending_attachments(draft.attachments)
    except ValueError as exc:
        assert "Unsupported attachment type" in str(exc)
    else:
        raise AssertionError("unsupported attachment was accepted")


def test_validate_pending_attachments_rejects_unsupported_image_mime(
    tmp_path: Path,
) -> None:
    """Image-like files should still be limited to provider-supported formats."""

    vector = tmp_path / "diagram.svg"
    vector.write_text("<svg />", encoding="utf-8")
    draft = parse_attachment_drops(f"inspect {vector}", root=tmp_path)

    try:
        validate_pending_attachments(draft.attachments)
    except ValueError as exc:
        assert "Unsupported image MIME type" in str(exc)
    else:
        raise AssertionError("unsupported image MIME type was accepted")
