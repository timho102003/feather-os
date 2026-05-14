"""Attachment parsing, display, and provider-content helpers."""

from __future__ import annotations

import base64
import mimetypes
import shlex
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from feather.models import AttachmentKind, AttachmentRecord, PendingAttachment

MAX_DIRECT_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_INLINE_TEXT_ATTACHMENT_BYTES = 1 * 1024 * 1024
MAX_INLINE_TEXT_ATTACHMENT_CHARS = 120_000
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
SUPPORTED_PDF_MIME = "application/pdf"

_TEXT_EXTENSIONS = {
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_SENSITIVE_FILENAMES = {
    ".env",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
_SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
_SYSTEM_PATH_PREFIXES = (
    Path("/bin"),
    Path("/boot"),
    Path("/dev"),
    Path("/etc"),
    Path("/lib"),
    Path("/lib64"),
    Path("/proc"),
    Path("/root"),
    Path("/run"),
    Path("/sbin"),
    Path("/sys"),
    Path("/usr"),
    Path("/var"),
)


@dataclass(slots=True, frozen=True)
class AttachmentDraft:
    """Dropped-file parse result before persistence."""

    text: str
    attachments: tuple[PendingAttachment, ...]


def parse_attachment_drops(text: str, *, root: Path) -> AttachmentDraft:
    """Detect dropped absolute file paths or file URIs in a user message.

    Args:
        text: Raw composer text.
        root: Workspace root used to resolve relative paths.

    Returns:
        Cleaned user text plus pending attachments. Unknown, missing, and
        relative paths remain in the text so normal workspace path mentions
        are not swallowed as attachments.
    """

    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:
        return AttachmentDraft(text=text.strip(), attachments=())
    if not tokens:
        return AttachmentDraft(text=text.strip(), attachments=())

    kept: list[str] = []
    attachments: list[PendingAttachment] = []
    seen_paths: set[Path] = set()
    for token in tokens:
        path = _path_from_token(token, root=root)
        if path is None:
            recovered_paths, remainder = _recover_concatenated_paths(token, root=root)
            if not recovered_paths:
                kept.append(token)
                continue
            for recovered in recovered_paths:
                if recovered not in seen_paths:
                    attachments.append(_pending_attachment(recovered))
                    seen_paths.add(recovered)
            if remainder:
                kept.append(remainder)
            continue
        if path not in seen_paths:
            attachments.append(_pending_attachment(path))
            seen_paths.add(path)
    return AttachmentDraft(text=" ".join(kept).strip(), attachments=tuple(attachments))


def validate_pending_attachments(attachments: tuple[PendingAttachment, ...]) -> None:
    """Reject unsupported, unsafe, or oversized attachments before persistence."""

    total_bytes = sum(item.size_bytes for item in attachments)
    if total_bytes > MAX_DIRECT_ATTACHMENT_BYTES:
        raise ValueError(
            "Combined attachment size exceeds the 50 MB direct model-input limit."
        )
    for attachment in attachments:
        if attachment.size_bytes > MAX_DIRECT_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachment {attachment.original_name!r} exceeds the 50 MB direct model-input limit."
            )
        if _is_text_attachment(attachment) and attachment.size_bytes > MAX_INLINE_TEXT_ATTACHMENT_BYTES:
            raise ValueError(
                f"Text attachment {attachment.original_name!r} exceeds the 1 MB inline text limit."
            )
        if attachment.kind == AttachmentKind.IMAGE and not _is_supported_image_mime(
            attachment.mime_type
        ):
            raise ValueError(
                "Unsupported image MIME type for direct model input: "
                f"{attachment.original_name} ({attachment.mime_type}). "
                "Supported image types are PNG, JPEG, WEBP, and GIF."
            )
        if not _is_supported_attachment(attachment):
            raise ValueError(
                "Unsupported attachment type for direct model input: "
                f"{attachment.original_name} ({attachment.mime_type}). "
                "Supported types are images, PDFs, and text/code files."
            )


def render_attachment_message(
    text: str,
    attachments: tuple[PendingAttachment, ...] | list[AttachmentRecord],
) -> str:
    """Render user-visible text with compact attachment placeholders."""

    markers = attachment_markers(attachments)
    parts = [text.strip(), " ".join(markers)]
    return "\n".join(part for part in parts if part).strip()


def attachment_markers(
    attachments: tuple[PendingAttachment, ...] | list[AttachmentRecord],
) -> list[str]:
    """Return `[image #n]` / `[File #n]` markers in attachment order."""

    image_index = 0
    file_index = 0
    markers: list[str] = []
    for attachment in attachments:
        if attachment.kind == AttachmentKind.IMAGE:
            image_index += 1
            markers.append(f"[image #{image_index}]")
        else:
            file_index += 1
            markers.append(f"[File #{file_index}]")
    return markers


def build_attachment_content_blocks(
    *,
    root: Path,
    attachments: list[AttachmentRecord],
) -> list[dict[str, str]]:
    """Build Responses-API content blocks for saved attachments."""

    blocks: list[dict[str, str]] = []
    resolved_root = root.resolve()
    total_bytes = sum(item.size_bytes for item in attachments)
    if total_bytes > MAX_DIRECT_ATTACHMENT_BYTES:
        raise ValueError(
            "Combined attachment size exceeds the 50 MB direct model-input limit."
        )
    for attachment in attachments:
        if attachment.size_bytes > MAX_DIRECT_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachment {attachment.original_name!r} exceeds the 50 MB direct model-input limit."
            )
        path = (resolved_root / attachment.filepath).resolve()
        if not _is_under(path, resolved_root):
            raise ValueError(f"Attachment path escapes workspace: {attachment.filepath}")
        if not path.is_file():
            raise ValueError(f"Attachment file is missing: {attachment.filepath}")
        if attachment.kind == AttachmentKind.IMAGE:
            if not _is_supported_image_mime(attachment.mime_type):
                raise ValueError(
                    "Unsupported image MIME type for direct model input: "
                    f"{attachment.original_name} ({attachment.mime_type})."
                )
            data_url = _data_url(path, attachment.mime_type)
            blocks.append({"type": "input_image", "image_url": data_url})
            continue
        if attachment.mime_type == SUPPORTED_PDF_MIME:
            data_url = _data_url(path, attachment.mime_type)
            blocks.append(
                {
                    "type": "input_file",
                    "filename": attachment.original_name,
                    "file_data": data_url,
                }
            )
            continue
        if _is_text_record(attachment, path):
            blocks.append(
                {
                    "type": "input_text",
                    "text": _inline_text_attachment(path, attachment),
                }
            )
            continue
        raise ValueError(
            "Unsupported attachment type for direct model input: "
            f"{attachment.original_name} ({attachment.mime_type})."
        )
    return blocks


def _path_from_token(token: str, *, root: Path) -> Path | None:
    raw = token.strip()
    if not raw:
        return None
    is_file_uri = raw.startswith("file://")
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        raw = unquote(parsed.path)
    # ``Path.expanduser()`` raises ``RuntimeError("Could not determine home
    # directory.")`` when ``$HOME`` is unset (Python's pathlib does not
    # silently degrade — see ``_PosixFlavour.gethomedir``). Sub-agent
    # subprocesses occasionally launch with a stripped environment (CI
    # sandboxes, some container init systems) and would otherwise crash
    # on every inbound user message that happens to contain a ``~`` —
    # this codepath is exercised for ALL incoming text, not just text the
    # user intended as an attachment. Treat the failure as "this token
    # cannot be resolved into a file on this system" — exactly the
    # contract the rest of the function returns ``None`` for.
    try:
        candidate = Path(raw).expanduser()
    except RuntimeError:
        return None
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    if not _is_safe_attachment_path(resolved, root=root.resolve(), from_file_uri=is_file_uri):
        return None
    return resolved


def _recover_concatenated_paths(token: str, *, root: Path) -> tuple[list[Path], str]:
    """Recover path tokens accidentally pasted without separators.

    Some terminal drag/drop flows can duplicate a path as
    ``/path/file.pdf/path/file.pdf``. Treat only safe, existing prefixes as
    attachments and leave any non-path suffix as text.
    """

    rest = token
    paths: list[Path] = []
    while rest:
        match = _longest_existing_path_prefix(rest, root=root)
        if match is None:
            break
        recovered, consumed_chars = match
        if consumed_chars <= 0:
            break
        paths.append(recovered)
        rest = rest[consumed_chars:]
    return paths, rest.strip()


def _longest_existing_path_prefix(token: str, *, root: Path) -> tuple[Path, int] | None:
    raw = token.strip()
    if not raw.startswith(("/", "file://")):
        return None
    matches: list[tuple[Path, int]] = []
    for index, char in enumerate(raw):
        if char != "/":
            continue
        candidate = _path_from_token(raw[:index], root=root)
        if candidate is not None:
            matches.append((candidate, index))
    candidate = _path_from_token(raw, root=root)
    if candidate is not None:
        matches.append((candidate, len(raw)))
    if not matches:
        return None
    return max(matches, key=lambda match: match[1])


def _pending_attachment(path: Path) -> PendingAttachment:
    mime_type = _guess_mime_type(path)
    kind = AttachmentKind.IMAGE if mime_type.startswith("image/") else AttachmentKind.FILE
    return PendingAttachment(
        source_path=str(path),
        kind=kind,
        mime_type=mime_type,
        original_name=path.name,
        size_bytes=path.stat().st_size,
    )


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _data_url(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _is_safe_attachment_path(path: Path, *, root: Path, from_file_uri: bool) -> bool:
    if _is_under(path, root):
        return not _has_sensitive_component(path.relative_to(root).parts)
    if not from_file_uri:
        return False
    if any(path == prefix or prefix in path.parents for prefix in _SYSTEM_PATH_PREFIXES):
        return False
    return not _has_sensitive_component(path.parts)


def _has_sensitive_component(parts: tuple[str, ...]) -> bool:
    lowered = [part.lower() for part in parts]
    if any(part in {".ssh", ".gnupg", ".aws", ".config", ".gcloud"} for part in lowered):
        return True
    name = lowered[-1] if lowered else ""
    suffix = Path(name).suffix
    return (
        name in _SENSITIVE_FILENAMES
        or name.startswith(".env")
        or suffix in _SENSITIVE_SUFFIXES
    )


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_supported_attachment(attachment: PendingAttachment) -> bool:
    return (
        (
            attachment.kind == AttachmentKind.IMAGE
            and _is_supported_image_mime(attachment.mime_type)
        )
        or attachment.mime_type == SUPPORTED_PDF_MIME
        or _is_text_attachment(attachment)
    )


def _is_supported_image_mime(mime_type: str) -> bool:
    return mime_type.lower() in SUPPORTED_IMAGE_MIME_TYPES


def _is_text_attachment(attachment: PendingAttachment) -> bool:
    return attachment.mime_type.startswith("text/") or Path(
        attachment.original_name
    ).suffix.lower() in _TEXT_EXTENSIONS


def _is_text_record(attachment: AttachmentRecord, path: Path) -> bool:
    return attachment.mime_type.startswith("text/") or path.suffix.lower() in _TEXT_EXTENSIONS


def _inline_text_attachment(path: Path, attachment: AttachmentRecord) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_INLINE_TEXT_ATTACHMENT_CHARS:
        text = f"{text[:MAX_INLINE_TEXT_ATTACHMENT_CHARS].rstrip()}\n... [truncated]"
    return (
        f"[File: {attachment.original_name}; mime={attachment.mime_type}; "
        f"path={attachment.filepath}]\n{text}"
    )
