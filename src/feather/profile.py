"""Deterministic user-profile store backed by ``.feather/user.md``.

The store keeps a single markdown file with a YAML frontmatter of
structured fields and a free-form body. It is the canonical place where
the lead agent records stable facts about the user (name, role, ongoing
projects). Unlike semantic memory (Qdrant), this profile is always
included in the prompt, so updates are deterministic and immediate.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml


_FRONTMATTER_DELIM = "---"
_RESERVED_FIELDS = frozenset({"created_at", "updated_at"})
_MAX_FILE_BYTES = 16 * 1024
# Sequences that, if persisted into the profile body or a field value, would
# let a hostile user message forge or close the Feather system-prompt frame
# the lead and sub-agents see. Persisted prompt-injection is significantly
# worse than transient injection (it survives sessions and compaction), so
# we reject these unconditionally instead of escaping. The same names are
# used by :class:`feather.core.prompt_builder.PromptBuilder`.
_FORBIDDEN_CONTROL_TOKENS: tuple[str, ...] = (
    "</user_profile>",
    "<user_profile>",
    "</feather_system_prompt>",
    "<feather_system_prompt",
    "<static_cached_prefix>",
    "</static_cached_prefix>",
    "<dynamic_prompt_extensions>",
    "</dynamic_prompt_extensions>",
    "<agent_profile>",
    "</agent_profile>",
    "<available_tools>",
    "</available_tools>",
    "<available_skills>",
    "</available_skills>",
    "<available_mcp_servers>",
    "</available_mcp_servers>",
    "<long_term_memory>",
    "</long_term_memory>",
    "<loaded_skills>",
    "</loaded_skills>",
    "<dispatchable_agents>",
    "</dispatchable_agents>",
)


@dataclass(slots=True, frozen=True)
class UserProfile:
    """Parsed user profile with frontmatter fields and free-form body."""

    fields: Mapping[str, str] = field(default_factory=dict)
    body: str = ""

    @classmethod
    def empty(cls) -> "UserProfile":
        """Return the canonical empty profile."""

        return cls(fields={}, body="")


class UserProfileStore:
    """Read/write ``.feather/user.md`` with structured frontmatter.

    Loads are file-mtime cached so the per-turn prompt build does not
    re-parse YAML when nothing changed. CRUD writes go through an
    asyncio.Lock + atomic-rename to keep the file consistent under
    concurrent agent calls.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        # Cache key is (mtime_ns, st_size); ns precision plus size avoids
        # the same-second collision hazard on filesystems with 1s mtime
        # resolution (HFS+, some network FS), where two writes within the
        # same second would otherwise produce identical mtimes and cause
        # the cache to serve stale content.
        self._cache: tuple[tuple[int, int], UserProfile] | None = None

    @property
    def path(self) -> Path:
        """Return the backing file path."""

        return self._path

    def render(self) -> str:
        """Return the profile file contents verbatim, or ``""`` if absent.

        The prompt builder quotes this directly inside the
        ``<user_profile>`` block, so we keep formatting intact.
        """

        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8").rstrip() + "\n"

    def load(self) -> UserProfile:
        """Load the profile from disk; return an empty profile if absent."""

        if not self._path.exists():
            return UserProfile.empty()
        stat = self._path.stat()
        cache_key = (stat.st_mtime_ns, stat.st_size)
        if self._cache is not None and self._cache[0] == cache_key:
            return self._cache[1]
        text = self._path.read_text(encoding="utf-8")
        profile = self._parse(text)
        self._cache = (cache_key, profile)
        return profile

    @staticmethod
    def _parse(text: str) -> UserProfile:
        """Parse a markdown file with optional YAML frontmatter."""

        stripped = text.lstrip()
        if not stripped.startswith(_FRONTMATTER_DELIM):
            return UserProfile(fields={}, body=text.strip())
        without_leading = stripped[len(_FRONTMATTER_DELIM) :].lstrip("\n")
        end_marker = without_leading.find(f"\n{_FRONTMATTER_DELIM}")
        if end_marker == -1:
            return UserProfile(fields={}, body=text.strip())
        frontmatter_raw = without_leading[:end_marker]
        body = without_leading[end_marker + len(_FRONTMATTER_DELIM) + 1 :].lstrip("\n")
        try:
            parsed = yaml.safe_load(frontmatter_raw) or {}
        except yaml.YAMLError:
            return UserProfile(fields={}, body=text.strip())
        if not isinstance(parsed, dict):
            return UserProfile(fields={}, body=text.strip())
        fields_out: dict[str, str] = {}
        for key, value in parsed.items():
            fields_out[str(key)] = "" if value is None else str(value)
        body_text = body.rstrip()
        return UserProfile(
            fields=fields_out,
            body=body_text + ("\n" if body_text else ""),
        )

    async def create(self, field_name: str, value: str) -> None:
        """Create a new structured field. Raises if it already exists."""

        self._reject_control_sequences(value)
        async with self._lock:
            self._mutate(create={field_name: value})

    async def update(self, field_name: str, value: str) -> None:
        """Update an existing structured field. Raises if absent."""

        self._reject_control_sequences(value)
        async with self._lock:
            self._mutate(update={field_name: value})

    async def delete(self, field_name: str) -> None:
        """Delete an existing structured field. Raises if absent."""

        async with self._lock:
            self._mutate(delete={field_name})

    async def append_note(self, text: str) -> None:
        """Append a dated bullet under the ``## Notes`` section."""

        if not text.strip():
            raise ValueError("note text cannot be empty.")
        self._reject_control_sequences(text)
        async with self._lock:
            self._mutate(note=text.strip())

    @staticmethod
    def _reject_control_sequences(text: str) -> None:
        """Reject text that would forge Feather prompt-frame tags.

        The profile body is dumped verbatim into the system prompt; a
        hostile string containing ``</user_profile>`` (etc.) would let
        the user persist forged frame tags that re-injected on every
        future turn for both the lead and every sub-agent.
        """

        lowered = text.lower()
        for token in _FORBIDDEN_CONTROL_TOKENS:
            if token.lower() in lowered:
                raise ValueError(
                    f"value contains forbidden prompt control sequence "
                    f"`{token}`; reject and ask the user to rephrase."
                )

    def _mutate(
        self,
        *,
        create: Mapping[str, str] | None = None,
        update: Mapping[str, str] | None = None,
        delete: set[str] | None = None,
        note: str | None = None,
    ) -> None:
        """Apply one mutation atomically with a write-then-rename."""

        profile = self.load()
        fields_out = dict(profile.fields)
        body = profile.body
        now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        is_first_write = "created_at" not in fields_out
        if create:
            for key, value in create.items():
                self._reject_reserved(key)
                if key in fields_out:
                    raise ValueError(f"field `{key}` already exists.")
                fields_out[key] = value
        if update:
            for key, value in update.items():
                self._reject_reserved(key)
                if key not in fields_out:
                    raise ValueError(f"field `{key}` does not exist.")
                fields_out[key] = value
        if delete:
            for key in delete:
                self._reject_reserved(key)
                if key not in fields_out:
                    raise ValueError(f"field `{key}` does not exist.")
                del fields_out[key]
        if note:
            body = self._append_note_to_body(body, note, now_iso[:10])
        if is_first_write:
            fields_out["created_at"] = now_iso
        fields_out["updated_at"] = now_iso
        rendered = self._render_file(fields_out, body)
        if len(rendered.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ValueError(
                f"profile would exceed {_MAX_FILE_BYTES} bytes; clean up old notes first."
            )
        self._atomic_write(rendered)
        self._cache = None

    @staticmethod
    def _reject_reserved(key: str) -> None:
        if key in _RESERVED_FIELDS:
            raise ValueError(f"field `{key}` is reserved.")

    @staticmethod
    def _append_note_to_body(body: str, note: str, date_prefix: str) -> str:
        bullet = f"- {date_prefix}: {note}"
        if "## Notes" in body:
            return body.rstrip() + f"\n{bullet}\n"
        prefix = body.rstrip()
        if prefix:
            prefix += "\n\n"
        return f"{prefix}## Notes\n\n{bullet}\n"

    @staticmethod
    def _render_file(fields_in: Mapping[str, str], body: str) -> str:
        ordered_keys = [k for k in fields_in if k not in _RESERVED_FIELDS]
        ordered_keys.extend(k for k in _RESERVED_FIELDS if k in fields_in)
        frontmatter = yaml.safe_dump(
            {k: fields_in[k] for k in ordered_keys},
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ).rstrip()
        body_text = body.rstrip()
        if body_text:
            return f"---\n{frontmatter}\n---\n\n{body_text}\n"
        return f"---\n{frontmatter}\n---\n"

    def _atomic_write(self, content: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self._path)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass
