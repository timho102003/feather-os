"""Skill discovery and loading from layered sources.

The catalog walks one or more "skill source roots" in order; later sources
override earlier ones by skill name. Each root is a tree of
``<skill-name>/SKILL.md`` directories (one level deep). Sources may be:

* Real filesystem paths (:class:`pathlib.Path`) — used for global
  ``~/.feather/skills`` and project ``./.feather/skills`` overrides.
* :class:`importlib.resources.abc.Traversable` resources — used for the
  built-in skills bundled inside the wheel under
  :mod:`feather._resources.skills.built-in`.

Refs declared in a skill's frontmatter are resolved relative to the
``SKILL.md`` file inside the *same* source, so a packaged skill with
``refs: [./reference.md]`` continues to work after pip install.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from collections.abc import Sequence

import yaml

from feather.models import LoadedSkill, SkillMetadata


SkillSource = Path | Traversable


@dataclass(slots=True)
class Frontmatter:
    """Parsed frontmatter for a skill file."""

    data: dict
    body: str


@dataclass(slots=True, frozen=True)
class _SkillEntry:
    """Internal record tying a skill's parsed metadata to its source bundle.

    The catalog keeps these instead of re-walking every source on each
    ``load_skill`` call. ``parent`` is the skill directory (the one
    containing ``SKILL.md``) inside whatever source provided the skill;
    refs resolve relative to ``parent`` so packaged refs stay packaged
    and project refs stay project-local.
    """

    metadata: SkillMetadata
    parent: SkillSource


class SkillCatalog:
    """Discover and load skills from a layered set of sources.

    Args:
        sources: Source roots in priority order. Later sources override
            earlier ones when two skills share a ``name`` in their
            frontmatter. Each source is treated as the parent directory
            of one-level-deep skill bundles
            (``<root>/<skill-name>/SKILL.md``).
    """

    def __init__(self, sources: "SkillSource | Sequence[SkillSource]") -> None:
        if isinstance(sources, (str, Path)) or _looks_like_traversable(sources):
            self._sources: list[SkillSource] = [sources]  # type: ignore[list-item]
        else:
            self._sources = list(sources)  # type: ignore[arg-type]

    @property
    def sources(self) -> tuple[SkillSource, ...]:
        return tuple(self._sources)

    def list_metadata(self) -> list[SkillMetadata]:
        """List metadata for every available skill.

        Returns:
            One :class:`SkillMetadata` per skill, sorted by name. When
            two sources expose a skill with the same ``name``, the one
            from the later source wins.
        """

        return [entry.metadata for entry in self._collect().values()]

    def load_skill(self, skill_name: str) -> LoadedSkill:
        """Load one skill by name and inline any referenced files.

        Args:
            skill_name: Exact skill name from metadata.

        Returns:
            The loaded skill content.

        Raises:
            ValueError: If no source provides a skill with this name.
        """

        entries = self._collect()
        entry = entries.get(skill_name)
        if entry is None:
            raise ValueError(f"Unknown skill: {skill_name}")
        return self._load_entry(entry)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect(self) -> dict[str, _SkillEntry]:
        """Walk every source and return a name → entry map.

        Later sources overwrite earlier ones, so user-installed and
        project-local skills can shadow built-ins by reusing the
        ``name`` field in frontmatter.
        """

        collected: dict[str, _SkillEntry] = {}
        for root in self._sources:
            for skill_dir in _iter_skill_dirs(root):
                skill_md = skill_dir / "SKILL.md"
                if not _is_file(skill_md):
                    continue
                frontmatter = self._parse_frontmatter(_read_text(skill_md))
                metadata = SkillMetadata(
                    name=frontmatter.data["name"],
                    description=frontmatter.data["description"],
                    path=str(skill_md),
                    refs=list(frontmatter.data.get("refs", [])),
                )
                collected[metadata.name] = _SkillEntry(
                    metadata=metadata, parent=skill_dir
                )
        return dict(sorted(collected.items()))

    def _load_entry(self, entry: _SkillEntry) -> LoadedSkill:
        skill_md = entry.parent / "SKILL.md"
        frontmatter = self._parse_frontmatter(_read_text(skill_md))
        sections = [frontmatter.body.strip()]
        for ref in entry.metadata.refs:
            ref_text = _read_text(_resolve_ref(entry.parent, ref))
            sections.append(
                f"## Reference: {Path(ref).name}\n\n{ref_text.strip()}"
            )
        body = "\n\n".join(section for section in sections if section)
        return LoadedSkill(metadata=entry.metadata, content=body)

    def _parse_frontmatter(self, raw: str) -> Frontmatter:
        """Parse YAML frontmatter from a SKILL.md file.

        Args:
            raw: Full markdown file content.

        Returns:
            Parsed frontmatter and markdown body.

        Raises:
            ValueError: If the skill is missing valid frontmatter.
        """

        marker = "---\n"
        if not raw.startswith(marker):
            raise ValueError("Skill file is missing YAML frontmatter.")
        _, rest = raw.split(marker, 1)
        frontmatter_raw, body = rest.split(marker, 1)
        data = yaml.safe_load(frontmatter_raw) or {}
        if "name" not in data or "description" not in data:
            raise ValueError(
                "Skill frontmatter must include `name` and `description`."
            )
        return Frontmatter(data=data, body=body.strip())


# ---------------------------------------------------------------------------
# Source-agnostic helpers
# ---------------------------------------------------------------------------


def _iter_skill_dirs(root: SkillSource):
    """Yield each direct child directory of ``root``, if any."""
    if not _is_dir(root):
        return
    try:
        children = list(root.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return
    for child in children:
        if _is_dir(child):
            yield child


def _is_dir(node: SkillSource) -> bool:
    try:
        return node.is_dir()
    except (FileNotFoundError, NotADirectoryError):
        return False


def _is_file(node: SkillSource) -> bool:
    try:
        return node.is_file()
    except (FileNotFoundError, NotADirectoryError):
        return False


def _read_text(node: SkillSource) -> str:
    return node.read_text(encoding="utf-8")


def _looks_like_traversable(node: object) -> bool:
    """Duck-type check for :class:`importlib.resources.abc.Traversable`."""
    return (
        not isinstance(node, (list, tuple))
        and hasattr(node, "iterdir")
        and hasattr(node, "is_dir")
        and hasattr(node, "joinpath")
    )


def _resolve_ref(parent: SkillSource, ref: str) -> SkillSource:
    """Resolve a ref string relative to a skill's parent directory.

    Refs may begin with ``./`` (treated identically to bare paths) and
    may contain forward-slash separators. Backwards traversal (``..``)
    is rejected to keep packaged-resource semantics safe.
    """
    cleaned = ref.lstrip("./").replace("\\", "/")
    if any(part == ".." for part in cleaned.split("/")):
        raise ValueError(f"Skill ref must not contain '..': {ref!r}")
    target = parent
    for part in cleaned.split("/"):
        if part:
            target = target / part
    return target
