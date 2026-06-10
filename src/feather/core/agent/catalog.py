"""Discovery of agent YAML configs for the dispatchable-agents catalog."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from feather.config import load_agent_config
from feather.models import AgentConfig
from feather.resources import iter_packaged_agent_names

if TYPE_CHECKING:
    from feather.paths import FeatherPaths

logger = logging.getLogger(__name__)

_BUILTIN_ROLES = frozenset({"lead", "explore", "research", "validate"})
_CUSTOM_FILENAME_SUFFIX = "-custom"


@dataclass(slots=True, frozen=True)
class AgentCatalogEntry:
    """Metadata for one agent discovered in the config directory.

    ``is_lead`` / ``dispatchable`` are computed from ``role`` and mirror the
    defaults in :class:`CapabilityProfile` (a lead is the only non-dispatchable
    role), so the catalog reflects the same lead/sub-agent distinction the
    factory and running agent use.
    """

    name: str
    role: str
    description: str
    personality: str
    registered_tools: tuple[str, ...] = field(default_factory=tuple)
    is_builtin: bool = False

    @property
    def is_lead(self) -> bool:
        """True when this YAML declares a top-level lead agent."""
        return self.role == "lead"

    @property
    def dispatchable(self) -> bool:
        """True when spawn_agent may dispatch this agent (leads are not)."""
        return self.role != "lead"


class AgentCatalog:
    """Discover agent YAMLs across packaged, global, and project sources.

    Sources are unioned by name; project sources win over global, which
    win over packaged. Broken YAMLs are skipped with a warning so one
    malformed custom file does not crash the lead's prompt build. The
    catalog is scanned on demand — the lead rebuilds its prompt on every
    turn, so the catalog staying up to date without a reload step falls
    out naturally.
    """

    def __init__(self, root: Path, paths: "FeatherPaths | None" = None) -> None:
        self._root = Path(root)
        self._paths = paths

    def list_entries(self) -> list[AgentCatalogEntry]:
        """Return all discovered agents, sorted by name."""

        names = self._discover_names()
        entries: list[AgentCatalogEntry] = []
        for name in sorted(names):
            try:
                config = load_agent_config(self._root, name, paths=self._paths)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "agent_catalog: skipping unreadable agent yaml=%s error=%s",
                    f"{name}.yaml",
                    exc,
                )
                continue
            entries.append(self._to_entry(name, config))
        self._warn_on_naming_mismatch(entries)
        return entries

    def _discover_names(self) -> set[str]:
        """Union the agent names visible across every source.

        Packaged + global discovery is gated on ``paths`` being provided.
        Without ``paths`` the catalog stays project-only — this keeps
        legacy tests (which stage a minimal agents/ tree under tmp_path)
        from inheriting the bundled lead/explore/research/validate
        defaults. Production callers always pass ``paths`` and get the
        full union as a result.
        """

        names: set[str] = set()
        project_agents_dir = (self._root / "config" / "agents").resolve()
        if project_agents_dir.is_dir():
            for path in project_agents_dir.glob("*.yaml"):
                names.add(path.stem)
        if self._paths is not None:
            names.update(iter_packaged_agent_names())
            if self._paths.global_agents_dir.is_dir():
                for path in self._paths.global_agents_dir.glob("*.yaml"):
                    names.add(path.stem)
        return names

    def list_dispatchable(self) -> list[AgentCatalogEntry]:
        """Return catalog entries that can be passed to ``spawn_agent``."""

        return [entry for entry in self.list_entries() if self.is_dispatchable(entry)]

    def list_leads(self) -> list[AgentCatalogEntry]:
        """Return catalog entries that are leads (top-level, switchable).

        These are the agents a multi-lead UI offers as switchable roots; they
        are not dispatchable as sub-agents.
        """

        return [entry for entry in self.list_entries() if entry.is_lead]

    def get(self, agent_name: str) -> AgentCatalogEntry | None:
        """Return one catalog entry by filename slug, or ``None`` if missing."""

        for entry in self.list_entries():
            if entry.name == agent_name:
                return entry
        return None

    @staticmethod
    def is_dispatchable(entry: AgentCatalogEntry) -> bool:
        """Report whether ``entry`` should appear in the spawn_agent allow-list.

        Mirrors ``CapabilityProfile.dispatchable`` (computed at discovery time
        in :meth:`_to_entry`), so a lead is never dispatchable as a sub-agent.
        """

        return entry.dispatchable

    @staticmethod
    def is_valid_name(agent_name: str) -> bool:
        """Allow alnum + ``_`` + ``-`` only, so ``agent_name`` can't traverse paths."""

        if not agent_name:
            return False
        return all(ch.isalnum() or ch in ("_", "-") for ch in agent_name)

    def _to_entry(self, name: str, config: AgentConfig) -> AgentCatalogEntry:
        return AgentCatalogEntry(
            name=name,
            role=config.role,
            description=config.description,
            personality=config.personality,
            registered_tools=tuple(config.registered_tools),
            is_builtin=config.role in _BUILTIN_ROLES,
        )

    def _warn_on_naming_mismatch(self, entries: list[AgentCatalogEntry]) -> None:
        """Log a soft warning when ``role: custom`` files don't follow the naming convention."""

        for entry in entries:
            if entry.role == "custom" and not entry.name.endswith(_CUSTOM_FILENAME_SUFFIX):
                logger.warning(
                    "agent_catalog: custom agent %s.yaml should be named with the "
                    "'%s' suffix (e.g. '%s%s.yaml')",
                    entry.name,
                    _CUSTOM_FILENAME_SUFFIX,
                    entry.name,
                    _CUSTOM_FILENAME_SUFFIX,
                )
            elif entry.role != "custom" and entry.name.endswith(_CUSTOM_FILENAME_SUFFIX):
                logger.warning(
                    "agent_catalog: %s.yaml uses the custom filename suffix but "
                    "declares role=%s; pick one convention to avoid confusion",
                    entry.name,
                    entry.role,
                )


def render_catalog_block(entries: list[AgentCatalogEntry]) -> str:
    """Render a catalog list as a prompt-friendly block. Empty input → empty string."""

    dispatchable = [entry for entry in entries if AgentCatalog.is_dispatchable(entry)]
    if not dispatchable:
        return ""
    lines: list[str] = []
    for entry in dispatchable:
        tag = "builtin" if entry.is_builtin else "custom"
        description = entry.description or "(no description provided)"
        tools = ", ".join(entry.registered_tools) if entry.registered_tools else "(none)"
        lines.append(
            f"- `{entry.name}` [{tag}, role={entry.role}] — {description} | tools: {tools}"
        )
    return "\n".join(lines)
