"""Discovery of soul presets across packaged, global, and project sources.

A :class:`SoulLibrary` is to souls what :class:`AgentCatalog` is to agents: it
unions YAML definitions from three layers (project > global > packaged) and
hands back :class:`Soul` value objects. Unlike the agent catalog — which gates
packaged discovery on ``paths`` so legacy tests that stage a minimal agents/
tree don't inherit the bundled defaults — the **packaged souls always load**,
because the built-in 20-soul library *is* the product feature.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from feather.core.leads.soul import Soul, load_soul
from feather.resources import iter_packaged_soul_names, packaged_soul_yaml_text

if TYPE_CHECKING:
    from feather.paths import FeatherPaths

logger = logging.getLogger(__name__)

__all__ = ("SoulLibrary",)


class SoulLibrary:
    """Union soul presets from packaged + global + project sources.

    Sources overlay by ``id`` with **project > global > packaged** precedence
    (a full replace — souls are small, no deep-merge). One malformed YAML is
    skipped with a warning so a single bad custom soul never hides the rest.
    Scanned on demand; callers (``/lead souls``, ``GET /api/souls``, lead
    creation) are cold paths, never the agent loop.
    """

    def __init__(self, root: Path, paths: "FeatherPaths | None" = None) -> None:
        self._root = Path(root)
        self._paths = paths

    def list(self) -> list[Soul]:
        """Return every discovered soul, sorted by id."""

        by_id: dict[str, Soul] = {}
        # packaged (always) → global (if paths) → project: later layers win.
        for soul_id in iter_packaged_soul_names():
            self._load_into(by_id, soul_id, packaged_soul_yaml_text(soul_id), source="packaged")
        if self._paths is not None:
            self._load_dir(by_id, self._paths.global_souls_dir, source="global")
        self._load_dir(by_id, self._root / "config" / "souls", source="project")
        return [by_id[key] for key in sorted(by_id)]

    def get(self, soul_id: str) -> Soul | None:
        """Return one soul by id, or ``None`` if absent."""

        for soul in self.list():
            if soul.id == soul_id:
                return soul
        return None

    def _load_dir(self, by_id: dict[str, Soul], directory: Path, *, source: str) -> None:
        if not directory.is_dir():
            return
        for path in directory.glob("*.yaml"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("soul_library: cannot read %s error=%s", path, exc)
                continue
            self._load_into(by_id, path.stem, text, source=source)

    @staticmethod
    def _load_into(by_id: dict[str, Soul], soul_id: str, text: str, *, source: str) -> None:
        try:
            raw = yaml.safe_load(text) or {}
            by_id[soul_id] = load_soul(soul_id, raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "soul_library: skipping unreadable soul id=%s source=%s error=%s",
                soul_id, source, exc,
            )
