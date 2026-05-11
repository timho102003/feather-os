"""Comment-preserving writer for Feather config YAML.

Uses ``ruamel.yaml`` in round-trip mode so existing comments, blank
lines, and key ordering survive a write. Writes are atomic via
tmp-file + rename so an interrupted write never leaves a half-written
``app.yaml`` on disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def write_yaml_value(file: Path, yaml_path: list[str], value: Any) -> None:
    """Write ``value`` into ``file`` at the dotted ``yaml_path``.

    If ``file`` does not exist, it is created with an empty mapping
    before the write. Intermediate mappings are created as needed
    (so a first write of a nested leaf works on a sparse global
    overlay). Comments and ordering on existing nodes are preserved.

    Args:
        file: Target YAML file.
        yaml_path: Sequence of keys leading to the leaf.
        value: New scalar / list value to write.

    Raises:
        ValueError: If ``yaml_path`` is empty.
    """

    if not yaml_path:
        raise ValueError("yaml_path must be non-empty")

    file.parent.mkdir(parents=True, exist_ok=True)
    yaml = _yaml()
    if file.exists():
        data = yaml.load(file.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    cursor: Any = data
    for key in yaml_path[:-1]:
        existing = cursor.get(key) if hasattr(cursor, "get") else None
        if existing is None:
            cursor[key] = {}
        cursor = cursor[key]
    cursor[yaml_path[-1]] = value

    buffer = io.StringIO()
    yaml.dump(data, buffer)
    new_text = buffer.getvalue()

    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(file)


def delete_yaml_value(file: Path, yaml_path: list[str]) -> bool:
    """Remove the leaf at ``yaml_path`` from ``file``.

    Args:
        file: Target YAML file.
        yaml_path: Sequence of keys leading to the leaf.

    Returns:
        True if a value was removed; False if the file or key did not
        exist.
    """

    if not file.exists():
        return False
    yaml = _yaml()
    data = yaml.load(file.read_text(encoding="utf-8")) or {}

    cursor: Any = data
    for key in yaml_path[:-1]:
        if not hasattr(cursor, "__contains__") or key not in cursor:
            return False
        cursor = cursor[key]
    if not hasattr(cursor, "__contains__") or yaml_path[-1] not in cursor:
        return False
    del cursor[yaml_path[-1]]

    buffer = io.StringIO()
    yaml.dump(data, buffer)
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(buffer.getvalue(), encoding="utf-8")
    tmp.replace(file)
    return True
