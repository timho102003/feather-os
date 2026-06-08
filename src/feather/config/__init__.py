"""Feather configuration subsystem.

Groups what were the scattered ``config_*`` top-level modules: layered YAML
loading (:mod:`loader`), the editable-field registry (:mod:`schema`), the sync
service (:mod:`service`), the comment-preserving writer (:mod:`writer`), config
path resolution (:mod:`resolver`), the ``/config`` slash handler (:mod:`slash`),
the model-picker catalog (:mod:`model_catalog`), dotenv loading (:mod:`env`),
and app path resolution (:mod:`app_paths`).

The loader's public API is re-exported here so ``from feather.config import
load_app_config`` keeps resolving for the many existing callers; every other
submodule is imported by its deep path (mirroring the ``core/*`` layout).
"""

from __future__ import annotations

from feather.config.loader import load_agent_config, load_app_config

__all__ = ("load_agent_config", "load_app_config")
