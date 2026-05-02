# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Distribution rename to **`feather-os`** on PyPI; import name stays `feather`.
- `hatchling` build backend with `hatch-vcs` for git-tag-driven versioning.
- `feather.paths.FeatherPaths`: single chokepoint for global vs project
  path resolution. Walk-up project detection with `~` as the stop boundary.
- `feather.resources`: `importlib.resources` accessors for the bundled
  config / agents / built-in skills.
- Layered config + agent loaders: packaged → project-staged → user-global
  override (deep-merged for `app.yaml`, first-hit-wins for agent YAML).
- Multi-source `SkillCatalog`: packaged + global + project sources, with
  override-by-name semantics and source-aware ref resolution.
- New CLI subcommands: `feather init`, `feather init-memory`,
  `feather stop-memory`, `feather remove-memory [--purge]`. Plus
  `--version`, `--project`, and `FEATHER_HOME` / `FEATHER_PROJECT_ROOT`
  environment overrides.
- Onboarding wizard now skips memory questions when run with
  `feather_paths`: presence of the global memory marker is the opt-in.
- `WriteFileTool` extends its writable whitelist with the global config
  + skills directories when constructed with a `FeatherPaths`.
- PEP 561 `py.typed` marker; PEP 639 SPDX license expression.

### Changed

- Bundled assets (`config/`, built-in skills) moved into
  `src/feather/_resources/` so editable installs and built wheels see
  identical layouts via `importlib.resources`.

### Removed

- `Dockerfile`, `compose.yaml`, `.dockerignore`. The new
  `feather init-memory` shells out to `docker run` directly when
  long-term memory is requested.

### Migration notes

- The previous "clone the repo and run `uv run feather`" workflow still
  works for development. Production users should `pip install feather-os`.
- `feather init-memory` writes `~/.feather/state/memory.json`; the
  onboarding wizard reads that marker. Re-running `feather onboard` no
  longer asks "Enable long-term memory?" — that decision is now the
  init-memory step itself.
