"""First-run setup + legacy migration (cold-path; never cross-process).

The onboarding wizard (:mod:`wizard`) and the legacy-artifact migration
(:mod:`migration`) live here. Submodules are imported by their deep path,
mirroring the ``core/*`` layout; ``feather.onboarding`` remains as a thin
top-level shim re-exporting :mod:`feather.setup.wizard`.
"""
