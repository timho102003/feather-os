"""External-service integrations (cold + cross-process).

Each integration is a self-contained subpackage imported by its deep path
(mirroring the ``core/*`` layout): :mod:`feather.integrations.mcp` for remote
Model Context Protocol servers and :mod:`feather.integrations.attachments` for
chat attachment ingest (drops + PDF text extraction).
"""
