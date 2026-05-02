"""Long-term memory subsystem for Feather.

Provides vector-store-backed memory of user facts, preferences, and decisions
across sessions. Extraction runs as an async background job after each agent
turn; retrieval contextualizes the latest conversation into a query and injects
top-K matches into the agent prompt.

The subsystem is gated on ``QDRANT_URL`` and ``GEMINI_API_KEY`` environment
variables and opt-in per-agent via ``AgentConfig.memory_enabled``. When gated
off, ``NoOpMemoryTrigger`` and ``NoOpMemoryReader`` are wired so ``BaseAgent``
behaves identically to the no-memory baseline.
"""
