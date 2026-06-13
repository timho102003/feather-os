"""Declarative registry of selectable LLM providers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from feather.providers.base import BaseLLMProvider
from feather.providers.claude_provider import ClaudeMessagesProvider
from feather.providers.openai_provider import OpenAIResponsesProvider
from feather.providers.openrouter_provider import OpenRouterChatProvider

if TYPE_CHECKING:
    from feather.models import AppConfig

__all__ = ("PROVIDER_NAMES", "ProviderSpec", "provider_spec")


@dataclass(slots=True, frozen=True)
class ProviderSpec:
    """Everything dispatch sites need to know about one provider, in one row.

    ``config_block`` returns the provider's app.yaml section (None when the
    operator didn't configure it); callers own their missing-block error
    messages so existing wording is preserved. ``build`` assumes the block
    is present (openai's block always exists on AppConfig).
    """

    name: str
    build: Callable[[Any], BaseLLMProvider]
    config_block: Callable[[Any], Any | None]
    default_model: Callable[[Any], str]
    supports_multimodal: Callable[[Any], bool]


_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        build=lambda cfg: OpenAIResponsesProvider(cfg.openai),
        config_block=lambda cfg: cfg.openai,
        default_model=lambda cfg: cfg.openai.model,
        supports_multimodal=lambda cfg: True,
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        build=lambda cfg: OpenRouterChatProvider(cfg.openrouter),
        config_block=lambda cfg: cfg.openrouter,
        default_model=lambda cfg: (
            cfg.openrouter.model if cfg.openrouter is not None else cfg.openai.model
        ),
        supports_multimodal=lambda cfg: (
            cfg.openrouter.supports_multimodal if cfg.openrouter is not None else False
        ),
    ),
    "claude": ProviderSpec(
        name="claude",
        build=lambda cfg: ClaudeMessagesProvider(cfg.claude),
        config_block=lambda cfg: cfg.claude,
        default_model=lambda cfg: (
            cfg.claude.model if cfg.claude is not None else cfg.openai.model
        ),
        supports_multimodal=lambda cfg: (
            cfg.claude.supports_multimodal if cfg.claude is not None else False
        ),
    ),
}

PROVIDER_NAMES: tuple[str, ...] = tuple(_SPECS)


def provider_spec(name: str) -> ProviderSpec | None:
    """Look up a spec by normalized provider name."""

    return _SPECS.get(name)
