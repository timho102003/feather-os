"""Tests for OpenAI Responses API structured-output translation.

Covers :class:`ProviderRequestConfig.response_schema` flowing through
:meth:`OpenAIResponsesProvider._build_request_kwargs` into the ``text.format``
JSON-schema payload with ``strict=True``.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from feather.models import OpenAIConfig, ProviderRequestConfig
from feather.providers.openai_provider import OpenAIResponsesProvider


# A non-memory Pydantic model kept local to this test file so the test isn't
# coupled to the memory subsystem.
class _NestedBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    weight: float


class _SimpleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["A", "B"]
    detail: str = Field(description="free text")
    block: _NestedBlock


def _openai_config() -> OpenAIConfig:
    return OpenAIConfig(
        api_key_env="OPENAI_API_KEY",
        model="gpt-4.1-mini",
        max_output_tokens=1234,
        temperature=0.1,
        parallel_tool_calls=True,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        store=True,
        reasoning=None,
    )


def _build_provider(monkeypatch: pytest.MonkeyPatch) -> OpenAIResponsesProvider:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAIResponsesProvider(_openai_config())


# Backward-compat ------------------------------------------------------------


def test_request_has_no_text_key_when_response_schema_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing call-sites without response_schema must emit no `text` key."""
    provider = _build_provider(monkeypatch)
    req = provider._build_request_kwargs(
        instructions="go",
        input_items=[],
        tools=[],
        previous_response_id=None,
    )
    assert "text" not in req


def test_response_schema_none_on_request_config_keeps_backward_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _build_provider(monkeypatch)
    req = provider._build_request_kwargs(
        instructions="go",
        input_items=[],
        tools=[],
        previous_response_id=None,
        request_config=ProviderRequestConfig(model="gpt-4.1-mini"),
    )
    assert "text" not in req


# response_schema translation ------------------------------------------------


def test_response_schema_produces_strict_json_schema_text_block(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _build_provider(monkeypatch)
    req = provider._build_request_kwargs(
        instructions="",
        input_items=[],
        tools=[],
        previous_response_id=None,
        request_config=ProviderRequestConfig(response_schema=_SimpleResponse),
    )
    assert "text" in req
    text = req["text"]
    assert text["format"]["type"] == "json_schema"
    assert text["format"]["strict"] is True
    assert text["format"]["name"] == "_SimpleResponse"
    assert "schema" in text["format"]


def test_response_schema_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _build_provider(monkeypatch)
    req = provider._build_request_kwargs(
        instructions="",
        input_items=[],
        tools=[],
        previous_response_id=None,
        request_config=ProviderRequestConfig(
            response_schema=_SimpleResponse,
            response_schema_name="ExtractionResponse",
        ),
    )
    assert req["text"]["format"]["name"] == "ExtractionResponse"


def test_schema_has_additional_properties_false_on_every_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_harden_strict_schema must set additionalProperties:false on every nested object."""
    provider = _build_provider(monkeypatch)
    req = provider._build_request_kwargs(
        instructions="",
        input_items=[],
        tools=[],
        previous_response_id=None,
        request_config=ProviderRequestConfig(response_schema=_SimpleResponse),
    )
    schema = req["text"]["format"]["schema"]

    # Top-level object.
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False

    # Find the referenced $defs block (pydantic emits nested models under $defs).
    defs = schema.get("$defs") or {}
    assert "_NestedBlock" in defs
    nested = defs["_NestedBlock"]
    assert nested.get("type") == "object"
    assert nested.get("additionalProperties") is False


def test_harden_schema_does_not_overwrite_existing_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a schema already has additionalProperties:false it must be left alone, not reset."""
    from feather.providers.openai_provider import _harden_strict_schema

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    before = dict(schema)
    _harden_strict_schema(schema)
    assert schema["additionalProperties"] is False
    # other keys unchanged
    for k, v in before.items():
        if k != "additionalProperties":
            assert schema[k] == v


def test_harden_schema_walks_nested_arrays_of_objects() -> None:
    """Nested objects inside arrays must also get additionalProperties:false."""
    from feather.providers.openai_provider import _harden_strict_schema

    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"k": {"type": "string"}},
                    "required": ["k"],
                },
            }
        },
        "required": ["items"],
    }
    _harden_strict_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["items"]["additionalProperties"] is False
