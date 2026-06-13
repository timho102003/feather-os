"""OpenAI Responses API provider."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

from feather.models import (
    EventHandler,
    MCPConfig,
    ModelTurn,
    OpenAIConfig,
    ProviderRequestConfig,
    RuntimeEvent,
    ToolCall,
)
from feather.integrations.mcp.client import openai_mcp_tools
from feather.providers.base import BaseLLMProvider
from feather.providers.schema_utils import harden_strict_schema as _harden_strict_schema

logger = logging.getLogger(__name__)


class OpenAIStreamIdleTimeoutError(TimeoutError):
    """Raised when an OpenAI Responses stream stalls longer than the configured idle budget.

    The upstream HTTP connection returns 200 OK before streaming begins, so a
    silent stall mid-stream is otherwise indistinguishable from a slow model.
    Surfacing this as a dedicated error lets the agent loop log it cleanly
    instead of hanging the CLI indefinitely.
    """


class OpenAIStreamError(RuntimeError):
    """Raised when an OpenAI Responses stream terminates in a non-completed state.

    The upstream SDK's ``get_final_response()`` only handles ``response.completed``
    and throws a generic ``RuntimeError("Didn't receive a response.completed
    event.")`` for every other terminal condition — most commonly
    ``response.incomplete`` with ``reason=max_output_tokens`` when the model
    burns through its output budget on reasoning tokens before finishing. This
    typed error surfaces the real reason (incomplete / failed / error / no
    terminal event) so the agent loop logs something actionable instead of the
    SDK's opaque message.
    """


class OpenAIResponsesProvider(BaseLLMProvider):
    """Thin adapter over the OpenAI Responses API."""

    def __init__(
        self, config: OpenAIConfig, *, mcp_config: MCPConfig | None = None
    ) -> None:
        self._config = config
        self._mcp_config = mcp_config or MCPConfig()
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing required environment variable: {config.api_key_env}")
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        event_handler: EventHandler | None = None,
        request_config: ProviderRequestConfig | None = None,
    ) -> ModelTurn:
        """Run one streamed Responses API call.

        Args:
            instructions: Full system instructions.
            input_items: Newly appended input items.
            tools: Tool definitions for the current agent.
            previous_response_id: Previous OpenAI response ID for stateful continuation.
            event_handler: Optional event sink used by the CLI.

        Returns:
            Normalized model turn.

        Note:
            ``request_config.cache_prefix`` is intentionally unused — OpenAI's
            Responses API caches prompt prefixes automatically (≥1024 tokens),
            so the static-first ordering already in ``instructions`` suffices;
            no explicit breakpoint is needed.
        """

        request = self._build_request_kwargs(
            instructions=instructions,
            input_items=input_items,
            tools=tools,
            previous_response_id=previous_response_id,
            request_config=request_config,
        )
        logger.info(
            "openai request model=%s reasoning=%s previous_response_id=%s tools=%s",
            request.get("model"),
            request.get("reasoning"),
            previous_response_id,
            [
                tool.get("name") or tool.get("server_label") or tool.get("type")
                for tool in request.get("tools", [])
            ],
        )
        idle_timeout = self._config.stream_idle_timeout_seconds
        model_name = request.get("model")
        # Track terminal events ourselves rather than calling
        # stream.get_final_response(), which only understands
        # response.completed and raises a generic RuntimeError for any
        # other exit path. The server always emits exactly one terminal
        # event — response.completed, response.incomplete, or
        # response.failed — and can also emit a top-level `error` event
        # mid-stream. Capturing them explicitly lets us surface the real
        # reason (e.g. max_output_tokens) in a structured error.
        response: Any = None
        terminal_event_type: str | None = None
        async with self._client.responses.stream(**request) as stream:
            iterator = stream.__aiter__()
            while True:
                try:
                    event = await asyncio.wait_for(
                        iterator.__anext__(), timeout=idle_timeout
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise OpenAIStreamIdleTimeoutError(
                        f"openai stream idle >{idle_timeout:.0f}s "
                        f"model={model_name} "
                        f"previous_response_id={previous_response_id}"
                    ) from exc
                if event.type == "response.output_text.delta":
                    if event_handler is not None:
                        event_handler(
                            RuntimeEvent(kind="assistant_text_delta", text=event.delta)
                        )
                elif event.type in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    response = getattr(event, "response", None)
                    terminal_event_type = event.type
                elif event.type == "error":
                    code = getattr(event, "code", None)
                    message = getattr(event, "message", None) or "unknown"
                    raise OpenAIStreamError(
                        f"openai stream error model={model_name} "
                        f"previous_response_id={previous_response_id} "
                        f"code={code!r} message={message!r}"
                    )

        if terminal_event_type == "response.failed":
            error_obj = getattr(response, "error", None)
            code = getattr(error_obj, "code", None)
            message = getattr(error_obj, "message", None)
            raise OpenAIStreamError(
                f"openai response failed model={model_name} "
                f"previous_response_id={previous_response_id} "
                f"response_id={getattr(response, 'id', None)} "
                f"code={code!r} message={message!r}"
            )

        if terminal_event_type == "response.incomplete":
            detail = getattr(response, "incomplete_details", None)
            reason = getattr(detail, "reason", None) if detail is not None else None
            logger.warning(
                "openai response incomplete model=%s reason=%s id=%s output_chars=%s",
                model_name,
                reason,
                getattr(response, "id", None),
                len(getattr(response, "output_text", "") or ""),
            )
            raise OpenAIStreamError(
                f"openai response incomplete model={model_name} "
                f"previous_response_id={previous_response_id} "
                f"response_id={getattr(response, 'id', None)} "
                f"reason={reason} "
                f"(bump max_output_tokens or shorten the task; see config/app.yaml)"
            )

        if response is None:
            raise OpenAIStreamError(
                "openai stream ended without a terminal event "
                "(no response.completed/incomplete/failed) "
                f"model={model_name} previous_response_id={previous_response_id}"
            )

        tool_calls: list[ToolCall] = []
        for item in response.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"raw_arguments": item.arguments}
            tool_calls.append(ToolCall(call_id=item.call_id, name=item.name, arguments=arguments))

        usage = response.usage.model_dump() if response.usage is not None else None
        logger.info(
            "openai response id=%s tool_calls=%s output_chars=%s",
            response.id,
            len(tool_calls),
            len(response.output_text or ""),
        )
        return ModelTurn(
            response_id=response.id,
            output_text=response.output_text or "",
            tool_calls=tool_calls,
            usage=usage,
        )

    def _build_request_kwargs(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        request_config: ProviderRequestConfig | None = None,
    ) -> dict[str, Any]:
        """Construct the OpenAI request payload.

        Args:
            instructions: Full system instructions.
            input_items: Newly appended input items.
            tools: Tool definitions for the current agent.
            previous_response_id: Previous OpenAI response ID for stateful continuation.

        Returns:
            Keyword arguments for `responses.stream`.
        """

        active_config = request_config or ProviderRequestConfig()
        reasoning_config = active_config.reasoning if active_config.reasoning is not None else self._config.reasoning

        model_name = active_config.model or self._config.model
        request: dict[str, Any] = {
            "model": model_name,
            "instructions": instructions,
            "input": input_items,
            "max_output_tokens": active_config.max_output_tokens or self._config.max_output_tokens,
            "store": self._config.store,
            "truncation": "auto",
        }
        mcp_servers = active_config.mcp_servers
        request_tools = list(tools)
        request_tools.extend(openai_mcp_tools(mcp_servers))
        if request_tools:
            request["tools"] = request_tools
            request["parallel_tool_calls"] = self._config.parallel_tool_calls
        temperature = self._config.temperature if active_config.temperature is None else active_config.temperature
        if self._supports_temperature(model_name, reasoning_config):
            request["temperature"] = temperature
        if active_config.response_schema is not None:
            schema_cls = active_config.response_schema
            schema = schema_cls.model_json_schema()
            _harden_strict_schema(schema)
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": active_config.response_schema_name or schema_cls.__name__,
                    "schema": schema,
                    "strict": True,
                }
            }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        if self._config.prompt_cache_key:
            request["prompt_cache_key"] = self._config.prompt_cache_key
        if self._config.prompt_cache_retention:
            request["prompt_cache_retention"] = self._normalize_prompt_cache_retention(
                self._config.prompt_cache_retention
            )
        if reasoning_config is not None:
            reasoning: dict[str, Any] = {}
            if reasoning_config.effort is not None:
                reasoning["effort"] = reasoning_config.effort
            if reasoning_config.summary is not None:
                reasoning["summary"] = reasoning_config.summary
            if reasoning:
                request["reasoning"] = reasoning
        return request

    def _supports_temperature(self, model_name: str, reasoning_config: Any) -> bool:
        """Return whether `temperature` should be sent for the given model."""

        normalized = model_name.strip().lower()
        if not normalized.startswith("gpt-5"):
            return True

        effort = None
        if reasoning_config is not None:
            effort = getattr(reasoning_config, "effort", None)

        if normalized.startswith(("gpt-5.1", "gpt-5.2")) and effort == "none":
            return True
        return False

    def _normalize_prompt_cache_retention(self, value: str) -> str:
        """Normalize prompt cache retention to the API's accepted values.

        Args:
            value: Raw configured retention value.

        Returns:
            Normalized retention value.

        Raises:
            ValueError: If the value is unsupported.
        """

        normalized = value.strip().lower().replace("-", "_")
        if normalized in {"in_memory", "24h"}:
            return normalized
        raise ValueError(
            "Unsupported prompt_cache_retention. Expected one of: in_memory, 24h."
        )
