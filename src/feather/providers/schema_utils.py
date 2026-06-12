"""Shared JSON-schema strict-mode hardening for providers."""

from __future__ import annotations

from typing import Any

__all__ = ("harden_strict_schema",)


def harden_strict_schema(schema: dict[str, Any]) -> None:
    """Walk ``schema`` in-place and enforce strict-mode invariants.

    OpenAI's strict JSON-schema mode (and Anthropic's equivalent) require every
    ``type:"object"`` node to (a) set ``additionalProperties:false`` and (b) list
    **every** property name in ``required``. Pydantic does (a) at the root by
    default but not always for nested objects or for objects inside arrays /
    ``$defs``, and it drops fields with defaults from ``required`` — both rejected
    by strict mode. This walker enforces both invariants throughout the schema
    graph (``$defs``, ``properties``, ``items``, ``anyOf``/``oneOf``/``allOf``).
    Optional behavior is expressed via ``["T", "null"]`` unions.

    Args:
        schema: JSON-schema dict to mutate in place.
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                if "additionalProperties" not in node:
                    node["additionalProperties"] = False
                properties = node.get("properties")
                if isinstance(properties, dict) and properties:
                    # Force `required` to list every property — strict mode
                    # rejects schemas where any property is missing.
                    node["required"] = list(properties.keys())
            for key in ("properties", "$defs", "definitions", "patternProperties"):
                sub = node.get(key)
                if isinstance(sub, dict):
                    for child in sub.values():
                        _walk(child)
            for key in ("items", "additionalItems", "contains"):
                sub = node.get(key)
                if isinstance(sub, (dict, list)):
                    _walk(sub)
            for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
                sub = node.get(key)
                if isinstance(sub, list):
                    for child in sub:
                        _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(schema)
