"""Tests for feather.providers.schema_utils.harden_strict_schema.

Targets the shared implementation directly rather than going through a
provider, so that coverage is not coupled to any one provider's plumbing.
"""

from __future__ import annotations

from feather.providers.schema_utils import harden_strict_schema


# ------------------------------------------------------------------ happy path


def test_harden_sets_additional_properties_false_on_root_object() -> None:
    schema: dict = {"type": "object", "properties": {"a": {"type": "string"}}}
    harden_strict_schema(schema)
    assert schema["additionalProperties"] is False


def test_harden_sets_required_to_all_property_names() -> None:
    schema: dict = {
        "type": "object",
        "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
    }
    harden_strict_schema(schema)
    assert set(schema["required"]) == {"x", "y"}


def test_harden_walks_defs_nested_object() -> None:
    """Objects under $defs must also receive additionalProperties:false."""
    schema: dict = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Child"}},
        "$defs": {
            "Child": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
            }
        },
    }
    harden_strict_schema(schema)
    assert schema["$defs"]["Child"]["additionalProperties"] is False
    assert schema["$defs"]["Child"]["required"] == ["label"]


def test_harden_walks_anyof_branches() -> None:
    """Objects inside anyOf/oneOf must also be hardened."""
    schema: dict = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "number"}}},
        ]
    }
    harden_strict_schema(schema)
    for branch in schema["anyOf"]:
        assert branch.get("additionalProperties") is False


def test_harden_walks_nested_array_items() -> None:
    """Objects nested inside array items must also get additionalProperties:false."""
    schema: dict = {
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
    harden_strict_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["items"]["additionalProperties"] is False


def test_harden_idempotent_on_already_hardened_schema() -> None:
    """Running harden_strict_schema twice must leave the schema unchanged."""
    schema: dict = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    harden_strict_schema(schema)
    harden_strict_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["a"]


# ------------------------------------------------------------------ edge / failure cases


def test_harden_does_not_overwrite_existing_additional_properties_false() -> None:
    """An already-set additionalProperties:false must not be reset."""
    schema: dict = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    before = dict(schema)
    harden_strict_schema(schema)
    assert schema["additionalProperties"] is False
    for k, v in before.items():
        assert schema[k] == v


def test_harden_does_not_add_required_when_properties_absent() -> None:
    """An object node with no properties must not gain a required key."""
    schema: dict = {"type": "object"}
    harden_strict_schema(schema)
    assert "required" not in schema
    assert schema.get("additionalProperties") is False


def test_harden_does_not_touch_non_object_nodes() -> None:
    """String/array/number schema nodes must be left untouched."""
    schema: dict = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    harden_strict_schema(schema)
    assert "additionalProperties" not in schema["properties"]["name"]
    assert "additionalProperties" not in schema["properties"]["count"]


def test_harden_handles_list_at_top_level() -> None:
    """A top-level list of schemas must be walked recursively."""
    schemas: list = [
        {"type": "object", "properties": {"a": {"type": "string"}}},
        {"type": "string"},
    ]
    # harden_strict_schema accepts dict; pass via wrapper to test list branch
    wrapper: dict = {"anyOf": schemas}
    harden_strict_schema(wrapper)
    assert schemas[0]["additionalProperties"] is False
    # string node untouched
    assert "additionalProperties" not in schemas[1]
