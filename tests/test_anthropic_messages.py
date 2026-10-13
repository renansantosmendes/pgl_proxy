from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.anthropic_messages import (
    ALLOWED_ANTHROPIC_MODELS,
    format_sse_event,
    sanitize_messages_payload,
    usage_details,
)
from app.main import validate_model


def test_only_haiku_is_allowed():
    assert ALLOWED_ANTHROPIC_MODELS == {"claude-haiku-4-5"}
    validate_model("claude-haiku-4-5", ALLOWED_ANTHROPIC_MODELS)


def test_sanitize_drops_unknown_fields_and_keeps_allowed_ones():
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "messages": [],
        "system": "s",
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "metadata": {"user_id": "x"},
        "service_tier": "priority",
        "mcp_servers": [{"type": "url", "url": "https://example.com"}],
    }

    sanitized = sanitize_messages_payload(payload, max_tokens_limit=2048)

    assert set(sanitized) == {"model", "max_tokens", "messages", "system", "thinking"}


@pytest.mark.parametrize(
    ("max_tokens", "expected"), [(100, 100), (2048, 2048), (64000, 2048), (None, 2048)]
)
def test_sanitize_caps_max_tokens(max_tokens, expected):
    payload = {} if max_tokens is None else {"max_tokens": max_tokens}

    assert sanitize_messages_payload(payload, max_tokens_limit=2048)["max_tokens"] == expected


@pytest.mark.parametrize("max_tokens", [True, -1, 0, 1.5, "100"])
def test_sanitize_rejects_invalid_max_tokens(max_tokens):
    with pytest.raises(HTTPException) as exc_info:
        sanitize_messages_payload({"max_tokens": max_tokens}, max_tokens_limit=2048)

    assert exc_info.value.status_code == 400


def test_sanitize_accepts_explicit_custom_tools():
    payload = {"max_tokens": 10, "tools": [{"type": "custom", "name": "t", "input_schema": {}}]}

    assert sanitize_messages_payload(payload, max_tokens_limit=2048)["tools"]


def test_usage_details_maps_anthropic_fields_and_skips_missing_ones():
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=4,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=2,
    )

    assert usage_details(usage) == {"input": 10, "output": 4, "cache_creation_input_tokens": 2}
    assert usage_details(None) == {}


def test_format_sse_event_matches_anthropic_wire_format():
    assert format_sse_event("message_stop", '{"type":"message_stop"}') == (
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
