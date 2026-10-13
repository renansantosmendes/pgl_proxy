import json

import anthropic
import httpx2
import pytest
from anthropic.types import Message, RawMessageStreamEvent
from pydantic import TypeAdapter

from app import main as app_main
from conftest import ACTIVE_REGISTRATION_NUMBER, make_token

HAIKU = "claude-haiku-4-5"
REQUEST = {
    "model": HAIKU,
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is the capital of Brazil?"}],
}

_event_adapter = TypeAdapter(RawMessageStreamEvent)


def make_message(text: str = "Brasília") -> Message:
    """Build a real `anthropic.types.Message`, as the SDK would return."""
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": HAIKU,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }
    )


def make_events(text_chunks: list[str]) -> list:
    """Build the raw stream events Anthropic sends for a text answer."""
    raw_events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": HAIKU,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    raw_events += [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk}}
        for chunk in text_chunks
    ]
    raw_events += [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    return [_event_adapter.validate_python(event) for event in raw_events]


class FakeAnthropicStream:
    """Stand-in for the SDK's async event stream, optionally failing midway."""

    def __init__(self, events, error=None):
        self._events = events
        self._error = error

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event
        if self._error is not None:
            raise self._error


def make_status_error(error_class, status_code: int, message: str):
    """Build an Anthropic SDK status error carrying Anthropic's error body."""
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    response = httpx2.Response(
        status_code,
        json=body,
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return error_class(message, response=response, body=body)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split an SSE body into (event name, JSON data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ---------------------------------------------------------------------------
# Authentication and configuration
# ---------------------------------------------------------------------------


def test_missing_credentials_return_401(client):
    response = client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 401


def test_bearer_token_is_also_accepted(
    client, mock_is_enrollment_active, mock_check_rate_limit, fake_langfuse, mock_anthropic_create
):
    mock_anthropic_create.return_value = make_message()
    client.headers["Authorization"] = f"Bearer {make_token()}"

    response = client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 200


def test_inactive_registration_number_returns_403(client, mock_is_enrollment_active):
    client.headers["x-api-key"] = make_token("00000000")

    response = client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 403


def test_rate_limit_is_shared_and_returns_429(anthropic_client, mock_check_rate_limit):
    mock_check_rate_limit.return_value = False

    response = anthropic_client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 429


def test_returns_503_when_anthropic_is_not_configured(anthropic_client, monkeypatch):
    monkeypatch.setattr(app_main, "anthropic_client", None)

    response = anthropic_client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_missing_model_returns_400(anthropic_client):
    response = anthropic_client.post(
        "/v1/messages", json={key: value for key, value in REQUEST.items() if key != "model"}
    )

    assert response.status_code == 400


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-4o-mini"])
def test_models_outside_the_allowlist_return_403(anthropic_client, model):
    response = anthropic_client.post("/v1/messages", json={**REQUEST, "model": model})

    assert response.status_code == 403
    assert model in response.json()["detail"]


def test_haiku_is_not_accepted_on_chat_completions(authenticated_client):
    response = authenticated_client.post(
        "/v1/chat/completions", json={"model": HAIKU, "messages": []}
    )

    assert response.status_code == 403


@pytest.mark.parametrize("max_tokens", [0, -5, "100", True, 1.5])
def test_invalid_max_tokens_returns_400(anthropic_client, max_tokens):
    response = anthropic_client.post("/v1/messages", json={**REQUEST, "max_tokens": max_tokens})

    assert response.status_code == 400


@pytest.mark.parametrize("max_tokens", [None, 64000])
def test_missing_or_excessive_max_tokens_is_capped(
    anthropic_client, mock_anthropic_create, max_tokens
):
    mock_anthropic_create.return_value = make_message()
    payload = {**REQUEST, "max_tokens": max_tokens}
    if max_tokens is None:
        del payload["max_tokens"]

    response = anthropic_client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    assert mock_anthropic_create.await_args.kwargs["max_tokens"] == app_main.MAX_TOKENS_LIMIT


def test_server_tools_are_rejected(anthropic_client):
    response = anthropic_client.post(
        "/v1/messages",
        json={**REQUEST, "tools": [{"type": "web_search_20260209", "name": "web_search"}]},
    )

    assert response.status_code == 400
    assert "web_search_20260209" in response.json()["detail"]


def test_oversized_body_returns_413(anthropic_client):
    huge_content = "x" * (app_main.MAX_REQUEST_BODY_BYTES + 1)

    response = anthropic_client.post(
        "/v1/messages",
        json={**REQUEST, "messages": [{"role": "user", "content": huge_content}]},
    )

    assert response.status_code == 413


def test_non_object_json_body_returns_400(anthropic_client):
    response = anthropic_client.post("/v1/messages", json=[1, 2, 3])

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------


def test_message_is_forwarded_and_returned_unchanged(
    anthropic_client, mock_anthropic_create, fake_langfuse
):
    mock_anthropic_create.return_value = make_message("Brasília")
    custom_tool = {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }

    response = anthropic_client.post(
        "/v1/messages",
        json={
            **REQUEST,
            "system": "Answer briefly.",
            "tools": [custom_tool],
            "metadata": {"user_id": "someone-else"},
            "service_tier": "priority",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == [{"type": "text", "text": "Brasília"}]
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 3}
    mock_anthropic_create.assert_awaited_once_with(
        **REQUEST, system="Answer briefly.", tools=[custom_tool]
    )

    generation = fake_langfuse.generations[0]
    assert generation.start_kwargs["model"] == HAIKU
    assert generation.start_kwargs["as_type"] == "generation"
    assert generation.merged_updates()["usage_details"] == {"input": 12, "output": 3}
    assert generation.ended


def test_stream_is_relayed_in_anthropic_sse_format(
    anthropic_client, mock_anthropic_create, fake_langfuse
):
    mock_anthropic_create.return_value = FakeAnthropicStream(make_events(["Bras", "ília"]))

    response = anthropic_client.post("/v1/messages", json={**REQUEST, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert all(name == data["type"] for name, data in events)
    assert events[2][1]["delta"] == {"type": "text_delta", "text": "Bras"}

    generation = fake_langfuse.generations[0]
    recorded = generation.merged_updates()
    assert recorded["output"] == "Brasília"
    assert recorded["usage_details"] == {"input": 12, "output": 5}
    assert "completion_start_time" in recorded
    assert generation.ended


def test_failure_midway_through_stream_sends_error_event(
    anthropic_client, mock_anthropic_create, fake_langfuse
):
    mock_anthropic_create.return_value = FakeAnthropicStream(
        make_events(["Bras"])[:3],
        error=anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
    )

    response = anthropic_client.post("/v1/messages", json={**REQUEST, "stream": True})

    events = parse_sse(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["type"] == "api_error"
    assert fake_langfuse.generations[0].merged_updates()["level"] == "ERROR"
    assert fake_langfuse.generations[0].ended


# ---------------------------------------------------------------------------
# Upstream errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_class", "status_code"),
    [(anthropic.BadRequestError, 400), (anthropic.RateLimitError, 429)],
)
def test_student_request_errors_are_relayed(
    anthropic_client, mock_anthropic_create, fake_langfuse, error_class, status_code
):
    mock_anthropic_create.side_effect = make_status_error(error_class, status_code, "bad input")

    response = anthropic_client.post("/v1/messages", json=REQUEST)

    assert response.status_code == status_code
    assert response.json()["error"]["message"] == "bad input"
    assert fake_langfuse.generations[0].merged_updates()["level"] == "ERROR"
    assert fake_langfuse.generations[0].ended


@pytest.mark.parametrize(
    ("error_class", "status_code"),
    [
        (anthropic.AuthenticationError, 401),
        (anthropic.PermissionDeniedError, 403),
        (anthropic.InternalServerError, 500),
    ],
)
def test_proxy_side_upstream_errors_become_502(
    anthropic_client, mock_anthropic_create, error_class, status_code
):
    mock_anthropic_create.side_effect = make_status_error(error_class, status_code, "secret detail")

    response = anthropic_client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 502
    assert response.json()["type"] == "error"
    assert "secret detail" not in response.text


def test_connection_errors_become_502(anthropic_client, mock_anthropic_create):
    mock_anthropic_create.side_effect = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    response = anthropic_client.post("/v1/messages", json=REQUEST)

    assert response.status_code == 502


def test_generation_is_tagged_with_registration_number(
    anthropic_client, mock_anthropic_create, fake_langfuse, monkeypatch
):
    seen_user_ids = []
    real_propagate = app_main.propagate_attributes

    def spy_propagate_attributes(**kwargs):
        seen_user_ids.append(kwargs.get("user_id"))
        return real_propagate(**kwargs)

    monkeypatch.setattr(app_main, "propagate_attributes", spy_propagate_attributes)
    mock_anthropic_create.return_value = make_message()

    anthropic_client.post("/v1/messages", json=REQUEST)

    assert seen_user_ids == [ACTIVE_REGISTRATION_NUMBER]


# ---------------------------------------------------------------------------
# End to end with the real anthropic SDK (what ChatAnthropic uses)
# ---------------------------------------------------------------------------


def make_sdk_client() -> anthropic.AsyncAnthropic:
    """An `anthropic` SDK client that talks to the proxy app in-process."""
    return anthropic.AsyncAnthropic(
        api_key=make_token(),
        base_url="http://proxy.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app_main.app)),
    )


async def test_anthropic_sdk_can_use_the_proxy(
    mock_is_enrollment_active, mock_check_rate_limit, fake_langfuse, mock_anthropic_create
):
    mock_anthropic_create.return_value = make_message("Brasília")

    message = await make_sdk_client().messages.create(**REQUEST)

    assert message.content[0].text == "Brasília"
    assert message.usage.output_tokens == 3


async def test_anthropic_sdk_can_stream_through_the_proxy(
    mock_is_enrollment_active, mock_check_rate_limit, fake_langfuse, mock_anthropic_create
):
    mock_anthropic_create.return_value = FakeAnthropicStream(make_events(["Bras", "ília"]))

    async with make_sdk_client().messages.stream(**REQUEST) as stream:
        text = "".join([chunk async for chunk in stream.text_stream])
        final_message = await stream.get_final_message()

    assert text == "Brasília"
    assert final_message.stop_reason == "end_turn"


async def test_anthropic_sdk_raises_typed_errors_from_the_proxy(
    mock_is_enrollment_active, mock_check_rate_limit, fake_langfuse
):
    with pytest.raises(anthropic.PermissionDeniedError):
        await make_sdk_client().messages.create(**{**REQUEST, "model": "claude-opus-5-5"})
