import json
from types import SimpleNamespace

import pytest


def make_completion(content: str = "hello from mock"):
    """Build a stand-in for OpenAI's ChatCompletion, matching the
    `.model_dump()` interface `app.main` relies on."""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return SimpleNamespace(model_dump=lambda: payload)


class FakeStream:
    """Stand-in for OpenAI's async streaming response."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk


def make_chunk(content: str):
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}}],
    }
    return SimpleNamespace(model_dump_json=lambda: json.dumps(payload))


def test_missing_authorization_returns_401(client):
    response = client.post(
        "/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []}
    )

    assert response.status_code == 401


def test_inactive_or_unknown_matricula_returns_403(client, mock_is_enrollment_active):
    client.headers["Authorization"] = "Bearer 00000000"

    response = client.post(
        "/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []}
    )

    assert response.status_code == 403


def test_missing_model_returns_400(authenticated_client):
    response = authenticated_client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 400


def test_disallowed_model_returns_403(authenticated_client):
    response = authenticated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": []},
    )

    assert response.status_code == 403


def test_allowed_model_forwards_to_openai_and_returns_completion(
    authenticated_client, mock_openai_create
):
    mock_openai_create.return_value = make_completion("42")

    response = authenticated_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the answer?"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "42"

    mock_openai_create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the answer?"}],
    )


def test_streaming_request_relays_chunks_as_sse(authenticated_client, mock_openai_create):
    mock_openai_create.return_value = FakeStream(
        [make_chunk("Hel"), make_chunk("lo")]
    )

    response = authenticated_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert body.count("data: ") == 3  # two chunks + final [DONE]
    assert body.strip().endswith("data: [DONE]")
    assert '"content": "Hel"' in body
    assert '"content": "lo"' in body


def test_openai_error_propagates_as_500(authenticated_client, mock_openai_create):
    mock_openai_create.side_effect = RuntimeError("upstream failure")

    with pytest.raises(RuntimeError):
        authenticated_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
