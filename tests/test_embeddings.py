from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.main as app_main
from conftest import ACTIVE_REGISTRATION_NUMBER


def make_embeddings(count: int = 1):
    """Stand-in for OpenAI's embeddings response, with the `.model_dump()`
    interface `app.main` relies on."""
    payload = {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": index, "embedding": [0.1, 0.2, 0.3]}
            for index in range(count)
        ],
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    return SimpleNamespace(model_dump=lambda: payload)


@pytest.fixture
def mock_openai_embeddings(monkeypatch):
    """Replace the OpenAI embeddings call with an AsyncMock, so no real call is made."""
    mock = AsyncMock(return_value=make_embeddings())
    monkeypatch.setattr(app_main.openai_client.embeddings, "create", mock)
    return mock


def test_missing_authorization_returns_401(client):
    response = client.post(
        "/v1/embeddings", json={"model": "text-embedding-3-small", "input": "oi"}
    )

    assert response.status_code == 401


def test_allowed_model_forwards_to_openai_and_returns_vectors(
    authenticated_client, mock_openai_embeddings
):
    response = authenticated_client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": ["primeiro", "segundo"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 1
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    mock_openai_embeddings.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["primeiro", "segundo"],
        metadata={"registration_number": ACTIVE_REGISTRATION_NUMBER},
    )


def test_a_single_text_is_accepted_and_unknown_fields_are_dropped(
    authenticated_client, mock_openai_embeddings
):
    response = authenticated_client.post(
        "/v1/embeddings",
        json={
            "model": "text-embedding-3-small",
            "input": "um texto",
            "dimensions": 256,
            "user": "outra-pessoa",
        },
    )

    assert response.status_code == 200
    kwargs = mock_openai_embeddings.await_args.kwargs
    assert kwargs["input"] == "um texto"
    assert kwargs["dimensions"] == 256
    assert "user" not in kwargs


@pytest.mark.parametrize("model", ["text-embedding-3-large", "text-embedding-ada-002", "gpt-4o-mini"])
def test_models_outside_the_allowlist_return_403(
    authenticated_client, mock_openai_embeddings, model
):
    response = authenticated_client.post(
        "/v1/embeddings", json={"model": model, "input": "oi"}
    )

    assert response.status_code == 403
    mock_openai_embeddings.assert_not_awaited()


def test_missing_model_returns_400(authenticated_client, mock_openai_embeddings):
    response = authenticated_client.post("/v1/embeddings", json={"input": "oi"})

    assert response.status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"model": "text-embedding-3-small", "input": []},
        {"model": "text-embedding-3-small", "input": [1, 2, 3]},
        {"model": "text-embedding-3-small", "input": 42},
        {"model": "text-embedding-3-small", "input": ["x"] * 2049},
        {"model": "text-embedding-3-small", "input": "a" * 300_001},
        {"model": "text-embedding-3-small", "input": "oi", "dimensions": 0},
        {"model": "text-embedding-3-small", "input": "oi", "dimensions": 5000},
    ],
)
def test_inputs_outside_the_limits_return_400(
    authenticated_client, mock_openai_embeddings, body
):
    response = authenticated_client.post("/v1/embeddings", json=body)

    assert response.status_code == 400
    mock_openai_embeddings.assert_not_awaited()


def test_a_long_document_within_the_limits_is_forwarded(
    authenticated_client, mock_openai_embeddings
):
    text = "ação " * 40_000

    response = authenticated_client.post(
        "/v1/embeddings", json={"model": "text-embedding-3-small", "input": [text]}
    )

    assert response.status_code == 200
    mock_openai_embeddings.assert_awaited_once()


def test_embeddings_use_their_own_rate_limit(
    authenticated_client, mock_openai_embeddings, mock_check_rate_limit
):
    authenticated_client.post(
        "/v1/embeddings", json={"model": "text-embedding-3-small", "input": "oi"}
    )

    mock_check_rate_limit.assert_awaited_once_with(
        ACTIVE_REGISTRATION_NUMBER,
        app_main.EMBEDDING_RATE_LIMIT_MAX_REQUESTS,
        app_main.EMBEDDING_RATE_LIMIT_WINDOW_SECONDS,
        kind="embeddings",
    )


def test_exhausted_embeddings_quota_returns_429_without_calling_openai(
    authenticated_client, mock_openai_embeddings, mock_check_rate_limit
):
    mock_check_rate_limit.return_value = False

    response = authenticated_client.post(
        "/v1/embeddings", json={"model": "text-embedding-3-small", "input": "oi"}
    )

    assert response.status_code == 429
    mock_openai_embeddings.assert_not_awaited()


def test_chat_keeps_the_chat_rate_limit(
    authenticated_client, mock_check_rate_limit, mock_openai_create
):
    from conftest import ACTIVE_REGISTRATION_NUMBER as REG

    mock_openai_create.return_value = SimpleNamespace(
        model_dump=lambda: {"id": "x", "object": "chat.completion", "choices": []}
    )
    authenticated_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "oi"}]},
    )

    mock_check_rate_limit.assert_awaited_once_with(
        REG, app_main.RATE_LIMIT_MAX_REQUESTS, app_main.RATE_LIMIT_WINDOW_SECONDS
    )
