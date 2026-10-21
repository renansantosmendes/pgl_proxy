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
        {"model": "text-embedding-3-small", "input": ["x"] * 101},
        {"model": "text-embedding-3-small", "input": "a" * 50_001},
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
