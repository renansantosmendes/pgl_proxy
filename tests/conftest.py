import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

# Environment variables must exist before `app.main` is imported, since the
# module reads OPENAI_API_KEY at import time (os.environ["OPENAI_API_KEY"]).
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ.setdefault(
    "NEON_DATABASE_URL", "postgresql://user:pass@localhost/dummy"
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main as app_main  # noqa: E402

#: A matricula/senha pair used by tests that need a valid, active student
#: without touching the real Neon database.
ACTIVE_MATRICULA = "20230001"
ACTIVE_SENHA = "correct-senha"


@pytest.fixture
def client():
    return TestClient(app_main.app)


@pytest.fixture
def mock_verify_student_credentials(monkeypatch):
    """Replace the Neon credential check with an AsyncMock.

    Defaults to treating `ACTIVE_MATRICULA`/`ACTIVE_SENHA` as valid and
    everything else as invalid/inactive, mirroring what a real lookup
    would do. Tests can override `mock.side_effect` for other scenarios.
    """
    mock = AsyncMock(
        side_effect=lambda matricula, senha: matricula == ACTIVE_MATRICULA
        and senha == ACTIVE_SENHA
    )
    monkeypatch.setattr(app_main, "verify_student_credentials", mock)
    return mock


@pytest.fixture
def mock_check_rate_limit(monkeypatch):
    """Replace the Neon rate limiter with an AsyncMock that always allows.

    Tests that need to simulate an exhausted quota can set
    `mock.return_value = False` (or a `side_effect`).
    """
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr(app_main, "check_and_increment_rate_limit", mock)
    return mock


@pytest.fixture
def authenticated_client(client, mock_verify_student_credentials, mock_check_rate_limit):
    """A TestClient that already sends a valid matricula:senha as the api_key."""
    client.headers["Authorization"] = f"Bearer {ACTIVE_MATRICULA}:{ACTIVE_SENHA}"
    return client


@pytest.fixture
def mock_register_password(monkeypatch):
    """Replace the Neon register-password call with an AsyncMock.

    Defaults to returning "ok". Tests override `mock.return_value` for
    "not_found_or_inactive" / "already_registered" scenarios.
    """
    mock = AsyncMock(return_value="ok")
    monkeypatch.setattr(app_main, "register_password", mock)
    return mock


@pytest.fixture
def mock_check_keyed_rate_limit(monkeypatch):
    """Replace the Neon keyed rate limiter with an AsyncMock that always allows."""
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr(app_main, "check_and_increment_keyed_rate_limit", mock)
    return mock


@pytest.fixture
def mock_openai_create(monkeypatch):
    """Replace the OpenAI chat completions call with an AsyncMock.

    Tests configure `mock.return_value` or `mock.side_effect` to control
    what the "OpenAI API" hands back, without ever making a real call.
    """
    mock = AsyncMock()
    monkeypatch.setattr(app_main.openai_client.chat.completions, "create", mock)
    return mock
