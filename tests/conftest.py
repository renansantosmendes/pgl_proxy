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

#: A matricula used by tests that need a valid, active student without
#: touching the real Neon database.
ACTIVE_MATRICULA = "20230001"


@pytest.fixture
def client():
    return TestClient(app_main.app)


@pytest.fixture
def mock_is_enrollment_active(monkeypatch):
    """Replace the Neon lookup with an AsyncMock.

    Defaults to treating `ACTIVE_MATRICULA` as active and everything else
    as inactive/unknown, mirroring what a real lookup would do. Tests can
    override `mock.side_effect` for other scenarios.
    """
    mock = AsyncMock(side_effect=lambda matricula: matricula == ACTIVE_MATRICULA)
    monkeypatch.setattr(app_main, "is_enrollment_active", mock)
    return mock


@pytest.fixture
def authenticated_client(client, mock_is_enrollment_active):
    """A TestClient that already sends a valid matricula as the api_key."""
    client.headers["Authorization"] = f"Bearer {ACTIVE_MATRICULA}"
    return client


@pytest.fixture
def mock_openai_create(monkeypatch):
    """Replace the OpenAI chat completions call with an AsyncMock.

    Tests configure `mock.return_value` or `mock.side_effect` to control
    what the "OpenAI API" hands back, without ever making a real call.
    """
    mock = AsyncMock()
    monkeypatch.setattr(app_main.openai_client.chat.completions, "create", mock)
    return mock
