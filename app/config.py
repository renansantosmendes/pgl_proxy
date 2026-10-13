"""Environment-derived configuration shared across the app and scripts."""

import os
from typing import Optional

#: The real OpenAI API key. This is the only place in the whole system where
#: it exists, and it must be provided via an environment variable so it
#: never ends up committed to source control or handed to a student.
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

#: The real Anthropic API key, used only by the ``/v1/messages`` endpoint.
#: Optional: when it is not set, the proxy still serves OpenAI models and
#: answers ``/v1/messages`` with 503, so existing deployments keep working.
ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY") or None

#: Neon Postgres connection string used to look up which student
#: registration numbers are allowed to use the proxy.
NEON_DATABASE_URL: str = os.environ["NEON_DATABASE_URL"]

#: Shared HS256 signing secret for the JWTs students present as their
#: api_key. Must be the exact same value configured on pgl_auth_server
#: (the service that actually issues these tokens after verifying
#: registration_number/senha) — this app only verifies signatures, it
#: never issues tokens or sees a student's password.
JWT_SECRET_KEY: str = os.environ["JWT_SECRET_KEY"]
