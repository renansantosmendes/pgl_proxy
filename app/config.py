"""Environment-derived configuration shared across the app and scripts."""

import os

#: The real OpenAI API key. This is the only place in the whole system where
#: it exists, and it must be provided via an environment variable so it
#: never ends up committed to source control or handed to a student.
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

#: Neon Postgres connection string used to look up which student matriculas
#: are allowed to use the proxy.
NEON_DATABASE_URL: str = os.environ["NEON_DATABASE_URL"]

#: Comma-separated list of origins allowed to call the browser-facing
#: endpoints (currently just /v1/register) via CORS, e.g. the deployed
#: pgl-registry-front URL. Empty by default, which leaves those endpoints
#: unreachable from any browser page until explicitly configured.
ALLOWED_FRONTEND_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
