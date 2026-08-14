"""Environment-derived configuration shared across the app and scripts."""

import os

#: The real OpenAI API key. This is the only place in the whole system where
#: it exists, and it must be provided via an environment variable so it
#: never ends up committed to source control or handed to a student.
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

#: Neon Postgres connection string used to look up which student matriculas
#: are allowed to use the proxy.
NEON_DATABASE_URL: str = os.environ["NEON_DATABASE_URL"]

#: Shared HS256 signing secret for the JWTs students present as their
#: api_key. Must be the exact same value configured on pgl_auth_server
#: (the service that actually issues these tokens after verifying
#: matricula/senha) — this app only verifies signatures, it never issues
#: tokens or sees a student's password.
JWT_SECRET_KEY: str = os.environ["JWT_SECRET_KEY"]
