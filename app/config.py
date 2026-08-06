"""Environment-derived configuration shared across the app and scripts."""

import os

#: The real OpenAI API key. This is the only place in the whole system where
#: it exists, and it must be provided via an environment variable so it
#: never ends up committed to source control or handed to a student.
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

#: Neon Postgres connection string used to look up which student matriculas
#: are allowed to use the proxy.
NEON_DATABASE_URL: str = os.environ["NEON_DATABASE_URL"]
