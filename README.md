# PGL OpenAI Proxy

A lightweight FastAPI reverse proxy that sits between students' LangChain
clients and the OpenAI API, so the real OpenAI API key is never exposed to
students. Every request is traced in [Langfuse](https://langfuse.com) for
monitoring (cost, latency, prompts, completions, errors).

## How it works

1. Students obtain a short-lived JWT from
   [`pgl_auth_server`](https://pgl-auth-server.vercel.app) — typically via
   the `pgl_auth` PyPI package (`PGLAuthClient().login(registration_number, senha)`),
   which itself verifies registration_number/senha (bcrypt, `pgl_auth.students`) and
   confirms the registration number is enrolled/active in `pgl_proxy.students`. This
   proxy is never told the student's password.
2. Students configure LangChain's `ChatOpenAI` (or the raw OpenAI SDK) to
   point at this proxy (`base_url="http://your-server:8000/v1"`), setting
   that JWT as the `api_key`. The SDK sends it as an
   `Authorization: Bearer <token>` header.
3. This server verifies the token's signature and expiry (`JWT_SECRET_KEY`,
   shared with `pgl_auth_server`), re-checks the registration number is still active
   in Neon (defense in depth against a mid-window deactivation, since the
   token itself stays valid for its whole 4h TTL), validates the requested
   model against an allowlist, then uses the **real** OpenAI API key (read
   from an environment variable) to call OpenAI via the
   Langfuse-instrumented OpenAI SDK client.
4. The real OpenAI key lives only on the server and is never sent to, or
   seen by, student code.
5. Every call is automatically traced in Langfuse: prompts, completions,
   token usage, latency, and errors — each trace tagged with the student's
   registration number as `user_id` (via `langfuse.propagate_attributes`), so you can
   filter/group usage per student in the Langfuse dashboard.

This repo owns *enrollment* (who's taking the course and currently active)
but not *credentials* — passwords and token issuance live in the separate
`pgl_auth` (client library) / `pgl_auth_server` (token-issuing API) repos.

## Project structure

```
app/
  main.py             FastAPI application (the proxy itself)
  db.py                Neon connection pool + registration-number/rate-limit lookups
  config.py            Environment-derived configuration
db/
  schema.sql            DDL for the pgl_proxy.students table
scripts/
  init_db.py             Creates the schema/table in Neon (run once)
  manage_students.py      CLI to add/activate/deactivate/list registration numbers
tests/
  conftest.py        Shared fixtures (test client, mocked OpenAI/Neon calls)
  test_health.py      Tests for GET /health
  test_auth.py         Tests for the JWT authentication dependency
  test_validate_model.py    Tests for the model allowlist logic
  test_chat_completions.py  Tests for POST /v1/chat/completions
.github/workflows/
  ci.yml              GitHub Actions pipeline (runs the test suite)
vercel.json           Vercel deployment configuration
requirements.txt      Runtime dependencies
requirements-dev.txt  Runtime + test dependencies
```

## API

### `POST /v1/chat/completions`

Mirrors OpenAI's `POST /v1/chat/completions` so LangChain's `ChatOpenAI` (or
the raw OpenAI SDK) can talk to it without any code changes beyond setting
`base_url` and `api_key` (the student's JWT from `pgl_auth_server`). Accepts
the same JSON body as OpenAI (`model`, `messages`, `stream`, etc.).

- Returns `401` if no token was sent, it's malformed, invalid, or expired.
- Returns `403` if the registration number is inactive, or if the requested model is
  not in the allowlist.
- Returns `429` if the registration number has exceeded its request rate limit.
- Returns `413` if the request body is too large.
- Returns `400` if the `model` field is missing, the body isn't valid JSON,
  or `max_tokens`/`n` exceed their caps.
- Returns the chat completion JSON for regular requests, or relays a
  server-sent events stream when `"stream": true` is set.

### `GET /health`

Simple liveness check. Returns `{"status": "ok"}`.

## Allowed models

The set of models students may request is defined in `ALLOWED_MODELS` in
[`app/main.py`](app/main.py). Requests for any other model are rejected
before they reach OpenAI, keeping costs predictable.

## Abuse protection

Beyond the model allowlist, [`app/main.py`](app/main.py) applies a few more
guardrails before forwarding a request to OpenAI:

- **Payload allowlist** (`ALLOWED_PAYLOAD_FIELDS`): only known OpenAI chat
  completion fields are forwarded — anything else in the request body is
  dropped, so a student can't smuggle in unexpected fields (including a
  `metadata` field, which would otherwise collide with the one the proxy
  adds itself for Langfuse).
- **Cost caps**: `max_tokens`/`max_completion_tokens` above
  `MAX_TOKENS_LIMIT` (2048) or `n` above `MAX_N` (1) are rejected with
  `400`, capping the worst case cost of a single request.
- **Request body size limit**: bodies over `MAX_REQUEST_BODY_BYTES`
  (256 KiB) are rejected with `413` before being parsed.
- **Per-registration-number rate limit**: `RATE_LIMIT_MAX_REQUESTS` (20) requests per
  `RATE_LIMIT_WINDOW_SECONDS` (60) window, enforced atomically in Neon
  (`pgl_proxy.rate_limits`, see [`app/db.py`](app/db.py)) — this works
  correctly even across multiple serverless instances, unlike an in-memory
  counter would.

## Student enrollment (Neon)

Only registration numbers registered and marked active in `pgl_proxy.students` may
authenticate (whether getting a token from `pgl_auth_server`, or using this
proxy with one). This repo owns that table; it does not store credentials.

1. **Create the schema/tables** (once, or whenever
   [`db/schema.sql`](db/schema.sql) changes — it's idempotent and never
   drops existing tables or rows):

   ```bash
   python -m scripts.init_db
   ```

   This creates `pgl_proxy.students`:

   | column          | type          | notes                                  |
   |-----------------|---------------|------------------------------------------|
   | `registration_number` | `text`  | primary key, the enrollment number     |
   | `id`            | `uuid`        | auto-generated surrogate id            |
   | `full_name`     | `text`        | from Canvas "Nome"                     |
   | `login_id`      | `text`        | from Canvas "ID de Login"               |
   | `sis_id`        | `text`        | from Canvas "ID do SIS"                 |
   | `course`        | `text`        | from Canvas "Seção"                     |
   | `role`          | `text`        | from Canvas "Papel"                     |
   | `is_active`     | `boolean`     | defaults to `true`                     |
   | `last_modified` | `timestamptz` | auto-updated by a trigger on `UPDATE`  |

   ...and `pgl_proxy.rate_limits`, used by the abuse-protection rate
   limiter (see above):

   | column          | type          | notes                                  |
   |-----------------|---------------|------------------------------------------|
   | `registration_number` | `text`  | primary key, references `students`     |
   | `window_start`  | `timestamptz` | start of the current fixed window      |
   | `request_count` | `int`         | requests counted in the current window |

2. **Manage registration numbers** with the companion CLI:

   ```bash
   python -m scripts.manage_students add 20231001 20231002
   python -m scripts.manage_students deactivate 20231001
   python -m scripts.manage_students activate 20231001
   python -m scripts.manage_students list
   ```

   For a full roster export from Canvas ("People" page → export as CSV,
   or transcribed into [`db/roster.csv`](db/roster.csv) with columns
   `registration_number,full_name,login_id,sis_id,course,role`), bulk-import it
   instead — this upserts by registration_number and marks every imported row active:

   ```bash
   python -m scripts.import_students_csv db/roster.csv
   ```

3. Once enrolled, students register a password and log in through
   `pgl_auth_server` (see that repo, or the `pgl_auth` PyPI package), and
   set the resulting JWT as `api_key`, e.g.:

   ```python
   from pgl_auth import PGLAuthClient
   token = PGLAuthClient().login(registration_number="20231001", senha="...")

   ChatOpenAI(base_url="http://your-server:8000/v1", api_key=token)
   ```

## Running locally

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
export NEON_DATABASE_URL="postgresql://..."
export JWT_SECRET_KEY="..."  # must match pgl_auth_server's JWT_SECRET_KEY

python -m scripts.init_db  # once, to create the schema/table
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Alternatively, put these values in a local `.env` file (already excluded
from version control by `.gitignore`) and load it with your tool of choice
before starting the server.

## Testing

Install the dev dependencies and run the suite with `pytest`:

```bash
pip install -r requirements-dev.txt
pytest -v
```

The test suite mocks both the OpenAI client and the Neon lookup entirely
(see [`tests/conftest.py`](tests/conftest.py)), so no real API key,
database, or network access is required to run it. Test JWTs are minted
locally with a fixed test `JWT_SECRET_KEY`, matching how `pgl_auth_server`
signs real ones. It covers:

- The `/health` endpoint.
- The JWT authentication dependency (`require_active_registration_number`):
  missing/malformed header, invalid/wrong-secret/expired token,
  inactive/unknown registration number, valid token.
- The model allowlist validation (`validate_model`).
- `/v1/chat/completions` for missing auth, missing/disallowed models,
  successful non-streaming completions, streaming (SSE) completions, and
  upstream error propagation.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the test suite
on every push and pull request to `main`, against Python 3.11, 3.12, and
3.13.

## Deployment (Vercel)

[`vercel.json`](vercel.json) deploys `app/main.py` as a Python serverless
function using `@vercel/python`, routing every path to the FastAPI app.

Set the following environment variables in the Vercel project settings
before deploying:

- `OPENAI_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`
- `NEON_DATABASE_URL`
- `JWT_SECRET_KEY` — must be the exact same value configured on
  `pgl_auth_server`, since that's the service that actually signs the
  tokens this proxy verifies.

> **Streaming caveat:** Vercel serverless functions have an execution time
> limit (10s on the free plan, longer on paid plans). If streamed OpenAI
> responses run longer than that limit, consider hosting the proxy on a
> platform with long-lived processes instead (e.g. Render, Fly.io,
> Railway).
