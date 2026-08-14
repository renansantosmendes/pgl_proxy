# PGL OpenAI Proxy

A lightweight FastAPI reverse proxy that sits between students' LangChain
clients and the OpenAI API, so the real OpenAI API key is never exposed to
students. Every request is traced in [Langfuse](https://langfuse.com) for
monitoring (cost, latency, prompts, completions, errors).

## How it works

1. Students configure LangChain's `ChatOpenAI` (or the raw OpenAI SDK) to
   point at this proxy (`base_url="http://your-server:8000/v1"`), setting
   `"<matricula>:<senha>"` (enrollment number and password) as the
   `api_key`. The SDK sends it as an
   `Authorization: Bearer <matricula>:<senha>` header.
2. The server checks that matricula/senha against the `pgl_proxy.students`
   table in [Neon](https://neon.tech) (matricula must exist, be marked
   active, and senha must match its stored hash), then validates the
   requested model against an allowlist, then uses the
   **real** OpenAI API key (read from an environment variable) to call
   OpenAI via the Langfuse-instrumented OpenAI SDK client.
3. The real OpenAI key lives only on the server and is never sent to, or
   seen by, student code.
4. Every call is automatically traced in Langfuse: prompts, completions,
   token usage, latency, and errors — each trace tagged with the student's
   matricula as `user_id` (via `langfuse.propagate_attributes`), so you can
   filter/group usage per student in the Langfuse dashboard.

## Project structure

```
app/
  main.py             FastAPI application (the proxy itself)
  db.py                Neon connection pool + matricula lookup
  config.py            Environment-derived configuration
db/
  schema.sql            DDL for the pgl_proxy.students table
scripts/
  init_db.py             Creates the schema/table in Neon (run once)
  manage_students.py      CLI to add/activate/deactivate/list matriculas,
                           and set/reset their passwords
tests/
  conftest.py        Shared fixtures (test client, mocked OpenAI/Neon calls)
  test_health.py      Tests for GET /health
  test_auth.py         Tests for the matricula authentication dependency
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
`base_url` and `api_key` (the student's `"matricula:senha"`). Accepts the
same JSON body as OpenAI (`model`, `messages`, `stream`, etc.).

- Returns `401` if no matricula/senha was sent (missing/malformed
  `Authorization` header, or `api_key` isn't `"matricula:senha"`).
- Returns `403` if the matricula/senha is invalid, the matricula is
  inactive, or the requested model is not in the allowlist.
- Returns `429` if the matricula has exceeded its request rate limit.
- Returns `413` if the request body is too large.
- Returns `400` if the `model` field is missing, the body isn't valid JSON,
  or `max_tokens`/`n` exceed their caps.
- Returns the chat completion JSON for regular requests, or relays a
  server-sent events stream when `"stream": true` is set.

### `POST /v1/register`

Lets a student set their password for the first time. Meant to be called
from a browser (e.g. the companion `pgl-registry-front` page) via
`fetch()`, not from student LangChain/OpenAI SDK code — so it's the only
endpoint with CORS enabled, restricted to the origins listed in
`ALLOWED_FRONTEND_ORIGINS`.

Body: `{"matricula": "20231001", "senha": "..."}` (senha 8–72 chars).

- Returns `200 {"status": "ok"}` on success.
- Returns `404` if the matricula isn't registered/active in
  `pgl_proxy.students` (see [Student authentication](#student-authentication-neon)
  below — it must already exist, e.g. from `import_students_csv`).
- Returns `409` if that matricula already has a password set. There is no
  self-service reset: matricula alone isn't a secret, so anyone could
  otherwise hijack another student's account by "resetting" it. Use
  `python -m scripts.manage_students set-password <matricula>` to reset
  one as an instructor.
- Returns `429` if the calling IP has registered too many times recently
  (`REGISTER_RATE_LIMIT_MAX_REQUESTS` per `REGISTER_RATE_LIMIT_WINDOW_SECONDS`,
  see [`app/main.py`](app/main.py)) — this also throttles brute-forcing
  which matriculas exist.
- Returns `422` if `senha` is missing or outside 8–72 characters.

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
- **Per-matricula rate limit**: `RATE_LIMIT_MAX_REQUESTS` (20) requests per
  `RATE_LIMIT_WINDOW_SECONDS` (60) window, enforced atomically in Neon
  (`pgl_proxy.rate_limits`, see [`app/db.py`](app/db.py)) — this works
  correctly even across multiple serverless instances, unlike an in-memory
  counter would.

## Student authentication (Neon)

Only matriculas registered, marked active, and with a password set in Neon
may use the proxy.

1. **Create the schema/tables** (once, or whenever
   [`db/schema.sql`](db/schema.sql) changes — it's idempotent and never
   drops existing tables or rows):

   ```bash
   python -m scripts.init_db
   ```

   This creates `pgl_proxy.students`:

   | column          | type          | notes                                  |
   |-----------------|---------------|------------------------------------------|
   | `matricula`     | `text`        | primary key, the enrollment number     |
   | `id`            | `uuid`        | auto-generated surrogate id            |
   | `full_name`     | `text`        | from Canvas "Nome"                     |
   | `login_id`      | `text`        | from Canvas "ID de Login"               |
   | `sis_id`        | `text`        | from Canvas "ID do SIS"                 |
   | `course`        | `text`        | from Canvas "Seção"                     |
   | `role`          | `text`        | from Canvas "Papel"                     |
   | `password_hash` | `text`        | bcrypt hash, set via `manage_students set-password` |
   | `is_active`     | `boolean`     | defaults to `true`                     |
   | `last_modified` | `timestamptz` | auto-updated by a trigger on `UPDATE`  |

   ...and `pgl_proxy.rate_limits`, used by the abuse-protection rate
   limiter (see above):

   | column          | type          | notes                                  |
   |-----------------|---------------|------------------------------------------|
   | `matricula`     | `text`        | primary key, references `students`     |
   | `window_start`  | `timestamptz` | start of the current fixed window      |
   | `request_count` | `int`         | requests counted in the current window |

2. **Manage matriculas** with the companion CLI:

   ```bash
   python -m scripts.manage_students add 20231001 20231002
   python -m scripts.manage_students deactivate 20231001
   python -m scripts.manage_students activate 20231001
   python -m scripts.manage_students set-password 20231001
   python -m scripts.manage_students list
   ```

   `set-password` prompts (via `getpass`, so it isn't echoed or stored in
   shell history) for the student's senha and stores its bcrypt hash. A
   matricula cannot authenticate until a password has been set.

   For a full roster export from Canvas ("People" page → export as CSV,
   or transcribed into [`db/roster.csv`](db/roster.csv) with columns
   `matricula,full_name,login_id,sis_id,course,role`), bulk-import it
   instead — this upserts by matricula and marks every imported row active:

   ```bash
   python -m scripts.import_students_csv db/roster.csv
   ```

3. Students set `"matricula:senha"` as `api_key` when configuring their
   client, e.g.
   `ChatOpenAI(base_url="http://your-server:8000/v1", api_key="20231001:their-senha")`.

## Running locally

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
export NEON_DATABASE_URL="postgresql://..."

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
database, or network access is required to run it. It covers:

- The `/health` endpoint.
- The matricula authentication dependency (`require_active_matricula`):
  missing/malformed header, inactive/unknown matricula, valid matricula.
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
- `ALLOWED_FRONTEND_ORIGINS` — comma-separated origins allowed to call
  `/v1/register` via CORS, e.g. `https://pgl-registry-front.vercel.app`.
  Leave unset to keep `/v1/register` unreachable from any browser page.

> **Streaming caveat:** Vercel serverless functions have an execution time
> limit (10s on the free plan, longer on paid plans). If streamed OpenAI
> responses run longer than that limit, consider hosting the proxy on a
> platform with long-lived processes instead (e.g. Render, Fly.io,
> Railway).
