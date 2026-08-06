# PGL OpenAI Proxy

A lightweight FastAPI reverse proxy that sits between students' LangChain
clients and the OpenAI API, so the real OpenAI API key is never exposed to
students. Every request is traced in [Langfuse](https://langfuse.com) for
monitoring (cost, latency, prompts, completions, errors).

## How it works

1. Students configure LangChain's `ChatOpenAI` (or the raw OpenAI SDK) to
   point at this proxy (`base_url="http://your-server:8000/v1"`), setting
   their **matricula** (enrollment number) as the `api_key`. The SDK sends
   it as an `Authorization: Bearer <matricula>` header.
2. The server checks that matricula against the `pgl_proxy.students` table
   in [Neon](https://neon.tech) (must exist and be marked active), then
   validates the requested model against an allowlist, then uses the
   **real** OpenAI API key (read from an environment variable) to call
   OpenAI via the Langfuse-instrumented OpenAI SDK client.
3. The real OpenAI key lives only on the server and is never sent to, or
   seen by, student code.
4. Every call is automatically traced in Langfuse: prompts, completions,
   token usage, latency, and errors — each trace tagged with the student's
   matricula (`langfuse_user_id`), so you can filter/group usage per
   student in the Langfuse dashboard.

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
  manage_students.py      CLI to add/activate/deactivate/list matriculas
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
`base_url` and `api_key` (the student's matricula). Accepts the same JSON
body as OpenAI (`model`, `messages`, `stream`, etc.).

- Returns `401` if no matricula was sent (missing/malformed `Authorization`
  header).
- Returns `403` if the matricula is not registered or is inactive, or if
  the requested model is not in the allowlist.
- Returns `400` if the `model` field is missing.
- Returns the chat completion JSON for regular requests, or relays a
  server-sent events stream when `"stream": true` is set.

### `GET /health`

Simple liveness check. Returns `{"status": "ok"}`.

## Allowed models

The set of models students may request is defined in `ALLOWED_MODELS` in
[`app/main.py`](app/main.py). Requests for any other model are rejected
before they reach OpenAI, keeping costs predictable.

## Student authentication (Neon)

Only matriculas registered and marked active in Neon may use the proxy.

1. **Create the schema/table** (once, or whenever
   [`db/schema.sql`](db/schema.sql) changes):

   ```bash
   python -m scripts.init_db
   ```

   This creates the `pgl_proxy.students` table (dropping it first if it
   already exists, so re-running this wipes existing rows):

   | column          | type          | notes                                  |
   |-----------------|---------------|------------------------------------------|
   | `matricula`     | `text`        | primary key, the enrollment number     |
   | `id`            | `uuid`        | auto-generated surrogate id            |
   | `full_name`     | `text`        | from Canvas "Nome"                     |
   | `login_id`      | `text`        | from Canvas "ID de Login"               |
   | `sis_id`        | `text`        | from Canvas "ID do SIS"                 |
   | `course`        | `text`        | from Canvas "Seção"                     |
   | `role`          | `text`        | from Canvas "Papel"                     |
   | `is_active`     | `boolean`     | defaults to `true`                     |
   | `last_modified` | `timestamptz` | auto-updated by a trigger on `UPDATE`  |

2. **Manage matriculas** with the companion CLI:

   ```bash
   python -m scripts.manage_students add 20231001 20231002
   python -m scripts.manage_students deactivate 20231001
   python -m scripts.manage_students activate 20231001
   python -m scripts.manage_students list
   ```

   For a full roster export from Canvas ("People" page → export as CSV,
   or transcribed into [`db/roster.csv`](db/roster.csv) with columns
   `matricula,full_name,login_id,sis_id,course,role`), bulk-import it
   instead — this upserts by matricula and marks every imported row active:

   ```bash
   python -m scripts.import_students_csv db/roster.csv
   ```

3. Students set their matricula as `api_key` when configuring their client,
   e.g. `ChatOpenAI(base_url="http://your-server:8000/v1", api_key="20231001")`.

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

> **Streaming caveat:** Vercel serverless functions have an execution time
> limit (10s on the free plan, longer on paid plans). If streamed OpenAI
> responses run longer than that limit, consider hosting the proxy on a
> platform with long-lived processes instead (e.g. Render, Fly.io,
> Railway).
