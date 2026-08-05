# Classroom OpenAI Proxy

A lightweight FastAPI reverse proxy that sits between students' LangChain
clients and the OpenAI API, so the real OpenAI API key is never exposed to
students. Every request is traced in [Langfuse](https://langfuse.com) for
monitoring (cost, latency, prompts, completions, errors).

## How it works

1. Students configure LangChain's `ChatOpenAI` (or the raw OpenAI SDK) to
   point at this proxy (`base_url="http://your-server:8000/v1"`). The
   `api_key` field can be any placeholder string, since it is never
   forwarded to OpenAI.
2. The server validates the requested model against an allowlist, then uses
   the **real** OpenAI API key (read from an environment variable) to call
   OpenAI via the Langfuse-instrumented OpenAI SDK client.
3. The real key lives only on the server and is never sent to, or seen by,
   student code.
4. Every call is automatically traced in Langfuse: prompts, completions,
   token usage, latency, and errors.

> **Note:** Per-student authentication (a token per student) was
> intentionally left out of this version to keep things simple. Without it,
> anyone who can reach the server can use it. Add authentication (a shared
> class secret, per-student tokens, or an API gateway) before exposing this
> publicly.

## Project structure

```
app/
  main.py           FastAPI application (the proxy itself)
tests/
  conftest.py        Shared fixtures (test client, mocked OpenAI client)
  test_health.py      Tests for GET /health
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
`base_url`. Accepts the same JSON body as OpenAI (`model`, `messages`,
`stream`, etc.).

- Returns `400` if the `model` field is missing.
- Returns `403` if the requested model is not in the allowlist.
- Returns the chat completion JSON for regular requests, or relays a
  server-sent events stream when `"stream": true` is set.

### `GET /health`

Simple liveness check. Returns `{"status": "ok"}`.

## Allowed models

The set of models students may request is defined in `ALLOWED_MODELS` in
[`app/main.py`](app/main.py). Requests for any other model are rejected
before they reach OpenAI, keeping costs predictable.

## Running locally

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL

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

The test suite mocks the OpenAI client entirely (see
[`tests/conftest.py`](tests/conftest.py)), so no real API key or network
access is required to run it. It covers:

- The `/health` endpoint.
- The model allowlist validation (`validate_model`).
- `/v1/chat/completions` for missing/disallowed models, successful
  non-streaming completions, streaming (SSE) completions, and upstream
  error propagation.

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

> **Streaming caveat:** Vercel serverless functions have an execution time
> limit (10s on the free plan, longer on paid plans). If streamed OpenAI
> responses run longer than that limit, consider hosting the proxy on a
> platform with long-lived processes instead (e.g. Render, Fly.io,
> Railway).
