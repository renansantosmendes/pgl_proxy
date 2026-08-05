"""
proxy_openai.py

A FastAPI reverse proxy that sits between students' LangChain clients and the
OpenAI API, so the real OpenAI API key is never exposed to students. All
requests are also traced in Langfuse for monitoring (cost, latency, prompts,
completions, errors).

How it works
------------
1. Students configure LangChain's ``ChatOpenAI`` to point at this proxy
   (``base_url="http://your-server:8000/v1"``). The ``api_key`` field can be
   any placeholder string, since it is never forwarded to OpenAI.
2. This server validates the requested model against an allowlist, then uses
   the REAL OpenAI API key (read from an environment variable) to call
   OpenAI via the Langfuse-instrumented OpenAI SDK client.
3. The real key lives only on this server and is never sent to, or seen by,
   student code.
4. Every call is automatically traced in Langfuse: prompts, completions,
   token usage, latency, and errors.

Note
----
Per-student authentication (token per student) was intentionally removed
for now to keep this version simple. Without it, anyone who can reach this
server can use it. Add authentication (e.g. shared class secret, per-student
tokens, or an API gateway) before exposing this publicly.

Running locally
----------------
    pip install fastapi uvicorn langfuse openai
    export OPENAI_API_KEY="sk-..."
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
    uvicorn proxy_openai:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

# The Langfuse-wrapped OpenAI client is a drop-in replacement for the
# official `openai` SDK client: every call made through it is automatically
# traced in Langfuse (prompt, completion, token usage, latency, errors).
from langfuse.openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: The real OpenAI API key. This is the only place in the whole system where
#: it exists, and it must be provided via an environment variable so it
#: never ends up committed to source control or handed to a student.
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

#: Set of model names students are allowed to request. Any request for a
#: model not in this set is rejected before it ever reaches OpenAI, which
#: keeps per-class costs predictable and prevents students from using
#: expensive models by accident (or on purpose).
ALLOWED_MODELS: set[str] = {
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-nano",
    "gpt-5.4-mini",
}

#: Langfuse-instrumented OpenAI client used for every request. Langfuse
#: picks up its own credentials (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
#: LANGFUSE_HOST) from environment variables automatically.
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_model(model_name: Optional[str]) -> None:
    """Ensure the requested model is on the class's allowlist.

    Parameters
    ----------
    model_name:
        The value of the ``"model"`` field from the student's request body,
        or ``None`` if the field was omitted entirely.

    Raises
    ------
    HTTPException
        400 if ``model_name`` is missing, or 403 if it is not present in
        ``ALLOWED_MODELS``.
    """
    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required")

    if model_name not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Model '{model_name}' is not allowed. "
                f"Allowed models: {sorted(ALLOWED_MODELS)}"
            ),
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PGL OpenAI Proxy",
    description=(
        "OpenAI-compatible proxy that restricts which models students can "
        "use, forwards requests to OpenAI using a key that is never exposed "
        "to student code, and traces every call in Langfuse for monitoring."
    ),
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward a chat completion request to OpenAI on behalf of a student.

    This endpoint mirrors OpenAI's ``POST /v1/chat/completions`` so that
    LangChain's ``ChatOpenAI`` (or the raw OpenAI SDK) can talk to it
    without any code changes beyond setting ``base_url``. Every call is
    traced in Langfuse automatically by the underlying client.

    Parameters
    ----------
    request:
        The incoming FastAPI request. Its JSON body is expected to match
        the OpenAI chat completions request schema (model, messages,
        stream, etc.).

    Returns
    -------
    dict | StreamingResponse
        The chat completion response for non-streaming requests, or a
        ``StreamingResponse`` that relays OpenAI's server-sent events for
        streaming requests (``"stream": true`` in the request body).

    Raises
    ------
    HTTPException
        400 if the request has no model field, or 403 if the requested
        model is not in ``ALLOWED_MODELS``.
    """
    payload = await request.json()
    validate_model(payload.get("model"))

    is_streaming = payload.get("stream", False)

    if is_streaming:
        return StreamingResponse(
            _stream_chat_completion(payload),
            media_type="text/event-stream",
        )

    completion = await openai_client.chat.completions.create(**payload)
    return JSONResponse(content=completion.model_dump())


async def _stream_chat_completion(payload: dict):
    """Relay a streaming chat completion from OpenAI chunk by chunk.

    Parameters
    ----------
    payload:
        The JSON body of the original student request, forwarded as-is
        (with ``stream`` already set to ``true``) to the OpenAI SDK.

    Yields
    ------
    bytes
        Each chunk formatted as a server-sent event, in the same shape
        OpenAI's own streaming API produces, so the student's client can
        parse it the way it expects.
    """
    stream = await openai_client.chat.completions.create(**payload)
    async for chunk in stream:
        yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness check endpoint.

    Returns
    -------
    dict[str, str]
        ``{"status": "ok"}`` if the server is up and able to respond.
    """
    return {"status": "ok"}