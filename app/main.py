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

Authentication
--------------
Students authenticate with their matricula (enrollment number), passed as
the `api_key` in their OpenAI/LangChain client (`ChatOpenAI(api_key="...")`).
The SDK sends it as `Authorization: Bearer <matricula>`, which this server
checks against the `pgl_proxy.students` table in Neon: the matricula must
exist and be marked active. See db/schema.sql and scripts/init_db.py to set
up that table, and scripts/manage_students.py to register matriculas.

Running locally
----------------
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
    export NEON_DATABASE_URL="postgresql://..."
    python -m scripts.init_db  # once, to create the schema/table
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

# The Langfuse-wrapped OpenAI client is a drop-in replacement for the
# official `openai` SDK client: every call made through it is automatically
# traced in Langfuse (prompt, completion, token usage, latency, errors).
from langfuse.openai import AsyncOpenAI

from app.config import OPENAI_API_KEY
from app.db import close_pool, is_enrollment_active

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


async def require_active_matricula(
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Resolve and validate the student's matricula from the request.

    Students pass their matricula as the `api_key` in their OpenAI/LangChain
    client, which the SDK sends as an `Authorization: Bearer <matricula>`
    header. This looks up that matricula in Neon and confirms it is
    registered and active.

    Parameters
    ----------
    authorization:
        The raw `Authorization` header, expected to be
        ``"Bearer <matricula>"``.

    Returns
    -------
    str
        The validated matricula.

    Raises
    ------
    HTTPException
        401 if the header is missing or malformed, or 403 if the matricula
        is not registered or is inactive.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token: set your matricula as the api_key",
        )

    matricula = authorization[len("Bearer "):].strip()
    if not matricula:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token: set your matricula as the api_key",
        )

    if not await is_enrollment_active(matricula):
        raise HTTPException(
            status_code=403,
            detail="Matricula not recognized or inactive",
        )

    return matricula


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_pool()


app = FastAPI(
    title="PGL OpenAI Proxy",
    description=(
        "OpenAI-compatible proxy that restricts which models students can "
        "use, forwards requests to OpenAI using a key that is never exposed "
        "to student code, and traces every call in Langfuse for monitoring."
    ),
    lifespan=lifespan,
)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    matricula: str = Depends(require_active_matricula),
):
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
        401 if the matricula (sent as the client's ``api_key``) is missing,
        403 if the matricula is not registered/active or the requested
        model is not in ``ALLOWED_MODELS``, or 400 if the request has no
        model field.
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