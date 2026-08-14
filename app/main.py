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
   token usage, latency, and errors. Each trace is tagged with the
   student's matricula as the trace's `user_id` (via
   `langfuse.propagate_attributes`), so usage in the Langfuse dashboard can
   be filtered/grouped per student.

Authentication
--------------
Students authenticate with `"<matricula>:<senha>"` (enrollment number and
password), passed as the `api_key` in their OpenAI/LangChain client
(`ChatOpenAI(api_key="matricula:senha")`). The SDK sends it as
`Authorization: Bearer <matricula>:<senha>`, which this server checks
against the `pgl_proxy.students` table in Neon: the matricula must exist,
be marked active, and `senha` must match its stored bcrypt hash. See
db/schema.sql and scripts/init_db.py to set up that table, and
scripts/manage_students.py to register matriculas and set passwords.

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

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

# The Langfuse-wrapped OpenAI client is a drop-in replacement for the
# official `openai` SDK client: every call made through it is automatically
# traced in Langfuse (prompt, completion, token usage, latency, errors).
from langfuse import propagate_attributes
from langfuse.openai import AsyncOpenAI

from app.config import OPENAI_API_KEY
from app.db import (
    check_and_increment_rate_limit,
    close_pool,
    verify_student_credentials,
)

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

#: The only OpenAI chat completion fields the proxy will forward. Anything
#: else in the student's request body is dropped, which both keeps unknown
#: fields from reaching the OpenAI SDK call and prevents a student payload
#: from colliding with the `langfuse_*` kwargs this proxy adds itself.
ALLOWED_PAYLOAD_FIELDS: set[str] = {
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
}

#: Upper bound on tokens generated per request, to cap cost per call.
MAX_TOKENS_LIMIT = 2048

#: Upper bound on how many completions a single request may ask for.
MAX_N = 1

#: Requests larger than this are rejected before being parsed as JSON.
MAX_REQUEST_BODY_BYTES = 256 * 1024  # 256 KiB

#: Per-matricula request budget enforced in Neon (see app/db.py).
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

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
    """Resolve and validate the student's matricula/senha from the request.

    Students pass ``"<matricula>:<senha>"`` as the `api_key` in their
    OpenAI/LangChain client, which the SDK sends as an
    ``Authorization: Bearer <matricula>:<senha>`` header. This looks up
    the matricula in Neon and confirms it is registered, active, and that
    `senha` matches its stored password.

    Parameters
    ----------
    authorization:
        The raw `Authorization` header, expected to be
        ``"Bearer <matricula>:<senha>"``.

    Returns
    -------
    str
        The validated matricula.

    Raises
    ------
    HTTPException
        401 if the header is missing/malformed or the credentials are
        invalid, or 403 if the matricula is not registered or is inactive.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token: set 'matricula:senha' as the api_key",
        )

    token = authorization[len("Bearer "):].strip()
    matricula, _, senha = token.partition(":")
    matricula = matricula.strip()
    senha = senha.strip()
    if not matricula or not senha:
        raise HTTPException(
            status_code=401,
            detail="Malformed bearer token: expected api_key='matricula:senha'",
        )

    if not await verify_student_credentials(matricula, senha):
        raise HTTPException(
            status_code=403,
            detail="Invalid matricula/senha, or matricula is inactive",
        )

    return matricula


async def enforce_rate_limit(
    matricula: str = Depends(require_active_matricula),
) -> str:
    """Consume one unit of `matricula`'s request budget, or reject.

    Chains onto `require_active_matricula`, so callers get both
    authentication and rate limiting from a single dependency.

    Raises
    ------
    HTTPException
        429 if `matricula` has already made `RATE_LIMIT_MAX_REQUESTS`
        requests within the last `RATE_LIMIT_WINDOW_SECONDS`.
    """
    allowed = await check_and_increment_rate_limit(
        matricula, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS}s"
            ),
        )
    return matricula


def sanitize_payload(payload: dict) -> dict:
    """Drop unknown fields and enforce cost caps on the request body.

    Parameters
    ----------
    payload:
        The student's raw JSON request body.

    Returns
    -------
    dict
        Only the keys in `ALLOWED_PAYLOAD_FIELDS`, ready to forward to the
        OpenAI SDK.

    Raises
    ------
    HTTPException
        400 if `max_tokens`/`max_completion_tokens` exceeds
        `MAX_TOKENS_LIMIT`, or `n` exceeds `MAX_N`.
    """
    sanitized = {k: v for k, v in payload.items() if k in ALLOWED_PAYLOAD_FIELDS}

    for field in ("max_tokens", "max_completion_tokens"):
        value = sanitized.get(field)
        if isinstance(value, int) and value > MAX_TOKENS_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=f"'{field}' must be <= {MAX_TOKENS_LIMIT}",
            )

    n = sanitized.get("n")
    if isinstance(n, int) and n > MAX_N:
        raise HTTPException(status_code=400, detail=f"'n' must be <= {MAX_N}")

    return sanitized


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
    matricula: str = Depends(enforce_rate_limit),
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
        401 if the matricula/senha (sent as the client's ``api_key``) is
        missing or malformed, 403 if the credentials are invalid, the
        matricula is inactive, or the requested model is not in
        ``ALLOWED_MODELS``, 429 if the matricula's rate limit was
        exceeded, 413 if the request body is too large, or 400 if the
        request has no model field or exceeds a cost cap.
    """
    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
        )

    try:
        payload = sanitize_payload(json.loads(body))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Request body is not valid JSON"
        ) from None

    validate_model(payload.get("model"))

    is_streaming = payload.get("stream", False)

    if is_streaming:
        return StreamingResponse(
            _stream_chat_completion(payload, matricula),
            media_type="text/event-stream",
        )

    with propagate_attributes(user_id=matricula):
        completion = await openai_client.chat.completions.create(
            **payload,
            metadata={"matricula": matricula},
        )
    return JSONResponse(content=completion.model_dump())


async def _stream_chat_completion(payload: dict, matricula: str):
    """Relay a streaming chat completion from OpenAI chunk by chunk.

    Parameters
    ----------
    payload:
        The JSON body of the original student request, forwarded as-is
        (with ``stream`` already set to ``true``) to the OpenAI SDK.
    matricula:
        The requesting student's matricula, attached to the Langfuse trace
        so usage can be attributed to them.

    Yields
    ------
    bytes
        Each chunk formatted as a server-sent event, in the same shape
        OpenAI's own streaming API produces, so the student's client can
        parse it the way it expects.
    """
    with propagate_attributes(user_id=matricula):
        stream = await openai_client.chat.completions.create(
            **payload,
            metadata={"matricula": matricula},
        )
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