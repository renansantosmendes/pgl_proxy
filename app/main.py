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
   student's registration number as the trace's `user_id` (via
   `langfuse.propagate_attributes`), so usage in the Langfuse dashboard can
   be filtered/grouped per student.

Authentication
--------------
Students authenticate with a short-lived JWT, passed as the `api_key` in
their OpenAI/LangChain client (`ChatOpenAI(api_key=token)`). The SDK sends
it as `Authorization: Bearer <token>`. This server never sees a student's
password: the token is obtained separately, by the student, from
`pgl_auth_server` (via the `pgl_auth` PyPI package or its `/api/login`
endpoint directly), which verifies registration_number/senha against
`pgl_auth.students` and signs a JWT (HS256, ``sub`` = registration_number,
4h TTL). This server only:

1. Verifies the token's signature and expiry using the same `JWT_SECRET_KEY`
   (must be configured identically on both services).
2. Re-checks that the registration number is still active in
   `pgl_proxy.students` on every request — a token can outlive a
   mid-window deactivation, so this is the defense-in-depth check against
   that.

See db/schema.sql and scripts/init_db.py to set up `pgl_proxy.students`, and
scripts/manage_students.py to manage which registration numbers are
enrolled/active (this repo only owns enrollment status, not credentials).

Running locally
----------------
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
    export NEON_DATABASE_URL="postgresql://..."
    export JWT_SECRET_KEY="..."  # must match pgl_auth_server's JWT_SECRET_KEY
    python -m scripts.init_db  # once, to create the schema/table
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import json
from contextlib import asynccontextmanager
from typing import Optional

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

# The Langfuse-wrapped OpenAI client is a drop-in replacement for the
# official `openai` SDK client: every call made through it is automatically
# traced in Langfuse (prompt, completion, token usage, latency, errors).
from langfuse import propagate_attributes
from langfuse.openai import AsyncOpenAI

from app.config import JWT_SECRET_KEY, OPENAI_API_KEY
from app.db import check_and_increment_rate_limit, close_pool, is_enrollment_active

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

#: Per-registration_number request budget enforced in Neon (see app/db.py).
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


async def require_active_registration_number(
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Resolve and validate the student's registration number from a bearer JWT.

    Students pass the JWT issued by `pgl_auth_server` (after it verifies
    their registration_number/senha) as the `api_key` in their
    OpenAI/LangChain client, which the SDK sends as an
    `Authorization: Bearer <token>` header. This verifies the token's
    signature/expiry, then re-checks that the registration number is
    still active in Neon.

    Parameters
    ----------
    authorization:
        The raw `Authorization` header, expected to be
        ``"Bearer <jwt>"``.

    Returns
    -------
    str
        The validated registration number (the token's ``sub`` claim).

    Raises
    ------
    HTTPException
        401 if the header is missing/malformed or the token is invalid or
        expired, or 403 if the registration number is not registered or is
        inactive.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token: set your pgl_auth access token as the api_key",
        )

    token = authorization[len("Bearer "):].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token: set your pgl_auth access token as the api_key",
        )

    try:
        claims = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired: log in again with pgl_auth to get a new one",
        ) from None
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token") from None

    registration_number = claims.get("sub")
    if not registration_number:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not await is_enrollment_active(registration_number):
        raise HTTPException(
            status_code=403,
            detail="Registration number not recognized or inactive",
        )

    return registration_number


async def enforce_rate_limit(
    registration_number: str = Depends(require_active_registration_number),
) -> str:
    """Consume one unit of `registration_number`'s request budget, or reject.

    Chains onto `require_active_registration_number`, so callers get both
    authentication and rate limiting from a single dependency.

    Raises
    ------
    HTTPException
        429 if `registration_number` has already made
        `RATE_LIMIT_MAX_REQUESTS` requests within the last
        `RATE_LIMIT_WINDOW_SECONDS`.
    """
    allowed = await check_and_increment_rate_limit(
        registration_number, RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS}s"
            ),
        )
    return registration_number


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
    registration_number: str = Depends(enforce_rate_limit),
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
        401 if the bearer token (sent as the client's ``api_key``) is
        missing, malformed, invalid, or expired, 403 if the registration
        number is inactive or the requested model is not in
        ``ALLOWED_MODELS``, 429 if the registration number's rate limit
        was exceeded, 413 if the request body is too large, or 400 if the
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
            _stream_chat_completion(payload, registration_number),
            media_type="text/event-stream",
        )

    with propagate_attributes(user_id=registration_number):
        completion = await openai_client.chat.completions.create(
            **payload,
            metadata={"registration_number": registration_number},
        )
    return JSONResponse(content=completion.model_dump())


async def _stream_chat_completion(payload: dict, registration_number: str):
    """Relay a streaming chat completion from OpenAI chunk by chunk.

    Parameters
    ----------
    payload:
        The JSON body of the original student request, forwarded as-is
        (with ``stream`` already set to ``true``) to the OpenAI SDK.
    registration_number:
        The requesting student's registration number, attached to the
        Langfuse trace so usage can be attributed to them.

    Yields
    ------
    bytes
        Each chunk formatted as a server-sent event, in the same shape
        OpenAI's own streaming API produces, so the student's client can
        parse it the way it expects.
    """
    with propagate_attributes(user_id=registration_number):
        stream = await openai_client.chat.completions.create(
            **payload,
            metadata={"registration_number": registration_number},
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
