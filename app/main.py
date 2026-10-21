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

#: Embedding models students may request. text-embedding-3-small is the
#: cheapest current OpenAI embedding model, so per-class costs stay predictable.
ALLOWED_EMBEDDING_MODELS: set[str] = {
    "text-embedding-3-small",
}

#: The only embeddings request fields the proxy will forward.
ALLOWED_EMBEDDING_FIELDS: set[str] = {
    "model",
    "input",
    "encoding_format",
    "dimensions",
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

#: Largest number of texts in one embeddings request (the OpenAI limit).
MAX_EMBEDDING_INPUTS = 2048

#: Largest total number of characters in one embeddings request (about
#: 75,000 tokens, well under the OpenAI limit of 300,000 tokens per request).
MAX_EMBEDDING_CHARS = 300_000

#: Largest vector size a student may ask for.
MAX_EMBEDDING_DIMENSIONS = 1536

#: Largest request body of an embeddings request; it must hold MAX_EMBEDDING_CHARS
#: of accented text plus JSON escapes.
MAX_EMBEDDING_BODY_BYTES = 2 * 1024 * 1024

#: Requests larger than this are rejected before being parsed as JSON.
MAX_REQUEST_BODY_BYTES = 256 * 1024  # 256 KiB

#: Per-registration_number request budget enforced in Neon (see app/db.py).
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

#: Embeddings have their own quota: a student indexing documents sends many
#: texts per request, so the chat limit would block them too early.
EMBEDDING_RATE_LIMIT_MAX_REQUESTS = 300
EMBEDDING_RATE_LIMIT_WINDOW_SECONDS = 60

#: Langfuse-instrumented OpenAI client used for every request. Langfuse
#: picks up its own credentials (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
#: LANGFUSE_HOST) from environment variables automatically.
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def validate_model(
    model_name: Optional[str], allowed_models: Optional[set[str]] = None
) -> None:
    """Ensure the requested model is on the class's allowlist.

    Parameters
    ----------
    model_name:
        The value of the "model" field from the student's request body,
        or "None" if the field was omitted entirely.
    allowed_models:
        The allowlist of the endpoint being called. Defaults to
        "ALLOWED_MODELS" (the chat completions models).

    Raises
    ------
    HTTPException
        400 if "model_name" is missing, or 403 if it is not present in
        "allowed_models".
    """
    if allowed_models is None:
        allowed_models = ALLOWED_MODELS

    if not model_name:
        raise HTTPException(status_code=400, detail="Field 'model' is required")

    if model_name not in allowed_models:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Model '{model_name}' is not allowed. "
                f"Allowed models: {sorted(allowed_models)}"
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


async def enforce_embedding_rate_limit(
    registration_number: str = Depends(require_active_registration_number),
) -> str:
    """Consume one unit of `registration_number`'s embeddings budget, or reject.

    Uses its own quota (`EMBEDDING_RATE_LIMIT_MAX_REQUESTS` per
    `EMBEDDING_RATE_LIMIT_WINDOW_SECONDS`), separate from the chat quota.

    Raises
    ------
    HTTPException
        429 if `registration_number` has used its embeddings quota for the
        current window.
    """
    allowed = await check_and_increment_rate_limit(
        registration_number,
        EMBEDDING_RATE_LIMIT_MAX_REQUESTS,
        EMBEDDING_RATE_LIMIT_WINDOW_SECONDS,
        kind="embeddings",
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {EMBEDDING_RATE_LIMIT_MAX_REQUESTS} "
                f"embedding requests per {EMBEDDING_RATE_LIMIT_WINDOW_SECONDS}s"
            ),
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


async def _read_json_body(
    request: Request, max_bytes: int = MAX_REQUEST_BODY_BYTES
) -> dict:
    """Read the request body, enforcing the size limit, and parse it as JSON.

    Parameters
    ----------
    request:
        The incoming FastAPI request.

    Returns
    -------
    dict
        The parsed JSON object.

    Raises
    ------
    HTTPException
        413 if the body exceeds `MAX_REQUEST_BODY_BYTES`, or 400 if it is
        not a valid JSON object.
    """
    body = await request.body()
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Request body exceeds {max_bytes} bytes",
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Request body is not valid JSON"
        ) from None

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON object"
        )
    return payload


def sanitize_embedding_payload(payload: dict) -> dict:
    """Drop unknown fields and enforce the limits of an embeddings request.

    Parameters
    ----------
    payload:
        The student's parsed JSON request body.

    Returns
    -------
    dict
        Only the keys in `ALLOWED_EMBEDDING_FIELDS`, ready to forward to the
        OpenAI SDK.

    Raises
    ------
    HTTPException
        400 if `input` is not a text or a list of texts, if it has more than
        `MAX_EMBEDDING_INPUTS` items or more than `MAX_EMBEDDING_CHARS`
        characters, or if `dimensions` is outside 1..`MAX_EMBEDDING_DIMENSIONS`.
    """
    sanitized = {k: v for k, v in payload.items() if k in ALLOWED_EMBEDDING_FIELDS}

    texts = sanitized.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise HTTPException(
            status_code=400,
            detail="Field 'input' must be a text or a non-empty list of texts",
        )
    if len(texts) > MAX_EMBEDDING_INPUTS:
        raise HTTPException(
            status_code=400,
            detail=f"'input' must have at most {MAX_EMBEDDING_INPUTS} texts",
        )
    if sum(len(t) for t in texts) > MAX_EMBEDDING_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"'input' must have at most {MAX_EMBEDDING_CHARS} characters in total",
        )

    dimensions = sanitized.get("dimensions")
    if dimensions is not None and (
        not isinstance(dimensions, int)
        or not 1 <= dimensions <= MAX_EMBEDDING_DIMENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"'dimensions' must be an integer between 1 and {MAX_EMBEDDING_DIMENSIONS}",
        )

    return sanitized


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
    payload = sanitize_payload(await _read_json_body(request))

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


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    registration_number: str = Depends(enforce_embedding_rate_limit),
):
    """Forward an embeddings request to OpenAI on behalf of a student.

    This endpoint mirrors OpenAI's ``POST /v1/embeddings``, so the OpenAI SDK
    or LangChain's ``OpenAIEmbeddings`` can call it by changing only
    ``base_url``. Only the models in `ALLOWED_EMBEDDING_MODELS` are accepted,
    and every call is traced in Langfuse by the underlying client.

    Parameters
    ----------
    request:
        The incoming FastAPI request. Its JSON body is expected to match the
        OpenAI embeddings request schema (model, input, dimensions...).

    Returns
    -------
    dict
        The embeddings response, as OpenAI returns it.

    Raises
    ------
    HTTPException
        401 if the bearer token is missing or invalid, 403 if the registration
        number is inactive or the model is not in `ALLOWED_EMBEDDING_MODELS`,
        429 if the rate limit was exceeded, 413 if the body is too large, or
        400 if the input breaks one of the limits.
    """
    payload = sanitize_embedding_payload(
        await _read_json_body(request, MAX_EMBEDDING_BODY_BYTES)
    )

    validate_model(payload.get("model"), ALLOWED_EMBEDDING_MODELS)

    with propagate_attributes(user_id=registration_number):
        result = await openai_client.embeddings.create(
            **payload,
            metadata={"registration_number": registration_number},
        )
    return JSONResponse(content=result.model_dump())


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
