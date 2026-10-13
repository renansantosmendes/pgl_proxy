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

Anthropic models
----------------
``POST /v1/messages`` mirrors Anthropic's Messages API for LangChain's
``ChatAnthropic`` / the ``anthropic`` SDK, accepting the same JWT in the
``x-api-key`` header. Only ``claude-haiku-4-5`` is allowed; see
``app/anthropic_messages.py`` for the guardrails. It is enabled only when
ANTHROPIC_API_KEY is set.

Running locally
----------------
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."  # optional, enables /v1/messages
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
    export NEON_DATABASE_URL="postgresql://..."
    export JWT_SECRET_KEY="..."  # must match pgl_auth_server's JWT_SECRET_KEY
    python -m scripts.init_db  # once, to create the schema/table
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import datetime
import json
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

# The Langfuse-wrapped OpenAI client is a drop-in replacement for the
# official `openai` SDK client: every call made through it is automatically
# traced in Langfuse (prompt, completion, token usage, latency, errors).
from langfuse import get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI

from app.anthropic_messages import (
    ALLOWED_ANTHROPIC_MODELS,
    format_sse_event,
    generation_input,
    generation_parameters,
    sanitize_messages_payload,
    stream_error_event,
    upstream_error_response,
    usage_details,
)
from app.config import ANTHROPIC_API_KEY, JWT_SECRET_KEY, OPENAI_API_KEY
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

#: Anthropic client used by `/v1/messages`, or ``None`` when
#: ANTHROPIC_API_KEY is not configured. The key is passed explicitly so the
#: SDK never falls back to other credential sources on the server.
anthropic_client: Optional[anthropic.AsyncAnthropic] = (
    anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
)


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
        The value of the ``"model"`` field from the student's request body,
        or ``None`` if the field was omitted entirely.
    allowed_models:
        The allowlist for the endpoint being called. Defaults to
        ``ALLOWED_MODELS`` (the OpenAI models of `/v1/chat/completions`).

    Raises
    ------
    HTTPException
        400 if ``model_name`` is missing, or 403 if it is not present in
        ``allowed_models``.
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

    return await _registration_number_from_token(token)


async def require_active_registration_number_from_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
) -> str:
    """Resolve the student's registration number for Anthropic-style clients.

    LangChain's ``ChatAnthropic`` and the ``anthropic`` SDK send the
    ``api_key`` in the ``x-api-key`` header instead of ``Authorization``.
    This accepts either: ``x-api-key: <jwt>`` first, falling back to
    ``Authorization: Bearer <jwt>``.

    Parameters
    ----------
    authorization:
        The raw `Authorization` header, if any.
    x_api_key:
        The raw `x-api-key` header, if any.

    Returns
    -------
    str
        The validated registration number (the token's ``sub`` claim).

    Raises
    ------
    HTTPException
        401 if no token was sent or it is invalid or expired, or 403 if the
        registration number is not registered or is inactive.
    """
    if x_api_key and x_api_key.strip():
        return await _registration_number_from_token(x_api_key.strip())
    return await require_active_registration_number(authorization=authorization)


async def _registration_number_from_token(token: str) -> str:
    """Verify a pgl_auth JWT and check its registration number is active.

    Parameters
    ----------
    token:
        The encoded JWT issued by `pgl_auth_server`.

    Returns
    -------
    str
        The validated registration number (the token's ``sub`` claim).

    Raises
    ------
    HTTPException
        401 if the token is invalid or expired, or 403 if the registration
        number is not registered or is inactive.
    """
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
    return await _consume_rate_limit(registration_number)


async def enforce_rate_limit_for_api_key(
    registration_number: str = Depends(require_active_registration_number_from_api_key),
) -> str:
    """Same as `enforce_rate_limit`, for clients that send `x-api-key`.

    The budget is shared with `/v1/chat/completions`: a student has one
    request quota regardless of which provider they call.

    Raises
    ------
    HTTPException
        429 if the registration number's quota for the current window is
        used up.
    """
    return await _consume_rate_limit(registration_number)


async def _consume_rate_limit(registration_number: str) -> str:
    """Count one request against `registration_number`'s quota, or reject.

    Raises
    ------
    HTTPException
        429 if the quota for the current window is already used up.
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


async def _read_json_body(request: Request) -> dict:
    """Read the request body, enforcing the size limit, and parse it as JSON.

    Raises
    ------
    HTTPException
        413 if the body exceeds `MAX_REQUEST_BODY_BYTES`, or 400 if it is
        not a valid JSON object.
    """
    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
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


@app.exception_handler(anthropic.APIError)
async def handle_anthropic_error(request: Request, error: anthropic.APIError):
    """Turn Anthropic SDK errors into Anthropic-shaped error responses.

    See `app.anthropic_messages.upstream_error_response` for which upstream
    statuses are relayed and which become 502.
    """
    return upstream_error_response(error)


@app.post("/v1/messages")
async def messages(
    request: Request,
    registration_number: str = Depends(enforce_rate_limit_for_api_key),
):
    """Forward an Anthropic Messages API request on behalf of a student.

    Mirrors Anthropic's ``POST /v1/messages`` so LangChain's
    ``ChatAnthropic`` (or the ``anthropic`` SDK) works by setting
    ``base_url`` to this proxy and ``api_key`` to the student's pgl_auth
    JWT. The body is forwarded as-is after the field allowlist and cost
    caps; the response (or event stream) comes back unchanged. Each call is
    recorded in Langfuse as a generation tagged with the registration
    number.

    Returns
    -------
    JSONResponse | StreamingResponse
        The Anthropic ``Message`` for regular requests, or Anthropic's
        server-sent events when ``"stream": true`` is set.

    Raises
    ------
    HTTPException
        401/403 for authentication, 429 for the rate limit, 413 for an
        oversized body, 400 for invalid JSON, a missing or excessive
        ``max_tokens``, or a server tool, 403 for a model outside
        ``ALLOWED_ANTHROPIC_MODELS``, and 503 when ANTHROPIC_API_KEY is not
        configured.
    """
    if anthropic_client is None:
        raise HTTPException(
            status_code=503, detail="Anthropic models are not enabled on this proxy"
        )

    payload = sanitize_messages_payload(
        await _read_json_body(request), MAX_TOKENS_LIMIT
    )
    validate_model(payload.get("model"), ALLOWED_ANTHROPIC_MODELS)

    langfuse = get_client()
    with propagate_attributes(user_id=registration_number):
        generation = langfuse.start_observation(
            name="anthropic-messages",
            as_type="generation",
            model=payload["model"],
            input=generation_input(payload),
            model_parameters=generation_parameters(payload),
        )

    try:
        response = await anthropic_client.messages.create(**payload)
    except anthropic.APIError as error:
        generation.update(level="ERROR", status_message=str(error))
        generation.end()
        raise

    if payload.get("stream"):
        return StreamingResponse(
            _relay_anthropic_stream(response, generation),
            media_type="text/event-stream",
        )

    generation.update(
        output=response.to_dict(mode="json").get("content"),
        usage_details=usage_details(response.usage),
    )
    generation.end()
    return JSONResponse(content=response.to_dict(mode="json"))


async def _relay_anthropic_stream(stream, generation):
    """Relay Anthropic's event stream unchanged, recording usage in Langfuse.

    Parameters
    ----------
    stream:
        The async iterator of events returned by
        ``messages.create(stream=True)``.
    generation:
        The Langfuse generation opened for this request; it is ended when
        the stream finishes, fails, or the client disconnects.

    Yields
    ------
    bytes
        Each event as a server-sent event in Anthropic's wire format, or a
        final ``error`` event if the upstream stream fails midway.
    """
    output_parts: list[str] = []
    usage: dict[str, int] = {}
    first_token_seen = False
    try:
        async for event in stream:
            if event.type == "message_start":
                usage.update(usage_details(event.message.usage))
            elif event.type == "content_block_delta":
                if not first_token_seen:
                    first_token_seen = True
                    generation.update(
                        completion_start_time=datetime.datetime.now(datetime.timezone.utc)
                    )
                if event.delta.type == "text_delta":
                    output_parts.append(event.delta.text)
            elif event.type == "message_delta":
                usage.update(usage_details(event.usage))
            yield format_sse_event(event.type, event.to_json(indent=None))
    except anthropic.APIError as error:
        generation.update(level="ERROR", status_message=str(error))
        yield stream_error_event("Upstream Anthropic stream failed")
    finally:
        generation.update(output="".join(output_parts), usage_details=usage)
        generation.end()


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness check endpoint.

    Returns
    -------
    dict[str, str]
        ``{"status": "ok"}`` if the server is up and able to respond.
    """
    return {"status": "ok"}
