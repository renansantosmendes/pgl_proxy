"""Helpers for the Anthropic-compatible ``POST /v1/messages`` endpoint.

Students point LangChain's ``ChatAnthropic`` (or the raw ``anthropic`` SDK)
at this proxy, so the request and response bodies follow Anthropic's
Messages API exactly. These helpers hold the pure logic around that
passthrough: which models and fields are allowed, how cost caps are
enforced, how upstream errors are surfaced to the student, and how the
server-sent events stream is re-encoded. The FastAPI route and the
Anthropic client themselves live in ``app/main.py``.
"""

import json
from typing import Any

import anthropic
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Anthropic models students may request through ``/v1/messages``. Claude
#: Haiku 4.5 is the cheapest current Claude model ($1 / $5 per million
#: input / output tokens), which keeps per-class costs predictable.
ALLOWED_ANTHROPIC_MODELS: set[str] = {
    "claude-haiku-4-5",
}

#: The only Messages API fields the proxy will forward. Anything else is
#: dropped, so a student can't smuggle in billable or account-level options
#: (``metadata``, ``service_tier``, ``container``, ``mcp_servers``, ...).
ALLOWED_MESSAGES_FIELDS: set[str] = {
    "model",
    "messages",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
}

#: Tool ``type`` values accepted in the ``tools`` array. Only client-side
#: (custom) tools are allowed: Anthropic server tools such as web search or
#: code execution are billed per use on top of tokens.
ALLOWED_TOOL_TYPES: set[str] = {"custom"}

#: Upstream statuses that describe a problem with the student's request and
#: are relayed unchanged, so the student's SDK raises the matching error.
RELAYED_CLIENT_ERROR_STATUSES: set[int] = {400, 404, 413, 422, 429}


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def sanitize_messages_payload(payload: dict, max_tokens_limit: int) -> dict:
    """Drop unknown fields and enforce cost caps on a Messages API body.

    Parameters
    ----------
    payload:
        The student's raw JSON request body.
    max_tokens_limit:
        Upper bound accepted for ``max_tokens``.

    Unlike ``/v1/chat/completions``, a ``max_tokens`` above the limit is
    lowered to the limit instead of rejected, and a missing one is set to
    the limit: LangChain's ``ChatAnthropic`` sends the model's maximum
    output (64000 for Claude Haiku 4.5) unless the student sets it, so
    rejecting would break every default call. The cost cap is the same
    either way.

    Returns
    -------
    dict
        Only the keys in ``ALLOWED_MESSAGES_FIELDS``, with ``max_tokens``
        capped, ready to forward to the Anthropic SDK.

    Raises
    ------
    HTTPException
        400 if ``max_tokens`` is not a positive integer, or if ``tools``
        contains a server tool.
    """
    sanitized = {k: v for k, v in payload.items() if k in ALLOWED_MESSAGES_FIELDS}

    max_tokens = sanitized.get("max_tokens", max_tokens_limit)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="Field 'max_tokens' must be a positive integer",
        )
    sanitized["max_tokens"] = min(max_tokens, max_tokens_limit)

    for tool in sanitized.get("tools") or []:
        tool_type = tool.get("type", "custom") if isinstance(tool, dict) else None
        if tool_type not in ALLOWED_TOOL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Tool type '{tool_type}' is not allowed: only custom "
                    "(client-side) tools are supported"
                ),
            )

    return sanitized


def generation_input(payload: dict) -> dict:
    """Select the part of the request worth recording as the trace input.

    Parameters
    ----------
    payload:
        The sanitized Messages API request body.

    Returns
    -------
    dict
        ``system``, ``messages`` and ``tools`` (when present).
    """
    return {
        key: payload[key] for key in ("system", "messages", "tools") if key in payload
    }


def generation_parameters(payload: dict) -> dict[str, Any]:
    """Extract the sampling parameters recorded on the trace.

    Parameters
    ----------
    payload:
        The sanitized Messages API request body.

    Returns
    -------
    dict[str, Any]
        Scalar request parameters such as ``max_tokens`` and
        ``temperature``.
    """
    return {
        key: payload[key]
        for key in ("max_tokens", "temperature", "top_p", "top_k", "stream")
        if key in payload
    }


def usage_details(usage: Any) -> dict[str, int]:
    """Convert an Anthropic ``usage`` object into Langfuse usage details.

    Parameters
    ----------
    usage:
        ``Message.usage`` (or the ``usage`` of a ``message_start`` /
        ``message_delta`` stream event).

    Returns
    -------
    dict[str, int]
        Token counts keyed the way Langfuse prices Anthropic models.
    """
    if usage is None:
        return {}
    fields = {
        "input": "input_tokens",
        "output": "output_tokens",
        "cache_read_input_tokens": "cache_read_input_tokens",
        "cache_creation_input_tokens": "cache_creation_input_tokens",
    }
    details = {}
    for langfuse_key, anthropic_attribute in fields.items():
        value = getattr(usage, anthropic_attribute, None)
        if isinstance(value, int):
            details[langfuse_key] = value
    return details


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def format_sse_event(event_type: str, data: str) -> bytes:
    """Encode one server-sent event in the shape Anthropic's API streams.

    Parameters
    ----------
    event_type:
        The event name (``message_start``, ``content_block_delta``, ...).
    data:
        The event payload, already serialized as JSON.

    Returns
    -------
    bytes
        ``event: <type>\\ndata: <json>\\n\\n`` encoded as UTF-8.
    """
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


def error_body(error_type: str, message: str) -> dict:
    """Build an error body in Anthropic's format.

    Parameters
    ----------
    error_type:
        Anthropic error type, e.g. ``api_error``.
    message:
        Human-readable explanation.

    Returns
    -------
    dict
        ``{"type": "error", "error": {"type": ..., "message": ...}}``.
    """
    return {"type": "error", "error": {"type": error_type, "message": message}}


def stream_error_event(message: str) -> bytes:
    """Encode an ``error`` event for failures after streaming has begun.

    Parameters
    ----------
    message:
        Human-readable explanation sent to the student.

    Returns
    -------
    bytes
        A server-sent ``error`` event.
    """
    return format_sse_event("error", json.dumps(error_body("api_error", message)))


def upstream_error_response(error: anthropic.APIError) -> JSONResponse:
    """Translate an Anthropic SDK error into the response sent to the student.

    Request problems (bad input, unknown model, rate limits) are relayed
    with Anthropic's own status and body. Problems with the proxy's own
    credentials or with Anthropic's availability become ``502`` so they
    are not mistaken for the student's authentication failing.

    Parameters
    ----------
    error:
        Exception raised by the Anthropic SDK.

    Returns
    -------
    JSONResponse
        Error response in Anthropic's format.
    """
    if isinstance(error, anthropic.APIStatusError):
        if error.status_code in RELAYED_CLIENT_ERROR_STATUSES:
            body = error.body if isinstance(error.body, dict) else error_body(
                "invalid_request_error", error.message
            )
            return JSONResponse(status_code=error.status_code, content=body)
        if error.status_code in (401, 403):
            return JSONResponse(
                status_code=502,
                content=error_body(
                    "api_error", "The proxy's upstream credentials were rejected"
                ),
            )
    return JSONResponse(
        status_code=502,
        content=error_body("api_error", "Upstream Anthropic API is unavailable"),
    )
