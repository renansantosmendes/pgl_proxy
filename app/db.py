"""Neon-backed lookup of which student registration numbers may use the
proxy, and enforcement of a per-registration_number request rate limit."""

import datetime
from typing import Optional

import asyncpg

from app.config import NEON_DATABASE_URL

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(NEON_DATABASE_URL, min_size=1, max_size=5)
    return _pool


async def close_pool() -> None:
    """Close the shared connection pool, if one was created."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def is_enrollment_active(registration_number: str) -> bool:
    """Check whether `registration_number` is registered and active in Neon.

    Used as defense in depth after JWT verification: a token stays valid
    for its whole TTL (see pgl_auth_server, 4h) even if the student is
    deactivated mid-window, so every request re-checks the live roster.

    Parameters
    ----------
    registration_number:
        The student enrollment number to look up.

    Returns
    -------
    bool
        True if a row exists for `registration_number` with
        `is_active = true`, False otherwise (including when the
        registration number is unknown).
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT is_active FROM pgl_proxy.students WHERE registration_number = $1",
        registration_number,
    )
    return row is not None and row["is_active"]


async def check_and_increment_rate_limit(
    registration_number: str, max_requests: int, window_seconds: int
) -> bool:
    """Atomically check and consume one request of `registration_number`'s quota.

    Implements a fixed-window counter in `pgl_proxy.rate_limits`: the row
    is locked with `FOR UPDATE` before it's read and written, so concurrent
    requests from the same registration number can't race past the limit.

    Parameters
    ----------
    registration_number:
        The student enrollment number making the request.
    max_requests:
        How many requests `registration_number` may make within `window_seconds`.
    window_seconds:
        Length of the fixed window, in seconds.

    Returns
    -------
    bool
        True if the request is within the limit (and has been counted),
        False if `registration_number` has already used up its quota for
        the current window (the request should be rejected, and is not
        counted again).
    """
    pool = await get_pool()
    window = datetime.timedelta(seconds=window_seconds)

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT window_start, request_count
                FROM pgl_proxy.rate_limits
                WHERE registration_number = $1
                FOR UPDATE
                """,
                registration_number,
            )

            now = datetime.datetime.now(datetime.timezone.utc)

            if row is None:
                await conn.execute(
                    """
                    INSERT INTO pgl_proxy.rate_limits (registration_number, window_start, request_count)
                    VALUES ($1, $2, 1)
                    """,
                    registration_number,
                    now,
                )
                return True

            window_expired = now - row["window_start"] >= window

            if window_expired:
                await conn.execute(
                    """
                    UPDATE pgl_proxy.rate_limits
                    SET window_start = $2, request_count = 1
                    WHERE registration_number = $1
                    """,
                    registration_number,
                    now,
                )
                return True

            if row["request_count"] >= max_requests:
                return False

            await conn.execute(
                """
                UPDATE pgl_proxy.rate_limits
                SET request_count = request_count + 1
                WHERE registration_number = $1
                """,
                registration_number,
            )
            return True
