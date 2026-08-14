"""Neon-backed lookup of which student matriculas may use the proxy, and
enforcement of a per-matricula request rate limit."""

import datetime
from typing import Optional

import asyncpg
import bcrypt

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


async def verify_student_credentials(matricula: str, senha: str) -> bool:
    """Check whether `matricula`/`senha` is a valid, active student login.

    Parameters
    ----------
    matricula:
        The student enrollment number to look up.
    senha:
        The plaintext password to verify against the stored bcrypt hash.

    Returns
    -------
    bool
        True if `matricula` exists, is active, has a password set, and
        `senha` matches its stored hash. False otherwise (including when
        the matricula is unknown or has no password configured).
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT is_active, password_hash FROM pgl_proxy.students WHERE matricula = $1",
        matricula,
    )
    if row is None or not row["is_active"] or not row["password_hash"]:
        return False

    return bcrypt.checkpw(senha.encode("utf-8"), row["password_hash"].encode("utf-8"))


async def register_password(matricula: str, senha: str) -> str:
    """Set `matricula`'s password, but only on its first registration.

    Locks the student's row before checking, so two concurrent register
    calls for the same matricula can't both succeed.

    Parameters
    ----------
    matricula:
        The student enrollment number to register a password for.
    senha:
        The plaintext password to hash and store.

    Returns
    -------
    str
        ``"ok"`` if the password was set, ``"not_found_or_inactive"`` if
        `matricula` isn't a registered/active student, or
        ``"already_registered"`` if it already has a password (use the
        admin CLI to reset it instead).
    """
    pool = await get_pool()
    password_hash = bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT is_active, password_hash
                FROM pgl_proxy.students
                WHERE matricula = $1
                FOR UPDATE
                """,
                matricula,
            )
            if row is None or not row["is_active"]:
                return "not_found_or_inactive"
            if row["password_hash"]:
                return "already_registered"

            await conn.execute(
                "UPDATE pgl_proxy.students SET password_hash = $2 WHERE matricula = $1",
                matricula,
                password_hash,
            )
            return "ok"


async def check_and_increment_keyed_rate_limit(
    key: str, max_requests: int, window_seconds: int
) -> bool:
    """Like `check_and_increment_rate_limit`, but for an arbitrary text key.

    Backs endpoints that must be rate-limited before (or regardless of)
    resolving a valid matricula, e.g. by client IP on `/v1/register`.
    Stored separately in `pgl_proxy.auth_rate_limits`, which isn't FK'd to
    `students`, so it accepts keys that aren't matriculas.

    Parameters
    ----------
    key:
        Arbitrary rate-limit bucket identifier (e.g. ``f"register-ip:{ip}"``).
    max_requests:
        How many requests `key` may make within `window_seconds`.
    window_seconds:
        Length of the fixed window, in seconds.

    Returns
    -------
    bool
        True if the request is within the limit (and has been counted),
        False if `key` has already used up its quota for the current
        window.
    """
    pool = await get_pool()
    window = datetime.timedelta(seconds=window_seconds)

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT window_start, request_count
                FROM pgl_proxy.auth_rate_limits
                WHERE limit_key = $1
                FOR UPDATE
                """,
                key,
            )

            now = datetime.datetime.now(datetime.timezone.utc)

            if row is None:
                await conn.execute(
                    """
                    INSERT INTO pgl_proxy.auth_rate_limits (limit_key, window_start, request_count)
                    VALUES ($1, $2, 1)
                    """,
                    key,
                    now,
                )
                return True

            window_expired = now - row["window_start"] >= window

            if window_expired:
                await conn.execute(
                    """
                    UPDATE pgl_proxy.auth_rate_limits
                    SET window_start = $2, request_count = 1
                    WHERE limit_key = $1
                    """,
                    key,
                    now,
                )
                return True

            if row["request_count"] >= max_requests:
                return False

            await conn.execute(
                """
                UPDATE pgl_proxy.auth_rate_limits
                SET request_count = request_count + 1
                WHERE limit_key = $1
                """,
                key,
            )
            return True


async def check_and_increment_rate_limit(
    matricula: str, max_requests: int, window_seconds: int
) -> bool:
    """Atomically check and consume one request of `matricula`'s quota.

    Implements a fixed-window counter in `pgl_proxy.rate_limits`: the row
    is locked with `FOR UPDATE` before it's read and written, so concurrent
    requests from the same matricula can't race past the limit.

    Parameters
    ----------
    matricula:
        The student enrollment number making the request.
    max_requests:
        How many requests `matricula` may make within `window_seconds`.
    window_seconds:
        Length of the fixed window, in seconds.

    Returns
    -------
    bool
        True if the request is within the limit (and has been counted),
        False if `matricula` has already used up its quota for the current
        window (the request should be rejected, and is not counted again).
    """
    pool = await get_pool()
    window = datetime.timedelta(seconds=window_seconds)

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT window_start, request_count
                FROM pgl_proxy.rate_limits
                WHERE matricula = $1
                FOR UPDATE
                """,
                matricula,
            )

            now = datetime.datetime.now(datetime.timezone.utc)

            if row is None:
                await conn.execute(
                    """
                    INSERT INTO pgl_proxy.rate_limits (matricula, window_start, request_count)
                    VALUES ($1, $2, 1)
                    """,
                    matricula,
                    now,
                )
                return True

            window_expired = now - row["window_start"] >= window

            if window_expired:
                await conn.execute(
                    """
                    UPDATE pgl_proxy.rate_limits
                    SET window_start = $2, request_count = 1
                    WHERE matricula = $1
                    """,
                    matricula,
                    now,
                )
                return True

            if row["request_count"] >= max_requests:
                return False

            await conn.execute(
                """
                UPDATE pgl_proxy.rate_limits
                SET request_count = request_count + 1
                WHERE matricula = $1
                """,
                matricula,
            )
            return True
