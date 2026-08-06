"""Neon-backed lookup of which student matriculas may use the proxy."""

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


async def is_enrollment_active(matricula: str) -> bool:
    """Check whether `matricula` is registered and active in Neon.

    Parameters
    ----------
    matricula:
        The student enrollment number to look up.

    Returns
    -------
    bool
        True if a row exists for `matricula` with `is_active = true`,
        False otherwise (including when the matricula is unknown).
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT is_active FROM pgl_proxy.students WHERE matricula = $1",
        matricula,
    )
    return row is not None and row["is_active"]
