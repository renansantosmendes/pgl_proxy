"""
init_db.py

Creates the `pgl_proxy` schema and its tables (`students`, `rate_limits`)
in Neon. Idempotent — safe to run multiple times, including after
db/schema.sql changes; it never drops existing tables or rows.

Usage
-----
    python -m scripts.init_db
"""

import asyncio
from pathlib import Path

import asyncpg

from app.config import NEON_DATABASE_URL

SCHEMA_SQL_PATH = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


async def main() -> None:
    schema_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")

    conn = await asyncpg.connect(NEON_DATABASE_URL)
    try:
        await conn.execute(schema_sql)
    finally:
        await conn.close()

    print("Schema 'pgl_proxy' and its tables are ready.")


if __name__ == "__main__":
    asyncio.run(main())
