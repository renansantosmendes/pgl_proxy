"""
init_db.py

Creates the `pgl_proxy` schema and the `students` table in Neon. Run this
once before starting the proxy for the first time.

WARNING: db/schema.sql drops and recreates the `students` table, so
re-running this script wipes any rows already registered. Re-register
matriculas with scripts/manage_students.py afterwards.

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

    print("Schema 'pgl_proxy' and table 'students' are ready.")


if __name__ == "__main__":
    asyncio.run(main())
