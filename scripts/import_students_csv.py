"""
import_students_csv.py

Bulk-loads a roster CSV (matricula, full_name, login_id, sis_id, course,
role) into pgl_proxy.students, upserting by matricula. Every imported row
is marked active.

Usage
-----
    python -m scripts.import_students_csv db/roster.csv
"""

import asyncio
import csv
import sys
from pathlib import Path

import asyncpg

from app.config import NEON_DATABASE_URL


def read_roster(csv_path: Path) -> list[tuple[str, str, str, str, str, str]]:
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (
                row["matricula"],
                row["full_name"],
                row["login_id"],
                row["sis_id"],
                row["course"],
                row["role"],
            )
            for row in reader
        ]


async def main(csv_path: Path) -> None:
    rows = read_roster(csv_path)

    conn = await asyncpg.connect(NEON_DATABASE_URL)
    try:
        await conn.executemany(
            """
            INSERT INTO pgl_proxy.students
                (matricula, full_name, login_id, sis_id, course, role, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, TRUE)
            ON CONFLICT (matricula) DO UPDATE SET
                full_name = EXCLUDED.full_name,
                login_id  = EXCLUDED.login_id,
                sis_id    = EXCLUDED.sis_id,
                course    = EXCLUDED.course,
                role      = EXCLUDED.role,
                is_active = TRUE
            """,
            rows,
        )
    finally:
        await conn.close()

    print(f"Imported {len(rows)} rows from {csv_path}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.import_students_csv <csv_path>")
        sys.exit(1)

    asyncio.run(main(Path(sys.argv[1])))
