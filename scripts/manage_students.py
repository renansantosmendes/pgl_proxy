"""
manage_students.py

Small CLI to register and manage which student matriculas may use the
proxy. Run `python -m scripts.init_db` first to create the table.

Usage
-----
    python -m scripts.manage_students add 20231001 20231002
    python -m scripts.manage_students deactivate 20231001
    python -m scripts.manage_students activate 20231001
    python -m scripts.manage_students list

Passwords aren't managed here: students set/verify their own via
pgl_auth_server (`pgl_auth.students`), not this repo.
"""

import argparse
import asyncio

import asyncpg

from app.config import NEON_DATABASE_URL


async def add(matriculas: list[str]) -> None:
    conn = await asyncpg.connect(NEON_DATABASE_URL)
    try:
        await conn.executemany(
            """
            INSERT INTO pgl_proxy.students (matricula)
            VALUES ($1)
            ON CONFLICT (matricula) DO NOTHING
            """,
            [(m,) for m in matriculas],
        )
    finally:
        await conn.close()
    print(f"Added (or already present): {', '.join(matriculas)}")


async def set_active(matriculas: list[str], is_active: bool) -> None:
    conn = await asyncpg.connect(NEON_DATABASE_URL)
    try:
        await conn.executemany(
            "UPDATE pgl_proxy.students SET is_active = $2 WHERE matricula = $1",
            [(m, is_active) for m in matriculas],
        )
    finally:
        await conn.close()
    state = "activated" if is_active else "deactivated"
    print(f"{state.capitalize()}: {', '.join(matriculas)}")


async def list_students() -> None:
    conn = await asyncpg.connect(NEON_DATABASE_URL)
    try:
        rows = await conn.fetch(
            "SELECT matricula, is_active, last_modified "
            "FROM pgl_proxy.students ORDER BY matricula"
        )
    finally:
        await conn.close()

    if not rows:
        print("No students registered yet.")
        return

    for row in rows:
        status = "active" if row["is_active"] else "inactive"
        print(f"{row['matricula']}\t{status}\t{row['last_modified']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Register one or more matriculas")
    add_parser.add_argument("matriculas", nargs="+")

    activate_parser = subparsers.add_parser("activate", help="Mark matriculas active")
    activate_parser.add_argument("matriculas", nargs="+")

    deactivate_parser = subparsers.add_parser(
        "deactivate", help="Mark matriculas inactive"
    )
    deactivate_parser.add_argument("matriculas", nargs="+")

    subparsers.add_parser("list", help="List all registered matriculas")

    args = parser.parse_args()

    if args.command == "add":
        asyncio.run(add(args.matriculas))
    elif args.command == "activate":
        asyncio.run(set_active(args.matriculas, True))
    elif args.command == "deactivate":
        asyncio.run(set_active(args.matriculas, False))
    elif args.command == "list":
        asyncio.run(list_students())


if __name__ == "__main__":
    main()
