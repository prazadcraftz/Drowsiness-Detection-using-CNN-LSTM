"""Read-only inspection: recent sessions, one summary, or one event list."""

import argparse
import json
from pathlib import Path
import sqlite3


def query(path, *, session_id=None, events=False, limit=10):
    # mode=ro also ensures a typo cannot create a new database.
    connection = sqlite3.connect(Path(path).expanduser().resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        if events:
            if session_id is None:
                raise ValueError("--events requires --session-id")
            rows = connection.execute("SELECT * FROM events WHERE session_id=? ORDER BY started_at_source_seconds,id", (session_id,))
        elif session_id is not None:
            rows = connection.execute("SELECT s.*, m.* FROM sessions s LEFT JOIN session_metrics m ON m.session_id=s.id WHERE s.id=?", (session_id,))
        else:
            if limit < 1:
                raise ValueError("--limit must be positive")
            rows = connection.execute("SELECT id,source_name,started_at_utc,finalized FROM sessions ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path")
    parser.add_argument("--session-id", type=int)
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    try:
        print(json.dumps(query(args.db_path, session_id=args.session_id, events=args.events, limit=args.limit), indent=2))
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
