"""
SQLite DB-call demos — the `fetch_place_fast*` helpers, extracted from the trip
planner demo into their own module.

Both read ONE joined row (places + stays + costs) for a destination, reusing a
single module-level connection (`_shared_db`) — the "connection reuse" pattern,
so there is no connect_to_database() on every call. They differ only in HOW they
read:

  fetch_place_fast   → grabs the lock manually, swaps row_factory to sqlite3.Row
  fetch_place_fast_2 → calls query_one() (lock handled internally), zips columns
"""

from __future__ import annotations

from local_demo_db import sqlite_demo_db

# Module-level DB connection — created once, reused by all tool calls.
_shared_db = sqlite_demo_db.connect_to_database()


def fetch_place_fast(place: str) -> dict | None:
    """Same as fetch_place_2 but reuses a shared DB connection.

    The only difference: no connect_to_database() on every call.
    This avoids repeated connection setup and "Seeded table..." noise.
    """
    import sqlite3
    with _shared_db._lock:
        original_factory = _shared_db._conn.row_factory
        _shared_db._conn.row_factory = sqlite3.Row
        try:
            cur = _shared_db._conn.cursor()

            # Main Query
            row = cur.execute("""
                SELECT p.*, s.stay_name, s.room_price_per_night,
                    c.travel_mode, c.travel_cost_per_person, c.daily_spend_per_person
                FROM places p
                JOIN stays s ON s.place_id = p.place_id
                JOIN costs c ON c.place_id = p.place_id
                WHERE UPPER(p.place_id) = UPPER(?) OR UPPER(p.name) = UPPER(?)
            """, (place, place)).fetchone()
        finally:
            # restore even if the query raises, so the shared connection isn't
            # left stuck on sqlite3.Row for every later query
            _shared_db._conn.row_factory = original_factory
    return dict(row) if row else None


def fetch_place_fast_2(place: str) -> dict | None:
    """fetch_place style — uses query_one() (thread lock handled internally)
    + manual zip for dict conversion. Reuses _shared_db (no repeated connects).

    Contrast with fetch_place_fast above:
      fetch_place_fast  → grabs _lock manually, swaps row_factory to sqlite3.Row
      fetch_place_fast_2 → calls query_one() which locks internally, zips cols
    """
    cols = (
        "place_id", "name", "trip_type", "best_months",
        "stay_name", "room_price_per_night",
        "travel_mode", "travel_cost_per_person", "daily_spend_per_person",
    )
    row = _shared_db.query_one("""
        SELECT p.place_id, p.name, p.trip_type, p.best_months,
               s.stay_name, s.room_price_per_night,
               c.travel_mode, c.travel_cost_per_person, c.daily_spend_per_person
        FROM places p
        JOIN stays s ON s.place_id = p.place_id
        JOIN costs c ON c.place_id = p.place_id
        WHERE UPPER(p.place_id) = UPPER(?) OR UPPER(p.name) = UPPER(?)
    """, (place, place))
    return dict(zip(cols, row)) if row else None

def list_places() -> list[dict]:
    """A short list of every destination."""
    import sqlite3
    with _shared_db._lock:
        original_factory = _shared_db._conn.row_factory
        _shared_db._conn.row_factory = sqlite3.Row
        try:
            cur = _shared_db._conn.cursor()

            # Main Query
            result = [dict(r) for r in cur.execute(
                    "SELECT place_id, name, trip_type, best_months FROM places ORDER BY trip_type, name"
                ).fetchall()]
        finally:
            # restore even if the query raises (see fetch_place_fast)
            _shared_db._conn.row_factory = original_factory

    return result