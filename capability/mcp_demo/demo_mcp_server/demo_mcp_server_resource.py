from local_demo_db import sqlite_demo_db

# Module-level DB connection — created once, reused by all tool calls.
_shared_db = sqlite_demo_db.connect_to_database()

def fetch_place(place: str) -> dict | None:
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


# =============================================================================
# MCP RESOURCES: read-only data at a URI. The APP decides when to read them,
#    not the LLM. Our agent (lesson 10) reads trips://destinations at the start
#    of every chat and puts it in the system prompt.
# =============================================================================
import json
from capability.mcp_demo.demo_mcp_server.fast_mcp_server_obj import mcp

@mcp.resource("trips://destinations", mime_type="application/json")
def destinations_resource() -> str:
    """All destinations with their trip type and best months."""
    return json.dumps(list_places(), ensure_ascii=False)


@mcp.resource("trips://place/{place_id}", mime_type="application/json")
def place_resource(place_id: str) -> str:
    """One destination's full record, e.g. trips://place/HAMPI (a URI template)."""
    return json.dumps(fetch_place(place_id) or {"error": f"{place_id} not found"}, ensure_ascii=False)
