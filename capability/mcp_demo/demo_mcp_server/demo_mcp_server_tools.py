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
# MCP TOOLS: the LLM decides when to call these
#    FastMCP turns the type hints into a JSON Schema and the docstring into the
#    description. That description is ALL the LLM knows about the tool, so be clear.
# =============================================================================
from typing import Annotated
from pydantic import Field
import math
from capability.mcp_demo.demo_mcp_server.fast_mcp_server_obj import mcp

@mcp.tool()
def db_lookup(
    place: Annotated[str, Field(description="Place name like 'Goa' or id like 'PONDY'")],
) -> dict:
    """Fetch a destination's trip type, best months, stay, room price, travel cost and daily spend from the database."""
    record = fetch_place(place)
    return record if record else {"error": f"{place} not found. Use list_destinations to see valid places."}


@mcp.tool()
def list_destinations() -> list[dict]:
    """Return all destinations in the database."""
    return list_places()


@mcp.tool()
def compute_trip_cost(
    travel_cost_per_person: Annotated[float, Field(ge=0, description="Return travel cost for one person, in rupees")],
    room_price_per_night: Annotated[float, Field(ge=0, description="Price of one room for one night (a room fits 2)")],
    daily_spend_per_person: Annotated[float, Field(ge=0, description="Food, local travel and fun per person per day")],
    people: Annotated[int, Field(ge=1, le=20, description="Number of friends travelling (1-20)")],
    nights: Annotated[int, Field(ge=1, le=30, description="Number of nights (1-30)")],
    budget_per_person: Annotated[float | None, Field(description="Optional budget per person, in rupees")] = None,
) -> dict:
    """
    Compute the trip cost per person with exact arithmetic.
    One room fits 2 people. Returns: rooms_needed, stay_share_per_person,
    food_and_fun_per_person, total_per_person, and (if a budget is given)
    within_budget and money_left_per_person (negative = over budget).
    """
    rooms_needed = math.ceil(people / 2)
    stay_share = round(rooms_needed * room_price_per_night * nights / people, 2)
    food_and_fun = round(daily_spend_per_person * nights, 2)
    total = round(travel_cost_per_person + stay_share + food_and_fun, 2)
    result = {
        "rooms_needed": rooms_needed,
        "stay_share_per_person": stay_share,
        "food_and_fun_per_person": food_and_fun,
        "total_per_person": total,
    }
    if budget_per_person is not None:
        result["within_budget"] = total <= budget_per_person
        result["money_left_per_person"] = round(budget_per_person - total, 2)
    return result
