"""
SQLite demo database — CSV-driven, auto-seeded, thread-safe wrapper.

Provides SQLiteDemoDB, a lightweight SQLite wrapper that:
  1. Auto-creates tables from every *.csv in the data/ folder on first connect
  2. Auto-detects column types (INTEGER / REAL / TEXT) from CSV values
  3. Supports primary keys via a ``*`` header convention (single or composite)
  4. Guards all operations with a threading.Lock() for concurrent access
  5. Exposes simple query / execute / drop_tables / print helpers

CSV primary-key convention (in the header row):
  - ``place_id*,name,trip_type``           →  PRIMARY KEY (place_id)
  - ``order_id*,line_id*,qty``             →  PRIMARY KEY (order_id, line_id)
  - No ``*`` at all                        →  table created without a PRIMARY KEY

Usage as a library (from consumer code):

    from local_demo_db.sqlite_demo_db import connect_to_database

    db = connect_to_database()              # auto-seeded from data/*.csv
    row  = db.query_one("SELECT …", (…,))   # read one row
    rows = db.query("SELECT …")             # read many rows
    db.execute("UPDATE …", (…,))            # write (auto-commits)
    dump = db.select_all_tables()            # {table: [rows]}
    db.print_all_tables()                    # pretty-print all tables
    db.drop_tables("old_table")             # DDL drop (one or more)
    db.close()

Run directly to see all operations in action:

    uv run --active python -m local_demo_db.sqlite_demo_db
"""

from __future__ import annotations

import csv
import sqlite3
import threading
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent
_DATA_DIR = _DIR / "data"
_DEFAULT_DB = _DIR / "sqlite-demo.db"


# ---------------------------------------------------------------------------
# Type inference — sample every value in a column to pick INTEGER / REAL / TEXT
# ---------------------------------------------------------------------------

def _infer_column_type(values: list[str]) -> str:
    """Infer SQLite column type from a list of string values read from CSV.

    Tries INTEGER first, then REAL, falls back to TEXT.
    Empty strings are skipped (they don't rule out a numeric type).
    """
    dominated_by_int = True
    dominated_by_float = True

    for v in values:
        if v == "":
            continue
        try:
            int(v)
        except ValueError:
            dominated_by_int = False
            try:
                float(v)
            except ValueError:
                dominated_by_float = False
                break

    if dominated_by_int:
        return "INTEGER"
    if dominated_by_float:
        return "REAL"
    return "TEXT"


def _cast_value(value: str, col_type: str) -> Any:
    """Cast a CSV string to the Python type matching the inferred SQLite type."""
    if value == "":
        return None
    if col_type == "INTEGER":
        return int(value)
    if col_type == "REAL":
        return float(value)
    return value


# ---------------------------------------------------------------------------
# CSV → SQLite: read a .csv, infer schema, create table, insert rows
# ---------------------------------------------------------------------------

def _seed_from_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    """Read *csv_path*, auto-detect schema, create + populate a table.

    Table name = CSV filename without extension (e.g. inventory.csv → inventory).

    Primary-key convention:
        Append ``*`` to one or more header names to mark them as the PRIMARY KEY.
        - Single PK:    ``place_id*,name,trip_type``  →  ``PRIMARY KEY (place_id)``
        - Composite PK: ``order_id*,line_id*,qty``    →  ``PRIMARY KEY (order_id, line_id)``
        - No ``*`` at all: table is created without a PRIMARY KEY (perfectly valid).

    Returns the number of rows inserted.
    """
    table_name = csv_path.stem

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_headers = [h.strip() for h in next(reader)]
        raw_rows = [row for row in reader if any(cell.strip() for cell in row)]

    # Detect PK columns (marked with trailing *) and strip the marker
    pk_cols: list[str] = []
    headers: list[str] = []
    for rh in raw_headers:
        if rh.endswith("*"):
            clean = rh[:-1]
            pk_cols.append(clean)
            headers.append(clean)
        else:
            headers.append(rh)

    col_values: dict[str, list[str]] = {h: [] for h in headers}
    for row in raw_rows:
        for h, v in zip(headers, row):
            col_values[h].append(v.strip())

    col_types = {h: _infer_column_type(col_values[h]) for h in headers}

    col_defs = ", ".join(f"{h} {col_types[h]}" for h in headers)
    if pk_cols:
        pk_clause = ", ".join(pk_cols)
        col_defs += f", PRIMARY KEY ({pk_clause})"
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

    # Skip seeding if the table already has rows (avoids duplicates on restart)
    existing = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    if existing:
        return 0

    placeholders = ", ".join("?" for _ in headers)
    typed_rows = [
        tuple(_cast_value(v.strip(), col_types[h]) for h, v in zip(headers, row))
        for row in raw_rows
    ]
    conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", typed_rows)
    return len(typed_rows)


# ---------------------------------------------------------------------------
# SQLiteDemoDB — single unified class (thread-safe, auto-seeds on connect)
# ---------------------------------------------------------------------------

class SQLiteDemoDB:
    """Thread-safe SQLite wrapper that auto-seeds from CSV on first connect.

    Usage (same pattern everywhere — consumer code AND __main__):

        db = connect_to_database()              # auto-seeded, thread-safe
        row = db.query_one("SELECT …", (…,))    # read one row
        rows = db.query("SELECT …")             # read many rows
        db.execute("UPDATE …", (…,))            # write (auto-commits)
        dump = db.select_all_tables()            # {table: [rows]}
        db.close()

    Or as a context manager:

        with connect_to_database() as db:
            ...
    """

    def __init__(self, path: str | Path = _DEFAULT_DB,
                 csv_dir: str | Path | None = None):
        self._path = Path(path)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False) # sqlite3.connect(":memory:")
        self._lock = threading.Lock()
        self._seed(csv_dir)

    # -- auto-seed (private) ------------------------------------------------

    def _seed(self, csv_dir: str | Path | None = None) -> None:
        csv_dir = Path(csv_dir) if csv_dir else _DATA_DIR
        csv_files = sorted(csv_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {csv_dir}")
            return
        with self._lock:
            for csv_file in csv_files:
                count = _seed_from_csv(self._conn, csv_file)
                print(f"  Seeded table '{csv_file.stem}' — {count} rows from {csv_file.name}")
            self._conn.commit()

    # -- read/Fetch/Retrieve ---------------------------------------------------------------

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()) -> tuple | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def select_all_tables(self) -> dict[str, list[tuple]]:
        """Return every row from every table in the database.

        Returns:
            ``{table_name: [(row, ...), ...], ...}``
        """
        with self._lock:
            tables = [
                row[0]
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            return {t: self._conn.execute(f"SELECT * FROM {t}").fetchall() for t in tables}

    def print_all_tables(self) -> None:
        """Print every row from every table — reuses select_all_tables() internally."""
        for table, rows in self.select_all_tables().items():
            print(f"  [{table}] {len(rows)} rows")
            for r in rows:
                print(f"    {r}")

    # -- DDL, DML: write --------------------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.rowcount

    # -- DDL: write --------------------------------------------------------------

    def drop_tables(self, *table_names: str) -> None:
        """Drop one or more tables by name. No-op for tables that don't exist."""
        with self._lock:
            for table_name in table_names:
                self._conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self._conn.commit()

    # -- demo: showcase all DB operations ------------------------------------

    def demo_raw_db_operations(self) -> None:
        """Walk through every public method so you can see what the class can do.

        Covers: query, query_one, execute (INSERT/UPDATE/DELETE),
        select_all_tables, drop_table.
        """
        print("\n========== demo_raw_db_operations ==========\n")

        # ── READ: all tables at a glance ──
        print("── select_all_tables() ──")
        self.print_all_tables()

        # ── READ: single row lookup with query_one ──
        print("\n── query_one: look up Goa's stay ──")
        row = self.query_one(
            "SELECT stay_name, room_price_per_night FROM stays WHERE place_id = ?",
            ("GOA",),
        )
        print(f"  {row}")

        # ── READ: multi-row query with JOIN ──
        print("\n── query: JOIN places + costs for all destinations ──")
        rows = self.query("""
            SELECT p.name, p.trip_type, c.travel_mode, c.travel_cost_per_person
            FROM places p
            JOIN costs c ON p.place_id = c.place_id
        """)
        for r in rows:
            print(f"  {r}")

        # ── WRITE: INSERT a new destination ──
        print("\n── execute: INSERT a new place (Pondicherry) ──")
        self.execute(
            "INSERT OR IGNORE INTO places VALUES (?,?,?,?,?)",
            ("PONDI", "Pondicherry", "Beach", "Puducherry", "Oct–Mar"),
        )
        self.execute(
            "INSERT OR IGNORE INTO stays VALUES (?,?,?)",
            ("PONDI", "French Quarter B&B", 2200),
        )
        self.execute(
            "INSERT OR IGNORE INTO costs VALUES (?,?,?,?)",
            ("PONDI", "Bus", 800, 900),
        )
        print(f"  places now: {self.query('SELECT * FROM places')}")

        # ── WRITE: UPDATE a price ──
        print("\n── execute: UPDATE Goa room price 3000 → 3500 ──")
        affected = self.execute(
            "UPDATE stays SET room_price_per_night = ? WHERE place_id = ?",
            (3500, "GOA"),
        )
        print(f"  rows affected: {affected}")
        print(f"  verify: {self.query_one('SELECT * FROM stays WHERE place_id = ?', ('GOA',))}")

        # ── WRITE: DELETE the row we just added ──
        print("\n── execute: DELETE Pondicherry ──")
        for tbl in ("costs", "stays", "places"):
            self.execute(f"DELETE FROM {tbl} WHERE place_id = ?", ("PONDI",))
        print(f"  places after delete: {self.query('SELECT place_id FROM places')}")

        # ── DDL: drop_tables ──
        # print("\n── drop_table: drop 'costs' table, then print_all_tables ──")
        # self.drop_tables("costs")

        self.print_all_tables()

        print("\n========== demo complete ==========\n")
        return self

    def demo_raw_db_operations_2(self) -> None:
        print("\n========== demo_raw_db_operations_2 ==========\n")

        with self._lock:
            # save original
            original_factory = self._conn.row_factory

            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS places (
                place_id TEXT PRIMARY KEY, name TEXT, trip_type TEXT, state TEXT, best_months TEXT
            );
            CREATE TABLE IF NOT EXISTS stays (
                place_id TEXT PRIMARY KEY, stay_name TEXT, room_price_per_night REAL,
                FOREIGN KEY (place_id) REFERENCES places(place_id)
            );
            CREATE TABLE IF NOT EXISTS costs (
                place_id TEXT PRIMARY KEY, travel_mode TEXT,
                travel_cost_per_person REAL, daily_spend_per_person REAL,
                FOREIGN KEY (place_id) REFERENCES places(place_id)
            );
            """)
            cur.executemany("INSERT OR IGNORE INTO places VALUES (?,?,?,?,?)", [
                ("GOA",       "Goa",         "Beach",     "Goa",              "Nov–Feb"),
                ("GOKARNA",   "Gokarna",     "Beach",     "Karnataka",        "Oct–Mar"),
                ("PONDY",     "Pondicherry", "Beach",     "Puducherry",       "Oct–Mar"),
                ("COORG",     "Coorg",       "Hills",     "Karnataka",        "Oct–Jun"),
                ("OOTY",      "Ooty",        "Hills",     "Tamil Nadu",       "Mar–Jun"),
                ("MANALI",    "Manali",      "Hills",     "Himachal Pradesh", "Mar–Jun"),
                ("HAMPI",     "Hampi",       "Heritage",  "Karnataka",        "Oct–Feb"),
                ("JAIPUR",    "Jaipur",      "Heritage",  "Rajasthan",        "Oct–Mar"),
                ("RISHIKESH", "Rishikesh",   "Adventure", "Uttarakhand",      "Oct–Jun"),
                ("DANDELI",   "Dandeli",     "Adventure", "Karnataka",        "Oct–May"),
            ])
            cur.executemany("INSERT OR IGNORE INTO stays VALUES (?,?,?)", [
                ("GOA",       "Sea Breeze Guesthouse",   3000), ("GOKARNA", "Om Beach Huts",          1800),
                ("PONDY",     "French Quarter Homestay", 2500), ("COORG",   "Coffee Estate Cottage",  2800),
                ("OOTY",      "Nilgiri View Lodge",      2200), ("MANALI",  "Old Manali Cafe Stay",   2500),
                ("HAMPI",     "Boulder View Guesthouse", 1500), ("JAIPUR",  "Pink City Haveli",       2400),
                ("RISHIKESH", "Ganga Riverside Camp",    1600), ("DANDELI", "Jungle River Camp",      2000),
            ])
            cur.executemany("INSERT OR IGNORE INTO costs VALUES (?,?,?,?)", [
                ("GOA",       "Overnight train", 2800, 1500), ("GOKARNA", "Overnight bus",   1200,  900),
                ("PONDY",     "Bus",             1400, 1200), ("COORG",   "Bus",             1000, 1100),
                ("OOTY",      "Bus",             1000, 1000), ("MANALI",  "Flight + bus",    9000, 1300),
                ("HAMPI",     "Overnight train",  900,  800), ("JAIPUR",  "Flight",          7500, 1200),
                ("RISHIKESH", "Flight + cab",    8500, 1400), ("DANDELI", "Bus",             1100, 1600),
            ])
            self._conn.commit()

            # restore so subsequent queries return plain tuples
            self._conn.row_factory = original_factory

        self.print_all_tables()

        print("\n========== demo_raw_db_operations_2 complete ==========\n")
        return self
        

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteDemoDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SQLiteDemoDB({self._path})"


# ---------------------------------------------------------------------------
# Public factory — single entry point for consumer code
# ---------------------------------------------------------------------------

def connect_to_database(path: str | Path = _DEFAULT_DB,
                        csv_dir: str | Path | None = None) -> SQLiteDemoDB:
    """Connect to the demo database, auto-seeded from all *.csv files.

    Returns a thread-safe SQLiteDemoDB instance — plug any local DB path
    here without changing the caller's code.
    """
    return SQLiteDemoDB(path, csv_dir) 


# ---------------------------------------------------------------------------
# Run directly → same pattern as consumer code
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db = connect_to_database()
    db.demo_raw_db_operations()
    db.demo_raw_db_operations_2()
    db.drop_tables("costs", "stays", "places")  # as many as needed
    db.close()
