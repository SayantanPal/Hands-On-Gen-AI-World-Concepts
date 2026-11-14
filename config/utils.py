def banner(title: str) -> None:
    line = "=" * min(88, max(40, len(title) + 8))
    print(f"\n{line}\n  {title}\n{line}")

def event_banner(text: str) -> None:
    line = "-" * min(88, max(40, len(text) + 8))
    print(f"\n{line}\n  {text}\n{line}")

def single_line() -> None:
    print("-" * 88)

def step(text: str) -> None:
    print(f"\n--> {text}")


def prettify_in_table_display(rows: list[dict], columns: list[str] | None = None, max_width: int = 60) -> None:
    """Print a list of dicts as an aligned text table."""
    if not rows:
        print("(no rows)")
        return
    columns = columns or list(rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        s = "" if v is None else str(v)
        s = " ".join(s.split())
        return s if len(s) <= max_width else s[: max_width - 3] + "..."

    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print("\n")