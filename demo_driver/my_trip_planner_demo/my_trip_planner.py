"""
Trip Planner demo — RawLLM + tool calling (single-pass).

The model receives two tools:
  1. lookup_place   — fetch destination prices (simulated DB)
  2. compute_trip_cost — arithmetic: rooms, stay share, food, total

Flow:
  User asks a natural-language trip question
  → model calls lookup_place to get prices
  → model calls compute_trip_cost with those prices
  → model writes a human-friendly summary

Run with:  uv run --active python demo_driver/my_trip_planner_demo/my_trip_planner.py
"""

from __future__ import annotations

import math

from gen_ai_core import Prompt, RawLLM
from gen_ai_core.llm.langchain_llm import LangChainLLM
from tools_capability.demo_tools import compute_trip_cost
from local_demo_db import sqlite_demo_db



# ---------------------------------------------------------------------------
# Tool functions (plain Python — no @tool decorator needed for RawLLM)
# ---------------------------------------------------------------------------
# RawLLM tools are plain functions + a JSON Schema dict.
# The @tool decorator is only needed for LangChain / Strands agents.

DESTINATIONS = {
    "GOA":    {"trip_type": "Beach", "room_price_per_night": 3000, "travel_cost_per_person": 2800, "daily_spend_per_person": 1500},
    "COORG":  {"trip_type": "Hills", "room_price_per_night": 2800, "travel_cost_per_person": 1000, "daily_spend_per_person": 1100},
    "MANALI": {"trip_type": "Hills", "room_price_per_night": 2500, "travel_cost_per_person": 9000, "daily_spend_per_person": 1300},
}


def lookup_place(place: str) -> dict:
    """Look up a destination's prices."""
    record = DESTINATIONS.get(place.upper())
    if not record:
        return {"error": f"{place} not found. Available: {', '.join(DESTINATIONS)}"}
    return {"place": place.title(), **record}

import json
import time

# Module-level DB connection — created once, reused by all tool calls.
# This is the "connection reuse" pattern mentioned above.
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
        cur = _shared_db._conn.cursor()
        row = cur.execute("""
            SELECT p.*, s.stay_name, s.room_price_per_night,
                c.travel_mode, c.travel_cost_per_person, c.daily_spend_per_person
            FROM places p
            JOIN stays s ON s.place_id = p.place_id
            JOIN costs c ON c.place_id = p.place_id
            WHERE UPPER(p.place_id) = UPPER(?) OR UPPER(p.name) = UPPER(?)
        """, (place, place)).fetchone()
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


# ---------------------------------------------------------------------------
# Tool definitions — JSON Schema that the model reads to decide what to call
# ---------------------------------------------------------------------------
# Each dict has:  name, description, parameters (JSON Schema), function (callable)

TOOLS = [
    # {
    #     "name": "lookup_place",
    #     "description": "Look up a destination's trip type and prices (room, travel, daily spend).",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "place": {"type": "string", "description": "Destination name, e.g. Goa, Coorg, Manali"},
    #         },
    #         "required": ["place"],
    #     },
    #     "function": lookup_place,
    # },
    {
        "name": "fetch_place",
        "description": (
            "Look up a destination's full details from the database: "
            "trip type, best months, stay name, room price, travel mode, "
            "travel cost, and daily spend per person."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "Destination name or ID, e.g. Goa, COORG, Manali"},
            },
            "required": ["place"],
        },
        "function": fetch_place_fast,
    },
    {
        "name": "compute_trip_cost",
        "description": (
            "Compute rooms needed, stay share, food & fun, and total cost per person. "
            "One room fits 2 people. Optionally checks against a budget."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "travel_cost_per_person": {"type": "number"},
                "room_price_per_night":   {"type": "number"},
                "daily_spend_per_person": {"type": "number"},
                "people":                 {"type": "integer", "description": "Number of friends"},
                "nights":                 {"type": "integer"},
                "budget_per_person":      {"type": ["number", "null"], "description": "Optional budget cap per person"},
            },
            "required": ["travel_cost_per_person", "room_price_per_night", "daily_spend_per_person", "people", "nights"],
        },
        "function": compute_trip_cost,
    },
]

# name → tool dict for schema lookup; name → function for execution
TOOLS_MAP = {t["name"]: t for t in TOOLS}
TOOL_FN_MAP = {t["name"]: t["function"] for t in TOOLS}


# ===========================================================================
# Demo 2 — Illustrating the single-pass limitation of RawLLM tool calling
# ===========================================================================

#
# WHY THIS BREAKS:
#   RawLLM.generate_response_content() uses a single-pass pattern:
#     1. First API call  — tools available  → model returns tool_calls
#     2. Execute tools, append results as role:"tool" messages
#     3. Second API call — tools REMOVED (kwargs.pop("tools", None))
#        → the model MUST reply with plain text, it cannot call more tools.
#
#   This works fine when one round of tool calls gets everything the model
#   needs (e.g., "How much is Goa?" → one fetch_place call).
#
#   It breaks when the model needs MULTIPLE sequential rounds:
#     "Compare Coorg vs Manali" → model calls fetch_place("Coorg") first,
#     gets results back, then wants to call fetch_place("Manali") — but
#     tools have been removed so the API rejects the second call with:
#       400 - "Tool choice is none, but model called a tool"
#
# WHERE IN THE CODE:
#   gen_ai_core/llm/raw_llm.py → _gen_openai_compatible():
#     Line 275:  kwargs.pop("tools", None)   ← tools removed before 2nd call
#   Same pattern in _gen_anthropic() and _gen_gemini().
#
# TAKEAWAY:
#   Single-pass is simple and cheap (exactly 2 API calls), but it cannot
#   handle multi-step reasoning where the model's next tool call depends on
#   a previous tool call's result.

def demo_single_pass_limitation() -> None:
    """Show that single-pass tool calling crashes on multi-lookup questions.

    Questions 1 & 2 work (single destination → one round of tool calls).
    Question 3 fails (compare two destinations → needs two rounds).
    """
    print("\n" + "=" * 70)
    print("  DEMO: Single-pass limitation (RawLLM.generate_response_content)")
    print("=" * 70)

    print(f"[Trip Planner] Tools registered: {list(TOOLS_MAP)}\n")
    prompt_role = "a helpful trip cost planner for a group of friends"

    # --- Question 1: single destination ---
    # Works — one destination, one round of tool calls
    q1 = "How much will a 3-night Goa trip cost for 5 friends?"
    print(f"\n[OK] Q1: {q1}")
    # LangChain equivalent (returns dict instead of LLMResult):
    # LangChain equivalent (1-line swap — tools bound internally via .bind_tools()):
    lc_llm = LangChainLLM(tools=TOOLS)
    r1 = lc_llm.invoke_with_tools(Prompt(role=prompt_role, user_query=q1)) # r1 = llm.generate_response_content(Prompt(role=prompt_role, user_query=q1))
    if r1["tool_calls"]:
        print(f'     Tools used: {[tc["name"] for tc in r1["tool_calls"]]}')
    print(f'     A: {r1["text"][:200]}...')
    # no latency/token stats — invoke_with_tools() doesn't track them


    # Build a RawLLM with the tools
    llm = RawLLM(tools=TOOLS)
    print(f"\n[Trip Planner] {llm}")
    
    # --- Question 2: budget check ---
    # Works — budget check, still one destination → one round of tool calls
    q2 = "We have a budget of ₹8000 per person. Can 4 friends afford a 2-night trip to Manali?"
    print(f"\n[OK] Q2: {q2}")
    r2 = llm.generate_response_content(Prompt(role=prompt_role, user_query=q2))
    # LangChain equivalent:
    # r2 = llm.invoke_with_tools(Prompt(role=prompt_role, user_query=q2))
    if r2.tool_calls:
        print(f"     Tools used: {[tc['name'] for tc in r2.tool_calls]}")
    print(f"     A: {r2.text[:200]}...")
    print(f"     ({r2.latency_ms:.0f} ms, {r2.input_tokens} in / {r2.output_tokens} out)")

    # --- Question 3: compare two destinations ---
    # Crashes — two destinations, model needs two separate fetch_place calls
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper?"
    print(f"\n[FAIL] Q3: {q3}")
    try:
        r3 = llm.generate_response_content(Prompt(role=prompt_role, user_query=q3))
        # LangChain equivalent — same single-pass limitation applies:
        # r3 = llm.invoke_with_tools(Prompt(role=prompt_role, user_query=q3))
        # invoke_with_tools does exactly one round of tool execution,
        # so a compare question that needs two sequential fetches will
        # get incomplete data (only the first batch of tool calls runs).
        # It won't crash like RawLLM (no tools.pop), but the answer will
        # be wrong or partial. The ReAct loop is still needed for this.
        if r3.tool_calls:
            print(f"       Tools used: {[tc['name'] for tc in r3.tool_calls]}")
        print(f"       A: {r3.text[:200]}...")
        print(f"       ({r3.latency_ms:.0f} ms, {r3.input_tokens} in / {r3.output_tokens} out)")
    except Exception as e:
        # Expected: 400 - "Tool choice is none, but model called a tool"
        print(f"       CRASHED as expected: {type(e).__name__}: {e}")

    print("\n" + "-" * 70)
    print("  Single-pass cannot handle questions needing multiple tool rounds.")
    print("  Solution: a ReAct loop that keeps tools available until the model")
    print("  stops requesting them (see demo_react_loop below).")
    print("-" * 70 + "\n")


# ===========================================================================
# Demo 3 — Manual ReAct loop for RawLLM (multi-turn tool calling)
# ===========================================================================
#
# WHAT IS A ReAct LOOP?
#   ReAct = Reasoning + Acting.  The model alternates between:
#     - Reasoning: "I need to look up Coorg first, then Manali"
#     - Acting:    calls fetch_place("Coorg"), receives result
#     - Reasoning: "Now I have Coorg's data, let me get Manali"
#     - Acting:    calls fetch_place("Manali"), receives result
#     - Final:     produces a text answer (no more tool calls)
#
#   The loop keeps tools available on every turn and only stops when:
#     a. The model replies with plain text (no tool_calls) → done
#     b. max_iterations reached → safety exit to prevent infinite loops
#
# REFERENCE: gen_ai_core/agents/simple_react_agent.py
#   That file does the same loop but via LangChain's abstractions:
#     - LangChainLLM wraps the model with .bind_tools()
#     - response.tool_calls gives structured calls
#     - ToolMessage feeds results back
#     - Loop until no tool_calls or max_steps reached
#
#   Below we replicate that pattern using the raw OpenAI SDK directly,
#   so you can see every message and API call with no framework magic.
#
# COST TRADE-OFF:
#   Single-pass: exactly 2 API calls (cheap, but limited)
#   ReAct loop:  2 + 2N API calls in the worst case (N = tool rounds)
#   In practice, most questions need 1–3 rounds, so 2–8 API calls.
#
# PERFORMANCE NOTE ON fetch_place_2 AND THREAD LOCKS:
#   Q: "If fetch_place_2 is slow on repeated calls, is it the thread lock?"
#   A: No. The lock (threading.Lock) takes ~microseconds to acquire/release.
#      The real bottleneck is connect_to_database() creating a NEW SQLite
#      connection on every call. Two fixes:
#
#      1. REUSE THE CONNECTION (the big win):
#         Create the DB once and pass it in, or use a module-level singleton.
#         This is what InventoryTools does in tools_capability/demo_tools.py:
#         connect once in __init__, share self.db across all tool methods.
#
#      2. async/await (helps with LLM API calls, NOT with the DB lock):
#         The slow part of a ReAct loop is waiting for the LLM API response
#         (~1–3 seconds per round-trip). async lets you do other work while
#         waiting for the API, but it doesn't help the DB lock (which is
#         held for ~0.1 ms). In production:
#           - Keep the thread lock (or switch to asyncio.Lock for async code)
#           - Make the LLM API calls async (aiohttp / httpx)
#           - Reuse the DB connection (the actual fix for DB slowness)
#
#      3. NEVER remove the thread lock in production. Without it, two
#         concurrent tool calls can corrupt the SQLite connection or read
#         partial writes. The lock is cheap — removing it saves microseconds
#         but risks data corruption.



def _execute_tool(name: str, arguments: dict) -> str:
    """Run a tool function and return its result as a JSON string."""
    fn = TOOL_FN_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    result = fn(**arguments)
    return result if isinstance(result, str) else json.dumps(result)


def _openai_tool_defs() -> list[dict]:
    """Convert TOOLS into the OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOLS
    ]


def react_loop(
    llm: RawLLM,
    prompt: Prompt,
    *,
    max_iterations: int = 5,
) -> dict:
    """Manual ReAct loop using the raw OpenAI-compatible SDK client.

    Mirrors the pattern in gen_ai_core/agents/simple_react_agent.py:
      - simple_react_agent uses LangChain's .invoke() + ToolMessage
      - This function uses openai.ChatCompletion.create() directly

    The loop:
      1. Send messages + tools to the API
      2. If model returns tool_calls → execute them, append results, go to 1
      3. If model returns plain text → done
      4. If max_iterations reached  → safety exit

    Returns a dict with: text, tool_calls_log, iterations, latency_ms
    """
    t0 = time.perf_counter()
    messages = prompt.to_role_dicts()
    tool_defs = _openai_tool_defs()
    tool_calls_log: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        # Every call includes tools — the model can always request more
        resp = llm._client.chat.completions.create(
            model=llm._model,
            messages=messages,
            tools=tool_defs,
            temperature=llm._llm_temperature,
            max_tokens=llm._max_tokens,
        )
        msg = resp.choices[0].message

        # --- No tool calls? Model is done reasoning → return final text ---
        if not msg.tool_calls:
            return {
                "text": (msg.content or "").strip(),
                "tool_calls": tool_calls_log,
                "iterations": iteration,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        # --- Model wants to call tools → execute and feed results back ---
        # Build the assistant message with tool_calls (must include it so
        # the API knows which tool_call_ids the upcoming tool results match)
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            result = _execute_tool(fn_name, fn_args)
            tool_calls_log.append({"name": fn_name, "args": fn_args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
            print(f"    [ReAct round {iteration}] {fn_name}({fn_args}) → {result[:120]}...")

        # Loop continues — next iteration sends updated messages WITH tools

    # Safety exit — model kept calling tools beyond max_iterations
    return {
        "text": "Stopped: too many tool-call rounds without a final answer.",
        "tool_calls": tool_calls_log,
        "iterations": max_iterations,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


def demo_react_loop() -> None:
    """Show the ReAct loop solving the multi-lookup question that single-pass cannot.

    Compare with simple_react_agent.py (LangChain ReAct):
      ┌──────────────────────────┬──────────────────────────────────────┐
      │ simple_react_agent.py    │ react_loop() here                    │
      ├──────────────────────────┼──────────────────────────────────────┤
      │ LangChainLLM + .invoke() │ RawLLM._client (raw OpenAI SDK)     │
      │ response.tool_calls      │ msg.tool_calls (same structure)      │
      │ ToolMessage(content=...) │ {"role":"tool", "content":...}       │
      │ max_steps guard          │ max_iterations guard                 │
      │ tools stay bound always  │ tools in every kwargs (never popped) │
      └──────────────────────────┴──────────────────────────────────────┘
    """
    print("\n" + "=" * 70)
    print("  DEMO: ReAct loop (manual multi-turn tool calling with RawLLM)")
    print("=" * 70)

    llm = RawLLM(tools=TOOLS)
    prompt_role = "a helpful trip cost planner for a group of friends"

    # The question that crashed single-pass — now works with the loop
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper per person?"
    print(f"\nQ: {q3}")

    result = react_loop(
        llm,
        Prompt(role=prompt_role, user_query=q3),
        max_iterations=5,
    )

    print(f"\nA: {result['text']}")
    print(f"   Tools used: {[tc['name'] for tc in result['tool_calls']]}")
    print(f"   ReAct iterations: {result['iterations']}")
    print(f"   Total latency: {result['latency_ms']:.0f} ms")

    print("\n" + "-" * 70)
    print("  The ReAct loop kept tools available on every round, so the model")
    print("  could call fetch_place('Coorg'), get results, then call")
    print("  fetch_place('Manali') in the next round — no crash.")
    print("-" * 70 + "\n")


# ===========================================================================
# Demo 4 — Same ReAct loop, but with LangChain instead of raw SDK
# ===========================================================================
#
# WHAT'S REUSED (loose coupling):
#   _execute_tool()      — same function dispatch, same TOOL_FN_MAP
#   _openai_tool_defs()  — same schema; LangChain's .bind_tools() accepts it
#   Prompt.to_lc_messages() — Prompt already knows how to emit LangChain msgs
#
# WHAT CHANGES (only the LLM call + message types):
#   Raw SDK                              LangChain
#   ────────────────────────────         ────────────────────────────
#   llm._client.chat.completions.create  model.invoke(messages)
#   {"role":"tool", "content":...}       ToolMessage(content=...)
#   json.loads(tc.function.arguments)    tc["args"]  (already parsed)
#   msg.tool_calls[i].id                 tc["id"]

def react_loop_langchain(
    model,
    prompt: Prompt,
    *,
    max_iterations: int = 5,
) -> dict:
    """ReAct loop using a LangChain ChatModel (with tools already bound).

    Reuses _execute_tool() and TOOL_FN_MAP — only the message format differs.
    Compare with react_loop() above which uses the raw OpenAI SDK.
    """
    from langchain_core.messages import ToolMessage

    t0 = time.perf_counter()
    messages = prompt.to_lc_messages()
    tool_calls_log: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        response = model.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            text = getattr(response, "text", None) or response.content
            return {
                "text": (text or "").strip(),
                "tool_calls": tool_calls_log,
                "iterations": iteration,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }

        for tc in response.tool_calls:
            result = _execute_tool(tc["name"], tc["args"])
            tool_calls_log.append({"name": tc["name"], "args": tc["args"], "result": result})
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            print(f"    [LC ReAct round {iteration}] {tc['name']}({tc['args']}) → {result[:120]}...")

    return {
        "text": "Stopped: too many tool-call rounds without a final answer.",
        "tool_calls": tool_calls_log,
        "iterations": max_iterations,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


def demo_react_loop_langchain() -> None:
    """Same multi-lookup demo but driven by LangChain instead of raw SDK.

    Shows that the loop logic is identical — only the LLM call + message
    types change. _execute_tool, TOOL_FN_MAP, Prompt are all reused.
    """
    from gen_ai_core.llm.langchain_llm import LangChainLLM

    print("\n" + "=" * 70)
    print("  DEMO: ReAct loop (LangChain — same tools, same _execute_tool)")
    print("=" * 70)

    # .bind_tools() accepts OpenAI-format tool defs — reuse _openai_tool_defs()
    lc = LangChainLLM()
    model = lc._llm.bind_tools(_openai_tool_defs())

    prompt_role = "a helpful trip cost planner for a group of friends"
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper per person?"
    print(f"\nQ: {q3}")

    result = react_loop_langchain(
        model,
        Prompt(role=prompt_role, user_query=q3),
        max_iterations=5,
    )

    print(f"\nA: {result['text']}")
    print(f"   Tools used: {[tc['name'] for tc in result['tool_calls']]}")
    print(f"   ReAct iterations: {result['iterations']}")
    print(f"   Total latency: {result['latency_ms']:.0f} ms")

    print("\n" + "-" * 70)
    print("  Same result as react_loop (raw SDK), but using LangChain types.")
    print("  The tool functions and dispatch (_execute_tool) are shared.")
    print("-" * 70 + "\n")


def main() -> None:
    demo_single_pass_limitation()       # single-pass demo: Q1/Q2 work, Q3 crashes
    # demo_react_loop()                   # ReAct loop (raw SDK) solves Q3
    # demo_react_loop_langchain()           # ReAct loop (LangChain) — same tools, same result

if __name__ == "__main__":
    main()

