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
from demo_driver.my_trip_planner_demo.sqlite_db_calls_demo import fetch_place_fast, fetch_place_fast_2



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

def demo_single_pass_limitation_using_langchain_llm() -> None:
    """Same single-pass limitation demo, driven entirely by LangChainLLM.

    Q1 & Q2 work (single destination → one tool round).
    Q3 (compare two destinations → needs two rounds) does NOT crash like RawLLM
    — LangChain never pops the tools — but invoke_with_tools runs only ONE round,
    so the answer comes back empty / partial. Same single-pass limit, quieter
    face. (The RawLLM twin crashes loudly; see demo_single_pass_limitation_using_raw_llm.)
    """
    print("\n" + "=" * 70)
    print("  DEMO: Single-pass limitation (LangChainLLM.invoke_with_tools)")
    print("=" * 70)

    lc = LangChainLLM(tools=TOOLS)
    print(f"[Trip Planner] Tools registered: {list(TOOLS_MAP)}\n")
    prompt_role = "a helpful trip cost planner for a group of friends"

    # --- Question 1: single destination ---
    # Works — one destination, one round of tool calls
    q1 = "How much will a 3-night Goa trip cost for 5 friends?"
    print(f"\n[OK] Q1: {q1}")
    r1 = lc.invoke_with_tools(Prompt(role=prompt_role, user_query=q1))
    if r1["tool_calls"]:
        print(f'     Tools used: {[tc["name"] for tc in r1["tool_calls"]]}')
    print(f'     A: {r1["text"][:200]}...')
    # NOTE: invoke_with_tools returns a dict — no latency/token stats like RawLLM's LLMResult.

    # --- Question 2: budget check ---
    # Works — budget check, still one destination → one round of tool calls
    q2 = "We have a budget of ₹8000 per person. Can 4 friends afford a 2-night trip to Manali?"
    print(f"\n[OK] Q2: {q2}")
    r2 = lc.invoke_with_tools(Prompt(role=prompt_role, user_query=q2))
    if r2["tool_calls"]:
        print(f'     Tools used: {[tc["name"] for tc in r2["tool_calls"]]}')
    print(f'     A: {r2["text"][:200]}...')

    # --- Question 3: compare two destinations ---
    # No crash (LangChain keeps tools bound), but single-pass runs ONE round,
    # so the comparison is empty / partial — the quiet twin of RawLLM's 400 crash.
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper?"
    print(f"\n[FAIL] Q3: {q3}")
    r3 = lc.invoke_with_tools(Prompt(role=prompt_role, user_query=q3))
    if r3["tool_calls"]:
        print(f'       Tools used: {[tc["name"] for tc in r3["tool_calls"]]}')
    print(f'       A: {r3["text"][:200] or "(empty — only one tool round ran)"}...')
    print("       ^ No crash like RawLLM, but single-pass ran one round, so the")
    print("         comparison is incomplete / empty — the quieter failure mode.")

    print("\n" + "-" * 70)
    print("  Single-pass cannot handle questions needing multiple tool rounds.")
    print("  Fix: lc.invoke_with_multiple_tools_in_agentic_react_loop(...) keeps")
    print("  tools available every round (see react_loop_langchain / demo_react_loop_langchain).")
    print("-" * 70 + "\n")


def demo_single_pass_limitation_using_raw_llm() -> None:
    """Same single-pass limitation demo, but driven entirely by RawLLM.

    Q1 & Q2 work (single destination → one tool round).
    Q3 crashes (compare two destinations → needs two rounds): on the 2nd API
    call RawLLM removes the tools (kwargs.pop("tools")), so a further tool call
    is rejected with 400 - "Tool choice is none, but model called a tool".
    """
    print("\n" + "=" * 70)
    print("  DEMO: Single-pass limitation (RawLLM — raw provider SDK)")
    print("=" * 70)

    llm = RawLLM(tools=TOOLS)
    print(f"\n[Trip Planner] {llm}")
    print(f"[Trip Planner] Tools registered: {list(TOOLS_MAP)}\n")
    prompt_role = "a helpful trip cost planner for a group of friends"

    # --- Question 1: single destination ---
    # Works — one destination, one round of tool calls
    q1 = "How much will a 3-night Goa trip cost for 5 friends?"
    print(f"\n[OK] Q1: {q1}")
    r1 = llm.generate_response_content(Prompt(role=prompt_role, user_query=q1))
    if r1.tool_calls:
        print(f"     Tools used: {[tc['name'] for tc in r1.tool_calls]}")
    print(f"     A: {r1.text[:200]}...")
    print(f"     ({r1.latency_ms:.0f} ms, {r1.input_tokens} in / {r1.output_tokens} out)")

    # --- Question 2: budget check ---
    # Works — budget check, still one destination → one round of tool calls
    q2 = "We have a budget of ₹8000 per person. Can 4 friends afford a 2-night trip to Manali?"
    print(f"\n[OK] Q2: {q2}")
    r2 = llm.generate_response_content(Prompt(role=prompt_role, user_query=q2))
    if r2.tool_calls:
        print(f"     Tools used: {[tc['name'] for tc in r2.tool_calls]}")
    print(f"     A: {r2.text[:200]}...")
    print(f"     ({r2.latency_ms:.0f} ms, {r2.input_tokens} in / {r2.output_tokens} out)")

    # --- Question 3: compare two destinations ---
    # Crashes — two destinations, model needs two separate fetch_place rounds
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper?"
    print(f"\n[FAIL] Q3: {q3}")
    try:
        r3 = llm.generate_response_content(Prompt(role=prompt_role, user_query=q3))
        if r3.tool_calls:
            print(f"       Tools used: {[tc['name'] for tc in r3.tool_calls]}")
        print(f"       A: {r3.text[:200]}...")
        print(f"       ({r3.latency_ms:.0f} ms, {r3.input_tokens} in / {r3.output_tokens} out)")
    except Exception as e:
        # Expected: 400 - "Tool choice is none, but model called a tool"
        print(f"       CRASHED as expected: {type(e).__name__}: {e}")

    print("\n" + "-" * 70)
    print("  Single-pass cannot handle questions needing multiple tool rounds.")
    print("  Fix: llm.invoke_with_multiple_tools_in_agentic_react_loop(...) keeps")
    print("  tools available every round (see react_loop / demo_react_loop).")
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
    # LangChain @tool objects (StructuredTool) aren't plain-callable — call
    # them with .invoke({args}). Plain Python functions take kwargs directly.
    result = fn.invoke(arguments) if hasattr(fn, "invoke") else fn(**arguments)
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


def demo_react_loop_v2() -> None:
    """Same as demo_react_loop, but the loop lives INSIDE RawLLM.

    demo_react_loop() wires the loop by hand (the standalone react_loop() harness).
    v2 just calls RawLLM.invoke_with_multiple_tools_in_agentic_react_loop() — the
    identical loop, now packaged on the class. Same result, far less demo code.
    """
    print("\n" + "=" * 70)
    print("  DEMO: ReAct loop via RawLLM.invoke_with_multiple_tools_in_agentic_react_loop")
    print("=" * 70)

    llm = RawLLM(tools=TOOLS)
    prompt_role = "a helpful trip cost planner for a group of friends"
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper per person?"
    print(f"\nQ: {q3}")

    # The class method keeps tools bound every round — no manual harness needed.
    result = llm.invoke_with_multiple_tools_in_agentic_react_loop(
        Prompt(role=prompt_role, user_query=q3),
        max_iterations=5,
    )

    print(f"\nA: {result['text']}")
    print(f"   Tools used: {[tc['name'] for tc in result['tool_calls']]}")
    print(f"   ReAct iterations: {result['iterations']}")
    print(f"   Total latency: {result['latency_ms']:.0f} ms")

    print("\n" + "-" * 70)
    print("  Same multi-round result as demo_react_loop, but the loop is reused")
    print("  from RawLLM — the demo no longer hand-rolls react_loop().")
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


def demo_react_loop_langchain_v2() -> None:
    """Same as demo_react_loop_langchain, but the loop lives INSIDE LangChainLLM.

    demo_react_loop_langchain() hand-binds the model (lc._llm.bind_tools(...)) and
    drives react_loop_langchain() itself. v2 just constructs LangChainLLM(tools=TOOLS)
    and calls invoke_with_multiple_tools_in_agentic_react_loop() — binding + the
    multi-round loop are both handled by the class.
    """
    print("\n" + "=" * 70)
    print("  DEMO: ReAct loop via LangChainLLM.invoke_with_multiple_tools_in_agentic_react_loop")
    print("=" * 70)

    lc = LangChainLLM(tools=TOOLS)           # constructor binds tools + builds the execute-map
    prompt_role = "a helpful trip cost planner for a group of friends"
    q3 = "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper per person?"
    print(f"\nQ: {q3}")

    result = lc.invoke_with_multiple_tools_in_agentic_react_loop(
        Prompt(role=prompt_role, user_query=q3),
        max_iterations=5,
    )

    print(f"\nA: {result['text']}")
    print(f"   Tools used: {[tc['name'] for tc in result['tool_calls']]}")
    print(f"   ReAct iterations: {result['iterations']}")
    print(f"   Total latency: {result['latency_ms']:.0f} ms")

    print("\n" + "-" * 70)
    print("  Same multi-round result as demo_react_loop_langchain, but the loop is")
    print("  reused from LangChainLLM — no manual bind_tools, no hand-rolled loop.")
    print("-" * 70 + "\n")


# ===========================================================================
# Demo 5 — LangChain prompt template + chain (NO tool calling)
# ===========================================================================
#
# HOW THIS DIFFERS FROM DEMOS 2–4:
#   Demos 2–4 let the MODEL decide which tools to call. Here PYTHON does the
#   DB lookup and the arithmetic itself, then hands the finished numbers to
#   the model only to WRITE THE EXPLANATION. The model never calls a tool —
#   it just turns structured data into friendly prose. This is the safe way
#   to present already-computed numbers (no risk of the model re-doing math).
#
# WHAT A ChatPromptTemplate IS:
#   A reusable message template with {named} slots. You fill the slots with a
#   dict at .invoke() time, so one template serves any destination.
#
# THE PIPE (|) CHAIN:
#   prompt | llm | StrOutputParser()  builds a Runnable:
#     prompt          → fills the template from your dict
#     llm             → sends the filled messages to the model (BASE model,
#                       no tools bound — we don't want tool calls here)
#     StrOutputParser → extracts the plain string from the AIMessage
#   LangChainLLM.chain() already wires exactly this pipe for you.

def demo_prompt_template_chain_with_manual_tool_calling(place: str = "Goa", people: int = 5,
                               nights: int = 3, budget_per_person: float = 8000) -> None:
    """DB → tool (plain Python) → LangChain chain that explains the numbers."""
    from langchain_core.prompts import ChatPromptTemplate
    from gen_ai_core import Prompt   # already importable

    print("\n" + "=" * 70)
    print("  DEMO: ChatPromptTemplate + chain (Python computes, model explains)")
    print("=" * 70)

    # --- Step 1: DB lookup (plain Python, no LLM) ---
    db_record = fetch_place_fast(place)
    if not db_record:
        print(f"Place {place!r} not found.")
        return

    # --- Step 2: exact arithmetic via the tool, called as a LangChain tool ---
    # compute_trip_cost is a @tool (StructuredTool) → NOT plain-callable.
    # Use .invoke({dict}); it validates args against the schema, then runs.
    tool_result = compute_trip_cost.invoke({
        "travel_cost_per_person": db_record["travel_cost_per_person"],
        "room_price_per_night":   db_record["room_price_per_night"],
        "daily_spend_per_person": db_record["daily_spend_per_person"],
        "people":                 people,
        "nights":                 nights,
        "budget_per_person":      budget_per_person,
    })
    print(f"\n[Tool] compute_trip_cost → {tool_result}")


    # Merge the DB record + the tool's numbers into one flat dict of template
    # variables. compute_trip_cost returns *_per_person keys, so map them to
    # the template's shorter slot names. Extra keys in the dict are ignored.
    trip_explain_data_variables = {
        **db_record,
        "people": people,
        "nights": nights,

        **tool_result,          # ← replaces all 4 explicit retrievals; extras are harmless

        # for manual retrieval, you could uncomment the lines below and access each key individually
        # "rooms_needed_per_person": tool_result["rooms_needed_per_person"],
        # "stay_share_per_person":   tool_result["stay_share_per_person"],
        # "food_and_fun_per_person": tool_result["food_and_fun_per_person"],
        # "total_per_person":         tool_result["total_per_person"],
    }

    # --- Step 3.a.1: the reusable template ({slots} filled at invoke time) ---
    # RCTFC system prompt — note the {total_per_person} slot survives into the string
    # Intended slots → single braces: {total_per_person} ✅
    # Literal braces (e.g. a JSON example in output_format="Return {\"ok\": true}") → must be doubled: {{"ok": true}}, or ChatPromptTemplate throws a KeyError / missing-variable error.
    trip_explain_system_prompt = Prompt(
        role="a trip planner for a group of friends",
        task="Explain the cost breakdown below in simple words.",
        constraints=[
            "The numbers were calculated by a trusted tool — do NOT redo the math.",
            "End your reply with exactly: COST_PER_PERSON: {total_per_person}",
        ],
    ).get_base_system_prompt()

    trip_explain_sys_and_usr_prompt = ChatPromptTemplate.from_messages([
        ("system", trip_explain_system_prompt),                 # ← reused RCTFC scaffold, with a live {slot}
        ("human",
        "Trip: {name} ({trip_type}), best months {best_months}\n"
        "Group: {people} friends, {nights} nights at {stay_name}\n"
        "Travel: {travel_mode}, ₹{travel_cost_per_person} per person\n\n"
        "Tool results:\n"
        "  Rooms needed     : {rooms_needed_per_person}\n"
        "  Stay share       : ₹{stay_share_per_person}\n"
        "  Food and fun     : ₹{food_and_fun_per_person}\n"
        "  Total per person : ₹{total_per_person}"),
    ])

    # --- Step 3.a.2: build the chain (prompt | base_llm | StrOutputParser) ---
    # NOTE: .chain() returns a LangChain pipe (RunnableSequence) → only .invoke() works, NOT .invoke_with_tools().
    # To call tools, skip .chain() and use lc.invoke_with_tools(prompt, variables) directly (see demo_prompt_template_chain_v2).
    answer = LangChainLLM().chain(trip_explain_sys_and_usr_prompt).invoke(trip_explain_data_variables)

    
    '''
    # --- Step 3.b ---
    answer = LangChainLLM().chain(ChatPromptTemplate.from_messages([
                ("system", Prompt(
                        role="a trip planner for a group of friends",
                        task="Explain the cost breakdown below in simple words.",
                        constraints=[
                            "The numbers were calculated by a trusted tool — do NOT redo the math.",
                            "End your reply with exactly: COST_PER_PERSON: {total_per_person}",
                        ],
                    ).get_base_system_prompt()
                ),                 # ← reused RCTFC scaffold, with a live {slot}
                ("human",
                "Trip: {name} ({trip_type}), best months {best_months}\n"
                "Group: {people} friends, {nights} nights at {stay_name}\n"
                "Travel: {travel_mode}, ₹{travel_cost_per_person} per person\n\n"
                "Tool results:\n"
                "  Rooms needed     : {rooms_needed_per_person}\n"
                "  Stay share       : ₹{stay_share_per_person}\n"
                "  Food and fun     : ₹{food_and_fun_per_person}\n"
                "  Total per person : ₹{total_per_person}"
                ),
            ])).invoke({
                **db_record,
                "people": people,
                "nights": nights,
                **tool_result,          # ← replaces all 4 explicit retrievals; extras are harmless
            })
    '''

    '''
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser
    answer = (ChatPromptTemplate.from_messages([
                ("system", Prompt(
                        role="a trip planner for a group of friends",
                        task="Explain the cost breakdown below in simple words.",
                        constraints=[
                            "The numbers were calculated by a trusted tool — do NOT redo the math.",
                            "End your reply with exactly: COST_PER_PERSON: {total_per_person}",
                        ],
                    ).get_base_system_prompt()
                ),                 # ← reused RCTFC scaffold, with a live {slot}
                ("human",
                "Trip: {name} ({trip_type}), best months {best_months}\n"
                "Group: {people} friends, {nights} nights at {stay_name}\n"
                "Travel: {travel_mode}, ₹{travel_cost_per_person} per person\n\n"
                "Tool results:\n"
                "  Rooms needed     : {rooms_needed_per_person}\n"
                "  Stay share       : ₹{stay_share_per_person}\n"
                "  Food and fun     : ₹{food_and_fun_per_person}\n"
                "  Total per person : ₹{total_per_person}"
                ),
            ]) | ChatGroq(model="openai/gpt-oss-120b", temperature=0.0) 
               | StrOutputParser()).invoke({
                **db_record,
                "people": people,
                "nights": nights,
                **tool_result,          # ← replaces all 4 explicit retrievals; extras are harmless
            })
        
    '''

    print(f"\n[Chain] explanation:\n{answer}")
    print("\n" + "-" * 70)
    print("  No tool calling here — Python did the DB lookup + math, the model")
    print("  only turned the finished numbers into a human-friendly explanation.")
    print("-" * 70 + "\n")


# ===========================================================================
# Demo 5b — Pre-resolve the fetch, bind ONLY compute_trip_cost, single-pass
# ===========================================================================
#
# THE CONCEPT (collapsing the dependency graph in CODE instead of with a loop):
#   demo_single_pass_limitation showed that a fetch→compute question needs TWO
#   tool rounds, which single-pass cannot give — so it crashes or returns empty
#   unless the model gets lucky and batches a GUESSED compute (see README §4-6).
#
#   v2's trick: do the fetch in PYTHON (resolving the upstream dependency), then
#   bind ONLY compute_trip_cost. Now the model's graph is a single LEAF tool with
#   no upstream tool dependency:
#
#       [Python fetch]  →  compute_trip_cost  →  answer      (one tool layer)
#
#   A single tool round is enough — and it is RELIABLE, not luck-based, because:
#     • Only ONE tool is bound → nothing to mis-batch or mis-order (README §6).
#     • The REAL DB prices sit in the prompt → the compute runs on grounded data,
#       not a guess (avoids the silent-wrongness trap of README §4).
#
#   Contrast with v1 (demo_prompt_template_chain): v1 computed the numbers in
#   Python and the model only wrote prose (ZERO tool calls). v2 hands the model
#   the raw prices and lets it CALL the tool itself — exactly ONE tool round.
#
# REUSE: LangChainLLM(tools=[...]) binds the tool in its constructor, and
#   invoke_with_tools() runs the single-pass execute→summarise internally —
#   so this demo does NO manual bind_tools and NO hand-rolled loop.
#   PROMPT MIX: invoke_with_tools() now accepts a ChatPromptTemplate (+ variables),
#   so system = Prompt (RCTFC scaffold) and human = ChatPromptTemplate slots.

def demo_prompt_template_chain_with_llm_implicit_tool_calling(place: str = "Goa", people: int = 5,
                                  nights: int = 3, budget_per_person: float = 8000) -> None:
    """Pre-resolve the DB fetch in Python, bind ONLY compute_trip_cost, let the
    model make ONE grounded tool call — single-pass, no ReAct loop needed."""
    from langchain_core.prompts import ChatPromptTemplate
    from gen_ai_core import Prompt

    print("\n" + "=" * 70)
    print("  DEMO: pre-resolve fetch + bind only compute_trip_cost (single-pass)")
    print("=" * 70)

    # --- Step 1: DB lookup in Python — resolves the UPSTREAM dependency ---
    db_record = fetch_place_fast(place)
    if not db_record:
        print(f"Place {place!r} not found.")
        return

    # --- Step 2: system (RCTFC Prompt) + human (ChatPromptTemplate) ---
    # No tool-RESULT slots here (that was v1) — the model PRODUCES those by calling
    # the tool. We DO pass the raw DB prices as {slots} so the single compute call
    # is GROUNDED, not guessed.
    trip_planner_system_prompt = Prompt(
        role="a trip planner for a group of friends",
        task="Call the compute_trip_cost tool with the numbers below, then explain the result in simple words.",
        constraints=[
            "Use the compute_trip_cost tool for ALL arithmetic — do NOT do the math yourself.",
            "End your reply with a line: COST_PER_PERSON: <total per person from the tool>",
        ],
    ).get_base_system_prompt()

    trip_planner_prompt = ChatPromptTemplate.from_messages([
        ("system", trip_planner_system_prompt),                 # ← reused RCTFC scaffold
        ("human",
        "Trip: {name} ({trip_type}), best months {best_months}\n"
        "Group: {people} friends, {nights} nights at {stay_name}\n"
        "Prices from our database (pass these to compute_trip_cost):\n"
        "  Room per night         : ₹{room_price_per_night}\n"
        "  Travel per person      : ₹{travel_cost_per_person}\n"
        "  Daily spend per person : ₹{daily_spend_per_person}\n"
        "Budget per person        : ₹{budget_per_person}"),
    ])

    # --- Step 3: bind ONLY compute_trip_cost via the constructor, then let
    #            invoke_with_tools() format the template + do the single-pass
    #            execute/summarise. (Full reuse — no manual orchestration here.)
    # NOTE: we pass the BARE LangChain tool — no TOOLS/TOOLS_MAP dict wrapper
    #       needed. The StructuredTool carries its own .name + schema, and
    #       LangChainLLM builds its execute-map from that directly.
    result = LangChainLLM(tools=[compute_trip_cost]).invoke_with_tools(trip_planner_prompt, {
        **db_record,
        "people": people,
        "nights": nights,
        "budget_per_person": budget_per_person,
    })

    '''
    result = LangChainLLM(provider = "groq", model = "openai/gpt-oss-120b", tools=[compute_trip_cost]).invoke_with_tools(
                                                                        ChatPromptTemplate.from_messages([
                                                                            ("system", Prompt(
                                                                                                role="a trip planner for a group of friends",
                                                                                                task="Call the compute_trip_cost tool with the numbers below, then explain the result in simple words.",
                                                                                                constraints=[
                                                                                                    "Use the compute_trip_cost tool for ALL arithmetic — do NOT do the math yourself.",
                                                                                                    "End your reply with a line: COST_PER_PERSON: <total per person from the tool>",
                                                                                                ],
                                                                                            ).get_base_system_prompt()
                                                                            ),                 # ← reused RCTFC scaffold

                                                                            ("human",
                                                                                "Trip: {name} ({trip_type}), best months {best_months}\n"
                                                                                "Group: {people} friends, {nights} nights at {stay_name}\n"
                                                                                "Prices from our database (pass these to compute_trip_cost):\n"
                                                                                "  Room per night         : ₹{room_price_per_night}\n"
                                                                                "  Travel per person      : ₹{travel_cost_per_person}\n"
                                                                                "  Daily spend per person : ₹{daily_spend_per_person}\n"
                                                                                "Budget per person        : ₹{budget_per_person}"
                                                                            ),
                                                                        ]), 

                                                                        {
                                                                            **db_record,
                                                                            "people": people,
                                                                            "nights": nights,
                                                                            "budget_per_person": budget_per_person,
                                                                        }
                                                                    )
    '''

    # --- Step 4: report the single tool round + final answer ---
    for tc in result["tool_calls"]:
        print(f"    [single tool round] {tc['name']}({tc['args']}) → {tc['result'][:120]}...")
    print(f"\n[Answer]\n{result['text']}")
    print("\n" + "-" * 70)
    print("  ONE tool round sufficed — the fetch→compute dependency was resolved")
    print("  in Python, leaving a single leaf tool for the model. Grounded (real")
    print("  DB prices in the prompt), so no guess and no ReAct loop needed.")
    print("-" * 70 + "\n")

# Manual Multi-pipeline
def demo_prompt_template_chain_with_llm_implicit_tool_calling_and_with_manual_seq_pipeline_cascading(place: str = "Goa", people: int = 5, nights: int = 3, budget_per_person: float = 8000) -> None:
    
    # --- Step 1: DB lookup in Python — resolves the UPSTREAM dependency ---
    db_record = fetch_place_fast(place)
    if not db_record:
        print(f"Place {place!r} not found.")
        return
    
    # --- Step 2: call LLM with single tool --- 
    from langchain_core.prompts import ChatPromptTemplate
    from gen_ai_core import Prompt                                                                                                           
    user_trip_explain_result = LangChainLLM(
                                        # provider = "groq", model = "openai/gpt-oss-120b", 
                                        tools=[compute_trip_cost]).invoke_with_tools(
                                                                            ChatPromptTemplate.from_messages([
                                                                                ("system", Prompt(
                                                                                                    role="a trip planner for a group of friends",
                                                                                                    task="Call the compute_trip_cost tool with the numbers below, then explain the result in simple words.",
                                                                                                    constraints=[
                                                                                                        "Use the compute_trip_cost tool for ALL arithmetic — do NOT do the math yourself.",
                                                                                                        "End your reply with a line: COST_PER_PERSON: <total per person from the tool>",
                                                                                                    ],
                                                                                                ).get_base_system_prompt()
                                                                                ),                 # ← reused RCTFC scaffold
    
                                                                                ("human",
                                                                                    "Trip: {name} ({trip_type}), best months {best_months}\n"
                                                                                    "Group: {people} friends, {nights} nights at {stay_name}\n"
                                                                                    "Prices from our database (pass these to compute_trip_cost tool):\n"
                                                                                    "  Room per night           : ₹{room_price_per_night}\n"
                                                                                    "  Travel per person        : ₹{travel_cost_per_person}\n"
                                                                                    "  Daily spend per person   : ₹{daily_spend_per_person}\n"
                                                                                    "  Budget per person        : ₹{budget_per_person}"
                                                                                ),
                                                                            ]), 
    
                                                                            {
                                                                                **db_record,
                                                                                "people": people,
                                                                                "nights": nights,
                                                                                "budget_per_person": budget_per_person,
                                                                            }
                                                                        )

    budget_cost_breakdown = user_trip_explain_result['text']
    trip_decision_result = LangChainLLM(
                                        # provider = "groq", model = "openai/gpt-oss-120b", 
                                        # tools=[compute_trip_cost]
                                        ).invoke_with_tools(
                                                                ChatPromptTemplate.from_messages([
                                                                    ("system", Prompt(
                                                                                        role="the group's money-minded friend.",
                                                                                        context="Money left per person = budget - total cost per person.",
                                                                                        task="Check if within budget. If over budget, suggest one simple change (fewer nights, cheaper place, more friends).",
                                                                                        output_format="Say clearly: GO (within budget) or NOT YET (over budget).",
                                                                                    ).get_base_system_prompt()
                                                                    ),                 # ← reused RCTFC scaffold

                                                                    ("human",
                                                                        "Trip: {name} | Budget per person: ₹{budget_per_person}\n\n"
                                                                        "Cost breakdown:\n{budget_cost_breakdown}"
                                                                    ),
                                                                ]), 

                                                                {
                                                                    "name":                         db_record["name"],
                                                                    "budget_per_person":            budget_per_person,
                                                                    "budget_cost_breakdown":        budget_cost_breakdown,      # ← llm-1's output is llm-2's input
                                                                }
                                        )
    print(f"\n--- Go or Not Yet ---\n{trip_decision_result['text']}")


def main() -> None:
    # demo_single_pass_limitation_using_langchain_llm()       # single-pass (LangChain for Q1): Q1/Q2 work, Q3 crashes
    # demo_single_pass_limitation_using_raw_llm()           # single-pass (RawLLM throughout): Q1/Q2 work, Q3 crashes
    # demo_react_loop()                   # ReAct loop (raw SDK) solves Q3
    # demo_react_loop_v2()                # ReAct loop via RawLLM.invoke_with_multiple_tools_in_agentic_react_loop
    # demo_react_loop_langchain()           # ReAct loop (LangChain) — same tools, same result
    # demo_react_loop_langchain_v2()      # ReAct loop via LangChainLLM.invoke_with_multiple_tools_in_agentic_react_loop
    # demo_prompt_template_chain_with_manual_tool_calling()        # ChatPromptTemplate + chain — model explains, Python computes
    # demo_prompt_template_chain_with_llm_implicit_tool_calling()     # pre-resolve fetch + bind only compute_trip_cost (single-pass tool call)
    demo_prompt_template_chain_with_llm_implicit_tool_calling_and_with_manual_seq_pipeline_cascading()

if __name__ == "__main__":
    main()

