# Demo Guide — `sample_demo_my_trip_planner.py`

A quick map of every demo in [`sample_demo_my_trip_planner.py`](./sample_demo_my_trip_planner.py): what it runs, which wrapper/method it uses, whether the model calls tools, and what the "compare Coorg vs Manali" question (Q3 — the one that needs **two** tool rounds) does in each.

Toggle any of them in `main()`.

## All demos at a glance

| Demo function | Wrapper · method | Tool calling | Q3 (needs 2 rounds) | What it teaches |
|---|---|---|---|---|
| `demo_single_pass_limitation_using_langchain_llm` | `LangChainLLM` · `invoke_with_tools` | single-pass (1 round) | **silent empty/partial** (no crash — tools stay bound, but only one round runs) | single-pass limit, quiet failure mode |
| `demo_single_pass_limitation_using_raw_llm` | `RawLLM` · `generate_response_content` | single-pass (1 round) | **loud 400 crash** (`tools.pop` → "Tool choice is none, but model called a tool") | single-pass limit, loud failure mode |
| `demo_react_loop` | `RawLLM._client` via standalone `react_loop()` | multi-round loop | ✅ solved | the ReAct loop spelled out by hand (raw SDK) |
| `demo_react_loop_v2` | `RawLLM` · `invoke_with_multiple_tools_in_agentic_react_loop` | multi-round loop | ✅ solved | same loop, now **packaged on the class** |
| `demo_react_loop_langchain` | `LangChainLLM` (+ manual `bind_tools`) via `react_loop_langchain()` | multi-round loop | ✅ solved | the ReAct loop spelled out by hand (LangChain) |
| `demo_react_loop_langchain_v2` | `LangChainLLM` · `invoke_with_multiple_tools_in_agentic_react_loop` | multi-round loop | ✅ solved | same loop, now **packaged on the class** |
| `demo_prompt_template_chain` | `LangChainLLM` · `chain()` | **none** (Python computes) | n/a (Goa only) | `chain()` text pipe — model only writes prose |
| `demo_prompt_template_chain_v2` | `LangChainLLM` · `invoke_with_tools` (bind only `compute_trip_cost`) | single grounded round | n/a (Goa only) | pre-resolve the fetch in Python → one reliable tool round |

## The three questions

| | Question | Tool rounds needed | Single-pass? |
|---|---|---|---|
| **Q1** | "3-night Goa trip for 5 friends?" | 1 (one destination) | ✅ works |
| **Q2** | "₹8000 budget — can 4 afford 2 nights in Manali?" | 1 (one destination) | ✅ works |
| **Q3** | "Compare Coorg vs Manali — which is cheaper?" | 2+ (two destinations + a compare) | ❌ fails — needs a loop |

Q3 is the pivot: it's the question single-pass can't answer, and the one every ReAct-loop demo solves. Why it *structurally* needs ≥2 rounds is explained in [`LEARNER-NOTES.md`](./LEARNER-NOTES.md) §7.

## v1 vs v2 — same result, less code

The `*_v2` demos prove the hand-rolled loops (`react_loop` / `react_loop_langchain`) are now reusable methods on the classes:

| | v1 (`demo_react_loop*`) | v2 (`*_v2`) |
|---|---|---|
| Loop location | hand-rolled in the demo (`react_loop` / `react_loop_langchain`) | inside the class method `invoke_with_multiple_tools_in_agentic_react_loop` |
| LangChain binding | manual `lc._llm.bind_tools(...)` | constructor `LangChainLLM(tools=TOOLS)` |
| Demo code | ~30 lines of orchestration | one call |
| Result | identical | identical |

Both v1 and v2 are kept on purpose: v1 is the "see every message and API call" teaching version; v2 is the packaged reuse you'd actually ship.

## Where each method lives

- `RawLLM` / `LangChainLLM` and their methods: [`gen_ai_core/llm/`](../../gen_ai_core/llm/) — see [`USAGE.md`](../../gen_ai_core/llm/USAGE.md) for the full API reference.
- The single-pass vs batching vs ReAct concepts (dependency graphs, non-determinism, the two failure faces): [`LEARNER-NOTES.md`](./LEARNER-NOTES.md).
