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
| `demo_prompt_template_chain_with_manual_tool_calling` | `LangChainLLM` · `chain()` | **none** (Python computes) | n/a (Goa only) | `chain()` text pipe — model only writes prose |
| `demo_prompt_template_chain_with_llm_implicit_tool_calling` | `LangChainLLM` · `invoke_with_tools` (bind only `compute_trip_cost`) | single grounded round | n/a (Goa only) | pre-resolve the fetch in Python → one reliable tool round |
| `demo_prompt_template_chain_with_llm_implicit_tool_calling_and_with_manual_seq_pipeline_cascading` | two `LangChainLLM` calls in sequence (stage-1 `invoke_with_tools` w/ tool → stage-2 `invoke_with_tools` no tool) | stage 1: one grounded round · stage 2: none | n/a (Goa only) | **cascading pipeline** — stage-1's answer becomes stage-2's input |

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

## Cascading pipeline — chaining two LLM calls

`demo_prompt_template_chain_with_llm_implicit_tool_calling_and_with_manual_seq_pipeline_cascading`
shows a different shape from everything above: instead of fixing *one* question, it runs **two
LLM stages in sequence**, wiring the first's output into the second's input — a manual
prompt-chain (cascade).

```
DB fetch (Python)
      │
      ▼
Stage 1  LangChainLLM(tools=[compute_trip_cost]).invoke_with_tools(...)
         → model calls the tool, returns a cost breakdown + "COST_PER_PERSON: <n>"
      │   result["text"]  ──────────────────────────────┐
      ▼                                                  │ (handoff)
Stage 2  LangChainLLM().invoke_with_tools(...)   ← NO tools (pure judgment)
         prompt contains {budget_cost_breakdown} = stage-1's text
         → model replies "GO" / "NOT YET" + one suggested change
```

Key points for a learner:

- **Different roles per stage.** Stage 1 is "trip planner" *with* a tool; stage 2 is the
  "money-minded friend" *with no tool*. Each stage is its own `LangChainLLM` with its own
  prompt — you don't reuse one instance.
- **The handoff is manual.** `budget_cost_breakdown = stage1_result["text"]` then passed as a
  template variable to stage 2. *You* orchestrate the sequence in Python — no agent framework.
- **Stage 2 binds no tool on purpose.** It only judges from stage-1's prose, so a tool would
  be dead weight (and risk a stray guessed call). Pick tools per stage by what that stage needs.
- **How it differs from a ReAct loop:** a loop runs the *same* model over multiple tool rounds
  for *one* question; a cascade chains *different* LLM calls (different roles/prompts), each a
  complete answer feeding the next. Loop = depth within one task; cascade = breadth across tasks.

## Where each method lives

- `RawLLM` / `LangChainLLM` and their methods: [`gen_ai_core/llm/`](../../gen_ai_core/llm/) — see [`USAGE.md`](../../gen_ai_core/llm/USAGE.md) for the full API reference.
- The single-pass vs batching vs ReAct concepts (dependency graphs, non-determinism, the two failure faces): [`LEARNER-NOTES.md`](./LEARNER-NOTES.md).
