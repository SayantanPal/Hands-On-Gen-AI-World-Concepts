# `gen_ai_core.llm` — Usage Guide

Two LLM wrappers live here. They solve the *same* job (talk to a model, optionally call
tools) but through **different stacks**:

| File | Class | Stack | Reach for it when… |
|---|---|---|---|
| `raw_llm.py` | `RawLLM` | raw provider SDKs (OpenAI / Groq / Ollama / Anthropic / Gemini / SAP) | you want provider-agnostic calls with **no framework**, and to see every message/round yourself |
| `langchain_llm.py` | `LangChainLLM` | LangChain (`ChatOpenAI`, `ChatGroq`, …) | you want LangChain niceties — prompt templates, pipes (`\|`), `@tool` objects, ecosystem |

Both read provider/model/temperature from `config.settings` (your `.env`), so you rarely pass
them explicitly.

---

## 1. `RawLLM` — raw SDK, provider-agnostic

### Plain text
```python
from gen_ai_core import RawLLM, Prompt

llm = RawLLM()
res = llm.generate_response_content(Prompt(role="a helpful assistant", user_query="Hello!"))
print(res.text)                       # the answer
print(res.input_tokens, res.output_tokens, res.latency_ms)   # LLMResult carries stats
```

### With tools (single-pass)
Tools are **plain dicts** — `name`, `description`, `parameters` (JSON Schema), `function`:
```python
TOOLS = [{
    "name": "fetch_place",
    "description": "Look up a destination's prices.",
    "parameters": {"type": "object",
                   "properties": {"place": {"type": "string"}},
                   "required": ["place"]},
    "function": fetch_place,          # any callable (or a LangChain @tool — auto-detected)
}]

llm = RawLLM(tools=TOOLS)
res = llm.generate_response_content(Prompt(role="trip planner", user_query="Cost of Goa?"))
print(res.text, res.tool_calls)
```

> **Single-pass limitation:** `RawLLM` runs exactly **one** round of tool calls, then removes
> the tools and forces a text reply (`kwargs.pop("tools")`). A question that needs two
> sequential rounds (e.g. "compare A vs B") **crashes** with `400 - Tool choice is none, but
> model called a tool`. See `demo_driver/my_trip_planner_demo/LEARNER-NOTES.md` for the full story
> and the ReAct-loop fix.

### Multi-round (ReAct loop) — `invoke_with_multiple_tools_in_agentic_react_loop(...)`
The fix for the single-pass crash: keep tools in the request on **every** round until the
model stops calling them. Handles sequential / multi-entity questions.
```python
llm = RawLLM(tools=TOOLS)
res = llm.invoke_with_multiple_tools_in_agentic_react_loop(
    Prompt(role="trip planner", user_query="Compare Coorg vs Manali, 6 friends, 4 nights."),
    max_iterations=5,
)                                   # → {"text", "tool_calls", "iterations", "latency_ms"}
```
- Providers: **openai / groq / ollama** (chat.completions). Raises `NotImplementedError` for
  anthropic / gemini / sap (their message formats differ — use single-pass there).
- `prompt` is flexible: **Prompt | ChatPromptTemplate (+ variables) | role-dict list | str**.

---

## 2. `LangChainLLM` — LangChain wrapper, FOUR entry points

This is the part that trips people up. The class exposes four methods; **(b) and (c) look
similar but behave very differently around tools**, and **(d) is the multi-round loop.**

### (a) `invoke(prompt)` — plain text, no tools
```python
from gen_ai_core.llm.langchain_llm import LangChainLLM
from gen_ai_core import Prompt

lc = LangChainLLM()
text = lc.invoke(Prompt(role="a helpful assistant", user_query="Hello!"))   # → str
```
Uses the **base** model (`_llm_base`). Even if you constructed with `tools=`, `invoke()`
ignores them.

### (b) `chain(template)` — a LangChain pipe, **text only**
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([("system", "..."), ("human", "{question}")])
pipe = lc.chain(template)                 # → RunnableSequence  (prompt | base_llm | StrOutputParser)
answer = pipe.invoke({"question": "Hi"})  # → str
```

> ⚠️ **The key gotcha:** `chain()` returns a LangChain **`RunnableSequence`**, not a
> `LangChainLLM`. It supports **`.invoke()` only** — there is **no `.invoke_with_tools()`** on
> it, and it wires the **base** model (no tools) + a `StrOutputParser`. So:
> ```python
> lc.chain(template).invoke_with_tools(vars)   # ❌ AttributeError — RunnableSequence has no such method
> lc.chain(template).invoke(vars)              # ✅ text only, never calls a tool
> ```
> **To call tools, do NOT use `chain()`.** Call `invoke_with_tools()` directly (next section).

### (c) `invoke_with_tools(prompt, variables=None)` — single-pass tool calling
```python
lc = LangChainLLM(tools=[compute_trip_cost])        # bare @tool, or a RawLLM-style dict
result = lc.invoke_with_tools(prompt, variables)    # → {"text": ..., "tool_calls": [...]}
```
Uses the **tool-bound** model (`_llm`): one round of tool calls → execute → model summarises.
The `prompt` argument is **datatype-flexible** (normalised by `_coerce_to_lc_messages`):

| You pass… | How it's consumed |
|---|---|
| `Prompt` (our dataclass) | `.to_lc_messages()` — `variables` ignored |
| `ChatPromptTemplate` | `.invoke(variables)` fills the `{slots}` |
| `list[BaseMessage]` | used as-is (copied) |
| `str` | wrapped as one `HumanMessage` |

Tools may be **either** a bare LangChain tool (`StructuredTool`, carries its own `.name`) **or**
a RawLLM-style dict (`{"name","description","parameters","function"}`). Use the dict only when
the same list must *also* feed `RawLLM`; otherwise the bare tool is cleaner.

### (d) `invoke_with_multiple_tools_in_agentic_react_loop(prompt, variables=None, *, max_iterations=5)`
Multi-round version of (c): tools stay bound on **every** round until the model stops calling
them. Use when one tool round isn't enough (sequential / multi-entity questions).
```python
lc = LangChainLLM(tools=TOOLS)
res = lc.invoke_with_multiple_tools_in_agentic_react_loop(
    Prompt(role="trip planner", user_query="Compare Coorg vs Manali, 6 friends, 4 nights."),
    max_iterations=5,
)                                   # → {"text", "tool_calls", "iterations", "latency_ms"}
```
Same flexible `prompt` input as (c). This is the LangChain twin of
`RawLLM.invoke_with_multiple_tools_in_agentic_react_loop`.

---

## 3. `chain()` vs `invoke_with_tools()` — the one-screen cheat sheet

```
          ┌─────────────────────────────┬──────────────────────────────────────┐
          │ lc.chain(template).invoke()  │ lc.invoke_with_tools(prompt, vars)    │
          ├─────────────────────────────┼──────────────────────────────────────┤
 returns  │ str                          │ {"text", "tool_calls"}                │
 model    │ _llm_base  (NO tools)        │ _llm       (tools bound)              │
 tools?   │ never calls a tool           │ one round of tool calls, then summary │
 input    │ ChatPromptTemplate + vars    │ Prompt | template+vars | msgs | str   │
 use for  │ templated text generation    │ letting the model call a tool         │
          └─────────────────────────────┴──────────────────────────────────────┘
```

**Rule:** want the model to *write* → `chain()`. Want the model to *act* (call a tool) →
`invoke_with_tools()`. They are separate on purpose; `chain()` is a text pipe, not a tool loop.

---

## 4. `RawLLM` vs `LangChainLLM` — which wrapper?

| | `RawLLM` | `LangChainLLM` |
|---|---|---|
| Dependency | provider SDKs only | LangChain |
| Tool defs | dicts (`name`/`description`/`parameters`/`function`) | bare `@tool` **or** dicts |
| Text call | `generate_response_content(Prompt)` → `LLMResult` (+token stats) | `invoke(Prompt)` → `str` |
| Templated text | build messages yourself | `chain(ChatPromptTemplate)` pipe |
| Tool call (single-pass) | `generate_response_content` (tools in ctor) | `invoke_with_tools` |
| Token/latency stats | ✅ on `LLMResult` | ✖ (not tracked) |
| Multi-round tools (ReAct) | `invoke_with_multiple_tools_in_agentic_react_loop` (openai/groq/ollama) | `invoke_with_multiple_tools_in_agentic_react_loop` |

Both are **single-pass** by default for tools. For multi-round / multi-entity questions, call
`invoke_with_multiple_tools_in_agentic_react_loop(...)` on either class — it keeps tools bound
every round. The same loop is also shown step-by-step as `react_loop` / `react_loop_langchain`
in `demo_driver/my_trip_planner_demo/` (see its `LEARNER-NOTES.md`).

---

## 5. Worked demos

All of these live in `demo_driver/my_trip_planner_demo/sample_demo_my_trip_planner.py`:

- `demo_single_pass_limitation_using_langchain_llm` / `demo_single_pass_limitation_using_raw_llm`
  — single-pass; Q1/Q2 work, Q3 (compare) crashes. Why multi-round needs a loop.
- `react_loop` / `react_loop_langchain` — the loop that fixes multi-round (same logic now also
  lives on the classes as `invoke_with_multiple_tools_in_agentic_react_loop`).
- `demo_prompt_template_chain` — **`chain()`** path: Python computes, model only writes prose.
- `demo_prompt_template_chain_v2` — **`invoke_with_tools()`** path: pre-resolve the fetch in
  Python, bind one tool, one grounded tool round (no loop).
