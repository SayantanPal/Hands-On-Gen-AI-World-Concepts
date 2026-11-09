# Learner's Note — Non-Deterministic Tool Batching & the Single-Pass Trap

> **TL;DR** — Whether a question survives *single-pass* tool calling is **not a property
> of the question alone, nor of RawLLM vs LangChain**. It depends on a runtime decision
> the model makes freshly on every call: *"Do I fire all my tool calls in one batch, or
> do I call one tool, look at the result, then call the next?"* That decision is
> **non-deterministic**. Q1 and Q2 land on different sides of it from run to run. Q3 is
> the one question in this demo that is **structurally** immune to the batching trick —
> it will need multiple rounds *every single time*, and the note below explains exactly why.

---

## 1. What we actually observed

Running `demo_single_pass_limitation()` produced this (abbreviated):

```
[OK] Q1 (LangChainLLM.invoke_with_tools):
     "How much will a 3-night Goa trip cost for 5 friends?"
     Tools used: ['fetch_place']
     A: ...                       ← EMPTY answer (no crash, but no result either)

[OK] Q2 (RawLLM.generate_response_content):
     "We have a budget of ₹8000 per person. Can 4 friends afford a 2-night Manali trip?"
     Tools used: ['fetch_place']
     A: "Short answer: No ..."     ← FULL, correct answer

[FAIL] Q3 (RawLLM.generate_response_content):
     "Compare Coorg vs Manali for 6 friends, 4 nights. Which is cheaper?"
     CRASHED: 400 - Tool choice is none, but model called a tool
```

Three questions, same two tools (`fetch_place`, `compute_trip_cost`), same single-pass
engine family — **three different outcomes**. The rest of this note explains why.

---

## 2. The core concept: *parallel (batched)* vs *sequential (multi-round)* tool calls

When a model is given tools, each turn it can reply with **zero, one, or many** tool calls.

- **Parallel / batched** — the model emits *several* tool calls in **one** assistant
  message. The harness runs them all, feeds every result back together, and the model
  continues. This is what OpenAI calls *parallel function calling* (on by default;
  toggled with `parallel_tool_calls`).
- **Sequential / multi-round** — the model emits **one** tool call, waits to see its
  result, *then* decides what to call next. This needs a **loop** that keeps tools
  available across turns (a **ReAct loop**).

> From OpenAI's function-calling guide:
> *"The model may choose to call multiple functions in a single turn... developers manage
> dependencies by executing the tool-calling loop iteratively: receive calls, execute them,
> return results, and let the model decide whether additional calls are needed."*

**Single-pass** (what `RawLLM.generate_response_content()` and
`LangChainLLM.invoke_with_tools()` do) allows **exactly one** round of execution:

```
call 1  →  model emits tool_calls  →  we execute them  →  call 2 (tools REMOVED) → must be text
```

So single-pass only succeeds if **everything the model needs fits into that one batch**.

---

## 3. The right mental model: a *dependency graph*

Draw each question as a graph of "what must happen before what":

```
Q1 / Q2  (one destination)              Q3  (compare two destinations)

  fetch_place(X)                          fetch_place(Coorg)   fetch_place(Manali)
       │  (prices)                              │                    │
       ▼                                        ▼                    ▼
  compute_trip_cost(...prices...)        compute(Coorg)        compute(Manali)
       │                                        └────────┬───────────┘
       ▼                                                 ▼
   final text answer                         COMPARE → "which is cheaper?"  ← REDUCE step
```

The key feature of a graph edge (`A → B`) is a **data dependency**: `compute_trip_cost`
needs the *real prices* that only `fetch_place` can return. You cannot run `B` correctly
until `A`'s result exists.

A **batch** can only contain nodes that have **no unresolved dependency between them**.
That is the whole game.

---

## 4. Why Q2 "worked" in a single pass — the *lucky-guess batch*

Q2 has a `fetch → compute` dependency, just like Q1. So how did it finish in one round?

Because the model **did not actually wait for the fetch result**. It emitted a batch like:

```jsonc
// one assistant message, TWO tool calls fired together:
fetch_place("Manali")
compute_trip_cost(room_price_per_night=2500, travel_cost_per_person=9000,
                  daily_spend_per_person=1300, people=4, nights=2, budget_per_person=8000)
```

Notice the `compute_trip_cost` arguments are **already filled with numbers** — numbers the
model pulled from its **own training knowledge** of roughly what Manali costs, *not* from
the `fetch_place` call sitting right next to it. In effect the model **guessed** the prices
so it could collapse a 2-layer graph into a 1-layer batch.

### ⚠️ The deep, uncomfortable insight

**In that batched call, the `fetch_place` result arrived too late to matter.**
`compute_trip_cost` ran on **guessed** prices, and the fetch was essentially wasted.

It only *looked* correct because **this demo deliberately seeded the database with the same
well-known numbers the model already knows** (see `DESTINATIONS` and `data/*.csv` — Manali =
room 2500, travel 9000, daily 1300). The guess matched the DB *by construction*.

> **In the real world, your DB holds prices the model has never seen** (live inventory,
> your negotiated rates, today's surge pricing). A batched `compute` on guessed numbers
> would then return a **confidently wrong** answer — and single-pass would **never crash**,
> so you'd never notice. This is the dangerous failure mode: *silent wrongness beats a loud
> crash only until it ships to production.*

---

## 5. Why Q1 "failed" under LangChain single-pass — the *honest sequential* choice

Q1 is structurally identical to Q2 (`fetch → compute → answer`). Yet this run produced an
**empty answer**. Why?

Because on *this* run the model chose the **honest, sequential** strategy:

```
Round 1:  fetch_place("Goa")            ← "let me get the REAL prices first"
          (result comes back)
Round 2:  wants compute_trip_cost(...)  ← but invoke_with_tools() does only ONE round
```

`invoke_with_tools()` executes round 1, hands the fetch result back to the model, and the
model's *second* reply is... another tool call (`compute_trip_cost`). But single-pass has
no third turn, and the code reads `final.content`, which is **empty** because the model
answered with a tool call, not text. Result: `A: ...` (nothing).

This is the **same single-pass limitation as the Q3 crash**, wearing a quieter mask:

| Engine | When the model under-fetches in single-pass | Symptom |
|---|---|---|
| `RawLLM` (OpenAI/Groq raw) | 2nd turn has tools **removed** (`kwargs.pop("tools")`) but model still calls one | **Loud 400 crash** |
| `LangChainLLM.invoke_with_tools` | 2nd turn's tool call is simply never executed | **Silent empty answer** |

---

## 6. The headline: there are **TWO** dice rolls, not one

The difference between Q1 and Q2 was the model's **strategy**, not the question and not the
SDK. But "strategy" is really **two separate decisions the model re-rolls on every single
call** — and only one of them has an off switch.

```
Every tool-calling turn, the model silently decides:

  Layer 1 — BATCHING:            "How MANY tool calls do I fire this turn — one, or several?"
  Layer 2 — SELECTION & ORDER:   "WHICH tool do I fire, and in WHAT order?"
```

| Layer | Who controls it | Lever |
|---|---|---|
| **1 — Batching** | You *can* pin it | `parallel_tool_calls=False` → at most one tool per turn |
| **2 — Selection & ordering** | **Nobody.** The model decides fresh every call. | *There is no lever.* |

Here is the whole demo's behaviour seen through those two layers. Take the Manali question
and watch every path it can take.

### Setting A — `parallel_tool_calls = True` (the default, what we ran)

The model is *allowed* to fire several tools at once, but is never *forced* to. Three
different runs can produce three different outcomes:

```
Run ①  turn 1 fires:  [ fetch_place("Manali") ,  compute_trip_cost(room=2500, travel=9000, ...) ]
        → both in ONE batch. compute's numbers were GUESSED (the fetch result isn't back yet).
        → turn 2 summarises → looks grounded, but the fetch was decorative.
        → right ONLY because our DB happens to match the guess (see §4).

Run ②  turn 1 fires:  [ fetch_place("Manali") ]        ← just the fetch
        → turn 2 has tools removed → model still wants compute → CRASH / empty.

Run ③  turn 1 fires:  [ compute_trip_cost(room=2500, ...) ]   ← pure guess, no fetch at all
        → turn 2 summarises → "works", but fetch_place never even ran.
```

> **The sharp point you raised:** even with batching ON and the model calling *both* tools
> at once (Run ①), nothing makes it feed the fetch result into `compute`. The harness does
> exactly one round, so the fetch and the compute happen *simultaneously* — the compute
> can only use **guessed** numbers. **The grounding step is missing, so LLM
> non-determinism decides whether the answer is right.**

### Setting B — `parallel_tool_calls = False`

Batching (Layer 1) is now off: one tool per turn, guaranteed. But **Layer 2 still rolls** —
the model still chooses *which* one tool:

```
Run ④  turn 1:  fetch_place("Manali")   → turn 2 stuck (tools gone) → CRASH / empty.
Run ⑤  turn 1:  compute_trip_cost(guessed)  → turn 2 summarises → guess-based "success".
```

So forcing `False` deletes Runs ① and ③ (anything with two calls in a turn) but **cannot
delete Run ⑤** — the model is still free to *open* with a guessed `compute_trip_cost` and
skip the fetch entirely. **You closed Layer 1; Layer 2 is still wide open.**

### The one-line conclusion

`parallel_tool_calls` is a **Layer-1 lever only**. No setting — not `temperature=0`, not a
stern prompt — turns Layer 2 into a guarantee. (Those nudge the *odds*: a prompt like "use
the database for exact prices" makes a fetch-first run more *likely*, never *certain*.)

**You cannot configure correctness out of a missing loop.** As long as the harness stops
after one round, which run you get — grounded, crashed, or confidently-wrong — is left to
the dice.

---

## 7. Why Q3 is the one question that is **structurally guaranteed** to need a loop

Q2 escaped via a guess-and-batch. Why can't Q3 do the same trick? Two compounding reasons —
the second is the decisive, *structural* one.

### Reason A — it would have to guess *twice as much*, for a task that punishes guessing
To collapse Q3 into one batch, the model would have to hallucinate the **full price set for
BOTH** Coorg *and* Manali, correctly, simultaneously. That is a far longer-odds gamble than
Q2's single guess. And a question whose entire point is *"which is cheaper?"* is exactly the
kind of accuracy-sensitive task where models are trained to **fetch authoritative data
first** rather than wing it. So in practice the model fetches — and the moment it fetches
before computing, it has committed to a second round.

### Reason B — the question contains a **REDUCE step that cannot live in the input batch**
This is the ironclad part. "Which is cheaper?" is a **reduction/aggregation** over a set:

```
answer = argmin( cost(Coorg), cost(Manali) )
```

A reduction is the model's **reasoning performed *after* its inputs exist**. You cannot emit
"compare these two totals" as a tool call *in the same batch that is supposed to produce
those totals* — the totals don't exist yet when the batch is formed. The comparison is,
by definition, a **later turn**.

So even in the most aggressive imaginable batch, Q3's dependency graph has an irreducible
depth:

```
   (fetch ∥ fetch)  →  (compute ∥ compute)  →  REDUCE/compare
   └── layer 1 ──┘      └──── layer 2 ────┘     └─ layer 3 (pure reasoning) ─┘
```

There is **no flattening** of a 3-layer graph where the final layer is a reduction over the
outputs of the middle layer. A single round of tool execution can satisfy **at most one
layer of fetches plus a batch of guessed computes** — it can never also perform the
after-the-fact comparison. Hence:

> **Q3 needs ≥ 2 rounds on every run, for every model, at every temperature. It is not a
> matter of luck — it is a matter of graph shape.** Q1 and Q2 *can* get lucky; Q3 cannot.

(Contrast with Q1/Q2: a *single* `fetch → compute → answer` chain where the final "answer"
is just prose summarizing one compute. A model willing to guess can fold the fetch away and
produce that prose in round 2. Q3's final step isn't prose summarizing one result — it's a
decision *over multiple results*, which forces the extra round.)

---

## 8. The fix — a ReAct loop (keep tools available until the model stops asking)

The cure for all three failure shapes (crash, empty, silent-wrong) is the same: stop doing
one round, and **loop** — re-sending the tools on every turn — until the model replies with
plain text and no tool calls.

```
while model keeps emitting tool_calls (up to max_iterations):
    run the tool calls
    append results
    call the model again  ← tools STILL available
return the first turn that has text and no tool_calls
```

That is exactly what `react_loop()` (raw SDK) and `react_loop_langchain()` (LangChain) do in
`my_trip_planner.py`. In a ReAct loop:

- Q1 safely does `fetch → compute → answer` across rounds.
- Q2 works whether the model batches *or* sequences.
- Q3 does `fetch Coorg → fetch Manali → compute → compute → compare`, no crash.

ReAct trades cost for correctness: single-pass is exactly 2 API calls; a loop is `2 + 2N`
in the worst case (`N` = rounds). Most questions need 1–3 rounds.

---

## 9. Flashcards (quiz yourself)

1. **Q:** Did RawLLM beat LangChain on Q2, or did LangChain fail Q1?
   **A:** Neither — the SDKs were incidental. The model *batched* on Q2 and *sequenced* on
   Q1. A re-run can flip both.

2. **Q:** When a model batches `fetch_place` and `compute_trip_cost` together, where did
   `compute`'s price arguments come from?
   **A:** The model's **training-knowledge guess**, not the fetch result (which arrives too
   late in the same batch). Correct here *only* because the demo DB was seeded to match.

3. **Q:** Which is more dangerous in production — RawLLM's crash or LangChain's empty answer?
   **A:** Both are visible. The *truly* dangerous case is a **batched guess that doesn't
   crash and returns a confidently wrong number** when the real DB differs from the model's
   guess.

4. **Q:** Why can't Q3 ever be rescued by batching?
   **A:** Its final step is a **reduction** ("which is cheaper") over two computed totals.
   A reduction is reasoning done *after* inputs exist, so it can't share a batch with the
   calls that produce those inputs. The graph has an irreducible 3rd layer → ≥ 2 rounds
   always.

5. **Q:** Does `parallel_tool_calls = false` guarantee Q1/Q2 behave safely?
   **A:** No. It closes **Layer 1** (no batching), but **Layer 2** still rolls — the model
   can still open with a guessed `compute_trip_cost` and skip the fetch. The only thing that
   makes Layer-2 non-determinism *harmless* is a ReAct loop, not a setting.

---

## 10. References

- **OpenAI — Function Calling guide** (parallel function calling, `parallel_tool_calls`,
  iterative tool-call loop for dependencies):
  https://developers.openai.com/api/docs/guides/function-calling
- **LangChain — Tool calling** (models return a list of `tool_calls` on one `AIMessage`;
  execution and looping are the developer's responsibility):
  https://docs.langchain.com/oss/python/langchain/overview
- **In this repo:**
  - `my_trip_planner.py` → `demo_single_pass_limitation()` (Q1/Q2/Q3 single-pass)
  - `my_trip_planner.py` → `react_loop()` / `react_loop_langchain()` (the fix)
  - `gen_ai_core/llm/raw_llm.py` → `kwargs.pop("tools", None)` (why RawLLM crashes round 2)
  - `gen_ai_core/llm/langchain_llm.py` → `invoke_with_tools()` (single-pass; empty on
    under-fetch) vs a real loop
