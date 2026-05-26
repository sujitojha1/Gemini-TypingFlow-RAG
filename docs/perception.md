# Perception Layer — perception.py

## Overview

`perception.py` is the **goal-management orchestrator** of the agent loop. It runs at the
top of every iteration, looks at the user's original query, the current Memory hits, and the
run history so far, and returns an updated `Observation` — the current list of goals with
done/not-done status and optional artifact attachments.

```
Each agent loop iteration
        │
        ▼
   perception.observe()
        │  inputs: query, memory hits, run history, prior goals
        │  one LLM call (structured JSON output)
        ▼
   Observation(goals=[Goal, ...])
        │
        ▼
   decision.py  ← sees the first unfinished Goal
```

**What Perception never does:**

- Never reads artifact bytes — it sees handles and descriptors only.
- Never names a specific MCP tool — it speaks in intent (`fetch`, `make searchable`,
  `query the knowledge base`), leaving tool selection to `decision.py`.
- Never contracts the goal list — prior goals keep their slot and id forever.

> **Grep confirmation:** `grep 'index_document\|search_knowledge' perception.py` → zero
> matches. Tool names are absent from both the system prompt and the Python code.

---

## Public API — `observe()`

```python
def observe(
    query:       str,
    hits:        list[MemoryItem],
    history:     list[dict],
    prior_goals: list[Goal],
    run_id:      str,
) -> Observation:
```

| Parameter | What it contains |
|---|---|
| `query` | The original user query — unchanged across all iterations |
| `hits` | Up to 12 `MemoryItem` objects returned by `memory.read()` this iteration |
| `history` | All agent events so far (actions, answers, errors) — clipped to last 10 |
| `prior_goals` | The `Goal` list from the previous `Observation`; empty on iteration 1 |
| `run_id` | Forwarded for context; not used inside `observe()` directly |

Returns an `Observation` whose `goals` list is the **sole output** consumed by the rest of
the loop.

---

## Data Models

### `_GoalDelta` (internal)

The raw shape the LLM is asked to emit per goal. No `id` field — identity drift across
iterations is impossible because the model never sees or produces IDs.

```python
class _GoalDelta(BaseModel):
    text:           str             # max 240 chars — short imperative
    done:           bool = False
    send_artifact:  bool = False
    artifact_index: int | None = None
```

### `_PerceptionOutput` (internal)

The top-level structured response schema passed to the LLM.

```python
class _PerceptionOutput(BaseModel):
    goals: list[_GoalDelta] = Field(default_factory=list, max_length=10)
```

Max 10 goals enforced at the schema level — the LLM cannot emit an unbounded list.

### `Goal` (from `schemas.py`)

The stable public shape used everywhere else in the agent.

```python
Goal(
    id:                 str          # "g:a1b2c3d4" — assigned here, preserved forever
    text:               str
    done:               bool
    attach_artifact_id: str | None   # "art:…" — resolved from artifact_index
)
```

---

## What the LLM Sees

The user message is assembled in `observe()` and contains four sections:

```
USER QUERY:
  <original query>

PRIOR GOALS:
  [{"text": "...", "done": false, ...}, ...]

MEMORY HITS (handles + descriptors only, no raw bytes; `i` is the
artifact_index to pass back when send_artifact is true):
  [{"i": 0, "kind": "tool_outcome", "descriptor": "...", "artifact_id": "art:…"}, ...]

RUN HISTORY (last 10 events):
  [...]
```

### Memory hit rendering — `_snapshot_hits()`

Artifacts in the hit list are assigned a sequential integer index `i` (0, 1, 2 …).
Non-artifact hits show `i: null`. This integer is what the LLM puts in `artifact_index`
when it wants to attach a specific artifact to a goal.

```python
def _snapshot_hits(hits: list[MemoryItem]) -> list[dict]:
    art_pos = 0
    for h in hits[:12]:          # hard cap at 12 hits
        i = art_pos if h.artifact_id else None
        if h.artifact_id:
            art_pos += 1
        yield {i, kind, descriptor, keywords, artifact_id}
```

Only `descriptor` and `keywords` are shown — **never raw bytes**. The bytes live in the
artifact store and are attached by the outer loop only when `attach_artifact_id` is set.

### History clipping — `_snapshot_history()`

- Keeps the last **10** history entries.
- Clips any string value longer than **240 chars** with `"..."` to prevent the context
  window from being dominated by large tool outputs.

---

## System Prompt Rules

The system prompt (`_SYSTEM_PROMPT`) encodes Perception's operating contract. Key rules:

### Intent-only language

Perception must speak in **intent verbs**, never tool names:

> "You speak at the level of INTENT, not tool selection. Write each goal as a short
> imperative describing WHAT must happen, not WHICH tool will do it."

Allowed example verbs: `fetch`, `open`, `list`, `look up the time`, `convert currency`,
`save a note`, **`make this content searchable`**, **`query the existing knowledge base`**,
`extract`, `summarise`, `compare`, `synthesise`.

The last two bolded phrases are the vocabulary contracts for the two RAG tools — the prompt
uses plain English rather than `index_document` / `search_knowledge`, and `decision.py`
maps those phrases back to the actual tools.

### Four-step procedure

| Step | Condition | Action |
|---|---|---|
| **1. First call** | `prior_goals` empty | Decompose query into separate goals (one per distinct item/file); end with a synthesis goal if the query is a question |
| **2. Subsequent calls** | `prior_goals` non-empty | Copy prior goal texts verbatim; mark `done: true` when RUN HISTORY shows satisfaction |
| **3. Artifact attachment** | First unfinished goal needs prior content | Set `send_artifact: true`, `artifact_index: i` of the relevant Memory hit |
| **4. Pure fetch/compute** | Goal needs no prior artifact | Leave `send_artifact: false`, `artifact_index: null` |

### Knowledge-base short-circuit (Step 1)

When Memory already contains `fact` items with descriptors starting `[sandbox:` or `[art:`
(markers of previously-indexed document chunks), Perception must emit a
**"query the existing knowledge base"** goal rather than re-fetch or re-read the source.
This ensures `search_knowledge` is called instead of `index_document` on a file that is
already indexed.

---

## Post-LLM Safety Passes

The LLM runs at `temperature=1.0` (creative, not deterministic). Two post-processing passes
protect against failure modes.

### Pass 1 — Goal-count invariant & deduplication

```
raw_goals from LLM
    │
    ├─ first len(prior_goals) slots → kept verbatim (position = identity)
    │
    └─ appended slots (new goals added by LLM) →
           deduplicated against prior_texts (case-insensitive strip)
           blank texts dropped
```

**Why appended goals are allowed (NOTES_RUNS §6 (4)):**  
A previous hard-truncate to `len(prior_goals)` blocked a run where `list_dir` revealed five
papers. The goal list was locked to three placeholders emitted before the listing was known.
Now new goals can be appended; only reordering and contraction are forbidden.

**Why deduplication is needed:**  
At `temperature=1.0` the LLM occasionally re-emits a prior goal in the append zone.
Deduplication silently drops the duplicate rather than creating two goals with the same
intent.

### Pass 2 — Synthesis-goal done guard

```python
_SYNTHESIS_DONE_KW = (
    "evaluate", "select", "synthes", "compare", "decide", "recommend",
    "tell me which", "most appropriate", "analy", "pick", "choose",
    "summarise", "summarize", "answer", "identify", "determine",
    "extract", "explain", "describe",
)
```

If the LLM marks a synthesis-type goal `done: true` on a turn where no `answer` history
event exists for that goal (or the answer text is < 60 chars), `proposed_done` is forced
back to `False`. This prevents Perception from closing a goal that Decision hasn't actually
answered yet.

Note the vocabulary split:  
- `_SYNTHESIS_DONE_KW` — governs the done guard (excludes lightweight verbs like `list`,
  `tell`, `name`; those can be satisfied by a tool call alone).
- `_ARTIFACT_NEEDED_KW` — governs artifact attachment (a superset that includes `list`,
  `report`, `tell`, `name` because those goals still need the data from a prior artifact).

### Pass 3 — Artifact attachment safety net

After goals are finalised, the first unfinished goal is inspected. If it:
- contains a synthesis keyword from `_ARTIFACT_NEEDED_KW`, **or**
- is a fetch/read goal that references a prior result (`result`, `article`, `item`,
  `first`/`second`/`third`, `url`, `link`)

AND there are artifacts in the hit list AND the model forgot to set `send_artifact` — the
most recent artifact is **force-attached**. For fetch/read-result goals, the logic
preferentially picks a "search" artifact (the one whose descriptor contains the word
"search") over the most-recent one.

```
first unfinished goal
    │
    ├─ already has attach_artifact_id? → skip
    ├─ no artifacts in memory?         → skip
    │
    └─ needs_artifact = True?
            ├─ yes, and is fetch+result → prefer search artifact
            └─ yes, otherwise           → attach most-recent artifact
```

Only the **first** unfinished goal is processed; the loop breaks unconditionally after it.

---

## ID & Slot Stability

```python
gid = prior_goals[i].id if i < len(prior_goals) else new_id("g")
```

Goal IDs are assigned exactly once — on the first iteration when the goal is created — and
never change. The `answer` history events reference `goal_id` so the done guard can
correlate them. If IDs shifted, the guard would always fail to find the matching answer.

---

## LLM Call

```python
LLM().chat(
    prompt=...,
    system=_SYSTEM_PROMPT,
    auto_route="perception",
    response_format={
        "type":   "json_schema",
        "schema": _PerceptionOutput.model_json_schema(),
        "name":   "PerceptionOutput",
        "strict": True,
    },
    temperature=1.0,
)
```

- `auto_route="perception"` — lets the gateway select the model; no provider is pinned.
  Pinning (e.g. `provider="g"`) would disable failover: if Gemini hiccups the call crashes
  with no recovery path.
- `response_format` with `strict: True` — structured output; the gateway enforces the
  schema before returning.
- `temperature=1.0` — intentional; creative decomposition produces better goal variety than
  greedy decoding. The two post-processing passes (done guard, artifact safety net) absorb
  the failure modes.

### Fallback

If the LLM returns an empty or unparseable response, `observe()` returns a single-goal
`Observation` whose text is the raw user query:

```python
return Observation(goals=[Goal(id=new_id("g"), text=query)])
```

This keeps the agent loop alive rather than crashing — Decision will attempt to satisfy the
query directly.

---

## Relationship to Other Layers

| Layer | Receives from Perception | Sends to Perception |
|---|---|---|
| **Agent loop** | `Observation.goals` — picks first `done=False` goal | Passes `prior_goals`, `history`, Memory `hits` |
| **Decision** | First unfinished `Goal` + attached artifact bytes | Nothing — Decision does not update goals |
| **Memory** | — | Provides `hits` that shape artifact-index assignments |
| **Artifacts** | `attach_artifact_id` resolved to bytes by the loop | Nothing — Perception never reads bytes directly |
