# Agent Loop — agent7.py

## Overview

`agent7.py` is the top-level orchestrator. It runs a single async function `run(query)` that
loops up to **20 iterations**, driving four typed layers in sequence each turn until every
goal is satisfied or the iteration cap is reached.

```
memory.read  →  perception.observe  →  decision.next_step  →  action.execute  →  memory.record_outcome
```

---

## Startup Sequence

```python
ensure_gateway()          # auto-starts llm_gatewayV7 on port 8107 if not running
run_id = uuid4().hex[:8]  # unique ID for this run; attached to every memory write

memory.remember(query, source="user_query", run_id=run_id)
# Classifies and embeds the raw query so facts/preferences in it
# survive into future runs. Failures are swallowed — a bad query
# record must never abort the run.

stdio_client(mcp_server.py)   # spawns MCP server as a subprocess
session.list_tools()           # loads all 11 tools; passed to Decision every iteration
```

---

## Iteration Structure

Each iteration (1 → MAX_ITERATIONS = 20) executes exactly five steps:

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1  MEMORY READ                                        │
│          memory.read(query, history)                        │
│          vector search → keyword fallback → list[MemoryItem]│
├─────────────────────────────────────────────────────────────┤
│  Step 2  PERCEPTION                                         │
│          perception.observe(query, hits, history,           │
│                             prior_goals, run_id)            │
│          LLM call → emits/updates goal list → Observation   │
├─────────────────────────────────────────────────────────────┤
│  Step 3  DECISION                                           │
│          decision.next_step(goal, hits, attached,           │
│                             history, tools)                 │
│          LLM call → tool_call | answer                      │
├─────────────────────────────────────────────────────────────┤
│  Step 4  ACTION  (only when Decision emits tool_call)       │
│          action.execute(session, tool_call)                 │
│          MCP dispatch → (result_text, artifact_id | None)   │
├─────────────────────────────────────────────────────────────┤
│  Step 5  MEMORY WRITE                                       │
│          memory.record_outcome(tool_call, result,           │
│                                artifact_id, run_id, goal_id)│
│          embeds descriptor → FAISS + memory.json            │
└─────────────────────────────────────────────────────────────┘
```

---

## Goal Lifecycle

### Creation (iteration 1)
Perception receives an empty `prior_goals` list and decomposes the query into one or more
short imperative goals, each written at the level of **intent** (not tool names).

```
query: "What is the time in Tokyo and Bangalore?"

goals emitted:
  ○ g:a1b2c3d4 — Look up the current time in Tokyo
  ○ g:e5f6a7b8 — Look up the current time in Bangalore
  ○ g:c9d0e1f2 — Tell the user both times and the difference in hours
```

### Update (iterations 2+)
Perception re-receives the same goal list via `prior_goals`. Rules:
- Goals are identified by **position**, never by ID (LLMs cannot reliably track identity).
- Prior goal texts are copied **verbatim** into the same slot — no rewording.
- A goal is marked `done: true` when RUN HISTORY shows an action satisfying it.
- Once done, it stays done in every later iteration.
- New goals may only be **appended** at the end (e.g. after `list_dir` reveals unknown files).
- Duplicate appended goals are dropped.

### Synthesis guard
If a goal contains synthesis verbs (`answer`, `summarise`, `extract`, `compare`, etc.),
Perception cannot mark it done on the strength of a tool call alone. The history must
contain an `answer` event for that `goal_id` with `len(text) > 60`. If not, `done` is
forced back to `False`.

### Termination
The loop exits as soon as **any** of these is true:
1. `obs.all_done` — every goal has `done: true`
2. `obs.next_unfinished()` returns `None` — no open goal found
3. Iteration counter reaches `MAX_ITERATIONS` (20)

---

## Artifact Attachment

After Perception runs, the loop checks whether the current goal has `attach_artifact_id` set:

```python
if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
    blob = artifacts.get_bytes(goal.attach_artifact_id)
    attached.append((goal.attach_artifact_id, blob))
```

`attached` is passed directly to Decision so the LLM sees the raw bytes (e.g. a prior
fetch result containing URLs the next goal needs to act on). Perception sets
`attach_artifact_id` either from the LLM's output or from a safety-net heuristic that
force-attaches the most recent relevant artifact when the model forgets.

---

## History Record

Every iteration appends one or two entries to `history: list[dict]`:

| `kind` | When added | Key fields |
|---|---|---|
| `"answer"` | Decision emits a final answer | `goal_id`, `text` |
| `"action"` | Decision emits a tool call | `goal_id`, `tool`, `arguments`, `result_descriptor`, `artifact_id` |

History is passed to both Perception (last 10 events, clipped to 240 chars each) and
Decision (recent context) every iteration. It is **not** persisted — it lives only for
the duration of one `run()` call.

---

## What Each Layer Owns

| Layer | Owns | Sees |
|---|---|---|
| **Perception** | Goal list state across iterations | Query, memory hits (descriptors), history, prior goals |
| **Decision** | Single next action per turn | Current goal, memory hits, attached artifact bytes, history, tool schemas |
| **Action** | MCP dispatch, artifact promotion | Tool call from Decision |
| **Memory** | Durable facts + FAISS index | Query (on read); tool call + result (on write) |

Only Perception maintains state across iterations. All other layers are stateless per call.
