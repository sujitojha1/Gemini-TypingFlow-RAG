# Decision Layer — decision.py

## Overview

`decision.py` is the **tool-selection and execution layer** of the agent. It runs once per
iteration, after Perception has identified the first unfinished goal. Given that goal plus
the relevant Memory hits, recent history, and optionally the bytes of an attached artifact,
it makes exactly one binary choice:

```
(a) Answer  — synthesise a plain-text reply from available evidence
(b) Tool    — call exactly one MCP tool to perform external work
```

```
perception.observe()
    │  first unfinished Goal + attach_artifact_id
    ▼
decision.next_step()
    │  one LLM call  (temperature=0, tools=mcp_tools)
    ▼
DecisionOutput
    ├── answer: str              → agent loop records Answer event, marks goal done
    └── tool_call: ToolCall      → agent loop executes via MCP, records Action event
```

Decision does **not** update goals, does **not** loop internally, and does **not** know
whether a goal will be marked done — that is Perception's job on the next iteration.

---

## Public API — `next_step()`

```python
def next_step(
    goal:      Goal,
    hits:      list[MemoryItem],
    attached:  list[tuple[str, bytes]],
    history:   list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
```

| Parameter | What it contains |
|---|---|
| `goal` | The first unfinished `Goal` from the current `Observation` |
| `hits` | Memory items returned by `memory.read()` for the goal text |
| `attached` | List of `(artifact_id, raw_bytes)` pairs — resolved by the agent loop from `goal.attach_artifact_id` |
| `history` | All agent events so far; Decision sees the last 6 |
| `mcp_tools` | The live MCP tool schema list — passed directly to the LLM as the tools parameter |

Returns a `DecisionOutput` with exactly one of `answer` or `tool_call` populated.

---

## What the LLM Sees

The user message is assembled from four sections:

```
GOAL:
  <goal.text>

MEMORY HITS:
  - [fact] [sandbox:notes.md chunk 1/3] first 120 chars …
      chunk (sandbox:notes.md): full 600-char preview …
  - [tool_outcome] web_search(…) → art:abc123
      value: {"query": "…", "result_count": 5}

RECENT HISTORY:
  - iter 2: web_search (artifact art:abc123) → 5 results, titles: …
  - iter 3: ANSWER → …

ATTACHED ARTIFACTS:
--- art:abc123 ---
<raw bytes decoded to UTF-8, truncated to 30 KB head+tail if larger>
```

### Memory hit rendering — `_format_hits()`

Hits are capped at **10**. For each hit, the value payload is rendered according to its
shape:

| Value shape | What is shown |
|---|---|
| `value.raw` is a non-empty string | `raw: <first 200 chars>` |
| `value.chunk` is a non-empty string | `chunk (<source>): <first 600 chars>` |
| Other structured value | Compact JSON of all keys except `chunk`, capped at 240 chars |

The `chunk` preview (600 chars) is intentionally generous. When `search_knowledge` has
already populated Memory with indexed chunks, Decision must be able to synthesise directly
from the memory-hit list — without seeing the chunk text it would see that chunks exist but
couldn't read them, causing a `search_knowledge` loop.

### History rendering — `_format_history()`

Only the last **6** history entries are shown (Perception shows 10 — Decision's window is
tighter because the artifact bytes already provide context). The result descriptor for
`action` events is shown at **300 chars** (matching the cap applied by `agent7.py` when
storing it), not a shorter clip — a prior 140-char limit was hiding the tail of `list_dir`
outputs, causing Decision to treat partial file lists as complete.

### Attached artifact rendering — `_format_attached()`

Artifacts are decoded from bytes to UTF-8. If the decoded text exceeds **30 KB**
(`ATTACH_HEAD=20_000` + `ATTACH_TAIL=10_000`), a head-and-tail window is applied:

```
<first 20 000 chars>

...[truncated; full size N bytes]...

<last 10 000 chars>
```

This keeps large fetched pages within a comfortable model context while preserving both the
opening structure and the closing content.

---

## System Prompt — Rule Catalogue

The system prompt encodes twelve rules that govern Decision's choices. They are grouped
below by theme.

### Core dispatch rules

**RULE 1 — Never narrate**  
Answer or call a tool — never both. No commentary like "I'll now call fetch_url…".

**RULE 2 — Never invent a tool**  
Only call tools from the live `mcp_tools` list. The LLM cannot fabricate tool names.

**RULE 3 — Answer directly when satisfied; call a tool when not**  
If the goal is already satisfied by Memory + history, answer without a tool call.  
Exception: a `fetch`/`read`/`download` goal is **not** satisfied by a search snippet —
`fetch_url` must be called with the actual URL. A snippet is not the full page.

### Artifact handle rules

**RULE 5 — Artifact handles are NOT tool arguments**  
`art:…` strings are opaque handles. They must never be passed to `read_file`,
`list_dir`, `fetch_url`, or any other tool. If a goal needs an artifact's bytes, those
bytes already appear in the `ATTACHED ARTIFACTS` section — answer directly from them.

```
WRONG: read_file({"path": "art:abc1234"})
WRONG: fetch_url({"url": "art:abc1234"})
RIGHT: read the bytes already in ATTACHED ARTIFACTS and answer.
```

**RULE 6 — Sandbox tools operate on real files**  
`read_file` and `list_dir` operate on `sandbox/`, not on artifacts. Only call them when
the user has asked to read or list a real sandbox file by name.

### Answer quality rules

**RULE 7 — Answer substantively**  
Synthesis/extract/list/compare goals must produce at least 3 sentences or a list of items.
Not a one-liner acknowledgement.

**RULE 8 — Reminders and notes → `create_file`**  
"Remember X", "save X", "set a reminder" goals must call `create_file` (or `update_file`
for re-saves) under the sandbox. Decision must not reply that it cannot set reminders —
`create_file` is the mechanism.

### RAG tool rules

**RULE 9 — Indexing content → `index_document`**  
When the goal asks to make content *searchable for later turns or runs* (phrasings:
`index`, `ingest`, `make searchable`, `add to the knowledge base`, `load into memory`),
call `index_document`. `read_file` only returns bytes once; `index_document` writes chunks
into Memory so they survive across turns and runs.

**RULE 10 — Answering from indexed facts → `search_knowledge`**  
When the goal is to answer a question AND Memory hits already contain `fact` items with
descriptors starting `[sandbox:` or `[art:` (previously-indexed chunks), call
`search_knowledge` instead of re-fetching or re-reading the source. If chunk previews are
already visible in the Memory hits, synthesise directly — do not call `search_knowledge`
again.

**RULE 12 — Anti-loop limit on `search_knowledge`**  
For any synthesis/answer/summarise goal, `search_knowledge` may be called **at most twice**.
Decision counts prior `search_knowledge` calls in RECENT HISTORY; on the third attempt it
must answer from the best material found so far, even if imperfect.  
`read_file` must not be used as a substitute for answering a synthesis goal — the attached
artifacts and Memory hits are sufficient.

### Completeness rules

**RULE 11 — Complete "all / every / each" goals fully**  
When the goal quantifies a set (`index every .md file`, `fetch each result`), Decision
must count the items in the original list (from a prior `list_dir` or search action visible
in RECENT HISTORY) and compare against completed tool calls. If any item is unprocessed,
call the tool for the **next unprocessed item** — never answer or consider the goal done
until every item is covered.

---

## Vocabulary Contract with Perception

Decision's rules and Perception's system prompt share a paired vocabulary. When either side
changes, the other must be reviewed.

| Intent phrase (Perception emits) | Decision rule | MCP tool called |
|---|---|---|
| `make this content searchable` | RULE 9 | `index_document` |
| `query the existing knowledge base` | RULE 10 | `search_knowledge` |
| `fetch <URL>` / `read <file>` | RULE 3 exception | `fetch_url` / `read_file` |
| `save a note` / `set a reminder` | RULE 8 | `create_file` / `update_file` |
| `look up the time` | RULE 3 (tool needed) | `get_time` |
| `convert currency` | RULE 3 (tool needed) | `currency_convert` |

Perception never names tools; Decision never names intent verbs in its rules. The vocabulary
in the middle is what binds them. A future skill-abstraction layer will replace this implicit
coupling with explicit capability tags.

---

## LLM Call

```python
LLM().chat(
    prompt=prompt,
    system=_SYSTEM_PROMPT,
    cache_system=True,           # system prompt is stable across turns → cache it
    tools=mcp_tools,             # live tool schema list from MCP
    tool_choice="auto",          # model chooses answer vs tool call
    auto_route="decision",       # gateway picks provider; no pin
    temperature=0,               # deterministic — Decision must be reliable
    max_tokens=2048,
)
```

Key differences from the Perception call:

| | Perception | Decision |
|---|---|---|
| `temperature` | `1.0` (creative decomposition) | `0` (deterministic dispatch) |
| `cache_system` | not set | `True` — system prompt is stable and large; caching saves cost |
| `tools` | not set | live `mcp_tools` list |
| `response_format` | strict JSON schema | not set — model uses native tool-call or plain text |

### Tool call extraction

```python
tcs = reply.get("tool_calls") or []
if tcs:
    tc = tcs[0]          # only the first tool call is used; model is instructed to emit one
    return DecisionOutput(tool_call=ToolCall(name=tc["name"], arguments=tc.get("arguments") or {}))
return DecisionOutput(answer=(reply.get("text") or "").strip())
```

If the LLM emits multiple tool calls (which it should not given RULE 1), only `tcs[0]` is
used and the rest are silently dropped.

---

## `DecisionOutput` (from `schemas.py`)

```python
class DecisionOutput(BaseModel):
    answer:    str | None = None
    tool_call: ToolCall | None = None
```

Exactly one field is populated. The agent loop checks which one and dispatches accordingly:
- `answer` → recorded as an `answer` history event; Perception may then mark the goal done.
- `tool_call` → executed via MCP; result recorded as an `action` history event with an
  optional `artifact_id`.

---

## Relationship to Other Layers

| Layer | Sends to Decision | Receives from Decision |
|---|---|---|
| **Agent loop** | Goal, hits, attached bytes, history, mcp_tools | `DecisionOutput` — routes to MCP executor or answer recorder |
| **Perception** | Goal text (via agent loop) | Nothing — Perception sees Decision's output only as history events next iteration |
| **MCP server** | — | `ToolCall.name` + `ToolCall.arguments` dispatched over stdio |
| **Memory** | `hits` list | Nothing — Decision reads Memory indirectly through `_format_hits()` |
| **Artifacts** | `attached` bytes (resolved by loop) | Nothing — Decision reads artifact bytes from the formatted prompt, not the store |
