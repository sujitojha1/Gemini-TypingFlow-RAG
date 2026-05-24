# Memory Service — memory.py

## Overview

`memory.py` is the durable knowledge layer of the agent. It provides two public operations —
**read** (retrieval) and **write** (storage) — backed by a JSON flat-file store and a FAISS
vector index. Every write embeds the item's descriptor and appends the vector to FAISS;
every read queries FAISS first and falls back to keyword overlap when the vector path is empty.

```
memory.json          ← source of truth (list of MemoryItem, embeddings included)
state/index.faiss    ← FAISS binary index (768-dim inner-product, L2-normalised)
state/index_ids.json ← parallel list mapping FAISS integer position → item.id
```

---

## MemoryItem ([schemas.py:30-47](../schemas.py))

```python
class MemoryItem(BaseModel):
    id:               str               # "mem:a1b2c3d4"
    kind:             MemoryKind        # "fact" | "preference" | "tool_outcome" | "scratchpad"
    keywords:         list[str]         # 3-10 lowercase tokens for keyword fallback
    descriptor:       str               # one short human-readable line — what gets embedded
    value:            dict              # structured payload (entities, raw text, tool args, …)
    artifact_id:      str | None        # set when the result was promoted to the artifact store
    embedding:        list[float] | None  # 768-dim vector, set at write time; None for scratchpad
    source:           str               # "user_query" | "action" | "sandbox:…" | …
    run_id:           str               # ties every item back to the run that created it
    goal_id:          str | None        # ties tool_outcome items to the goal they served
    confidence:       float             # default 1.0
    created_at:       datetime
```

**Which kinds get an embedding:**

```python
_EMBEDDABLE_KINDS = {"fact", "preference", "tool_outcome"}
# scratchpad items are run-scoped / ephemeral — keyword search is enough
```

---

## Read Path — `read()`

```python
def read(query, history=None, *, kinds=None, top_k=8) -> list[MemoryItem]:
    vec_hits = _vector_search(query, kinds=kinds, top_k=top_k)
    if vec_hits:
        return vec_hits           # vector wins — keyword never runs
    return _keyword_search(query, history, kinds=kinds, top_k=top_k)
```

### Vector search — `_vector_search()`

```
query string
    │
    ▼
_try_embed(query, task_type="retrieval_query")
    └─ gateway.embed() → POST /v1/embed → Ollama (768-dim) or Gemini fallback
    └─ returns list[float] | None  (None if gateway is down)
    │
    ▼
_index()  ← loads FAISS from disk every call (cheap at S7 scale;
           ensures MCP subprocess writes are immediately visible)
    cold start: if index empty, rebuilds from embedded items in memory.json
    │
    ▼
idx.search(qvec, k = top_k*2 if kinds else top_k)
    ← fetches 2× buffer when a kind filter is active
      (FAISS doesn't know about kinds; extra slots absorb post-filter loss)
    returns [(item_id, cosine_score), ...]
    │
    ▼
_load() → dict[id → MemoryItem]
    filter by kind → cap at top_k → list[MemoryItem]
```

**Task-type asymmetry:**

| Operation | task_type | nomic prefix |
|---|---|---|
| Write (store) | `retrieval_document` | `"search_document: "` |
| Read (query) | `retrieval_query` | `"search_query: "` |

Query and document vectors live in compatible but asymmetric subspaces — standard dense retrieval practice.

### Keyword fallback — `_keyword_search()`

Activates only when vector search returns `[]` (gateway down, empty index, or all results
filtered out by kind). Scores by token overlap between the query and each item's
`keywords + descriptor`. The last 3 history entries are also tokenised and added to the
query token set to widen recall.

---

## Write Paths

All four write paths share the same terminal step: `_persist_item()`.

### `_persist_item()`

```python
def _persist_item(item: MemoryItem) -> MemoryItem:
    items = _load()
    items.append(item)
    _save(items)                                  # write memory.json
    if item.embedding is not None and item.kind in _EMBEDDABLE_KINDS:
        idx = _index()
        idx.add(item.id, item.embedding)          # append to FAISS
        idx.persist()                             # write index.faiss + index_ids.json
    return item
```

Both stores are updated in the same call. No async — writes are synchronous and blocking.

---

### `remember()` — LLM-classified write

Used for ambiguous free-form content (user queries, observations).

```
raw_text
    │
    ▼
_llm_classify(raw_text)   ← one gateway chat call (auto_route="memory", temp=1.0)
    returns {kind, descriptor, keywords, value}
    │
    ├─ classifier fails → _fallback_remember()  (deterministic, no LLM)
    │
    ▼
if kind in _EMBEDDABLE_KINDS:
    embedding = _try_embed(descriptor, "retrieval_document")
    │
    ▼
MemoryItem(kind, keywords, descriptor, value, embedding, …)
    │
    ▼
_persist_item()
```

**Classifier prompt rules (key excerpts):**
- `descriptor` must include concrete entities, dates, numbers — "Mom's birthday is 15 May 2026" is good; "birthday reminder" is bad.
- `value` must never be empty when identifiable entities exist; fallback is `{"raw": raw_text}`.
- `temperature=1.0` is intentional — creative classification. The safety net handles the failure mode where value comes back empty.

### `_fallback_remember()` — deterministic fallback

When the classifier LLM is unreachable. Defaults `kind="fact"`, extracts top-10 word tokens
as keywords, uses `raw_text[:200]` as descriptor. Still attempts embedding — if that also
fails, item is persisted without a vector and remains reachable through keyword search.

---

### `record_outcome()` — zero-LLM tool outcome write

Used by the agent loop after every `action.execute()`. No LLM call — kind is `tool_outcome`
by construction.

```
tool_call + result_text + artifact_id
    │
    ▼
keywords  = [tool_call.name] + tokens from string arguments (cap 10)
descriptor = "tool_name({args[:80]}) -> artifact {id}"
             or "tool_name({args[:80]}) -> {result_text[:120]}"
    │
    ▼
embedding = _try_embed(descriptor, "retrieval_document")
    │
    ▼
MemoryItem(kind="tool_outcome", value={tool, arguments, result_preview[:400]}, …)
    │
    ▼
_persist_item()
```

Outcome items are what make past tool calls retrievable by semantic similarity — e.g.
"what did we fetch earlier about X?" matches against the descriptor of a prior `fetch_url` call.

---

### `add_fact()` — direct fact write (document indexing)

Used by `index_document` in `mcp_server.py`. Kind is `fact` by construction; skips the LLM
classifier entirely. Embeds the descriptor and persists. This is the write path for every
chunk produced during document indexing.

---

## Embedding Strategy Summary

| Write path | Input to embedder | task_type |
|---|---|---|
| `remember()` | LLM-generated `descriptor` | `retrieval_document` |
| `_fallback_remember()` | `raw_text[:200]` | `retrieval_document` |
| `record_outcome()` | constructed `descriptor` | `retrieval_document` |
| `add_fact()` | caller-supplied `descriptor` | `retrieval_document` |

The **descriptor**, not the full content, is always what gets embedded. This keeps the
vector a tight semantic signal; the full payload lives in `value` and is returned after
lookup.

---

## Graceful Degradation

`_try_embed()` wraps the gateway call in a bare `except`:

```python
def _try_embed(text, task_type) -> list[float] | None:
    try:
        return list(_gateway_embed(text, task_type=task_type)["embedding"])
    except Exception as e:
        print(f"[memory] embedding failed ({e!r}); item written without vector")
        return None
```

An item with `embedding=None` is still persisted to `memory.json`. It will never appear in
FAISS results but remains fully reachable through the keyword fallback.

---

## Cross-Process Consistency

The MCP server (`mcp_server.py`) runs as a **separate subprocess**. When it calls
`index_document`, that process writes to the same `memory.json` and `index.faiss` files.
The main agent process handles this by calling `_index()` fresh on every read — it re-reads
`index.faiss` from disk each iteration rather than caching it in memory. At S7 scale the
overhead is negligible and consistency is guaranteed.
