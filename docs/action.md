# Action Layer — action.py & artifacts.py

## Overview

`action.py` is the **pure dispatcher** of the agent loop. It receives a `ToolCall` from
Decision, executes it over MCP, and decides what to do with the result:

- **Small result** (≤ 4 KB) — returned as a plain text descriptor; stored directly in
  Memory by the agent loop.
- **Large result** (> 4 KB) — bytes written to the artifact store; a short preview +
  handle returned as the descriptor.

```
decision.next_step() → DecisionOutput(tool_call=ToolCall(…))
        │
        ▼
action.execute(session, tool_call)
        │
        ├─ guard: artifact handle passed as path/url?  → return ERROR descriptor, None
        │
        ├─ session.call_tool()  ← MCP stdio call to mcp_server.py
        │
        ├─ result ≤ 4 KB?  → return (text, None)
        │
        └─ result > 4 KB?  → artifacts.put(bytes)
                               return ("[artifact art:…] preview: …", art_id)
```

No LLM is involved. Action's job is purely I/O: call the tool, gate on size, route the
bytes.

---

## `execute()` — the only public function

```python
async def execute(
    session:   ClientSession,
    tool_call: ToolCall,
) -> tuple[str, str | None]:
```

| Parameter | What it is |
|---|---|
| `session` | The live MCP `ClientSession` — the open stdio connection to `mcp_server.py` |
| `tool_call` | `ToolCall(name, arguments)` from `DecisionOutput` |

Returns `(descriptor, artifact_id_or_None)`.

Both values are passed back to the agent loop, which hands them to `memory.record_outcome()`
and attaches `artifact_id` to the history event.

---

## Artifact Handle Guard

Decision occasionally hallucinates that an `art:…` handle is a real file path or URL
(despite RULE 5 in the decision system prompt). Action catches this **before** the MCP
call wastes a round-trip:

```python
for arg_name in ("path", "url"):
    v = tool_call.arguments.get(arg_name)
    if isinstance(v, str) and v.startswith("art:"):
        return (
            f"ERROR: {arg_name}={v!r} is an artifact handle, not a path/URL. "
            f"Artifact bytes are attached by Perception when needed — answer "
            f"from ATTACHED ARTIFACTS instead of calling {tool_call.name}.",
            None,
        )
```

The error descriptor is recorded in history as if the tool had returned it. Perception
sees this on the next iteration and can keep the goal open; Decision reads the error in
RECENT HISTORY and corrects its approach.

Only `path` and `url` argument names are checked — the two fields that sandbox and web
tools accept. Other argument names are not guarded.

---

## MCP Dispatch — `session.call_tool()`

```python
result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
```

This is a single `await` over the MCP `ClientSession`. The session is managed by the
agent loop (`agent7.py`) and kept open for the full run — no reconnect per call.

### Result collation — `_result_to_text()`

MCP returns a `CallToolResult` whose `content` field is a list of content items
(typically `TextContent`). The function joins them into one string:

```python
def _result_to_text(result: Any) -> str:
    parts = []
    for c in result.content:
        text = getattr(c, "text", None)
        parts.append(text if text is not None else str(c))
    return "\n".join(parts)
```

`str(c)` is the fallback for non-text content items (e.g. image blobs, binary data).
In practice all current tools return `TextContent`; the fallback is a safety net for
future tools.

---

## Size Gate — `ARTIFACT_THRESHOLD_BYTES`

```python
ARTIFACT_THRESHOLD_BYTES = 4096   # ~1 A4 page of text
```

The byte count is measured on the **UTF-8 encoded** result string.

### Small path (≤ 4 KB)

The full text is returned as the descriptor. The agent loop writes it directly into
Memory via `memory.record_outcome()`. Decision will see it in RECENT HISTORY on the
next turn.

### Large path (> 4 KB)

```python
art_id = artifacts.put(
    text.encode("utf-8"),
    content_type="text/plain",
    source=f"mcp:{tool_call.name}",
    descriptor=f"{tool_call.name}({json.dumps(tool_call.arguments)[:80]}) → {nbytes} bytes",
)
descriptor = (
    f"[artifact {art_id}, {nbytes} bytes] preview: "
    + text[:240].replace("\n", " ")
    + ("..." if nbytes > 240 else "")
)
return descriptor, art_id
```

The descriptor stored in Memory is a **240-char preview** plus the handle, not the full
content. This keeps the Memory + Decision context window small. The bytes themselves are
in the artifact store, accessible only when Perception explicitly attaches them to a goal.

The artifact store descriptor (written to `state/artifacts/<digest>.json`) is separately
constructed as `"tool_name({args[:80]}) → N bytes"` — used for provenance, not for
Decision context.

---

## Artifact Store — `artifacts.py`

The artifact store is a **content-addressable file system** under `state/artifacts/`.
Every artifact is stored as two files keyed by the first 16 hex chars of its SHA-256:

```
state/artifacts/
    <digest>.bin    ← raw bytes (written once; never modified)
    <digest>.json   ← Artifact metadata (id, content_type, size_bytes, source, descriptor)
```

### `put()` — write with deduplication

```python
def put(blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
    digest = hashlib.sha256(blob).hexdigest()[:16]
    art_id = f"art:{digest}"
    if not bin_path.exists():
        bin_path.write_bytes(blob)
        meta_path.write_text(meta.model_dump_json(indent=2))
    return art_id
```

If the same bytes are stored twice (e.g. two fetches of the same cached page), the second
call is a no-op — `.bin` already exists, metadata is not overwritten, and the same
`art_id` is returned. Deduplication is free because the key is content-derived.

### `get_bytes()` — retrieve raw bytes

```python
def get_bytes(artifact_id: str) -> bytes:
    digest = artifact_id.removeprefix("art:")
    return (STORE / f"{digest}.bin").read_bytes()
```

Called by the agent loop when it resolves `goal.attach_artifact_id` into bytes to pass to
Decision. Also called by `index_document` in `mcp_server.py` when path starts with `art:`.

### `get_meta()` — retrieve metadata

```python
def get_meta(artifact_id: str) -> Artifact:
```

Returns the `Artifact` schema object (id, content_type, size_bytes, source, descriptor).
Used for provenance inspection; not called in the hot path.

### `exists()` — check presence

```python
def exists(artifact_id: str) -> bool:
```

Guards against double-resolution of handles that don't exist yet.

### `Artifact` schema (from `schemas.py`)

```python
class Artifact(BaseModel):
    id:           str    # "art:<16-hex>"
    content_type: str    # "text/plain"
    size_bytes:   int
    source:       str    # "mcp:<tool_name>"
    descriptor:   str    # short human-readable label written at put() time
```

---

## Data Flow Summary

```
Decision emits ToolCall(name="fetch_url", arguments={"url": "https://…"})
        │
        ▼
action.execute()
        │
        ├─ guard passes (no art: in path/url)
        ├─ session.call_tool("fetch_url", {"url": "https://…"})
        │       ← MCP roundtrip to mcp_server.py subprocess
        ├─ _result_to_text(result) → markdown string, e.g. 48 000 bytes
        │
        ├─ 48 000 > 4 096  → artifacts.put(bytes, source="mcp:fetch_url", …)
        │                    → art:9f3a1c7b2e4d8a06  stored to disk
        │
        └─ return ("[artifact art:9f3a1c7b2e4d8a06, 48000 bytes] preview: …", "art:9f3a1c7b2e4d8a06")

agent loop calls memory.record_outcome(tool_call, descriptor, art_id)
        └─ Memory item: kind=tool_outcome, artifact_id="art:9f3a1c7b2e4d8a06"

Next iteration: Perception sees hit with i=0, artifact_id="art:9f3a1c7b2e4d8a06"
        └─ sets goal.attach_artifact_id = "art:9f3a1c7b2e4d8a06"

Agent loop calls artifacts.get_bytes("art:9f3a1c7b2e4d8a06")
        └─ passes (art_id, bytes) to decision.next_step() as `attached`
```

The 48 KB of HTML touches **exactly one LLM call** — the Decision call where it is
attached. It is never re-fetched, and it is never in any other prompt.

---

## Relationship to Other Layers

| Layer | Sends to Action | Receives from Action |
|---|---|---|
| **Agent loop** | `ToolCall` from `DecisionOutput`; `ClientSession` | `(descriptor, artifact_id)` → passed to `memory.record_outcome()` |
| **MCP server** | — | Called via `session.call_tool()` over stdio |
| **Artifacts** | — | `put()` called for large results; `get_bytes()` called by loop for attachment |
| **Memory** | — | `record_outcome()` called by the loop with the descriptor and `artifact_id` |
| **Perception** | — | Sees `artifact_id` in Memory hits; sets `goal.attach_artifact_id` for Decision |
| **Decision** | — | Receives attached bytes via `_format_attached()` in `next_step()` |
