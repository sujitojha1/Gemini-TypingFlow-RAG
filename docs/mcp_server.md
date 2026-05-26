# MCP Server — mcp_server.py

## Overview

`mcp_server.py` is the **tool surface of the agent**. It runs as a child subprocess over
`stdio` transport and exposes eleven tools to the agent loop via the
[Model Context Protocol](https://modelcontextprotocol.io). The agent never imports these
functions directly — it calls them through MCP JSON-RPC, which means the server runs in its
own process with its own memory space.

```
agent loop (main process)
    │   MCP JSON-RPC over stdio
    ▼
mcp_server.py  (child subprocess)
    ├── web tools      → Tavily / DuckDuckGo / crawl4ai
    ├── utility tools  → get_time, currency_convert
    ├── file tools     → read_file, list_dir, create_file, update_file, edit_file
    └── RAG tools      → index_document, search_knowledge
                             │
                             ▼
                         memory.py / artifacts.py  (shared on-disk stores)
```

The server is started with:

```bash
python mcp_server.py          # stdio transport
```

---

## Tool Catalogue

### `web_search`

```python
def web_search(query: str, max_results: int = 5) -> list[dict]
```

Search the web using **Tavily** as the primary provider and **DuckDuckGo** as fallback.

| Field | What it means |
|---|---|
| `title` | Page title |
| `url` | Canonical URL |
| `snippet` | Tavily `content` field or DDG `body` |

**Provider selection logic:**

```
TAVILY_API_KEY set AND monthly count < 950?
    ├─ yes → _tavily_search()   (search_depth="advanced")
    │         success? → _bump("tavily"); return
    │         exception? → _bump("tavily", "errors"); fall through
    └─ no  → _ddg_search()      (tries backends: auto → html → lite)
```

- `max_results` is silently clamped to `[1, 5]` — **hard cap** at 5 (Tavily charges per
  result).
- Usage is tracked in `usage.json` with monthly rollover. The soft cap is **950/1000** —
  50 results kept as headroom.

---

### `fetch_url`

```python
async def fetch_url(url: str, timeout: int = 20) -> dict
```

Fetch a URL and return clean **Markdown** via `crawl4ai` (headless Chromium).

| Field | Notes |
|---|---|
| `status` | HTTP status code (defaults to 200 if crawl4ai doesn't surface it) |
| `content_type` | Always `"text/markdown"` |
| `length_bytes` | UTF-8 byte count of the returned text |
| `text` | The Markdown content |

**stdout protection:** crawl4ai uses Rich internally which writes directly to `fd 1`.
Since this server uses `stdio` transport, any stray output would corrupt the JSON-RPC stream.
The implementation redirects `fd 1 → fd 2` at the OS level around the crawl call, then
restores it:

```python
saved_fd = os.dup(1)
os.dup2(2, 1)          # stdout → stderr for the duration
try:
    async with AsyncWebCrawler(verbose=False) as crawler:
        r = await crawler.arun(url=url)
finally:
    os.dup2(saved_fd, 1)
    os.close(saved_fd)
```

**Markdown extraction priority:** `raw_markdown` → `fit_markdown` → `md` → `cleaned_html`
→ `html` → `""`. The result is always cast to `str` before return because crawl4ai's
`StringCompatibleMarkdown` is a subclass that Pydantic serialises as `{}`.

---

### `get_time`

```python
def get_time(timezone: str = "UTC") -> dict
```

Return the current wall-clock time in any valid **IANA timezone**.

| Field | Example |
|---|---|
| `iso` | `"2025-07-14T18:30:00+05:30"` |
| `human` | `"Monday, 14 July 2025 18:30:00 IST"` |
| `timezone` | `"Asia/Kolkata"` |
| `offset_hours` | `5.5` |

Raises `ZoneInfoNotFoundError` (from stdlib `zoneinfo`) if the timezone string is invalid.

---

### `currency_convert`

```python
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict
```

Convert money between any two **ISO-4217 currencies** using the free
[frankfurter.dev](https://api.frankfurter.dev) API (no API key required).

| Field | Notes |
|---|---|
| `amount` | Input amount |
| `from` / `to` | Uppercased ISO-3 codes |
| `rate` | `converted / amount` (exact exchange rate for 1 unit) |
| `converted` | Result in target currency |
| `date` | Date of the exchange rate |
| `source` | `"frankfurter.dev"` |

---

### `read_file`

```python
def read_file(path: str) -> dict
```

Read a UTF-8 text file from the **sandbox** (`./sandbox/`).

| Field | Notes |
|---|---|
| `path` | The path as passed in |
| `size_bytes` | File size on disk |
| `content` | Full file text |
| `encoding` | `"utf-8"` |

All file tools use `_safe(path)` to **resolve and validate** the path.
Any path that escapes the sandbox (e.g. `../../etc/passwd`) raises `ValueError`.

---

### `list_dir`

```python
def list_dir(path: str = ".") -> dict
```

List a directory inside the sandbox.

| Field | Notes |
|---|---|
| `path` | Directory path relative to sandbox |
| `count` | Total number of entries |
| `names` | Flat `list[str]` of names — survives truncation |
| `entries` | Full list of `{name, type, size_bytes}` dicts |

> **Why both `names` and `entries`?**  
> Early runs showed that when the MCP response was clipped by the agent's 300-char preview
> or `decision.py`'s slicing, only the first 2–3 `entries` dicts survived. The agent then
> incorrectly concluded the directory was fully listed. The flat `names` list keeps the
> full file count visible even under truncation.

---

### `create_file`

```python
def create_file(path: str, content: str) -> dict
```

Create a **new** file in the sandbox. Raises `ValueError` if the file already exists or if
the parent directory doesn't exist (no implicit `mkdir`).

Returns `{ok: true, path, size_bytes}`.

---

### `update_file`

```python
def update_file(path: str, content: str) -> dict
```

Overwrite an **existing** file in the sandbox. Raises `ValueError` if the file does not
exist. Use `create_file` for new files.

Returns `{ok: true, path, size_bytes}`.

---

### `edit_file`

```python
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict
```

Find-and-replace inside a sandbox file. The tool is intentionally **strict**:

- Raises if `find` is not found at all.
- Raises if `find` occurs more than once and `replace_all=False` — forces the caller to be
  explicit rather than silently replacing the wrong occurrence.

| Field | Notes |
|---|---|
| `replacements` | 1, or the full count when `replace_all=True` |
| `size_bytes` | Updated file size |

---

### `index_document`

```python
def index_document(path: str, chunk_size: int = 400, overlap: int = 80) -> dict
```

Chunk a **sandbox file** or **artifact** (`art:` prefix) and write every chunk into
[Memory](memory.md) as a `fact` record, making the content vector-searchable across all
future turns.

**When to use vs `read_file`:**

| Use case | Tool |
|---|---|
| One-shot inspection of a file this turn | `read_file` |
| Content must survive beyond this turn / be searchable later | `index_document` |

**Chunking algorithm** (`_chunk_text`):

```
words = text.split()
stride = chunk_size - overlap   # default: 400 - 80 = 320 words
window slides over [0, stride, 2*stride, …]
last window clipped at end of words
```

This is **sliding-window by word count** (Session 7). Semantic / sentence-aware chunking
is planned for Session 8.

**What gets written to Memory per chunk:**

```python
descriptor = f"[{source} chunk {i+1}/{total}] {chunk[:120]}"
value = {
    "chunk":        chunk,          # full text of this window
    "chunk_index":  i,
    "total_chunks": total,
    "source":       source,
}
```

All chunks from a single `index_document` call share the same `run_id`
(`"index-YYYYMMDDHHMMSS"`), making them easy to group.

Returns `{path, source, chunks_indexed, chunk_size, overlap}`.

---

### `search_knowledge`

```python
def search_knowledge(query: str, k: int = 5) -> list[dict]
```

Vector search over all indexed `fact` chunks in Memory. Delegates directly to
[`memory.read(query, kinds=["fact"], top_k=k)`](memory.md#read-path----read).

| Field per result | Notes |
|---|---|
| `id` | Memory item ID (`"mem:…"`) |
| `descriptor` | The one-line descriptor that was embedded |
| `source` | Origin — `"sandbox:path"` or `"art:id"` |
| `chunk_preview` | First 240 chars of the chunk text |
| `metadata` | `{chunk_index, total_chunks, source}` — everything in `value` except `"chunk"` |

> **Guidance:** Always prefer `search_knowledge` over re-fetching URLs or re-reading files
> when Memory already contains indexed chunks for the topic — that is the purpose of having
> run `index_document` earlier.

---

## Sandbox Security — `_safe()`

```python
def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p
```

Resolves symlinks and `..` segments **before** checking containment, so traversal attacks
like `"../../etc/passwd"` or symlink chains that point outside `./sandbox/` are caught and
rejected. All five file tools (`read_file`, `list_dir`, `create_file`, `update_file`,
`edit_file`) pass every path through `_safe()` before any I/O.

---

## Usage Tracking — `usage.json`

Tavily charges per search result. The server tracks monthly usage in `./usage.json`:

```json
{
  "month": "2025-07",
  "tavily":     { "count": 42, "errors": 1 },
  "duckduckgo": { "count": 7,  "errors": 0 }
}
```

- Resets automatically when the month rolls over.
- `MONTHLY_CAP = 950` — `_under_cap()` blocks Tavily calls once this threshold is reached,
  forcing DuckDuckGo fallback for the rest of the month.
- A `threading.Lock` (`_usage_lock`) serialises reads and writes — safe if multiple
  async tool calls are in flight simultaneously.

---

## Adding a New Tool

Adding a tool to this server is three steps:

### 1. Write the function and decorate it

```python
@mcp.tool()
def my_new_tool(param1: str, param2: int = 10) -> dict:
    """One-line docstring used as the tool description in the MCP schema.
    Include an Example: comment so the model knows how to call it."""
    # implementation
    return {"result": ...}
```

- The decorator is `@mcp.tool()` (from `mcp.server.fastmcp`).
- Type annotations on parameters become the JSON schema FastMCP advertises to clients.
- `async def` is supported; use it when calling async libraries (crawl4ai, httpx async).
- The return value is serialised to JSON automatically. Return a `dict` or `list[dict]`
  for structured results; plain `str` also works for simple responses.

### 2. Update the module docstring

The file's top docstring is the canonical tool roster. Add your tool name and a short
description to the list:

```python
"""
MCP server for EAGV3 Session 7.

Eleven tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file,
    index_document, search_knowledge,
    my_new_tool          # ← add here
...
"""
```

### 3. Restart the server

The server is launched as a subprocess by the agent — stop the agent and restart it, or
kill the server process directly. There is no hot-reload; `mcp_server.py` must be
re-executed for changes to take effect.

---

### Tool shape checklist

| Concern | Guidance |
|---|---|
| **Docstring** | First sentence = tool description shown to the model. Add `Example:` so the model can generate correct call syntax. |
| **Return type** | Prefer `dict` or `list[dict]`. Avoid bare `str` for structured data — the model reasons better over labelled fields. |
| **Errors** | Raise `ValueError` for caller errors (bad input, file not found). Unhandled exceptions propagate as MCP error responses. |
| **Sandbox files** | Always call `_safe(path)` before any file I/O. Never accept a raw `Path` from the model. |
| **External HTTP** | Use `httpx.Client` (sync) or `httpx.AsyncClient` (async) with an explicit `timeout`. |
| **stdio safety** | Any library that writes to stdout will corrupt the JSON-RPC stream. Redirect `fd 1 → fd 2` around the call (see `fetch_url` pattern). |
| **Usage tracking** | If the new tool calls a rate-limited API, add a provider entry to `_empty_usage()` and call `_bump()` in the tool. |

---

## Relationship to Memory & Artifacts

Two tools (`index_document`, `search_knowledge`) reach directly into the same on-disk stores
used by the main agent process. This is intentional:

```
mcp_server.py subprocess
    index_document → memory.add_fact()  → memory.json + index.faiss (on disk)
    search_knowledge → memory.read()    ← reads same files

main agent process
    memory.read() ← reads same memory.json + index.faiss (fresh from disk each call)
```

`memory.py` is imported into the server via a `sys.path.insert` so the code is shared, not
duplicated. Cross-process consistency is maintained because `memory._index()` re-reads
`index.faiss` from disk on every call rather than caching it in process memory — see
[memory.md § Cross-Process Consistency](memory.md#cross-process-consistency).
