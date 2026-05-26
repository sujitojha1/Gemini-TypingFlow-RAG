# LLM Gateway V7 — llm_gatewayV7/

## Overview

The gateway is a **FastAPI server** that sits between the agent and every LLM / embedding
provider. It abstracts away provider differences, enforces rate limits, and runs two
separate failover rings — one for **chat** (worker pool) and one for **routing** (router
pool). A third, lighter ring handles **embeddings**.

```
agent (gateway.py client)
    │
    ├── POST /v1/chat   → Router + RouterPool → one of: ollama, gemini, nvidia, groq,
    │                                           cerebras, openrouter, github
    ├── POST /v1/embed  → embed_with_failover → OllamaEmbedder → GeminiEmbedder (fallback)
    └── GET  /v1/*      → status, providers, capabilities, routers, calls, embedders
```

Run the gateway:

```bash
cd llm_gatewayV7
uvicorn main:app --host 0.0.0.0 --port 8107
# or: python main.py
```

Port is read from `GATEWAY_V7_PORT` (default `8107`).

---

## Directory Layout

```
llm_gatewayV7/
    main.py        ← FastAPI app, all HTTP endpoints, tier classifier, lifespan
    router.py      ← RateState, Router (worker pool), RouterPool (routing pool), LIMITS
    embedders.py   ← OllamaEmbedder, GeminiEmbedder, embed_with_failover, EmbedRateState
    providers.py   ← per-provider chat adapters (Ollama, Gemini, Nvidia, Groq, …)
    schemas.py     ← Pydantic request/response models
    cache.py       ← GeminiCache (5-min TTL, used for system-prompt caching)
    db.py          ← SQLite call log (gateway_v7.db), db.log_call(), db.aggregate()
    requirements.txt
    static/        ← dashboard.html, help.html
    tests/         ← test_embed.py, test_all_providers.py
```

---

## Two Separate Provider Pools

### Worker Pool — `Router`

Handles all `POST /v1/chat` requests. Seven providers, configured in `DEFAULT_ORDER`:

```python
DEFAULT_ORDER = ["ollama", "gemini", "nvidia", "groq", "cerebras", "openrouter", "github"]
```

Overridden at startup via `LLM_ORDER` env var.

### Router Pool — `RouterPool`

A **separate failover ring** for routing-decision LLM calls only. The routing classifier
needs a fast, cheap model that can emit one word (`TINY`, `LARGE`, or `HUGE`) — it must
never compete with worker quotas. Default order:

```python
DEFAULT_ROUTER_ORDER = ["cerebras", "groq", "nvidia", "github"]
```

Overridden via `ROUTER_ORDER` env var. The same provider keys (`cerebras`, `groq`, …) are
used, but rate state is tracked in a **separate `defaultdict(RateState)`** so a router call
doesn't consume a worker RPM slot.

---

## Rate State — `RateState`

Both pools use the same `RateState` class per provider. It tracks:

| Metric | Window |
|---|---|
| `calls_minute` | Sliding 60s deque of timestamps |
| `tokens_minute` | Sliding 60s deque of `(timestamp, tokens)` pairs |
| `calls_today` | Rolling daily counter (resets at UTC midnight) |
| `tokens_today` | Rolling daily token counter |
| `unavailable_until` | Backoff expiry timestamp (set by `mark_unavailable()`) |

`can_use(limits, est_tokens)` checks all five constraints in order and returns
`(bool, reason_string)`. `record(tokens)` is called on success; `mark_unavailable(secs, reason)` on failure.

### `LIMITS` per provider

```python
LIMITS = {
    "ollama":     {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0,   "max_ctx": 32000},
    "cerebras":   {"rpm": 30,   "rpd": 9999,    "tpm": 60000,    "cooldown": 2,   "max_ctx": 8000,  "tokens_per_day": 1_000_000},
    "groq":       {"rpm": 30,   "rpd": 1000,    "tpm": 6000,     "cooldown": 2,   "max_ctx": 100000},
    "nvidia":     {"rpm": 40,   "rpd": 9999,    "tpm": 100000,   "cooldown": 2,   "max_ctx": 100000},
    "gemini":     {"rpm": 15,   "rpd": 1000,    "tpm": 250000,   "cooldown": 4,   "max_ctx": 1000000},
    "openrouter": {"rpm": 20,   "rpd": 50,      "tpm": 99999999, "cooldown": 3,   "max_ctx": 100000},
    "github":     {"rpm": 10,   "rpd": 50,      "tpm": 99999999, "cooldown": 6,   "max_ctx": 8000},
}
```

`LIMITS` is the single source of truth for both worker and router rate-checking.

### Provider shortcuts

```python
SHORTCUTS = {
    "g": "gemini",  "gem": "gemini",
    "n": "nvidia",  "nv": "nvidia",
    "o": "ollama",  "oll": "ollama",
    "gr": "groq",   "c": "cerebras",
    "or": "openrouter",  "gh": "github",
}
```

The `resolve(name)` function normalises a shortcut to its canonical key.

---

## `POST /v1/chat` — Tier-Routing Flow

When `auto_route` is set (values: `"perception"` | `"memory"` | `"decision"`), the gateway
runs a two-stage dispatch:

```
1. _classify_tier(req, role, router_pool, prompt_text)
        │
        ├─ est_tokens > 8000? → HUGE immediately, skip router LLM call
        │
        ├─ try each RouterPool candidate in order
        │     POST router LLM: "token_count: N\nsample:\n<800-char envelope>"
        │     LLM emits one word: TINY | LARGE | HUGE
        │     sanity clamp: HUGE blocked if est_tokens ≤ 8000 (hallucination guard)
        │
        └─ all routers fail? → deterministic fallback: _tier_from_count(tokens)
                HUGE if > 8000, LARGE if ≥ 1000, TINY otherwise

2. tier → TIER_TO_ORDER lookup
        TINY:  ["github", "openrouter", "groq", "nvidia", "cerebras", "gemini", "ollama"]
        LARGE: ["gemini", "groq", "nvidia", "cerebras", "github", "openrouter", "ollama"]
        HUGE:  → HTTP 503 (Summarizer Agent not yet implemented)

3. Worker failover loop over tier candidates
        Router.pick(est_tokens, candidates, required_caps)
        → first provider that passes RateState.can_use() AND has required capabilities
        → provider.chat(…)
        → on ProviderError: _backoff_for(err) → mark_unavailable, try next
```

When `provider` is explicitly set, routing is bypassed — `Router.candidates(override)`
returns that provider directly.

---

## `POST /v1/embed` — Embedding Endpoint

```python
@app.post("/v1/embed")
async def embed(req: EmbedRequest):
```

### Request schema (`EmbedRequest`)

```python
class EmbedRequest(BaseModel):
    text:      str
    task_type: Literal["retrieval_document", "retrieval_query"] = "retrieval_document"
    provider:  Optional[str] = None   # "ollama" | configured fallback name
```

### Response schema (`EmbedResponse`)

```python
class EmbedResponse(BaseModel):
    provider:   str
    model:      str
    embedding:  list[float]      # 768-dim vector
    dim:        int              # always 768 for current providers
    latency_ms: int
    attempted:  list[dict]       # [{provider, reason}, …] for all tried providers
```

### Endpoint logic

```
1. Guard: no embedders configured → 503

2. Guard: len(req.text) > MAX_INPUT_CHARS (8000) → 413
   "Chunk the input and embed each chunk."

3. embed_with_failover(embedders, text, task_type, explicit=req.provider)
        │
        ├─ explicit set? → filter to that provider only
        │     failure → 502 (pinned, no fallback)
        │     rate-limited → 429
        │
        └─ no explicit → try Ollama first, then configured fallback
              success → db.log_call(call_role="embed") → return EmbedResponse
              all fail → 503
```

### Key constraint — fixed 768-dim

Both providers pin `EMBED_DIM = 768`. This keeps the FAISS index valid across a failover.
Changing either provider, or their model, invalidates all existing FAISS indexes — a
one-way trip.

---

## Embedder Architecture — `embedders.py`

### `EmbeddingProvider` (abstract base)

```python
class EmbeddingProvider:
    name:  str
    model: str
    state: EmbedRateState

    async def embed(self, text: str, task_type: TaskType) -> dict:
        # returns: {"embedding": list[float], "model": str, "dim": int}
        raise NotImplementedError
```

### `OllamaEmbedder` (primary, local)

- Hits `POST /api/embeddings` on `OLLAMA_URL` (default `http://localhost:11434`)
- Model: `EMBED_OLLAMA_MODEL` (default `nomic-embed-text`)
- Prepends nomic's required task prefix before embedding:
  - `retrieval_query` → `"search_query: " + text`
  - `retrieval_document` → `"search_document: " + text`
- No API key. `EmbedRateState(rpm=0, cooldown=0)` — no local rate-limiting.

### `GeminiEmbedder` (fallback, cloud)

- Hits `generativelanguage.googleapis.com/v1beta/models/{model}:embedContent`
- Model: `EMBED_FALLBACK_MODEL` (default `gemini-embedding-001`)
- `outputDimensionality=768` forces matching dimension with Ollama
- Task type is passed natively as `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`
- `EmbedRateState(rpm=5, cooldown=5.0)` — conservative; free tier is tight

### `EmbedRateState`

Lighter than worker `RateState` — no daily token budget, no TPM window:

| Check | Detail |
|---|---|
| Backoff | Exponential: `[5s, 10s, 15s]`. Steps forward on `mark_failure()`; resets to 0 on `record()` |
| Cooldown | Min seconds between calls (0 for Ollama, 5s for Gemini) |
| RPM | Sliding 60s window (disabled for Ollama with `rpm=0`) |

### `embed_with_failover()`

```python
async def embed_with_failover(
    embedders: list[EmbeddingProvider],
    text: str,
    task_type: TaskType,
    explicit: str | None = None,
) -> tuple[str, dict, list[dict], int]:   # (name, result, attempts, latency_ms)
```

For each candidate in order:
1. `state.can_use()` — skip if rate-limited / in backoff
2. `await e.embed(text, task_type)` — call the provider
3. Success → `state.record()` (reset backoff), return
4. Failure → `state.mark_failure(reason)` (bump backoff), try next

`explicit` set → only that provider is tried; failures raise directly (no silent fallback).

### `build_embedders()`

Called once at startup. Reads env vars and returns `(list[EmbeddingProvider], list[str])`:

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `EMBED_OLLAMA_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBED_FALLBACK_PROVIDER` | `gemini` | Name of the fallback |
| `EMBED_FALLBACK_MODEL` | `gemini-embedding-001` | Fallback model |
| `GEMINI_API_KEY` | _(required for Gemini)_ | Drops Gemini if unset |
| `EMBED_ORDER` | `ollama,gemini` | Failover order |

---

## Adding a New `/v1/embed` Provider

Adding a new embedding backend is four steps.

### Step 1 — Implement `EmbeddingProvider` in `embedders.py`

```python
class MyEmbedder(EmbeddingProvider):
    name = "my_provider"          # must be unique; used in EMBED_ORDER and provider= param
    model: str

    def __init__(self, api_key: str, model: str = "my-model",
                 rpm: int = 10, cooldown: float = 1.0):
        self.api_key = api_key
        self.model = model
        self.state = EmbedRateState(rpm=rpm, cooldown=cooldown)

    async def embed(self, text: str, task_type: TaskType) -> dict:
        # 1. Map task_type to provider's enum if needed
        # 2. POST to provider API
        # 3. Extract vector from response
        # 4. Raise EmbedderError on HTTP errors — always set `status=` so the
        #    endpoint can return 429 vs 502 vs 400 correctly
        ...
        return {"embedding": vec, "model": self.model, "dim": len(vec)}
```

**Rules for the return dict:**
- `"dim"` must equal `EMBED_DIM` (768) to stay compatible with the existing FAISS index.
- `"embedding"` must be a `list[float]`.
- Raise `EmbedderError(msg, status=<http_code>)` on any failure — never let raw `httpx`
  exceptions escape; the endpoint only catches `EmbedderError`.

### Step 2 — Register in `build_embedders()`

```python
def build_embedders():
    ...
    registry: dict[str, EmbeddingProvider] = {
        "ollama": OllamaEmbedder(ollama_model, ollama_url),
    }
    if fallback_provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if key:
            registry["gemini"] = GeminiEmbedder(key, fallback_model)

    # ← add here:
    my_key = os.getenv("MY_PROVIDER_API_KEY")
    if my_key:
        registry["my_provider"] = MyEmbedder(my_key)
    ...
```

The provider is silently omitted if its key is missing — consistent with how Gemini is
handled. No startup crash.

### Step 3 — Set env vars

```bash
EMBED_FALLBACK_PROVIDER=my_provider
EMBED_FALLBACK_MODEL=my-model
MY_PROVIDER_API_KEY=sk-…
EMBED_ORDER=ollama,my_provider   # optional; default is ollama + fallback
```

Or to add it as a tertiary option alongside the existing fallback:

```bash
EMBED_ORDER=ollama,gemini,my_provider
```

### Step 4 — Restart the gateway

```bash
uvicorn main:app --host 0.0.0.0 --port 8107
```

`build_embedders()` re-runs in `lifespan()`. No reload needed — the server recreates the
embedder list fresh on every start.

### Provider checklist

| Concern | Guidance |
|---|---|
| **Dimension** | Must return 768-dim vectors. Any other dimension breaks FAISS. |
| **`EmbedRateState`** | Set `rpm=0` for local/unlimited; set `cooldown` to the provider's minimum inter-call gap. |
| **`EmbedderError` status** | `400` → bad input; `429` → rate-limited; `≥500` → upstream error. The endpoint maps these to HTTP status codes. |
| **Task type** | Map `"retrieval_document"` / `"retrieval_query"` to the provider's vocabulary. Nomic uses string prefixes; Gemini uses `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` enum values. |
| **Timeout** | Wrap the `httpx` call with `timeout=60`. Long timeouts here block the request; short ones cause spurious retries. |
| **No silent truncation** | Raise `EmbedderError` if the text is too long for your provider — do not silently truncate. The 8000-char gate in the endpoint is already a hard cap; provider-side truncation would produce wrong vectors without error. |

---

## Startup — `lifespan()`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    app.state.cache          = GeminiCache(ttl_seconds=300)
    app.state.providers      = P.build_providers(app.state.cache)
    app.state.router         = Router(app.state.providers, ORDER)
    app.state.router_providers = P.build_router_providers()
    app.state.router_pool    = RouterPool(app.state.router_providers, ROUTER_ORDER)
    app.state.embedders, app.state.embed_order = E.build_embedders()
    yield
```

All pools are built once and stored on `app.state`. The two provider dicts
(`app.state.providers` and `app.state.router_providers`) are separate so router LLMs are
never accidentally used as workers.

---

## Observation & Status Endpoints

| Endpoint | What it shows |
|---|---|
| `GET /v1/status` | Worker pool live rate state + today's call aggregate |
| `GET /v1/routers` | Router pool state + tier-to-order table |
| `GET /v1/embedders` | Embedder order, models, `EMBED_DIM`, `MAX_INPUT_CHARS`, live rate state |
| `GET /v1/providers` | Provider list, shortcuts, LIMITS, models |
| `GET /v1/capabilities` | Per-provider capability flags (tools, reasoning, structured, caching) |
| `GET /v1/calls` | Recent call log from SQLite (limit, provider, status filters) |
| `GET /` | Dashboard HTML |

---

## Call Logging — `db.log_call()`

Every chat and embed call is written to `gateway_v7.db` (SQLite). Key fields:

| Field | Set for |
|---|---|
| `call_role` | `"worker"` / `"router_<role>"` / `"embed"` |
| `router_decision` | Tier string when auto_route was used |
| `embed_dim` | Set for embed calls |
| `attempted` | Semicolon-joined `provider:reason` string of skipped candidates |

`db.aggregate(call_role=...)` returns today's totals — used by `/v1/status`,
`/v1/routers`, and `/v1/embedders`.
