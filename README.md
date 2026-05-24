# Gemini-typingflow-RAG

## Code Organisation

```
Gemini-TypingFlow-RAG/
│
├── agent7.py           Orchestrator — the main loop (Perception → Decision → Action → Memory)
├── perception.py       Decomposes user query into goals; updates goal status each iteration
├── decision.py         One LLM call per turn; picks a tool call or emits a final answer
├── action.py           MCP dispatcher — executes the tool, pushes large results to artifacts
├── memory.py           Read/write service; vector search (FAISS) with keyword fallback
├── vector_index.py     FAISS IndexFlatIP wrapper; L2-normalised cosine similarity over 768-dim vectors
├── artifacts.py        Content-addressable byte store (sha256-keyed); Decision sees bytes, Memory holds handles
├── gateway.py          Bridge to llm_gatewayV7; auto-starts it if not running, exports LLM + embed()
├── mcp_server.py       MCP stdio server — 11 tools: web_search, fetch_url, file CRUD, index_document, search_knowledge
├── schemas.py          Shared Pydantic contracts: MemoryItem, Goal, Observation, ToolCall, DecisionOutput
│
└── llm_gatewayV7/      Self-contained FastAPI gateway (port 8107)
    ├── main.py         Routes: POST /v1/chat, /v1/embed; lifespan wires providers + embedders
    ├── embedders.py    OllamaEmbedder + GeminiEmbedder; failover ring with per-provider rate state
    ├── providers.py    Provider adapters (Ollama, Gemini, Groq, Nvidia, Cerebras, OpenRouter, GitHub)
    ├── router.py       LLM-based tier classifier (TINY / LARGE / HUGE) with token-count fallback
    ├── client.py       Python client — LLM().chat() and LLM().embed(); used by gateway.py
    ├── cache.py        Gemini prompt-cache layer (SHA-256-keyed system content reuse)
    ├── db.py           SQLite call log (provider, model, latency, status per request)
    └── schemas.py      Gateway-internal Pydantic models (ChatRequest/Response, EmbedRequest/Response)
```

### Data Flow

```
user query
    │
    ▼
Perception   reads Memory → emits/updates goal list
    │
    ▼
Decision     one LLM call → tool_call | answer
    │
    ▼
Action       dispatches tool via MCP server → result bytes → Artifact store
    │
    ▼
Memory       embeds descriptor → FAISS index  (768-dim, Ollama → Gemini failover)
    │
    └──────────────────────────────────────────────────────► next iteration
```

## Docs

Detailed read-throughs live in [`docs/`](docs/). Add one Markdown file per topic as you go.

| File | Contents |
|---|---|
| [class_notes.md](docs/class_notes.md) | Core RAG concepts, Session 7 architecture upgrades, and best practices |
| [requirement.md](requirement.md) | Project requirements |

## Running

```bash
# 1. Start the LLM gateway backend (port 8107)
cd llm_gatewayV7 && uv run main.py

# 2. Run the agent
uv run agent7.py
```