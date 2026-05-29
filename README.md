# Gemini-TypingFlow-RAG

A **Retrieval-Augmented Generation (RAG) system** built on the Session 7 cognitive
architecture. A Chrome extension automatically indexes every page you visit; a popup lets
you query the corpus with natural language; the agent answers from your indexed knowledge,
citing sources, with a three-tier confidence gate.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Setup Instructions](#2-setup-instructions)
3. [Corpus Manifest](#3-corpus-manifest)
4. [Base Query Traces (A–H)](#4-base-query-traces-ah)
5. [Custom Query Traces](#5-custom-query-traces)
6. [Compliance Checklist](#6-compliance-checklist)

---

## 1. Architecture Overview

### System Diagram

```
Browser
  │
  ├─ content.js (auto-injected)
  │     Extract text → chunk (400 words / 80 overlap)
  │     └──► background.js
  │               POST /v1/embed (task: retrieval_document)
  │               │         ▲
  │               │    llm_gatewayV7 :8107
  │               │    OllamaEmbedder (nomic-embed-text, 768-dim)
  │               │         └─► GeminiEmbedder (fallback)
  │               POST /store-chunk ──► rag_server.py :8108
  │                                         └─► FAISS IndexFlatIP
  │                                         └─► memory.json (corpus)
  │
  └─ popup.js (user query)
        POST /search (query text)
        └──► rag_server.py :8108
                  POST /v1/embed (task: retrieval_query)
                  └──► llm_gatewayV7 :8107
                  FAISS top-k search (cosine similarity)
                  Confidence gate → POST /v1/chat (LLM answer)
                  └──────────────────────────► popup (answer + sources)
```

### Agent Pipeline (agent7.py)

The autonomous agent runs the same four-layer loop:

```
user query
    │
    ▼
memory.read()     vector search (FAISS) → keyword fallback → MemoryItem list
    │
    ▼
perception.observe()   LLM call → goal list (intent only, never tool names)
    │
    ▼
decision.next_step()   LLM call → ToolCall | final answer
    │
    ▼
action.execute()       MCP stdio dispatch → mcp_server.py
    │                  result > 4 KB → artifact store
    ▼
memory.record_outcome() embed descriptor → FAISS + memory.json
    │
    └──────────────────────────────────────── next iteration (max 20)
```

### Four Session 7 Changes

| # | Change | Where |
|---|---|---|
| 1 | **`index_document` MCP tool** — chunks a file, embeds each piece (768-dim), saves as `fact` records in FAISS | `mcp_server.py` |
| 2 | **`search_knowledge` MCP tool** — vector search over the FAISS corpus; returns ranked chunks with cosine scores | `mcp_server.py` |
| 3 | **MemoryItem `embedding` field** — memory reads use vector similarity first; fall back to keyword overlap when index is empty or gateway is unreachable | `memory.py`, `vector_index.py` |
| 4 | **Cross-process FAISS persistence** — `index.faiss` + `index_ids.json` written synchronously to `state/`; reloaded on every read so agent and MCP subprocess share one index without locks | `memory.py`, `vector_index.py` |

### Perception Tool-Blindness Principle

**Perception never names a tool.** Its system prompt uses intent verbs only:
`fetch`, `make this content searchable`, `query the existing knowledge base`.
Decision maps those phrases to `index_document` / `search_knowledge` at call time.

Enforcement proof:

```bash
grep -r 'index_document\|search_knowledge\|web_search\|fetch_url' perception.py
# → zero matches
```

Tool-selection guidance lives exclusively in each tool's docstring (AC-2, NFR-8).
Nothing from Perception leaks into any system-prompt string.

---

## 2. Setup Instructions

### Prerequisites

| Component | Version |
|---|---|
| Python | ≥ 3.11 |
| [uv](https://github.com/astral-sh/uv) | any recent |
| [Ollama](https://ollama.com) | any recent |
| Chrome | ≥ 114 (Manifest V3) |
| `GEMINI_API_KEY` | Gemini API key (embedding fallback + chat) |

### Step 1 — Clone and install dependencies

```bash
git clone https://github.com/sujitojha1/Gemini-TypingFlow-RAG.git
cd Gemini-TypingFlow-RAG
uv sync
```

### Step 2 — Install Ollama and pull the embedding model

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull nomic-embed-text (768-dim, required for FAISS compatibility)
ollama pull nomic-embed-text

# Verify it responds
ollama run nomic-embed-text "test"
```

### Step 3 — Set environment variables

```bash
export GEMINI_API_KEY="your-key-here"

# Optional overrides (defaults shown)
export OLLAMA_URL="http://localhost:11434"
export EMBED_OLLAMA_MODEL="nomic-embed-text"
export GATEWAY_V7_PORT="8107"
```

### Step 4 — Start the LLM gateway (port 8107)

```bash
cd llm_gatewayV7
uv run main.py
# Gateway is ready when you see: "Uvicorn running on http://0.0.0.0:8107"
```

**Verify:**

```bash
curl http://127.0.0.1:8107/v1/embedders
# Should show nomic-embed-text and gemini-embedding-001 in the response
```

### Step 5 — Start rag_server.py (port 8108)

```bash
# From the project root (new terminal)
uv run rag_server.py
# Ready when you see: "Uvicorn running on http://127.0.0.1:8108"
```

**Verify:**

```bash
curl http://127.0.0.1:8108/health
# {"ok": true, "port": 8108, "mcp_tools": ["index_document", "search_knowledge", ...]}

curl http://127.0.0.1:8108/status
# {"page_count": N, "chunk_count": N, "last_indexed_url": "...", "index_size_bytes": N}
```

### Step 6 — Load the Chrome extension

1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer mode** (toggle top-right)
3. Click **Load unpacked**
4. Select the `extension/` folder in this repository
5. The **RAG Search** icon appears in the toolbar

**Verify extension is working:**

- Navigate to any Wikipedia article
- Open Chrome DevTools → Application → Service Workers → check `background.js` is active
- Check the DevTools console for: `[RAG bg] done — N stored, 0 failed, XXXms`

### Step 7 — Run the agent (optional, for base queries)

```bash
# From project root
uv run agent7.py "What is backpropagation and how does it relate to gradient descent?"
```

### Component Verification Checklist

| Component | Command | Expected |
|---|---|---|
| Ollama embed | `curl -s http://localhost:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}' \| jq '.embedding\|length'` | `768` |
| Gateway embed | `curl -s -X POST http://127.0.0.1:8107/v1/embed -H 'Content-Type: application/json' -d '{"text":"test"}' \| jq '.dim'` | `768` |
| rag_server health | `curl http://127.0.0.1:8108/health` | `{"ok": true, ...}` |
| FAISS index dim | `python3 -c "import faiss; print(faiss.read_index('state/index.faiss').d)"` | `768` |
| Compliance grep | `grep -r 'index_document\|search_knowledge' perception.py` | _(no output)_ |

---

## 3. Corpus Manifest

Corpus built with `scripts/build_corpus.py`.

| Metric | Value |
|---|---|
| **page_count** | 55 |
| **chunk_count** | 901 |
| **index_size_bytes** | 2,767,917 (~2.6 MB) |
| **Domains** | 5 (wikipedia, arxiv, python-docs, realpython, peps) |
| **Failed / skipped** | 0 |

### Indexed Pages

| # | URL | Title | Domain | Topics |
|---|-----|-------|--------|--------|
| 1 | [link](https://en.wikipedia.org/wiki/Backpropagation) | Backpropagation | wikipedia | neural-network-training, gradient |
| 2 | [link](https://en.wikipedia.org/wiki/Gradient_descent) | Gradient Descent | wikipedia | neural-network-training, gradient |
| 3 | [link](https://en.wikipedia.org/wiki/Loss_function) | Loss Function | wikipedia | neural-network-training |
| 4 | [link](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) | Vanishing Gradient Problem | wikipedia | gradient, neural-network-training |
| 5 | [link](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) | Stochastic Gradient Descent | wikipedia | gradient, neural-network-training |
| 6 | [link](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) | Transformer Architecture | wikipedia | transformer, attention |
| 7 | [link](https://en.wikipedia.org/wiki/Attention_(machine_learning)) | Attention Mechanism | wikipedia | attention, transformer |
| 8 | [link](https://en.wikipedia.org/wiki/BERT_(language_model)) | BERT Language Model | wikipedia | transformer, attention |
| 9 | [link](https://en.wikipedia.org/wiki/Long_short-term_memory) | Long Short-Term Memory (LSTM) | wikipedia | neural-network-training, gradient |
| 10 | [link](https://en.wikipedia.org/wiki/Recurrent_neural_network) | Recurrent Neural Network | wikipedia | neural-network-training |
| 11 | [link](https://en.wikipedia.org/wiki/Convolutional_neural_network) | Convolutional Neural Network | wikipedia | neural-network-training |
| 12 | [link](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) | Dropout (Neural Networks) | wikipedia | neural-network-training |
| 13 | [link](https://en.wikipedia.org/wiki/Batch_normalization) | Batch Normalization | wikipedia | neural-network-training, gradient |
| 14 | [link](https://en.wikipedia.org/wiki/Transfer_learning) | Transfer Learning | wikipedia | neural-network-training |
| 15 | [link](https://en.wikipedia.org/wiki/Large_language_model) | Large Language Model | wikipedia | transformer, attention |
| 16 | [link](https://en.wikipedia.org/wiki/Softmax_function) | Softmax Function | wikipedia | neural-network-training, attention |
| 17 | [link](https://en.wikipedia.org/wiki/Cross-entropy) | Cross-Entropy | wikipedia | neural-network-training |
| 18 | [link](https://en.wikipedia.org/wiki/Word_embedding) | Word Embedding | wikipedia | transformer |
| 19 | [link](https://en.wikipedia.org/wiki/Word2vec) | Word2Vec | wikipedia | transformer |
| 20 | [link](https://en.wikipedia.org/wiki/Regularization_(mathematics)) | Regularization | wikipedia | neural-network-training, gradient |
| 21 | [link](https://en.wikipedia.org/wiki/Generative_adversarial_network) | Generative Adversarial Network | wikipedia | neural-network-training |
| 22 | [link](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) | RLHF | wikipedia | transformer, gradient |
| 23 | [link](https://docs.python.org/3/library/asyncio.html) | asyncio — Asynchronous I/O | python-docs | python-async |
| 24 | [link](https://docs.python.org/3/library/asyncio-task.html) | asyncio — Coroutines and Tasks | python-docs | python-async |
| 25 | [link](https://docs.python.org/3/library/asyncio-eventloop.html) | asyncio — Event Loop | python-docs | python-async |
| 26 | [link](https://docs.python.org/3/library/asyncio-sync.html) | asyncio — Synchronisation Primitives | python-docs | python-async |
| 27 | [link](https://docs.python.org/3/library/asyncio-queue.html) | asyncio — Queues | python-docs | python-async |
| 28 | [link](https://docs.python.org/3/library/asyncio-stream.html) | asyncio — Streams | python-docs | python-async |
| 29 | [link](https://docs.python.org/3/library/asyncio-exceptions.html) | asyncio — Exceptions | python-docs | python-async |
| 30 | [link](https://docs.python.org/3/library/asyncio-dev.html) | asyncio — Developing with asyncio | python-docs | python-async |
| 31 | [link](https://docs.python.org/3/library/asyncio-subprocess.html) | asyncio — Subprocesses | python-docs | python-async |
| 32 | [link](https://docs.python.org/3/library/asyncio-protocol.html) | asyncio — Transports and Protocols | python-docs | python-async |
| 33 | [link](https://peps.python.org/pep-0492/) | PEP 492 — Coroutines with async and await | peps | python-async |
| 34 | [link](https://peps.python.org/pep-0525/) | PEP 525 — Asynchronous Generators | peps | python-async |
| 35 | [link](https://peps.python.org/pep-0530/) | PEP 530 — Asynchronous Comprehensions | peps | python-async |
| 36 | [link](https://peps.python.org/pep-3156/) | PEP 3156 — Asynchronous IO Support | peps | python-async |
| 37 | [link](https://peps.python.org/pep-0567/) | PEP 567 — Context Variables | peps | python-async |
| 38 | [link](https://arxiv.org/abs/1706.03762) | Attention Is All You Need | arxiv | transformer, attention |
| 39 | [link](https://arxiv.org/abs/1810.04805) | BERT: Pre-training of Deep Bidirectional Transformers | arxiv | transformer, attention |
| 40 | [link](https://arxiv.org/abs/2205.01068) | Chain-of-Thought Prompting Elicits Reasoning | arxiv | transformer |
| 41 | [link](https://arxiv.org/abs/2210.11610) | ReAct: Synergizing Reasoning and Acting in LLMs | arxiv | transformer |
| 42 | [link](https://arxiv.org/abs/2106.09685) | LoRA: Low-Rank Adaptation of Large Language Models | arxiv | transformer, gradient |
| 43 | [link](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization (DPO) | arxiv | gradient, neural-network-training |
| 44 | [link](https://arxiv.org/abs/2302.13971) | LLaMA: Open and Efficient Foundation Language Models | arxiv | transformer |
| 45 | [link](https://arxiv.org/abs/2005.14165) | GPT-3: Language Models are Few-Shot Learners | arxiv | transformer, attention |
| 46 | [link](https://arxiv.org/abs/1512.03385) | Deep Residual Learning for Image Recognition (ResNet) | arxiv | neural-network-training, gradient |
| 47 | [link](https://arxiv.org/abs/1409.0473) | NMT by Jointly Learning to Align and Translate | arxiv | attention, transformer |
| 48 | [link](https://arxiv.org/abs/1412.6980) | Adam: A Method for Stochastic Optimization | arxiv | gradient, neural-network-training |
| 49 | [link](https://arxiv.org/abs/1607.06450) | Layer Normalization | arxiv | neural-network-training, transformer |
| 50 | [link](https://realpython.com/async-io-python/) | Async IO in Python: A Complete Walkthrough | realpython | python-async |
| 51 | [link](https://realpython.com/python-concurrency/) | Speed Up Your Python Program With Concurrency | realpython | python-async |
| 52 | [link](https://realpython.com/python-async-features/) | Getting Started With Async Features in Python | realpython | python-async |
| 53 | [link](https://realpython.com/python-gil/) | What Is the Python Global Interpreter Lock (GIL)? | realpython | python-async |
| 54 | [link](https://realpython.com/learning-paths/python-concurrency-parallel-programming/) | Python Concurrency and Parallel Programming | realpython | python-async |
| 55 | [link](https://realpython.com/python-sleep/) | Python sleep(): How to Add Time Delays | realpython | python-async |

### Domain Breakdown

| Domain | Pages | % of corpus |
|--------|-------|-------------|
| wikipedia | 22 | 40 % |
| arxiv | 12 | 22 % |
| python-docs | 10 | 18 % |
| realpython | 6 | 11 % |
| peps | 5 | 9 % |

### Topic Coverage (CRP-3)

| Topic | Pages |
|-------|-------|
| Neural network training | 19 |
| Python async patterns | 21 |
| Transformer architecture | 16 |
| Gradient-related topics | 12 |
| Attention mechanisms | 9 |

---

## 4. Base Query Traces (A–H)

Base queries exercise all 8 core capabilities of the Session 7 agent. Each trace is saved
as `traces/base/query_{A-H}.json` and contains the full iteration history.

Run a query and save its trace:

```bash
uv run agent7.py "your query here" --trace traces/base/query_A.json
```

| ID | Query text | Iterations | Pass? | Trace file | Demonstrates |
|----|-----------|-----------|-------|------------|--------------|
| A | "What is the current time in Asia/Tokyo and Asia/Kolkata? Tell me the difference in hours." | 3 | ✓ | `traces/base/query_A.json` | Multi-step factual lookup; two parallel goals; synthesis |
| B | "Fetch the Wikipedia article on backpropagation and give me a 3-sentence summary." | 3 | ✓ | `traces/base/query_B.json` | `fetch_url` → artifact store → Decision synthesis from bytes |
| C | "Search the web for the latest FAISS benchmarks and summarise the top result." | 4 | ✓ | `traces/base/query_C.json` | `web_search` → `fetch_url` → artifact attach → LLM answer |
| D | "Create a file called notes.txt with the text 'RAG session 7 done' and then read it back." | 3 | ✓ | `traces/base/query_D.json` | `write_file` + `read_file`; two sequential goals |
| E | "List the files in the sandbox directory." | 2 | ✓ | `traces/base/query_E.json` | `list_dir`; single-goal; Perception short-circuits on done |
| F | "Index the file sandbox/Attention_Is_All_You_Need_ext.txt and confirm how many chunks were created." | 2 | ✓ | `traces/base/query_F.json` | `index_document`; Session 7 new tool |
| G | "Search the knowledge base for information about the vanishing gradient problem." | 2 | ✓ | `traces/base/query_G.json` | `search_knowledge`; FAISS retrieval; cosine score in answer |
| H | "What does asyncio.gather do and how is it different from asyncio.wait?" | 3 | ✓ | `traces/base/query_H.json` | RAG answer from corpus (python-docs domain); high-confidence gate |

**Notes:**
- All 8 queries completed within the 20-iteration cap (FR-6.2).
- Queries F and G directly exercise the two new Session 7 MCP tools.
- Query H demonstrates the full RAG pipeline: vector retrieval → confidence gate → LLM answer with source citation.

---

## 5. Custom Query Traces

Five custom queries target the 55-page corpus. Each was run in two modes:
- **with-index**: full FAISS retrieval + LLM answer
- **without-index**: index hidden → graceful fallback

Traces: `traces/custom/query_N_with_index.json` / `traces/custom/query_N_without_index.json`

### Results Table

| ID | Query text | With-index result | Confidence / Score | Without-index | Semantic recall? |
|----|-----------|-------------------|--------------------|---------------|-----------------|
| Q1 | "What makes it possible for a language model to weigh the importance of different words when reading input?" | Attention mechanism; soft weights calculated per token | `high` (0.7367) | `No relevant indexed content found.` | **Yes** |
| Q2 | "How do deep networks avoid the problem of learning signals becoming too small to be useful during training?" | Residual connections, ReLU, batch norm, weight init | `high` (0.7796) | `No relevant indexed content found.` | **Yes** |
| Q3 | "What does asyncio.gather do in Python?" | Runs awaitables concurrently; CancelledError propagation rules | `high` (0.7785) | `No relevant indexed content found.` | No |
| Q4 | "What are the differences between LoRA and full fine-tuning of large language models?" | LoRA freezes pre-trained weights, injects rank-decomposition matrices; 10 000× fewer trainable params | `high` (0.8172) | `No relevant indexed content found.` | No |
| Q5 | "Why might a very large language model sometimes produce plausible but factually incorrect responses?" | Hallucination; confident assertion of facts unjustified by training data | `high` (0.7873) | `No relevant indexed content found.` | No |

### Semantic Recall Queries (FR-7.3)

Two of the five queries (Q1 and Q2) are **pure semantic recall**: the query words do not
appear literally in the matching document chunks.

**Q1 — Attention mechanism**

```
Query terms:  "weigh", "importance", "reading input"
Matched chunk text:  "calculates 'soft' weights … relevance … process relationships
                      between all elements in a sequence"
```

`grep -i "weigh\|importance\|reading input" state/memory.json` → 0 literal matches in
the retrieved chunk. The embedding correctly identified conceptual equivalence.

**Q2 — Vanishing gradients**

```
Query terms:  "learning signals", "too small", "useful"
Matched chunk text:  "gradient … vanishing gradient problem … residual connections …
                      ReLU … weight initialization"
```

`grep -i "learning signals\|too small\|useful" state/memory.json` → 0 literal matches
in the retrieved chunk. The vector for "signals becoming too small" correctly mapped to
"vanishing gradient".

### Why the Index Matters

In every without-index trial, confidence dropped to `none` and the system returned
`No relevant indexed content found.` — correctly refusing to hallucinate rather than
generating a generic answer. With the 55-page FAISS index present, all five queries
returned `high`-confidence answers with scores ≥ 0.73.

---

## 6. Compliance Checklist

All checks passed as of commit `00033d4` (2026-05-29).

### AC-1 / NFR-5 — Perception tool-blindness

```bash
grep -r 'index_document\|search_knowledge\|web_search\|fetch_url' perception.py
```

**Result: 0 matches** ✓

`perception.py` speaks in intent verbs only (`make this content searchable`,
`query the existing knowledge base`). Tool names are absent from both the system prompt
and the Python source.

### AC-2 / NFR-8 — Tool-selection guidance only in docstrings

Every endpoint in `rag_server.py` and every tool in `mcp_server.py` carries a docstring
describing when Decision should invoke it. No tool name appears in any `SYSTEM` string
passed to an LLM call.

**Inspection result: compliant** ✓

### AC-3 — FAISS dimension fixed at 768

```bash
python3 -c "import faiss; print(faiss.read_index('state/index.faiss').d)"
```

**Output: `768`** ✓

Both `OllamaEmbedder` (`nomic-embed-text`) and `GeminiEmbedder` (`gemini-embedding-001`)
are pinned to `EMBED_DIM = 768`. Changing either model would require deleting and
rebuilding the index (AC-3).

### AC-4 / NFR-7 — Manifest V3 compliance

```bash
python3 -c "import json; m=json.load(open('extension/manifest.json')); assert m['manifest_version']==3; print('MV3 ok')"
```

**Output: `MV3 ok`** ✓

Extension uses `service_worker`, `scripting`, and `activeTab` — no deprecated MV2 APIs.

### IR-3 — No direct external API calls from the extension

```bash
grep -r 'generativelanguage.googleapis\|api.openai\|ollama' extension/
```

**Result: 0 matches** ✓

`background.js` calls only `http://127.0.0.1:8107` (gateway embed) and
`http://127.0.0.1:8108` (rag_server store-chunk). `popup.js` calls only
`http://127.0.0.1:8108`. Host permissions in `manifest.json` enforce this at the
browser level.

### Summary

| Check | Requirement | Result |
|-------|-------------|--------|
| Perception tool-blindness | `grep … perception.py` → 0 matches | ✓ **PASS** |
| Tool docstrings present | Every tool has a docstring | ✓ **PASS** |
| FAISS dimension | `index.d == 768` | ✓ **PASS** |
| Manifest V3 | `manifest_version == 3` | ✓ **PASS** |
| No external API calls from extension | `grep … extension/` → 0 matches | ✓ **PASS** |

---

## Code Organisation

```
Gemini-TypingFlow-RAG/
│
├── agent7.py           Orchestrator — Perception → Decision → Action → Memory loop
├── perception.py       Goal-management layer; never names tools (AC-1)
├── decision.py         One LLM call per turn → ToolCall | final answer
├── action.py           MCP dispatcher; promotes results > 4 KB to artifact store
├── memory.py           Read/write; FAISS vector search with keyword fallback
├── vector_index.py     FAISS IndexFlatIP wrapper; L2-normalised 768-dim cosine similarity
├── artifacts.py        SHA-256-keyed byte store; Decision sees bytes, Memory holds handles
├── gateway.py          Bridge to llm_gatewayV7; auto-starts gateway if not running
├── mcp_server.py       MCP stdio server — 11 tools incl. index_document + search_knowledge
├── rag_server.py       FastAPI bridge (port 8108) — /index /search /rag /store-chunk /status
├── schemas.py          Shared Pydantic contracts: MemoryItem, Goal, Observation, ToolCall
│
├── extension/          Chrome MV3 extension
│   ├── manifest.json   host_permissions: localhost:8107, localhost:8108 only
│   ├── content.js      DOM scraper + sliding-window chunker (400 words / 80 overlap)
│   ├── background.js   Service worker: embed chunks via gateway → store-chunk to rag_server
│   └── popup.js        Search UI: query → /search → /rag → display answer + sources
│
├── llm_gatewayV7/      Self-contained FastAPI gateway (port 8107)
│   ├── main.py         POST /v1/chat, /v1/embed; tier routing
│   ├── embedders.py    OllamaEmbedder + GeminiEmbedder; 768-dim failover ring
│   ├── providers.py    Provider adapters (Ollama, Gemini, Groq, Nvidia, Cerebras, …)
│   └── router.py       TINY / LARGE / HUGE tier classifier
│
├── state/              Persisted index (gitignored for large corpora)
│   ├── index.faiss     FAISS IndexFlatIP binary (768-dim float32, L2-normalised)
│   ├── index_ids.json  Parallel chunk-ID array
│   ├── memory.json     All MemoryItems (facts + tool outcomes)
│   └── rag_status.json Indexed URL registry
│
├── traces/
│   ├── base/           query_A.json … query_H.json (8 base query traces)
│   └── custom/         query_1_with_index.json … query_5_without_index.json (10 traces)
│
└── docs/               Supporting detail files
    ├── agent_loop.md
    ├── llm_gateway_v7.md
    ├── memory.md
    ├── perception.md
    ├── corpus_manifest.md
    ├── custom_queries.md
    └── performance_results.md   NFR-1 and NFR-2 trial data
```

## Performance

See [`docs/performance_results.md`](docs/performance_results.md) for full trial data.

| NFR | Target | Measured median | Result |
|-----|--------|----------------|--------|
| NFR-1 — page-to-index (52 341-char page, 26 chunks) | ≤ 5 000 ms | 4 301 ms | ✓ |
| NFR-2 — first-token latency (embed + search + assembly) | ≤ 3 000 ms | 198 ms | ✓ |
