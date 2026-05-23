# Chrome Extension RAG — EAG V3 Session 7
**Product Requirements Document (PRD) v1.0**  
*Last updated: 2026-05-23 | Author: Sujit Kumar Ojha | Status: Draft*

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [User Stories](#4-user-stories)
5. [Functional Requirements](#5-functional-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [System Architecture](#7-system-architecture)
8. [Data Models & Schemas](#8-data-models--schemas)
9. [API Contracts](#9-api-contracts)
10. [Acceptance Criteria](#10-acceptance-criteria)
11. [Corpus & Query Design](#11-corpus--query-design)
12. [Deliverables & Milestones](#12-deliverables--milestones)
13. [Out of Scope](#13-out-of-scope)
14. [References](#14-references)

---

## 1. Executive Summary

Build a **Chrome Extension** that implements a production-grade **Retrieval-Augmented Generation (RAG)** pipeline over a corpus of 50+ web pages visited by the user. The extension automatically indexes page content using FAISS-backed vector storage via the `llm_gatewayV7` embedding endpoint, and exposes a **popup search box** for semantic retrieval with LLM-generated answers.

This product satisfies the EAG V3 Session 7 assignment:
- Run all **8 base queries (A-H)** from Session 7 within stated iteration bounds
- Build a real RAG app (**Chrome plugin path**) over 50+ indexed pages
- Design **5 custom queries** where >=2 require pure semantic recall

---

## 2. Background & Motivation

Session 7 of EAG V3 introduced FAISS-backed vector memory into the four-role agentic architecture (Perception -> Decision -> Action -> Memory). The key insight: a `POST /v1/embed` endpoint on `llm_gatewayV7` produces 768-dimensional vectors from either local Ollama (`nomic-embed-text`) or Google Gemini (`gemini-embedding-001`) — both pinned to the same dimensionality so the FAISS index remains portable across providers.

The Chrome Extension path was chosen because:
- It produces a **live corpus** of pages the user actually reads (intrinsic value)
- It demonstrates the full RAG loop: ingest -> chunk -> embed -> index -> retrieve -> generate
- It exercises the MCP tool docstring discipline (tool selection in docstrings, not system prompts)
- It provides a visible, interactive UI to prove RAG quality vs. no-index baseline

---

## 3. Goals & Non-Goals

### 3.1 Goals

| ID | Goal |
|----|------|
| G-1 | Automatically index every page the user visits into a persistent FAISS vector store |
| G-2 | Expose a popup search box that answers queries using RAG (retrieve chunks -> LLM generate) |
| G-3 | Leverage `llm_gatewayV7` for all embedding and LLM calls (no direct model calls from extension) |
| G-4 | Persist the FAISS index across browser restarts |
| G-5 | Pass all 8 base Session 7 queries (A-H) through the existing `agent7.py` system |
| G-6 | Design 5 custom corpus queries — >=2 must demonstrate semantic recall |
| G-7 | Maintain Session 7 architectural rules: grep-clean Perception SYSTEM, tool names only in docstrings |

### 3.2 Non-Goals

- No server-side hosting — everything runs locally (extension + Python bridge + llm_gatewayV7)
- No multi-user support
- No hybrid BM25 + dense retrieval (dense only; BM25 is a Session 8 forward pointer)
- No cross-encoder reranking (also Session 8)
- No UI for manual corpus management

---

## 4. User Stories

| ID | As a... | I want to... | So that... | Priority |
|----|---------|-------------|-----------|----------|
| US-1 | researcher | have every page I visit automatically indexed | I can search my reading history semantically | P0 |
| US-2 | researcher | type a query in the popup and get an LLM answer with sources | I don't have to re-read visited pages | P0 |
| US-3 | researcher | see which pages contributed to each answer | I can verify and go deeper | P0 |
| US-4 | researcher | know when there is no relevant indexed content | the system does not hallucinate | P0 |
| US-5 | developer | run the 8 base Session 7 queries and see passing traces | I can prove the RAG architecture is correct | P0 |
| US-6 | developer | run 5 custom queries and compare with-index vs without-index | I can demonstrate RAG value | P0 |
| US-7 | researcher | see an index status indicator (page count, last URL) | I know the corpus is growing correctly | P1 |
| US-8 | researcher | re-index a changed page | the index stays fresh | P1 |

---

## 5. Functional Requirements

### FR-1: Automatic Page Indexing (Content Script)

| ID | Requirement |
|----|-------------|
| FR-1.1 | Content script MUST inject into every http/https page the user navigates to |
| FR-1.2 | MUST extract visible text via DOM traversal (strip scripts, styles, nav, footer) |
| FR-1.3 | MUST chunk text at **400 words** with **80-word overlap** (matching `mcp_server.py::_chunk_text`) |
| FR-1.4 | Each chunk MUST carry metadata: `{url, title, chunk_index, total_chunks, timestamp_iso}` |
| FR-1.5 | MUST skip pages < 200 words (navigation pages, error pages) |
| FR-1.6 | MUST detect duplicate URLs and send a `reindex` flag to replace existing vectors |

### FR-2: Embedding Pipeline (Background Service Worker)

| ID | Requirement |
|----|-------------|
| FR-2.1 | Service worker MUST call `POST /v1/embed` on `llm_gatewayV7` (port 8107) for every chunk |
| FR-2.2 | Task type MUST be `retrieval_document` for indexing, `retrieval_query` for search |
| FR-2.3 | MUST respect gateway rate-limit responses (HTTP 429) with exponential backoff |
| FR-2.4 | Failed embeddings MUST be retried up to 3 times before marking chunk as failed |
| FR-2.5 | Embedding dimension MUST be 768; worker MUST reject vectors of other sizes |

### FR-3: Vector Store (Local Python Bridge — `rag_server.py`)

| ID | Requirement |
|----|-------------|
| FR-3.1 | FAISS `IndexFlatIP` with L2-normalised vectors MUST be used (cosine similarity) |
| FR-3.2 | Persistence files: `state/index.faiss`, `state/index_ids.json`, `state/corpus.json` |
| FR-3.3 | Store MUST survive Python process restart (load from disk on startup) |
| FR-3.4 | `POST /index` accepts `{chunk_id, embedding, metadata}` and adds to FAISS |
| FR-3.5 | `POST /search` accepts `{embedding, k}` and returns top-k chunk IDs + scores |
| FR-3.6 | `GET /status` returns `{page_count, chunk_count, last_indexed_url, index_size_bytes}` |
| FR-3.7 | On duplicate URL with `reindex=true`, old vectors MUST be removed before re-adding |

### FR-4: Search Box & RAG Answer (Popup)

| ID | Requirement |
|----|-------------|
| FR-4.1 | Popup MUST display a text input (search box) and a Submit button |
| FR-4.2 | On submit, popup MUST embed the query via `POST /v1/embed` (`retrieval_query`) |
| FR-4.3 | Embedded query MUST be sent to `POST /search` on `rag_server.py` to retrieve top-5 chunks |
| FR-4.4 | Retrieved chunks MUST be assembled into a RAG prompt and sent to `POST /v1/chat` on gateway |
| FR-4.5 | Popup MUST display: LLM-generated answer + source cards (URL, title, snippet, score) |
| FR-4.6 | If all top-5 scores < 0.30 cosine similarity, show 'No relevant indexed content found' |
| FR-4.7 | Popup MUST show an index status badge: 'N pages indexed' |

### FR-5: Agentic Decision Layer

| ID | Requirement |
|----|-------------|
| FR-5.1 | `rag_server.py` MUST implement a confidence gate: max score >= 0.70 -> RAG; < 0.30 -> 'not indexed'; otherwise RAG + low-confidence disclaimer |
| FR-5.2 | RAG prompt MUST include top-5 chunk texts and instruct LLM to answer only from context |
| FR-5.3 | Tool selection logic MUST live only in `rag_server.py` function docstrings, not in any system-prompt string |

### FR-6: Base Session 7 Queries (A-H)

| ID | Requirement |
|----|-------------|
| FR-6.1 | All 8 queries (A-H) MUST be run verbatim against `agent7.py` |
| FR-6.2 | Each query MUST complete within the iteration bounds stated in the Session 7 course notes |
| FR-6.3 | Full traces MUST be saved to `traces/base/query_{A-H}.json` |

---

## 6. Non-Functional Requirements

| ID | Requirement | Target |
|----|------------|--------|
| NFR-1 | **Indexing latency** — page load to index write complete | <= 5s for pages <= 50k chars |
| NFR-2 | **Search latency** — submit to first token displayed | <= 3s (excl. LLM streaming) |
| NFR-3 | **Privacy** — page content MUST NOT leave local machine except to llm_gatewayV7 | No external calls from extension |
| NFR-4 | **Corpus size** — index MUST contain >= 50 unique pages before custom query traces | >= 50 pages |
| NFR-5 | **Grep test** — `grep -r 'index_document|search_knowledge|web_search|fetch_url' perception.py` returns 0 hits | 0 hits |
| NFR-6 | **Offline graceful degradation** — if llm_gatewayV7 is unreachable, popup shows a clear error | No silent failures |
| NFR-7 | **MV3 compliance** — extension MUST use Manifest V3 | MV3 manifest |

---

## 7. System Architecture

```
+---------------------------------------------------------------------------+
|                          Chrome Browser                                   |
|                                                                           |
|  +------------------+   +-------------------+   +---------------------+  |
|  |   content.js     |   |  popup.html/js    |   |   background.js     |  |
|  | DOM extraction   |   | Search box        |   |   (service worker)  |  |
|  | Text chunking    |-->| Results UI        |<--| Embed chunks        |  |
|  | Sends to BG      |   | Status badge      |   | Calls bridge        |  |
|  +------------------+   +-------------------+   +----------+----------+  |
+-------------------------------------------------+----------+--------------+
                                                             |
                                                     HTTP localhost:8200
                          +------------------------------+--+
|                             rag_server.py (FastAPI)        |
|  POST /index  --> FAISS add                               |
|  POST /search --> FAISS kNN + confidence gate             |
|  GET  /status --> corpus stats                            |
|  state/index.faiss  state/index_ids.json  corpus.json     |
+------------------------------+-----------------------------+
                               |
                       HTTP localhost:8107
   +----------------------------+-------------------------------+
   |         llm_gatewayV7  (FastAPI)                          |
   |  POST /v1/embed  --> 768-dim vector                       |
   |  POST /v1/chat   --> LLM answer (streamed)                |
   |  Embedders: nomic-embed-text (Ollama) / gemini (fallback) |
   +-----------------------------------------------------------+
```

### Component Responsibilities

| Component | Language | Port | Responsibility |
|-----------|----------|------|----------------|
| `content.js` | JS (MV3) | — | DOM scraping, text chunking, sends to background |
| `background.js` | JS (MV3 SW) | — | Receives chunks, calls gateway + bridge, manages queue |
| `popup.html/js` | HTML/JS | — | Search UI, results display, status badge |
| `options.html/js` | HTML/JS | — | Gateway URL setting, corpus stats |
| `rag_server.py` | Python 3.11 | 8200 | FAISS read/write, confidence gate, RAG prompt assembly |
| `llm_gatewayV7/main.py` | Python 3.11 | 8107 | Embedding + LLM routing (existing, no changes) |

---

## 8. Data Models & Schemas

### 8.1 ChunkRecord (`state/corpus.json`)
```json
{
  "chunk_id": "chunk_<url_hash>_<index>",
  "url": "https://example.com/article",
  "title": "Page Title",
  "chunk_index": 0,
  "total_chunks": 12,
  "text": "...400-word text window...",
  "timestamp_iso": "2026-05-23T10:30:00Z",
  "embedding_dim": 768,
  "provider": "ollama"
}
```

### 8.2 FAISS Index State

| File | Format | Contents |
|------|--------|----------|
| `state/index.faiss` | Binary (FAISS) | `IndexFlatIP` with L2-normalised 768-dim float32 vectors |
| `state/index_ids.json` | JSON array | `["chunk_id_0", "chunk_id_1", ...]` — parallel to FAISS row indices |
| `state/corpus.json` | JSON object | `{ "chunk_id": ChunkRecord, ... }` keyed by chunk_id |

### 8.3 RAG Prompt Template

```
System: You are a helpful assistant. Answer ONLY from the provided context chunks.
If the answer is not in the context, say 'I don't have that in your indexed pages.'

Context:
[CHUNK 1 - source: <title> (<url>)]
<text>
...
[CHUNK 5 - source: <title> (<url>)]
<text>

User: <query>
```

---

## 9. API Contracts

### 9.1 `rag_server.py` endpoints (port 8200)

**`POST /index`**
```json
// Request
{ "chunk_id": "string", "embedding": [0.12, -0.44], "metadata": { "url": "string", "title": "string", "chunk_index": 0, "total_chunks": 12, "text": "string", "timestamp_iso": "string", "reindex": false } }
// Response 200
{ "status": "indexed", "chunk_id": "string", "total_chunks_in_index": 423 }
```

**`POST /search`**
```json
// Request
{ "embedding": [0.12, -0.44], "k": 5 }
// Response 200
{ "results": [ { "chunk_id": "string", "score": 0.84, "text": "string", "url": "string", "title": "string" } ] }
```

**`GET /status`**
```json
// Response 200
{ "page_count": 52, "chunk_count": 623, "last_indexed_url": "https://...", "last_indexed_at": "2026-05-23T10:30:00Z", "index_size_bytes": 1920000 }
```

### 9.2 `llm_gatewayV7` endpoints (port 8107) — existing, no changes

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/embed` | POST | `{text, task_type}` -> `{embedding: float[768], model, dim}` |
| `/v1/chat` | POST | `{messages, system}` -> streamed token response |
| `/v1/status` | GET | Provider health and rate-limit state |

---

## 10. Acceptance Criteria

### AC-1: Indexing
- [ ] Visit 50 unique pages; `GET /status` returns `page_count >= 50`
- [ ] FAISS index files exist at `state/index.faiss`, `state/index_ids.json`, `state/corpus.json`
- [ ] Kill and restart `rag_server.py`; `GET /status` still returns `page_count >= 50` (persistence)
- [ ] Re-visit an indexed URL; chunk count for that URL resets to current content

### AC-2: Search & RAG
- [ ] Query with known indexed content returns answer citing correct source URLs
- [ ] Query with no indexed content returns 'No relevant indexed content found' (no hallucination)
- [ ] Semantic recall query (query words absent from chunks) returns correct chunks

### AC-3: Base Queries (A-H)
- [ ] All 8 queries complete within stated iteration bounds against `agent7.py`
- [ ] Trace files exist at `traces/base/query_A.json` ... `traces/base/query_H.json`

### AC-4: Custom Queries (1-5)
- [ ] 5 custom queries each have `_with_index.json` and `_without_index.json` traces
- [ ] Queries 2 and 3 (semantic recall) answer correctly WITH index, fail/degrade WITHOUT
- [ ] README contains a results comparison table

### AC-5: Architectural Rules
- [ ] `grep -r 'index_document\|search_knowledge\|web_search\|fetch_url' perception.py` -> 0 matches
- [ ] All new tools in `rag_server.py` have docstrings describing when to use them
- [ ] No direct Gemini/Ollama API calls from the Chrome extension (all via llm_gatewayV7)

---

## 11. Corpus & Query Design

### 11.1 Minimum 50-Page Corpus

| Domain | Count | Topics |
|--------|-------|--------|
| Wikipedia | 15 | Transformer, attention, LSTM, RAG, FAISS, embeddings, backprop, gradient descent |
| arXiv abstracts | 10 | Attention Is All You Need, ReAct, CoT, LoRA, RAG paper, FAISS paper |
| MDN / Python docs | 8 | asyncio, fetch API, WebWorkers, IndexedDB, promises |
| Research blogs | 10 | Karpathy, Lilian Weng (RAG/retrieval posts), Eugene Yan |
| Course materials | 7 | EAG V3 session pages on Axiom |
| **Total** | **50+** | |

### 11.2 Five Custom Queries

| # | Query | Semantic Recall? | Expected WITH index | Expected WITHOUT |
|---|-------|-----------------|--------------------|--------------------|
| Q1 | 'What pages did I read about neural network training?' | No | Titles + URLs of indexed ML pages | Empty / generic |
| Q2 | 'What do my indexed pages say about memory efficiency in transformers?' | **Yes** | KV-cache / attention complexity chunks — words differ from query | Generic LLM answer, no sources |
| Q3 | 'Which articles discuss the vanishing gradient issue?' | **Yes** | Backprop / exploding-gradient chunks — no literal match in chunks | Fails to cite sources |
| Q4 | 'Summarise what I've read about Python async patterns' | No | asyncio chunks from MDN + Python docs | Generic async overview |
| Q5 | 'What do my indexed pages say about transformer attention?' | No | Attention mechanism chunks from Wikipedia + arXiv | Generic attention explanation |

---

## 12. Deliverables & Milestones

| Milestone | Items | Target |
|-----------|-------|--------|
| **M0 — Study** | Watch Session 7 video; read course content; understand Session 6->7 delta | Day 1-2 |
| **M1 — Understand Code** | Read `memory.py`, `mcp_server.py`, `agent7.py`, `llm_gatewayV7/embedders.py`, `llm_gatewayV7/main.py` | Day 2-3 |
| **M2 — Base Queries** | Run queries A-H through `agent7.py`; save traces to `traces/base/` | Day 3-4 |
| **M3 — rag_server.py** | Build FastAPI bridge: `/index`, `/search`, `/status`, confidence gate | Day 4-5 |
| **M4 — Chrome Extension** | `manifest.json`, `content.js`, `background.js`, `popup.html/js` | Day 5-7 |
| **M5 — Integration** | End-to-end: page visit -> index -> popup search -> RAG answer | Day 7-8 |
| **M6 — Corpus** | Index >= 50 pages; verify persistence across restart | Day 8-9 |
| **M7 — Custom Queries** | Run 5 custom queries with/without index; save traces | Day 9-10 |
| **M8 — Submission** | README with corpus manifest; demo video; GitHub push | Day 10-11 |

### Repository File Structure

```
Gemini-TypingFlow-RAG/
├── requirement.md              <- this document
├── README.md                   <- corpus manifest + architecture + setup
├── llm_gatewayV7/              <- existing gateway (no changes)
│   ├── main.py
│   ├── embedders.py
│   ├── router.py
│   └── ...
├── chrome_extension/
│   ├── manifest.json           <- MV3
│   ├── content.js              <- DOM scraper + chunker
│   ├── background.js           <- service worker
│   ├── popup.html
│   ├── popup.js
│   ├── options.html
│   └── options.js
├── rag_server.py               <- FastAPI bridge (port 8200)
├── state/
│   ├── index.faiss
│   ├── index_ids.json
│   └── corpus.json
├── traces/
│   ├── base/
│   │   └── query_A.json ... query_H.json
│   └── custom/
│       ├── query_1_with_index.json
│       ├── query_1_without_index.json
│       └── ... (5 queries x 2)
├── agent7.py                   <- existing (no changes)
├── memory.py                   <- existing (no changes)
├── mcp_server.py               <- existing (no changes)
├── perception.py               <- existing (no changes)
├── decision.py                 <- existing (no changes)
└── action.py                   <- existing (no changes)
```

---

## 13. Out of Scope

| Feature | Reason deferred |
|---------|----------------|
| Hybrid retrieval (BM25 + dense) | Session 8 feature (Reciprocal Rank Fusion) |
| Cross-encoder reranking | Session 8 feature |
| Semantic chunking (LLM-guided chunk boundaries) | Session 8 feature |
| Cloud deployment | Local-only for this version |
| Multi-user / shared corpus | Single-user only |
| Browser sync across devices | Index is local filesystem |
| Fine-tuned embedding models | Using off-the-shelf nomic/Gemini only |
| Corpus management UI (delete/inspect pages) | Out of scope for v1 |

---

## 14. References

| Resource | Location |
|----------|----------|
| Session 7 course content | Axiom Learning OS — EAG V3 Session 7 |
| Existing agent code | `agent7.py`, `memory.py`, `mcp_server.py`, `perception.py`, `decision.py`, `action.py` |
| LLM Gateway | `llm_gatewayV7/main.py`, `llm_gatewayV7/embedders.py`, `llm_gatewayV7/router.py` |
| FAISS documentation | https://faiss.ai/ |
| Chrome Extension MV3 | https://developer.chrome.com/docs/extensions/mv3/ |
| nomic-embed-text | https://ollama.com/library/nomic-embed-text |
| gemini-embedding-001 | https://ai.google.dev/api/embeddings |
