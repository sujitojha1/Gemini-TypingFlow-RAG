# Software Requirements Specification
## Gemini-TypingFlow-RAG — Chrome Extension RAG Pipeline

| Field | Value |
|---|---|
| Document ID | SRS-GTR-001 |
| Version | 1.0 |
| Date | 2026-05-23 |
| Author | Sujit Kumar Ojha |
| Status | Draft |
| Standard | IEEE 830 / EARS (Easy Approach to Requirements Syntax) |

---

## Conventions

- **SHALL** — mandatory requirement
- **SHOULD** — recommended, non-mandatory
- **Priority** — M = Must Have · S = Should Have · C = Could Have (MoSCoW)
- **Verification** — T = Test · I = Inspection · A = Analysis · D = Demonstration
- **Source** — US = User Story · AC = Assignment Criterion · NFR = Non-Functional

---

## 1. User Stories (Source Baseline)

| ID | Role | Capability | Rationale |
|---|---|---|---|
| US-1 | Researcher | Have every visited page automatically indexed | Enable semantic search over reading history |
| US-2 | Researcher | Type a query in popup and receive an LLM answer with sources | Avoid re-reading visited pages |
| US-3 | Researcher | See which pages contributed to each answer | Verify and drill deeper |
| US-4 | Researcher | Receive explicit signal when no relevant content is indexed | Prevent hallucination |
| US-5 | Developer | Run all 8 base Session 7 queries and see passing traces | Prove RAG architecture is correct |
| US-6 | Developer | Run 5 custom queries with and without index and compare results | Demonstrate RAG value |
| US-7 | Researcher | See index status indicator (page count, last URL) | Confirm corpus is growing |
| US-8 | Researcher | Re-index a changed page | Keep index fresh |

---

## 2. Functional Requirements

### 2.1 Page Indexing (Content Script)

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-1.1 | The content script SHALL inject into every HTTP and HTTPS page the user navigates to. | M | US-1 | T |
| FR-1.2 | The content script SHALL extract visible text via DOM traversal, excluding scripts, styles, navigation, and footer elements. | M | US-1 | T |
| FR-1.3 | The content script SHALL chunk extracted text into windows of 400 words with an 80-word overlap. | M | US-1, US-2 | T |
| FR-1.4 | Each chunk SHALL carry metadata containing: `url`, `title`, `chunk_index`, `total_chunks`, `timestamp_iso`. | M | US-3 | I |
| FR-1.5 | The content script SHALL skip pages whose extracted text is fewer than 200 words. | M | US-4 | T |
| FR-1.6 | The content script SHALL detect duplicate URLs and set a `reindex` flag on re-submission to replace existing vectors. | S | US-8 | T |

### 2.2 Embedding Pipeline (Background Service Worker)

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-2.1 | The service worker SHALL call `POST /v1/embed` on `llm_gatewayV7` (port 8107) for every chunk. | M | US-1 | T |
| FR-2.2 | The service worker SHALL use task type `retrieval_document` when indexing chunks and `retrieval_query` when embedding search queries. | M | US-2 | T |
| FR-2.3 | The service worker SHALL respect HTTP 429 responses from the gateway by applying exponential backoff. | M | NFR-3 | T |
| FR-2.4 | The service worker SHALL retry failed embedding calls up to 3 times before marking the chunk as failed. | M | NFR-3 | T |
| FR-2.5 | The service worker SHALL reject any embedding vector whose dimension is not 768. | M | FR-3.1 | T |

### 2.3 Vector Store — Local Python Bridge (`rag_server.py`)

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-3.1 | `rag_server.py` SHALL use a FAISS `IndexFlatIP` index with L2-normalised float32 vectors to implement cosine similarity search. | M | US-2 | I |
| FR-3.2 | `rag_server.py` SHALL persist the index to three files: `state/index.faiss`, `state/index_ids.json`, `state/corpus.json`. | M | US-1, NFR-1 | T |
| FR-3.3 | `rag_server.py` SHALL load all three persistence files on startup so that the index survives a Python process restart. | M | US-1 | D |
| FR-3.4 | `rag_server.py` SHALL expose `POST /index` accepting `{chunk_id, embedding, metadata}` and appending the vector to the FAISS index. | M | US-1 | T |
| FR-3.5 | `rag_server.py` SHALL expose `POST /search` accepting `{embedding, k}` and returning the top-k chunk IDs with cosine similarity scores. | M | US-2 | T |
| FR-3.6 | `rag_server.py` SHALL expose `GET /status` returning `{page_count, chunk_count, last_indexed_url, index_size_bytes}`. | S | US-7 | T |
| FR-3.7 | When `POST /index` is called with `reindex=true` for an existing URL, `rag_server.py` SHALL remove all prior vectors for that URL before adding the new ones. | S | US-8 | T |

### 2.4 Popup Search & RAG Answer

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-4.1 | The popup SHALL display a text input field and a Submit button. | M | US-2 | I |
| FR-4.2 | On submit, the popup SHALL embed the query by calling `POST /v1/embed` (task type `retrieval_query`) on the gateway. | M | US-2 | T |
| FR-4.3 | The popup SHALL send the embedded query to `POST /search` on `rag_server.py` and retrieve the top-5 chunks. | M | US-2 | T |
| FR-4.4 | The popup SHALL assemble the top-5 chunks into a RAG prompt and send it to `POST /v1/chat` on the gateway to obtain an answer. | M | US-2 | D |
| FR-4.5 | The popup SHALL display the LLM-generated answer followed by source cards containing URL, title, chunk snippet, and similarity score. | M | US-3 | D |
| FR-4.6 | When all top-5 similarity scores are below 0.30, the popup SHALL display the message "No relevant indexed content found" and SHALL NOT call the LLM. | M | US-4 | T |
| FR-4.7 | The popup SHALL display an index status badge showing the total number of indexed pages. | S | US-7 | I |

### 2.5 Confidence Gate & RAG Decision Logic

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-5.1 | When the maximum similarity score among top-5 results is ≥ 0.70, the system SHALL proceed with full RAG and return the LLM answer with sources. | M | US-2, US-4 | T |
| FR-5.2 | When the maximum similarity score is between 0.30 (inclusive) and 0.70 (exclusive), the system SHALL return the LLM answer with a low-confidence disclaimer. | M | US-4 | T |
| FR-5.3 | When the maximum similarity score is below 0.30, the system SHALL return "No relevant indexed content found" without calling the LLM. | M | US-4 | T |
| FR-5.4 | The RAG prompt SHALL instruct the LLM to answer only from the provided context chunks and to state "I don't have that in your indexed pages" when the answer is absent. | M | US-4 | I |
| FR-5.5 | Tool selection logic SHALL reside exclusively in `rag_server.py` function docstrings and SHALL NOT appear in any system-prompt string. | M | AC-5 | I |

### 2.6 Base Session 7 Queries (Assignment Compliance)

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-6.1 | The system SHALL pass all 8 base queries (A through H) verbatim against `agent7.py`. | M | US-5, AC-3 | D |
| FR-6.2 | Each base query SHALL complete within the iteration bound stated in the Session 7 course notes. | M | US-5, AC-3 | D |
| FR-6.3 | Full execution traces for each base query SHALL be saved to `traces/base/query_{A-H}.json`. | M | AC-3 | I |

### 2.7 Custom Corpus Queries

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| FR-7.1 | The system SHALL support 5 custom queries against the indexed corpus, each answering correctly with the index present. | M | US-6, AC-4 | D |
| FR-7.2 | Each custom query SHALL fail or degrade to a generic answer when the index is absent (no-index baseline). | M | US-6, AC-4 | D |
| FR-7.3 | At least 2 of the 5 custom queries SHALL require semantic recall, defined as: the words in the query do not appear literally in the chunks that provide the correct answer. | M | AC-4 | A |
| FR-7.4 | Traces for each custom query SHALL be saved as `traces/custom/query_N_with_index.json` and `traces/custom/query_N_without_index.json`. | M | AC-4 | I |

---

## 3. Non-Functional Requirements

| ID | Requirement | Target | Priority | Trace | Verify |
|---|---|---|---|---|---|
| NFR-1 | The system SHALL complete the full pipeline from page load to index-write within 5 seconds for pages up to 50,000 characters. | ≤ 5 s | M | US-1 | T |
| NFR-2 | The system SHALL deliver the first token of a popup search response within 3 seconds of query submission, excluding LLM streaming latency. | ≤ 3 s | M | US-2 | T |
| NFR-3 | Page content SHALL NOT be transmitted to any host other than the local `llm_gatewayV7` instance. | Zero external calls | M | US-1 | I |
| NFR-4 | The FAISS index SHALL contain vectors for at least 50 unique pages before custom query traces are recorded. | ≥ 50 pages | M | US-6, AC-4 | D |
| NFR-5 | A `grep -r 'index_document\|search_knowledge\|web_search\|fetch_url' perception.py` command SHALL return zero matches. | 0 matches | M | AC-5 | I |
| NFR-6 | When `llm_gatewayV7` is unreachable, the popup SHALL display a clear error message and SHALL NOT fail silently. | Visible error | M | US-4 | T |
| NFR-7 | The Chrome extension SHALL comply with Manifest V3 (MV3). | MV3 | M | — | I |
| NFR-8 | All new tools in `rag_server.py` SHALL have docstrings that describe the conditions under which Decision should invoke them. | Present | M | FR-5.5, AC-5 | I |

---

## 4. Corpus Requirements

| ID | Requirement | Priority | Trace | Verify |
|---|---|---|---|---|
| CRP-1 | The indexed corpus SHALL contain at least 50 unique pages before the 5 custom query traces are recorded. | M | NFR-4 | D |
| CRP-2 | The corpus SHALL include pages from at least 4 distinct domains or topic categories. | M | AC-4 | I |
| CRP-3 | The corpus SHALL include pages covering: neural network training, transformer architecture, attention mechanisms, Python async patterns, and gradient-related topics. | M | FR-7.3 | I |

---

## 5. Traceability Matrix

| Requirement | US-1 | US-2 | US-3 | US-4 | US-5 | US-6 | US-7 | US-8 | AC-3 | AC-4 | AC-5 | NFR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FR-1.1 | ✓ | | | | | | | | | | | |
| FR-1.2 | ✓ | | | | | | | | | | | |
| FR-1.3 | ✓ | ✓ | | | | | | | | | | |
| FR-1.4 | | | ✓ | | | | | | | | | |
| FR-1.5 | | | | ✓ | | | | | | | | |
| FR-1.6 | ✓ | | | | | | | ✓ | | | | |
| FR-2.1 | ✓ | | | | | | | | | | | |
| FR-2.2 | | ✓ | | | | | | | | | | |
| FR-2.3 | | | | | | | | | | | | ✓ |
| FR-2.4 | | | | | | | | | | | | ✓ |
| FR-2.5 | | | | | | | | | | | | ✓ |
| FR-3.1 | | ✓ | | | | | | | | | | |
| FR-3.2 | ✓ | | | | | | | | | | | ✓ |
| FR-3.3 | ✓ | | | | | | | | | | | |
| FR-3.4 | ✓ | | | | | | | | | | | |
| FR-3.5 | | ✓ | | | | | | | | | | |
| FR-3.6 | | | | | | | ✓ | | | | | |
| FR-3.7 | | | | | | | | ✓ | | | | |
| FR-4.1 | | ✓ | | | | | | | | | | |
| FR-4.2 | | ✓ | | | | | | | | | | |
| FR-4.3 | | ✓ | | | | | | | | | | |
| FR-4.4 | | ✓ | | | | | | | | | | |
| FR-4.5 | | | ✓ | | | | | | | | | |
| FR-4.6 | | | | ✓ | | | | | | | | |
| FR-4.7 | | | | | | | ✓ | | | | | |
| FR-5.1 | | ✓ | | ✓ | | | | | | | | |
| FR-5.2 | | | | ✓ | | | | | | | | |
| FR-5.3 | | | | ✓ | | | | | | | | |
| FR-5.4 | | | | ✓ | | | | | | | | |
| FR-5.5 | | | | | | | | | | | ✓ | |
| FR-6.1 | | | | | ✓ | | | | ✓ | | | |
| FR-6.2 | | | | | ✓ | | | | ✓ | | | |
| FR-6.3 | | | | | | | | | ✓ | | | |
| FR-7.1 | | | | | | ✓ | | | | ✓ | | |
| FR-7.2 | | | | | | ✓ | | | | ✓ | | |
| FR-7.3 | | | | | | ✓ | | | | ✓ | | |
| FR-7.4 | | | | | | | | | | ✓ | | |
| CRP-1 | | | | | | ✓ | | | | ✓ | | ✓ |
| CRP-2 | | | | | | ✓ | | | | ✓ | | |
| CRP-3 | | | | | | ✓ | | | | ✓ | | |

---

## 6. Data Schema Requirements

| ID | Requirement | Priority | Verify |
|---|---|---|---|
| DS-1 | Each `ChunkRecord` in `state/corpus.json` SHALL contain: `chunk_id`, `url`, `title`, `chunk_index`, `total_chunks`, `text`, `timestamp_iso`, `embedding_dim`, `provider`. | M | I |
| DS-2 | `state/index_ids.json` SHALL be a JSON array of chunk ID strings whose order is parallel to the FAISS row indices. | M | I |
| DS-3 | `state/index.faiss` SHALL be a binary FAISS `IndexFlatIP` file storing 768-dimensional float32 L2-normalised vectors. | M | I |
| DS-4 | The `chunk_id` field SHALL follow the format `chunk_<url_hash>_<index>`. | M | I |

---

## 7. Interface Requirements

| ID | Requirement | Priority | Verify |
|---|---|---|---|
| IR-1 | `rag_server.py` SHALL listen on port 8200. | M | T |
| IR-2 | `llm_gatewayV7` SHALL be called exclusively on port 8107. | M | T |
| IR-3 | The Chrome extension SHALL make no direct calls to Gemini or Ollama APIs; all model interactions SHALL be routed through `llm_gatewayV7`. | M | I |
| IR-4 | The RAG system prompt template SHALL follow the structure: system block instructing LLM to answer only from context, numbered context chunks with source labels, then the user query. | M | I |

---

## 8. Architectural Constraints

| ID | Constraint | Priority | Verify |
|---|---|---|---|
| AC-1 | The grep test `grep -r 'index_document\|search_knowledge' perception.py` SHALL return zero matches at all times. | M | I |
| AC-2 | All tool-selection guidance for new MCP tools SHALL appear only in the respective function docstrings. | M | I |
| AC-3 | The FAISS index embedding dimension SHALL be fixed at 768 for the lifetime of any index instance; changing the embedding model requires deleting and rebuilding the index. | M | I |
| AC-4 | The extension SHALL comply with Chrome Manifest V3; no Manifest V2 APIs SHALL be used. | M | I |
| AC-5 | All system components SHALL run locally; no cloud hosting of `rag_server.py` or `llm_gatewayV7` is permitted for this version. | M | I |
