# requirement.md — Chrome Extension RAG
## EAG V3 Session 7 Assignment

## Overview

Build a Chrome Extension that implements a full RAG (Retrieval-Augmented Generation) pipeline over a corpus of 50+ web pages visited by the user. The extension indexes page content into a FAISS-backed vector store via the `llm_gatewayV7` embedding endpoint, and exposes a search box for semantic retrieval with LLM-generated answers.

This project satisfies all three assignment requirements:
1. Pass all eight base queries (A–H) from Session 7 within stated iteration bounds.
2. 2. Build a real RAG application (Chrome plugin path) over 50+ indexed pages.
   3. 3. Design five custom queries — at least two require semantic recall.
     
      4. ---
     
      5. ## Architectural Rules (Carried Over from Session 7)
     
      6. - **Grep test on Perception's SYSTEM**: Zero MCP tool names may appear inside `perception.py`'s SYSTEM prompt. Tool selection lives in Decision's docstrings only.
         - - **Separation of concerns**: Perception decomposes intent → goals. Decision selects tools. Action executes. Memory persists and retrieves.
           - - **Embedding model lock**: Once the FAISS index is built with `nomic-embed-text` (768-dim via Ollama) or `gemini-embedding-001` (768-dim via Gemini), the model must not change without clearing the index.
             - - **MCP tool docstrings are Decision's instruction surface**: New tools added for Chrome extension content ingestion must document their usage in the docstring, not in any system prompt.
              
               - ---

               ## System Architecture

               ### Components

               ```
               ┌─────────────────────────────────────────────────────────────────┐
               │                     Chrome Extension                            │
               │                                                                 │
               │  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
               │  │ content.js   │  │   popup.html /  │  │  background.js   │  │
               │  │ (page scraper│  │   popup.js      │  │  (service worker)│  │
               │  │  + chunker)  │  │  (search box +  │  │  (index trigger, │  │
               │  │              │  │   results UI)   │  │   RAG query)     │  │
               │  └──────┬───────┘  └────────┬────────┘  └────────┬─────────┘  │
               │         │                   │                     │            │
               └─────────┼───────────────────┼─────────────────────┼────────────┘
                         │                   │                     │
                         ▼                   ▼                     ▼
               ┌─────────────────────────────────────────────────────────────────┐
               │                 llm_gatewayV7 (FastAPI, port 8107)              │
               │                                                                 │
               │  POST /v1/chat    POST /v1/embed    GET /v1/status              │
               │  (LLM answer)     (768-dim vec)     (health check)              │
               │                                                                 │
               │  Providers: Ollama (nomic-embed-text) → Gemini fallback         │
               └─────────────────────────────────────────────────────────────────┘
                         │                   │
                         ▼                   ▼
               ┌─────────────────┐  ┌────────────────────────────────────────────┐
               │  LLM Answer     │  │  FAISS Vector Store + metadata.json        │
               │  (streamed via  │  │  state/index.faiss  (768-dim vectors)      │
               │   /v1/chat)     │  │  state/index_ids.json  (chunk IDs)         │
               │                 │  │  state/corpus.json  (full corpus manifest) │
               └─────────────────┘  └────────────────────────────────────────────┘
               ```

               ### Chrome Extension Files

               | File | Purpose |
               |------|--------|
               | `manifest.json` | MV3 manifest; declares permissions: `activeTab`, `storage`, `scripting` |
               | `content.js` | Injected into every page; extracts clean text via DOM traversal; chunks into 400-word / 80-word-overlap windows; sends chunks to background |
               | `background.js` | Service worker; receives chunks from content.js; calls `POST /v1/embed` on gateway; writes vectors + metadata to local FAISS via Python bridge; handles search queries from popup |
               | `popup.html` | Extension popup UI with search box, result list, and index status indicator |
               | `popup.js` | Handles search submission; sends query to background; renders ranked results with source URL and snippet |
               | `options.html` | Settings page: gateway URL, embedding model toggle, corpus size counter |
               | `rag_server.py` | Local Python bridge (FastAPI, port 8200); wraps FAISS read/write; exposes `POST /index` and `POST /search`; delegates embeddings to llm_gatewayV7 |

               ---

               ## Functional Requirements

               ### FR-1: Page Indexing
               - The extension MUST automatically extract and index the visible text content of every page the user visits (content script injection).
               - - Indexing MUST chunk text at 400 words with 80-word overlap, matching the `index_document` chunker in `mcp_server.py`.
                 - - Each chunk MUST store metadata: `{url, title, chunk_index, total_chunks, timestamp}`.
                   - - The corpus MUST reach 50+ unique pages before the five custom queries are run.
                     - - Duplicate URLs MUST be detected; re-indexing an already-indexed URL re-chunks and replaces existing vectors.
                      
                       - ### FR-2: Embedding Pipeline
                       - - All embeddings MUST be generated via `llm_gatewayV7`'s `POST /v1/embed` endpoint.
                         - - Primary embedder: `nomic-embed-text` (Ollama, local, 768-dim).
                           - - Fallback embedder: `gemini-embedding-001` (Google, 768-dim, with rate limiting & exponential backoff as implemented in `embedders.py`).
                             - - Embedding dimension is pinned at 768. Changing the model requires clearing the index.
                              
                               - ### FR-3: Vector Store
                               - - FAISS `IndexFlatIP` with L2-normalised vectors (cosine similarity).
                                 - - Persistence: `state/index.faiss`, `state/index_ids.json`, `state/corpus.json`.
                                   - - The store MUST survive extension restart and browser restart.
                                    
                                     - ### FR-4: Search Box (RAG Query)
                                     - - The popup MUST expose a search box accepting free-text queries.
                                       - - On submit, the query MUST be embedded via `POST /v1/embed` (task_type=`retrieval_query`).
                                         - - Top-5 nearest chunks MUST be retrieved from FAISS.
                                           - - Retrieved chunks MUST be assembled into a RAG context and passed to `POST /v1/chat` on llm_gatewayV7 for an LLM-generated answer.
                                             - - The popup MUST display: the LLM answer, and the top-5 source chunks with URL, title, and relevance score.
                                              
                                               - ### FR-5: Agentic Behaviour (Decision Layer)
                                               - - The background service worker MUST implement a simple Decision loop:
                                                 -   - If the query can be answered from indexed chunks with confidence ≥ 0.7, return RAG answer.
                                                     -   - If no relevant chunks exist (all scores < 0.3), return a "not indexed" message rather than hallucinating.
                                                         -   - The decision heuristic MUST be implemented in `background.js` / `rag_server.py` without exposing tool names in any Perception-equivalent system prompt.
                                                          
                                                             - ### FR-6: Eight Base Queries (A–H) from Session 7
                                                             - All eight queries defined in the Session 7 course content MUST be reproducible against the agent (agent7.py + mcp_server.py) within the iteration bounds stated. Traces MUST be saved to `traces/base/query_{A-H}.json`.
                                                          
                                                             - ### FR-7: Five Custom Queries
                                                             - - Five queries MUST be designed against the Chrome extension corpus.
                                                               - - Each query MUST: (a) answer correctly WITH the index, (b) fail or degrade WITHOUT the index.
                                                                 - - At least two queries MUST require semantic recall (query words absent from the answer chunks).
                                                                   - - Traces saved to `traces/custom/query_{1-5}.json` with and without index.
                                                                    
                                                                     - ---

                                                                     ## Non-Functional Requirements

                                                                     - **NFR-1 Performance**: Indexing a page MUST complete within 5 seconds for pages up to 50,000 characters.
                                                                     - - **NFR-2 Privacy**: Page content MUST NOT be sent to any external service other than the configured llm_gatewayV7 instance.
                                                                       - - **NFR-3 Offline Fallback**: If llm_gatewayV7 is unreachable, the extension MUST surface a clear error rather than silently failing.
                                                                         - - **NFR-4 Corpus Size**: The corpus MUST contain at least 50 indexed pages before the five custom query traces are recorded.
                                                                           - - **NFR-5 No Tool Names in System Prompts**: Architectural rule from Session 7 — grep test on any Perception/system-prompt string must return zero hits for MCP tool names.
                                                                            
                                                                             - ---

                                                                             ## Corpus Manifest (Minimum 50 Pages)

                                                                             The `README.md` MUST include a corpus manifest table:

                                                                             | # | URL | Title | Chunks | Indexed At |
                                                                             |---|-----|-------|--------|------------|
                                                                             | 1 | ... | ... | ... | ... |
                                                                             | ... | | | | |
                                                                             | 50+ | ... | ... | ... | ... |

                                                                             Suggested corpus domains: Wikipedia articles (technical topics), arXiv abstracts, MDN Web Docs pages, Python docs pages, research blog posts.

                                                                             ---

                                                                             ## Deliverables

                                                                             1. **GitHub Repository** containing:
                                                                             2.    - This `requirement.md`
                                                                                   -    - `README.md` with corpus manifest, architecture diagram, setup instructions
                                                                                        -    - `chrome_extension/` directory with all extension source files
                                                                                             -    - `rag_server.py` local Python bridge
                                                                                                  -    - `llm_gatewayV7/` (existing, unchanged or minimally modified)
                                                                                                       -    - `traces/base/query_{A-H}.json` — eight base query traces
                                                                                                            -    - `traces/custom/query_{1-5}_with_index.json` and `query_{1-5}_without_index.json`
                                                                                                                 -    - `state/corpus.json` — corpus manifest (auto-generated)
                                                                                                                  
                                                                                                                      - 2. **Short Demo Video** (2–5 minutes) showing:
                                                                                                                        3.    - Extension indexing a page live
                                                                                                                              -    - Search box query returning RAG answer with sources
                                                                                                                                   -    - At least one semantic recall query demonstration
                                                                                                                                    
                                                                                                                                        - ---
                                                                                                                                        
                                                                                                                                        ## Five Custom Queries (Semantic Recall Requirements)
                                                                                                                                        
                                                                                                                                        | # | Query | Semantic Recall? | Expected behaviour WITH index | Expected behaviour WITHOUT |
                                                                                                                                        |---|-------|-----------------|-------------------------------|---------------------------|
                                                                                                                                        | 1 | "What pages did I read about neural network training?" | No | Returns titles + URLs of indexed ML pages | Empty / hallucination |
                                                                                                                                        | 2 | "What do the pages I've indexed say about memory efficiency?" | Yes | Retrieves chunks about attention memory, KV cache, etc. — words differ from query | Generic answer, no sources |
                                                                                                                                        | 3 | "Which articles discuss the vanishing gradient issue?" | Yes | Retrieves chunks about backprop, exploding/vanishing gradients — no exact match | Fails to cite sources |
                                                                                                                                        | 4 | "Summarise what I've read about Python async patterns" | No | Retrieves asyncio chunks across multiple pages | No specific summary |
                                                                                                                                        | 5 | "What do my indexed pages say about transformer attention?" | No | Returns relevant attention mechanism chunks | Generic LLM response |
                                                                                                                                        
                                                                                                                                        ---
                                                                                                                                        
                                                                                                                                        ## Tech Stack
                                                                                                                                        
                                                                                                                                        - **Chrome Extension**: Manifest V3, Vanilla JS (content.js, background.js, popup.js)
                                                                                                                                        - - **Local Bridge**: Python 3.11+, FastAPI, FAISS-cpu, numpy, httpx
                                                                                                                                          - - **Gateway**: `llm_gatewayV7` (FastAPI, port 8107) — existing codebase
                                                                                                                                            - - **Embeddings**: `nomic-embed-text` via Ollama (primary) + `gemini-embedding-001` (fallback)
                                                                                                                                              - - **LLM**: Routed via llm_gatewayV7 auto-route (TINY/LARGE tier)
                                                                                                                                                - - **Vector Index**: FAISS `IndexFlatIP` with L2 normalisation
                                                                                                                                                  - - **Persistence**: Local filesystem under `state/`
                                                                                                                                                   
                                                                                                                                                    - ---
                                                                                                                                                    
                                                                                                                                                    ## References
                                                                                                                                                    
                                                                                                                                                    - Session 7 course content: EAG V3, Axiom Learning OS
                                                                                                                                                    - - Existing codebase: `memory.py`, `mcp_server.py`, `agent7.py`, `perception.py`, `decision.py`, `action.py`
                                                                                                                                                      - - Gateway: `llm_gatewayV7/main.py`, `llm_gatewayV7/embedders.py`, `llm_gatewayV7/router.py`
                                                                                                                                                        - - FAISS documentation: https://faiss.ai/
                                                                                                                                                          - - Chrome Extension MV3: https://developer.chrome.com/docs/extensions/mv3/
                                                                                                                                                            - 
