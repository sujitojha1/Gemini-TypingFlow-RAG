# Class Notes — Session 7: RAG + Vector Memory

## Core RAG Concepts and Components

**Embeddings**
Fixed-length numerical vectors that represent the semantic meaning of a piece of text. By mapping text into a 768-dimensional space (using models like Nomic via Ollama or Gemini), semantically similar concepts cluster together, whilst unrelated concepts sit further apart.

**Dense vs. Sparse Retrieval**
Dense embeddings strongly capture broad meanings and synonyms, whereas sparse embeddings (such as BM25) excel at exact keyword matches, codes, or IDs. Session 7 focuses purely on dense retrieval; production-grade applications require a hybrid approach combining both methods alongside a reranker.

**FAISS (Facebook AI Similarity Search)**
An in-memory vector search engine that rapidly finds stored vectors closest to a query vector using metrics like Euclidean distance (L2) or inner product / cosine similarity. It acts purely as a similarity index and relies on a separate database layer to map vector positions back to text chunks and metadata.

**Chunking**
Embedding an entire document dilutes its semantic meaning into a single average vector, so texts must be divided into smaller passages. The current implementation uses a heuristic sliding window of 400 words with an 80-word overlap to preserve context at boundaries. It does not yet respect natural semantic boundaries like paragraphs or equations — semantic chunking arrives in Session 8.

---

## Agent Architecture Upgrades

**Minimal Foundational Changes**
The four-role architecture from the previous session (Perception → Decision → Action → Memory) remains fully intact. The core agent loop and schemas are unchanged.

**New MCP Tools**
Two tools are added to `mcp_server.py`:
- `index_document` — chunks a file, embeds each piece, and saves them as `fact` records in Memory (FAISS-searchable).
- `search_knowledge` — executes a vector search over the indexed corpus and returns ranked chunks with provenance.

**Memory Service Update**
`MemoryItem` gains an optional `embedding` field. During reads, the agent first attempts a vector search, gracefully falling back to keyword overlap search if the gateway is unreachable or the index is empty.

**Cross-Process Consistency**
FAISS indices (`index.faiss`) and their parallel identifier lists (`index_ids.json`) are persisted to disk synchronously alongside `memory.json`. FAISS is reloaded from disk on every call so the main agent process instantly sees new chunks indexed by the MCP subprocess.

---

## Architectural Discipline and Best Practices

**Tool-Blindness in Perception**
Perception must decompose goals without knowing or naming the available tools. Pushing tool names into Perception causes unscalable context bloat. Tool selection guidance belongs strictly in Decision's system prompt and in each tool's docstring.

**Diagnostic Discipline**
When an agent role hallucinates, loops, or fails, resist the instinct to blindly patch its system prompt with a new rule. Instead, reconstruct exactly what the role saw in its rendered input. Failures are most frequently caused by missing fields or truncation in the upstream rendering layer, not by the LLM's logic.
