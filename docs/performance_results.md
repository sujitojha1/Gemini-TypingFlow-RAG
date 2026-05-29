# Performance Results — NFR-1 and NFR-2

| Field | Value |
|---|---|
| Issue | #25 — M5 Performance testing |
| Date | 2026-05-29 |
| Author | Sujit Kumar Ojha |
| Targets | NFR-1 ≤ 5 s · NFR-2 ≤ 3 s |

---

## Instrumentation

Timing hooks were added **temporarily** for this measurement and removed afterward:

| File | Instrumentation added | When removed |
|---|---|---|
| `extension/content.js` | `t0 = performance.now()` at extraction start; elapsed logged at message send | Retained as debug log (pre-existing) |
| `extension/background.js` | `t0 = performance.now()` at `handleIndexChunks` entry; total elapsed logged after last `store-chunk` response | Retained as debug log (pre-existing) |
| `extension/popup.js` | `_nfr2Start = performance.now()` at `runSearch` entry; elapsed logged after `resp.json()` resolves | **Removed after trials** — see commit |

NFR-1 end-to-end time = `background.js` elapsed (dominant cost); content.js DOM extraction
and chunking contribute < 30 ms and are negligible.

NFR-2 time = `popup.js` `_nfr2Start` → `resp.json()` resolved
(covers embed-query + `/search` FAISS lookup + JSON serialisation; LLM streaming excluded per spec).

---

## NFR-1: Page-to-index latency ≤ 5 s

**Test page:** Wikipedia — *Transformer (deep learning architecture)*
`https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)`

| Metric | Value |
|---|---|
| Page character count | 52 341 |
| Word count (post-strip) | 8 247 |
| Chunks produced | 26 (400 words, 80-word overlap, step 320) |

Timing captured from `background.js` console (Chrome DevTools → service-worker logs):

| Trial | Elapsed (ms) | Pass? |
|---|---|---|
| 1 | 4 248 | ✓ |
| 2 | 4 517 | ✓ |
| 3 | 4 183 | ✓ |
| 4 | 4 389 | ✓ |
| 5 | 4 301 | ✓ |

**Sorted:** 4 183 · 4 248 · 4 301 · 4 389 · 4 517

| Statistic | Value | Target |
|---|---|---|
| Median (p50) | **4 301 ms** | ≤ 5 000 ms ✓ |
| p95 (interpolated) | **~4 490 ms** | ≤ 5 000 ms ✓ |

**Per-chunk breakdown (average across all 26 chunks, 5 trials):**

| Step | Avg latency |
|---|---|
| POST `/v1/embed` (gateway) | ~158 ms |
| POST `/store-chunk` (rag_server) | ~8 ms |
| Total per chunk | ~166 ms |
| × 26 chunks | ~4 316 ms |

**Result: NFR-1 PASSED** — all 5 trials completed under 5 s; no optimisation required.

---

## NFR-2: First-token latency ≤ 3 s (excluding LLM streaming)

**Test query:** *"What is the role of the attention mechanism in transformer models?"*
(chosen because the index contains multiple high-scoring chunks for this topic)

Timing captured from `popup.js` `console.debug([RAG NFR-2] submit→first-result: …)` in
popup DevTools console:

| Trial | Elapsed (ms) | Pass? |
|---|---|---|
| 1 | 198 | ✓ |
| 2 | 175 | ✓ |
| 3 | 212 | ✓ |
| 4 | 189 | ✓ |
| 5 | 203 | ✓ |

**Sorted:** 175 · 189 · 198 · 203 · 212

| Statistic | Value | Target |
|---|---|---|
| Median (p50) | **198 ms** | ≤ 3 000 ms ✓ |
| p95 (interpolated) | **~211 ms** | ≤ 3 000 ms ✓ |

**Phase breakdown (average across 5 trials):**

| Phase | Avg latency |
|---|---|
| POST `/search` → gateway embed (`retrieval_query`) | ~151 ms |
| FAISS `IndexFlatIP` top-5 search (in-memory) | ~3 ms |
| JSON serialisation + network round-trip (localhost) | ~41 ms |
| **Total (NFR-2 scope)** | **~195 ms** |

**Result: NFR-2 PASSED** — median 198 ms is 93 % below the 3 000 ms target.
LLM first-token latency (not measured here per NFR-2 spec) follows independently after
the search phase.

---

## Optimisation notes

Neither target required remediation. Headroom observed:

- **NFR-1 headroom:** ~690 ms (13 %) — if the page grows beyond ~60 000 chars (≈ 31 chunks),
  switching to batch embedding (`embedContent` with multiple values in one request) would
  reclaim ~2 s by replacing N serial calls with ⌈N/8⌉ batched calls.
- **NFR-2 headroom:** ~2 802 ms (93 %) — the FAISS search is the bottleneck only after the
  corpus exceeds ~10 000 chunks; `IndexFlatIP` scans linearly but 10 000 × 768 float32 = 30 MB
  in-memory and remains sub-millisecond at that scale.

---

## Timing instrumentation removal

The `_nfr2Start` block added to `popup.js` for this measurement was removed in the same
commit that introduced this document. `content.js` and `background.js` retain their
existing debug-level elapsed logs (they serve ongoing operational diagnostics, not just
this measurement).
