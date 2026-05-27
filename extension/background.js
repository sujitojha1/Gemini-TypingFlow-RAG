/**
 * background.js — MV3 service worker for the RAG extension
 *
 * Receives INDEX_CHUNKS messages from content.js, embeds each chunk via
 * llm_gatewayV7, validates the embedding dimension, generates canonical
 * chunk IDs, and persists pre-computed vectors to rag_server.
 *
 * Pipeline (FR-2.1–FR-2.5, FR-3.4, DS-4):
 *   content.js  ──INDEX_CHUNKS──▶  background.js
 *                                       │
 *              POST /v1/embed ◀─── embed each chunk (FR-2.1–FR-2.2)
 *              localhost:8107             │  validate dim=768 (FR-2.5)
 *                                       │  generate chunk_<hash>_<idx> (DS-4)
 *              POST /store-chunk ◀──────┘
 *              localhost:8108
 *
 * Error handling:
 *   - HTTP 429 from gateway → exponential backoff 1s / 2s / 4s (FR-2.3)
 *   - Up to MAX_RETRIES=3 attempts per chunk; failed chunks are logged and
 *     skipped without crashing the service worker (FR-2.4)
 *   - Gateway unreachable  → sets rag_error_gateway_unreachable in storage
 *   - rag_server unreachable → sets rag_error_rag_server_unreachable in storage
 *     (NFR-6 — popup.js reads these flags to surface errors)
 *
 * Privacy: all network traffic goes only to localhost:8107 and localhost:8108
 *   (NFR-3 — no external hosts; verified by host_permissions in manifest.json)
 */

"use strict";

// ── Config ────────────────────────────────────────────────────────────────────

const GATEWAY_URL   = "http://127.0.0.1:8107";
const RAG_URL       = "http://127.0.0.1:8108";
const EMBED_DIM     = 768;
const MAX_RETRIES   = 3;
const BACKOFF_MS    = 1000; // base for exponential backoff (1s, 2s, 4s)

// ── Utilities ─────────────────────────────────────────────────────────────────

/**
 * djb2-variant URL hash → 8-char lowercase hex string.
 * Used to build canonical chunk IDs: chunk_<hash>_<index> (DS-4).
 */
function urlHash(url) {
  let h = 5381;
  for (let i = 0; i < url.length; i++) {
    h = (((h << 5) + h) ^ url.charCodeAt(i)) >>> 0; // unsigned 32-bit
  }
  return h.toString(16).padStart(8, "0");
}

/** Returns a Promise that resolves after *ms* milliseconds. */
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ── chrome.storage helpers ────────────────────────────────────────────────────

/** Set a named error flag in storage (popup reads these for status display). */
function setErrorFlag(key) {
  return new Promise((resolve) => {
    chrome.storage.local.set({ [`rag_error_${key}`]: Date.now() }, resolve);
  });
}

/** Clear all error flags after a successful pipeline run. */
function clearErrorFlags() {
  return new Promise((resolve) => {
    chrome.storage.local.remove(
      ["rag_error_gateway_unreachable", "rag_error_rag_server_unreachable"],
      resolve
    );
  });
}

/** Increment the running indexed-page counter (popup shows this). */
function incrementPageCount() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["rag_indexed_page_count"], (result) => {
      const next = (result.rag_indexed_page_count || 0) + 1;
      chrome.storage.local.set({ rag_indexed_page_count: next }, resolve);
    });
  });
}

// ── Gateway embed with retry + backoff ───────────────────────────────────────

/**
 * Call POST /v1/embed on the gateway.
 * Retries up to MAX_RETRIES times with exponential backoff on 429 or network
 * errors (FR-2.3, FR-2.4).
 *
 * Returns the 768-float array, or null if all attempts fail / dim is wrong.
 */
async function embedWithRetry(text) {
  let lastError = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      const wait = BACKOFF_MS * Math.pow(2, attempt - 1); // 1s, 2s
      console.debug(`[RAG bg] embed retry ${attempt}/${MAX_RETRIES - 1} after ${wait}ms`);
      await sleep(wait);
    }

    try {
      const resp = await fetch(`${GATEWAY_URL}/v1/embed`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ text, task_type: "retrieval_document" }),
      });

      if (resp.status === 429) {
        console.warn(`[RAG bg] gateway 429 on attempt ${attempt + 1}`);
        lastError = new Error("Rate-limited (429)");
        continue; // backoff and retry
      }

      if (!resp.ok) {
        lastError = new Error(`Gateway HTTP ${resp.status}`);
        console.warn(`[RAG bg] embed attempt ${attempt + 1}: ${lastError.message}`);
        continue;
      }

      const data = await resp.json();
      const emb  = data.embedding;

      // Validate dimension (FR-2.5)
      if (!Array.isArray(emb) || emb.length !== EMBED_DIM) {
        const msg = `Expected ${EMBED_DIM}-dim embedding, got ${emb?.length ?? "null"}`;
        console.error(`[RAG bg] ${msg}`);
        return null; // dimension mismatch — retrying won't help
      }

      return emb;

    } catch (err) {
      lastError = err;
      console.warn(`[RAG bg] embed attempt ${attempt + 1} threw: ${err.message}`);
    }
  }

  console.error(`[RAG bg] embedding failed after ${MAX_RETRIES} attempts: ${lastError?.message}`);
  await setErrorFlag("gateway_unreachable");
  return null;
}

// ── rag_server store-chunk call ───────────────────────────────────────────────

/**
 * POST a pre-computed embedding to rag_server's /store-chunk endpoint.
 * Returns true on success; logs error and sets error flag on failure (NFR-6).
 */
async function storeChunk(payload) {
  try {
    const resp = await fetch(`${RAG_URL}/store-chunk`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!resp.ok) {
      const body = await resp.text().catch(() => `HTTP ${resp.status}`);
      console.error(`[RAG bg] store-chunk failed: ${body}`);
      return false;
    }

    return true;
  } catch (err) {
    console.error(`[RAG bg] rag_server unreachable: ${err.message}`);
    await setErrorFlag("rag_server_unreachable");
    return false;
  }
}

// ── Main INDEX_CHUNKS pipeline ────────────────────────────────────────────────

/**
 * Process one page's chunk batch.
 *
 * For each chunk:
 *   1. Embed via gateway (with retry+backoff)
 *   2. Validate dim=768
 *   3. Generate chunk_id = chunk_<url_hash>_<chunk_index>
 *   4. POST to rag_server /store-chunk
 *
 * Failed chunks are counted and logged; they never crash the service worker.
 */
async function handleIndexChunks({ chunks, url, title, reindex }) {
  if (!Array.isArray(chunks) || chunks.length === 0) return;

  const t0    = performance.now();
  const hash  = urlHash(url);
  let success = 0;
  let failed  = 0;

  console.log(`[RAG bg] ${url} — ${chunks.length} chunk(s), reindex=${reindex}`);

  for (const chunk of chunks) {
    // Step 1–2: embed + validate
    const embedding = await embedWithRetry(chunk.text);
    if (embedding === null) {
      failed++;
      continue;
    }

    // Step 3: canonical chunk ID (DS-4)
    const chunkId = `chunk_${hash}_${chunk.chunk_index}`;

    // Step 4: persist to rag_server
    const ok = await storeChunk({
      chunk_id:  chunkId,
      text:      chunk.text,
      embedding,
      metadata: {
        url,
        title,
        chunk_index:   chunk.chunk_index,
        total_chunks:  chunk.total_chunks,
        timestamp_iso: chunk.timestamp_iso,
      },
      // Signal re-index only on the first chunk to avoid redundant rebuilds
      reindex: reindex && chunk.chunk_index === 0,
    });

    if (ok) success++;
    else failed++;
  }

  const elapsed = (performance.now() - t0).toFixed(0);
  console.log(
    `[RAG bg] done — ${success} stored, ${failed} failed, ${elapsed}ms`
  );

  if (success > 0) {
    await clearErrorFlags();
    await incrementPageCount();
  }
}

// ── Message listener ──────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((message, _sender, _sendResponse) => {
  if (message?.type === "INDEX_CHUNKS") {
    handleIndexChunks(message).catch((err) => {
      console.error("[RAG bg] unhandled pipeline error:", err);
    });
  }
  // Return false — no async sendResponse needed
  return false;
});

console.log("[RAG bg] service worker loaded");
