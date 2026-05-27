/**
 * content.js — DOM scraper and sliding-window chunker
 *
 * Injected into every HTTP/HTTPS page at document_idle.
 * Extracts visible body text, discards pages below the minimum word
 * threshold, chunks the remainder with a sliding window, attaches metadata
 * to each chunk, and sends the batch to the background service worker for
 * indexing via the RAG server.
 *
 * Pipeline (FR-1.1 – FR-1.6):
 *   1. Extract visible text — strip script/style/nav/header/footer/aside
 *   2. Skip pages with < 200 words (FR-1.5)
 *   3. Sliding-window chunker: 400 words, 80-word overlap (FR-1.3)
 *   4. Attach metadata per chunk: {url, title, chunk_index, total_chunks,
 *      timestamp_iso} (FR-1.4)
 *   5. Dedup via chrome.storage.local — re-visits set reindex=true (FR-1.6)
 *   6. Send {type: "INDEX_CHUNKS", chunks, url, title, reindex} to background
 *
 * Performance target: full extract + chunk ≤ 1 s for 50 000-char pages (NFR-1).
 * The synchronous extract+chunk path runs in well under 50 ms at that size;
 * the only async work is the storage lookup before sending the message.
 */

(() => {
  "use strict";

  // ── Constants ───────────────────────────────────────────────────────────────

  const CHUNK_WORDS   = 400;          // words per chunk window
  const OVERLAP_WORDS = 80;           // overlap between consecutive windows
  const MIN_WORDS     = 200;          // minimum words to bother indexing
  const STORAGE_KEY   = "rag_indexed_urls"; // key inside chrome.storage.local

  // Tags whose entire subtree should be removed before text extraction.
  const STRIP_TAGS = [
    "script", "style", "noscript",
    "nav", "header", "footer", "aside",
    "iframe", "svg", "canvas", "figure", "form",
  ];

  // ── DOM text extraction ──────────────────────────────────────────────────────

  /**
   * Clone the body, strip noisy subtrees, then harvest innerText.
   * Collapsing whitespace makes word-count and chunking consistent regardless
   * of indentation in the original markup.
   */
  function extractText() {
    const clone = document.body.cloneNode(true);

    STRIP_TAGS.forEach((tag) => {
      clone.querySelectorAll(tag).forEach((el) => el.remove());
    });

    const raw = clone.innerText || clone.textContent || "";
    // Collapse all whitespace (spaces, tabs, newlines) into single spaces
    return raw.replace(/[ \t\r\n]+/g, " ").trim();
  }

  // ── Sliding-window word chunker ──────────────────────────────────────────────

  /**
   * Split *text* into overlapping windows of CHUNK_WORDS words.
   * Advance by (CHUNK_WORDS − OVERLAP_WORDS) words per step so each window
   * shares OVERLAP_WORDS words with its neighbour, preserving context at
   * boundaries.
   *
   * @param {string} text  — pre-extracted, whitespace-collapsed page text
   * @returns {string[]}   — array of chunk strings
   */
  function chunkText(text) {
    const words  = text.split(" ");
    const step   = CHUNK_WORDS - OVERLAP_WORDS;   // 320 words per step
    const chunks = [];
    let start = 0;

    while (start < words.length) {
      const end = Math.min(start + CHUNK_WORDS, words.length);
      chunks.push(words.slice(start, end).join(" "));
      if (end >= words.length) break;
      start += step;
    }

    return chunks;
  }

  // ── URL dedup via chrome.storage.local ────────────────────────────────────────

  /**
   * Look up *url* in chrome.storage.local.
   * If not seen before, record it and return false (new URL → reindex=false).
   * If already present, return true (revisit → reindex=true).
   *
   * @param {string} url
   * @returns {Promise<boolean>} — true if this URL was previously indexed
   */
  function checkAndMarkUrl(url) {
    return new Promise((resolve) => {
      chrome.storage.local.get([STORAGE_KEY], (result) => {
        const indexed = result[STORAGE_KEY] || {};
        const seen    = Object.prototype.hasOwnProperty.call(indexed, url);
        if (!seen) {
          indexed[url] = new Date().toISOString();
          chrome.storage.local.set({ [STORAGE_KEY]: indexed });
        }
        resolve(seen);
      });
    });
  }

  // ── Main pipeline ────────────────────────────────────────────────────────────

  async function run() {
    const url   = location.href;
    const title = document.title || "";

    // Only process real web pages
    if (!url.startsWith("http://") && !url.startsWith("https://")) return;

    const t0   = performance.now();

    // Step 1: extract
    const text  = extractText();
    const words = text.split(" ").filter((w) => w.length > 0);

    // Step 2: word-count gate (FR-1.5)
    if (words.length < MIN_WORDS) {
      console.debug(
        `[RAG content] skip — only ${words.length} words (< ${MIN_WORDS}): ${url}`
      );
      return;
    }

    // Step 3: chunk
    const rawChunks   = chunkText(words.join(" "));
    const totalChunks = rawChunks.length;
    const timestamp   = new Date().toISOString();

    // Step 4: attach metadata (FR-1.4)
    const chunks = rawChunks.map((text, idx) => ({
      text,
      url,
      title,
      chunk_index:   idx,
      total_chunks:  totalChunks,
      timestamp_iso: timestamp,
    }));

    const elapsedSync = performance.now() - t0;

    // Step 5: dedup check (FR-1.6) — async, but only a storage read
    const reindex = await checkAndMarkUrl(url);

    const elapsed = performance.now() - t0;
    console.debug(
      `[RAG content] ${url} → ${totalChunks} chunk(s) ` +
      `(${elapsed.toFixed(1)} ms, reindex=${reindex}, ` +
      `sync=${elapsedSync.toFixed(1)} ms)`
    );

    // Step 6: send to background (FR-1.6, issue #19 provides the listener)
    chrome.runtime.sendMessage(
      {
        type:    "INDEX_CHUNKS",
        chunks,
        url,
        title,
        reindex,
      },
      // Callback silences the "no listener" console error before background.js exists
      (_response) => { void chrome.runtime.lastError; }
    );
  }

  // Run after the DOM is fully parsed and rendered (document_idle in manifest)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }

})();
