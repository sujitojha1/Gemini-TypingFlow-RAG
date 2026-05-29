/* popup.js — RAG Search extension */

const API = "http://127.0.0.1:8108";

// ── Utilities ────────────────────────────────────────────────────────────────

function setStatus(type, msg) {
  const el = document.getElementById("status");
  el.className = `status show ${type}`;
  el.innerHTML =
    type === "loading"
      ? `<div class="spinner"></div> ${msg}`
      : `<span>${type === "success" ? "✓" : "✗"}</span> ${msg}`;
}

function clearStatus() {
  const el = document.getElementById("status");
  el.className = "status";
}

function shortenSource(src) {
  try {
    const url = new URL(src);
    return url.hostname + (url.pathname.length > 1 ? url.pathname.slice(0, 28) : "");
  } catch {
    return src.length > 40 ? src.slice(0, 40) + "…" : src;
  }
}

// ── Index status badge (FR-4.7) ───────────────────────────────────────────────
// Polls GET /status on startup and after each successful index operation.

async function loadIndexStatus() {
  try {
    const resp = await fetch(`${API}/status`);
    if (!resp.ok) return;
    const data = await resp.json();
    const badge = document.getElementById("index-badge");
    const n = data.page_count || 0;
    badge.textContent = `${n} page${n !== 1 ? "s" : ""} indexed`;
    badge.classList.add("show");
  } catch {
    // rag_server unreachable — badge stays hidden; error flag check handles warning
  }
}

// ── Background error flags (NFR-6) ────────────────────────────────────────────
// background.js sets rag_error_* in chrome.storage when the gateway or rag_server
// are unreachable. Show a visible warning in the popup so failures are never silent.

function checkErrorFlags() {
  chrome.storage.local.get(
    ["rag_error_gateway_unreachable", "rag_error_rag_server_unreachable"],
    (result) => {
      const cutoff = Date.now() - 5 * 60 * 1000; // only warn for errors in last 5 min
      if ((result.rag_error_gateway_unreachable || 0) > cutoff) {
        setStatus("error", "Gateway (port 8107) unreachable — background indexing may be failing.");
      } else if ((result.rag_error_rag_server_unreachable || 0) > cutoff) {
        setStatus("error", "RAG server (port 8108) unreachable — background indexing may be failing.");
      }
    }
  );
}

// ── Page info ────────────────────────────────────────────────────────────────

async function loadPageInfo() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  const titleEl = document.getElementById("page-title");
  const faviconEl = document.getElementById("favicon");

  titleEl.textContent = tab.title || tab.url || "Unknown page";
  if (tab.favIconUrl) {
    faviconEl.src = tab.favIconUrl;
    faviconEl.style.display = "block";
  } else {
    faviconEl.style.display = "none";
  }
}

// ── Index button ─────────────────────────────────────────────────────────────

document.getElementById("btn-index").addEventListener("click", async () => {
  const btn = document.getElementById("btn-index");
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner"></div> Indexing…`;
  setStatus("loading", "Extracting page content…");

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) throw new Error("No active tab");

    // Extract text + metadata from the page
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => ({
        text: document.body.innerText || "",
        title: document.title || "",
        url: location.href,
      }),
    });

    if (!result?.text?.trim()) {
      throw new Error("Page has no readable text");
    }

    setStatus("loading", `Indexing ${result.title.slice(0, 32) || "page"}…`);

    const resp = await fetch(`${API}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: result.text,
        title: result.title,
        url: result.url,
      }),
    });

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    setStatus("success", `Indexed ${data.chunks_indexed} chunks ✓`);
    setTimeout(clearStatus, 3000);

    // Refresh page-count badge after a successful index
    loadIndexStatus();

  } catch (err) {
    setStatus("error", err.message || "Failed to index");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>⬆</span> Index This Page`;
  }
});

// ── Search ───────────────────────────────────────────────────────────────────

function triggerSearch() {
  const query = document.getElementById("search-input").value.trim();
  if (!query) { renderEmpty(); return; }
  runSearch(query);
}

document.getElementById("btn-search").addEventListener("click", triggerSearch);

document.getElementById("search-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") triggerSearch();
});

// FR-4.2–4.6: call POST /rag which handles embed (retrieval_query) + search +
// confidence gate + LLM answer assembly server-side.
async function runSearch(query) {
  const resultsEl = document.getElementById("results");
  const btn = document.getElementById("btn-search");

  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:11px;height:11px;border-width:2px;"></div>`;

  resultsEl.innerHTML = `
    <div class="empty-state">
      <div class="spinner" style="margin: 0 auto 8px; border-top-color: var(--accent-lt); border-color: var(--border);"></div>
      Searching…
    </div>`;

  try {
    // POST /rag: embeds query (retrieval_query), searches FAISS, applies 0.30/0.70
    // confidence gate, assembles RAG prompt, and calls the LLM — all server-side.
    const resp = await fetch(`${API}/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5 }),
    });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);

    const data = await resp.json();
    renderRagAnswer(data, query);

  } catch (err) {
    resultsEl.innerHTML = `
      <div class="empty-state" style="color: var(--error);">
        <span>⚠</span>${err.message || "Search failed"}
      </div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Search";
  }
}

// FR-4.5/4.6: render LLM answer with confidence label + source cards,
// or "No relevant indexed content found" when max_score < 0.30 (FR-4.6).
function renderRagAnswer(data, query) {
  const el = document.getElementById("results");
  const { confidence, answer, sources } = data;

  // FR-4.6: confidence=none means all scores < 0.30 — no LLM was called
  if (confidence === "none") {
    el.innerHTML = `
      <div class="empty-state">
        <span>🔍</span>No relevant indexed content found.
      </div>`;
    return;
  }

  // Split LLM answer from the medium-confidence disclaimer appended by /rag
  let llmAnswer = answer || "";
  let disclaimer = "";
  if (confidence === "medium") {
    const splitAt = llmAnswer.lastIndexOf("\n\n_(Confidence:");
    if (splitAt !== -1) {
      llmAnswer = llmAnswer.slice(0, splitAt);
      disclaimer = "Low confidence — indexed content is only partially relevant.";
    }
  }

  // Source cards (FR-4.5)
  const sourcesHtml = (sources || []).map((s) => {
    const raw = s.source || s.descriptor || "";
    const src = shortenSource(raw);
    return `
      <div class="result-card">
        <div class="result-source">
          <div class="result-dot"></div>
          <span>${escHtml(src)}</span>
        </div>
      </div>`;
  }).join("");

  el.innerHTML = `
    <div class="answer-card">
      <div class="answer-header">
        <span class="answer-label">Answer</span>
        <span class="confidence-pill ${escHtml(confidence)}">${escHtml(confidence)}</span>
      </div>
      <div class="answer-text">${escHtml(llmAnswer)}</div>
      ${disclaimer ? `<div class="answer-disclaimer">${escHtml(disclaimer)}</div>` : ""}
    </div>
    ${sourcesHtml ? `<div class="sources-label">Sources</div>${sourcesHtml}` : ""}
  `;
}

function renderEmpty() {
  document.getElementById("results").innerHTML = `
    <div class="empty-state">
      <span>📚</span>
      Index pages and search them here
    </div>`;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function highlight(html, query) {
  const words = query.split(/\s+/).filter((w) => w.length > 2);
  words.forEach((word) => {
    const re = new RegExp(`(${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    html = html.replace(re, `<mark style="background:rgba(124,58,237,0.3);color:var(--accent-lt);border-radius:3px;padding:0 2px;">$1</mark>`);
  });
  return html;
}

// ── Init ─────────────────────────────────────────────────────────────────────

loadPageInfo();
loadIndexStatus();   // FR-4.7: show live page count on open
checkErrorFlags();   // NFR-6: surface background indexing errors immediately
