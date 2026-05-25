/* popup.js — RAG Search extension */

const API = "http://127.0.0.1:8108";
let SEARCH_TIMER = null;

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

    // Auto-clear after 3s
    setTimeout(clearStatus, 3000);

  } catch (err) {
    setStatus("error", err.message || "Failed to index");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>⬆</span> Index This Page`;
  }
});

// ── Search ───────────────────────────────────────────────────────────────────

document.getElementById("search-input").addEventListener("input", (e) => {
  clearTimeout(SEARCH_TIMER);
  const query = e.target.value.trim();

  if (!query) {
    renderEmpty();
    return;
  }

  SEARCH_TIMER = setTimeout(() => runSearch(query), 320);
});

async function runSearch(query) {
  const resultsEl = document.getElementById("results");
  resultsEl.innerHTML = `
    <div class="empty-state">
      <div class="spinner" style="margin: 0 auto 8px; border-top-color: var(--accent-lt); border-color: var(--border);"></div>
      Searching…
    </div>`;

  try {
    const resp = await fetch(`${API}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5 }),
    });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);

    const data = await resp.json();
    renderResults(data.results, query);

  } catch (err) {
    resultsEl.innerHTML = `
      <div class="empty-state" style="color: var(--error);">
        <span>⚠</span>${err.message || "Search failed"}
      </div>`;
  }
}

function renderResults(results, query) {
  const el = document.getElementById("results");

  if (!results || results.length === 0) {
    el.innerHTML = `
      <div class="empty-state">
        <span>🔍</span>No results for "<em>${escHtml(query)}</em>"
      </div>`;
    return;
  }

  el.innerHTML = results.map((r) => {
    const src = shortenSource(r.source);
    const chunk = r.chunk_index != null
      ? ` · chunk ${r.chunk_index + 1}/${r.total_chunks}`
      : "";
    const preview = highlight(escHtml(r.preview.trim()), query);

    return `
      <div class="result-card">
        <div class="result-source">
          <div class="result-dot"></div>
          <span>${escHtml(src)}${escHtml(chunk)}</span>
        </div>
        <div class="result-preview">${preview}</div>
      </div>`;
  }).join("");
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
  // Highlight each query word in the preview text
  const words = query.split(/\s+/).filter((w) => w.length > 2);
  words.forEach((word) => {
    const re = new RegExp(`(${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    html = html.replace(re, `<mark style="background:rgba(124,58,237,0.3);color:var(--accent-lt);border-radius:3px;padding:0 2px;">$1</mark>`);
  });
  return html;
}

// ── Init ─────────────────────────────────────────────────────────────────────

loadPageInfo();
