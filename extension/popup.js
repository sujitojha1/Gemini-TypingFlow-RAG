/* popup.js — RAG Search extension */

const API = "http://127.0.0.1:8108";

// ── Markdown renderer ─────────────────────────────────────────────────────────
// Converts the LLM answer (which uses GitHub-flavoured markdown) to safe HTML.
// Pipeline: HTML-escape the raw string first, then apply markdown transforms
// so user-supplied content can never inject tags via the answer text.

function applyInline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+?)\*/g,  "<em>$1</em>")
    .replace(/`([^`\n]+?)`/g,    "<code>$1</code>");
}

function renderMarkdown(raw) {
  const esc = raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const lines  = esc.split("\n");
  const parts  = [];
  let listType = null;

  const closeList = () => {
    if (listType) { parts.push(`</${listType}>`); listType = null; }
  };

  for (const line of lines) {
    const t = line.trimEnd();

    // Numbered list: "1. " / "2. " …
    const olM = t.match(/^\s*\d+\.\s+(.*)/);
    if (olM) {
      if (listType !== "ol") { closeList(); parts.push("<ol>"); listType = "ol"; }
      parts.push(`<li>${applyInline(olM[1])}</li>`);
      continue;
    }

    // Bullet list: "* " / "- " / "• "
    const ulM = t.match(/^\s*[\*\-•]\s+(.*)/);
    if (ulM) {
      if (listType !== "ul") { closeList(); parts.push("<ul>"); listType = "ul"; }
      parts.push(`<li>${applyInline(ulM[1])}</li>`);
      continue;
    }

    // Empty line — paragraph break
    if (t.trim() === "") { closeList(); parts.push("<br>"); continue; }

    // Regular text — close any open list, emit paragraph
    closeList();
    parts.push(`<p>${applyInline(t)}</p>`);
  }

  closeList();
  return parts.join("");
}

// ── Source info extractor ─────────────────────────────────────────────────────
// Parses a source object from /rag into { title, url, domain, preview }.

function extractSourceInfo(source) {
  const raw        = source.source     || "";
  const descriptor = source.descriptor || "";

  // URL from descriptor: "URL: https://..."
  const urlMatch   = descriptor.match(/URL:\s*(https?:\/\/[^\s\]]+)/);
  // Title from descriptor: "Title: Some Title"
  const titleMatch = descriptor.match(/Title:\s*([^\n\[]{3,80})/);
  // Preview text: content after the closing bracket "[sandbox:...] <here>"
  const prevMatch  = descriptor.match(/\]\s+(.+)/);

  // URL: descriptor's explicit "URL: ..." wins; fall back to source.source if it is a URL
  // (browser-indexed pages store the page URL directly in source.source)
  const urlFromDesc   = urlMatch ? urlMatch[1] : null;
  const urlFromSource = /^https?:\/\//.test(raw) ? raw : null;
  const url           = urlFromDesc || urlFromSource;

  // Domain (used as chip label)
  let domain = "";
  if (url) {
    try { domain = new URL(url).hostname.replace(/^www\./, ""); } catch {}
  }

  // Human-readable name
  let name = raw;
  if (raw.startsWith("sandbox:")) {
    // Corpus file: "sandbox:Foo_Bar_Baz_ext.txt" → "Foo Bar Baz"
    name = raw
      .replace(/^sandbox:/, "")
      .replace(/_ext\.txt$/, "")
      .replace(/_/g, " ")
      .replace(/\s{2,}/g, " ")
      .trim();
    if (name.length > 48) name = name.slice(0, 48) + "…";
  } else if (urlFromSource) {
    // Browser-indexed page: derive a readable label from the URL path
    try {
      const u    = new URL(raw);
      const slug = u.pathname.replace(/\/$/, "").split("/").pop() || "";
      name = (slug ? slug.replace(/[-_]/g, " ") : domain).slice(0, 48);
    } catch {
      name = domain || raw.slice(0, 48);
    }
  }

  const title   = titleMatch ? titleMatch[1].trim() : (domain || name);
  const preview = prevMatch  ? prevMatch[1].trim().slice(0, 160) : "";

  return { title, url, domain, preview, name };
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setStatus(type, msg) {
  const el = document.getElementById("status");
  el.className = `status show ${type}`;
  el.innerHTML =
    type === "loading"
      ? `<div class="spinner"></div> ${msg}`
      : `<span>${type === "success" ? "✓" : "✗"}</span> ${msg}`;
}

function clearStatus() {
  document.getElementById("status").className = "status";
}

// ── Index status badge (FR-4.7) ───────────────────────────────────────────────

async function loadIndexStatus() {
  try {
    const resp = await fetch(`${API}/status`);
    if (!resp.ok) return;
    const data = await resp.json();
    const badge = document.getElementById("index-badge");
    const n = data.page_count || 0;
    badge.textContent = `${n} page${n !== 1 ? "s" : ""} indexed`;
    badge.classList.add("show");
  } catch { /* gateway unreachable — badge stays hidden */ }
}

// ── Background error flags (NFR-6) ────────────────────────────────────────────

function checkErrorFlags() {
  chrome.storage.local.get(
    ["rag_error_gateway_unreachable", "rag_error_rag_server_unreachable"],
    (result) => {
      const cutoff = Date.now() - 5 * 60 * 1000;
      if ((result.rag_error_gateway_unreachable || 0) > cutoff) {
        setStatus("error", "Gateway (port 8107) unreachable — indexing may be failing.");
      } else if ((result.rag_error_rag_server_unreachable || 0) > cutoff) {
        setStatus("error", "RAG server (port 8108) unreachable — indexing may be failing.");
      }
    }
  );
}

// ── Page info ────────────────────────────────────────────────────────────────

async function loadPageInfo() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  document.getElementById("page-title").textContent = tab.title || tab.url || "Unknown page";
  const fav = document.getElementById("favicon");
  if (tab.favIconUrl) { fav.src = tab.favIconUrl; fav.style.display = "block"; }
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

    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => ({ text: document.body.innerText || "", title: document.title || "", url: location.href }),
    });

    if (!result?.text?.trim()) throw new Error("Page has no readable text");

    setStatus("loading", `Indexing ${result.title.slice(0, 32) || "page"}…`);

    const resp = await fetch(`${API}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: result.text, title: result.title, url: result.url }),
    });

    if (!resp.ok) throw new Error((await resp.text()) || `Server error ${resp.status}`);

    const data = await resp.json();
    setStatus("success", `Indexed ${data.chunks_indexed} chunks ✓`);
    setTimeout(clearStatus, 3000);
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
  const q = document.getElementById("search-input").value.trim();
  if (!q) { renderEmpty(); return; }
  runSearch(q);
}

document.getElementById("btn-search").addEventListener("click", triggerSearch);
document.getElementById("search-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") triggerSearch();
});

async function runSearch(query) {
  const resultsEl = document.getElementById("results");
  const btn       = document.getElementById("btn-search");

  btn.disabled = true;
  btn.innerHTML = `<div class="spinner" style="width:11px;height:11px;border-width:2px;"></div>`;
  resultsEl.innerHTML = `
    <div class="empty-state">
      <div class="spinner" style="margin:0 auto 8px;border-top-color:var(--accent);border-color:var(--border);"></div>
      Thinking…
    </div>`;

  try {
    const resp = await fetch(`${API}/rag`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k: 5 }),
    });
    if (!resp.ok) throw new Error(`Server error ${resp.status}`);
    renderRagAnswer(await resp.json());
  } catch (err) {
    resultsEl.innerHTML = `
      <div class="empty-state" style="color:var(--error);">
        <span>⚠</span>${escHtml(err.message || "Search failed")}
      </div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

// ── RAG answer renderer ───────────────────────────────────────────────────────

// Tracks which chip is currently expanded so we can toggle it off.
let _activeChip = null;

function renderRagAnswer(data) {
  const el         = document.getElementById("results");
  const { confidence, answer, sources } = data;

  // FR-4.6: no relevant content
  if (confidence === "none") {
    el.innerHTML = `
      <div class="empty-state">
        <span>🔍</span>No relevant indexed content found.
      </div>`;
    return;
  }

  // Strip medium-confidence disclaimer appended by /rag endpoint
  let llmAnswer  = answer || "";
  let disclaimer = "";
  if (confidence === "medium") {
    const splitAt = llmAnswer.lastIndexOf("\n\n_(Confidence:");
    if (splitAt !== -1) {
      llmAnswer  = llmAnswer.slice(0, splitAt);
      disclaimer = "Low confidence — indexed content is only partially relevant.";
    }
  }

  // ── Build source chips + detail panels ──────────────────────────────────────
  const srcs = (sources || []).map((s, i) => {
    const info    = extractSourceInfo(s);
    const chipId  = `src-chip-${i}`;
    const detailId= `src-detail-${i}`;
    const label   = info.domain || info.name.split(" ").slice(0, 3).join(" ");

    const chipHtml = `
      <button class="source-chip" id="${chipId}" data-detail="${detailId}" aria-expanded="false">
        <span class="chip-num">${i + 1}</span>
        ${escHtml(label)}
      </button>`;

    const linkHtml = info.url
      ? `<a class="source-link" href="${escHtml(info.url)}" target="_blank" rel="noopener">
           ↗ ${escHtml(info.domain || info.url.slice(0, 50))}
         </a>`
      : `<span style="font-size:10.5px;color:var(--muted);">📄 local corpus</span>`;

    const detailHtml = `
      <div class="source-detail" id="${detailId}">
        <div class="source-detail-title">
          <span class="src-num">${i + 1}</span>
          ${escHtml(info.title)}
        </div>
        ${info.preview ? `<div class="source-detail-preview">${escHtml(info.preview)}</div>` : ""}
        ${linkHtml}
      </div>`;

    return { chipHtml, detailHtml };
  });

  const chipsRow   = srcs.map(s => s.chipHtml).join("");
  const detailRows = srcs.map(s => s.detailHtml).join("");

  const sourcesBlock = srcs.length ? `
    <div class="sources-header">
      <span class="sources-title">Sources</span>
    </div>
    <div class="source-chips">${chipsRow}</div>
    ${detailRows}
  ` : "";

  el.innerHTML = `
    <div class="answer-card">
      <div class="answer-header">
        <span class="answer-label">Answer</span>
        <span class="confidence-pill ${escHtml(confidence)}">${escHtml(confidence)}</span>
      </div>
      <div class="answer-text">${renderMarkdown(llmAnswer)}</div>
      ${disclaimer ? `<div class="answer-disclaimer">⚠ ${escHtml(disclaimer)}</div>` : ""}
      ${sourcesBlock}
    </div>
  `;

  // ── Wire source chip click handlers ─────────────────────────────────────────
  _activeChip = null;
  el.querySelectorAll(".source-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const detailId = chip.dataset.detail;
      const detail   = document.getElementById(detailId);
      const isOpen   = chip.classList.contains("active");

      // Close any previously open chip
      if (_activeChip && _activeChip !== chip) {
        _activeChip.classList.remove("active");
        _activeChip.setAttribute("aria-expanded", "false");
        const prevDetail = document.getElementById(_activeChip.dataset.detail);
        if (prevDetail) prevDetail.classList.remove("show");
      }

      // Toggle this chip
      if (isOpen) {
        chip.classList.remove("active");
        chip.setAttribute("aria-expanded", "false");
        detail.classList.remove("show");
        _activeChip = null;
      } else {
        chip.classList.add("active");
        chip.setAttribute("aria-expanded", "true");
        detail.classList.add("show");
        _activeChip = chip;
      }
    });
  });
}

function renderEmpty() {
  document.getElementById("results").innerHTML = `
    <div class="empty-state">
      <span>◈</span>
      Index pages, then ask questions
    </div>`;
}

// ── Init ─────────────────────────────────────────────────────────────────────

loadPageInfo();
loadIndexStatus();
checkErrorFlags();
