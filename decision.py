"""Decision: one LLM call per turn.

Given the current goal, the relevant memory hits (descriptors only), the
recent history, and optionally the raw bytes of an artifact Perception
attached to this goal, the model picks ONE of:

  (a) answer in plain text — the answer may itself be summarisation,
      extraction, comparison, translation, or any other semantic work the
      LLM does on the attached content;
  (b) call exactly one MCP tool from the available tool list.

There is no taxonomy of "operation kinds". The model decides what it is
doing. Decision just routes the dispatch.
"""

from __future__ import annotations

import json

from gateway import LLM, ensure_gateway
from schemas import DecisionOutput, Goal, MemoryItem, ToolCall


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the Decision layer of an agent.
Inputs you receive: ONE current goal, the relevant memory snippets,
recent history, and optionally the raw bytes of one attached artifact.

Choose EXACTLY ONE response:
  (a) Reply with the final answer to this goal as plain text. If the
      goal asks you to summarise, extract, compare, or transform the
      attached content, do that work inside your reply. DO NOT use this
      option for fetch/read goals.
  (b) Call exactly ONE tool from the available MCP tools when you need
      external work (fetching, file ops, time, currency, web search).
      If the goal asks to fetch/read a specific result, you MUST call
      a tool (e.g. fetch_url) using the URL from the attached artifacts.

━━━ RULES ━━━

RULE 1 — Never narrate. Answer or call a tool, never both.

RULE 2 — Never invent a tool that is not in the tool list.

RULE 3 — Answer directly when satisfied
  • If the goal is already satisfied by the memory hits + history, answer
    directly without calling a tool. HOWEVER, if the goal explicitly asks
    to 'fetch', 'read', or 'download' a URL/result, it is NOT satisfied
    until you actually call the `fetch_url` tool to download the full page.
    A search snippet from an attached artifact is NOT the full page.

RULE 4 — Follow fetch/read instructions
  • If the goal instructs you to fetch, read, or download something (e.g.,
    'Fetch the first result'), you MUST call the appropriate tool
    (like fetch_url) using the URL or path found in the attached
    artifacts or memory. Do NOT just quote the snippet as an answer.

RULE 5 — Artifact handles are NOT for tools
  • Artifact handles (strings starting with `art:`) are NOT file paths,
    URLs, or tool arguments. NEVER pass an `art:...` value to read_file,
    list_dir, fetch_url, or ANY other tool. If a goal needs the bytes of
    an artifact, those bytes will already appear in the ATTACHED
    ARTIFACTS section of your input — answer directly from that text.
    WRONG:  read_file({"path": "art:abc1234"})
    WRONG:  fetch_url({"url": "art:abc1234"})
    RIGHT:  read the bytes already in ATTACHED ARTIFACTS and answer.

RULE 6 — Sandbox tools
  • read_file and list_dir operate on the local sandbox/ directory, not
    artifacts. Only call them when the user has asked you to read/list a
    real sandbox file by name.

RULE 7 — Answer substantively
  • Answer using whatever is in front of you: memory hits, history, and
    any attached artifact bytes. Be substantive — at least 3 sentences
    or a list of items when the goal is to extract/list/select/compare.

RULE 8 — Setting reminders
  • For 'remember X', 'save X', 'set a reminder', 'note X' style goals,
    call create_file (or update_file when re-saving) under the sandbox
    with a filename describing the topic. Do NOT reply that you cannot
    set reminders — create_file IS how you set them.

RULE 9 — Indexing content
  • When the goal asks to make a file's or fetched content's contents
    SEARCHABLE for later turns or runs (phrasings like 'index', 'ingest',
    'make searchable', 'add to the knowledge base', 'load into memory'),
    call `index_document`. `read_file` only returns the bytes once and
    then discards them; `index_document` chunks the content and writes
    the chunks into Memory so they survive across turns and runs. Use
    `read_file` only for one-shot inspection of a known sandbox file.

RULE 10 — Using indexed facts
  • When the goal asks to ANSWER a question and the MEMORY HITS already
    contain `fact` items whose descriptors begin with `[sandbox:` or
    `[art:` (those are previously-indexed chunks of source documents),
    call `search_knowledge` against the question rather than re-fetching
    the URL or re-reading the file. The indexed chunks are why the
    corpus was indexed in the first place; re-fetching is wasted work.
    The chunk text for each indexed hit is shown inline under the hit's
    descriptor (`chunk: ...`); synthesise directly from those previews
    rather than re-issuing the same vector query.
"""


# How much attached content to send to the model per turn. Most LARGE-tier
# workers handle 30 KB comfortably; truncate above that and let the model
# work with a head-and-tail window.
ATTACH_HEAD = 20_000
ATTACH_TAIL = 10_000


# ── helpers ───────────────────────────────────────────────────────────────


def _format_hits(hits: list[MemoryItem]) -> str:
    # Surface enough of each hit's `value` for Decision to anchor on it.
    # NOTES_RUNS §6 (2) handled the classifier-fact case (`value.raw` such
    # as the birthday date). NOTES_FIX §3 extends this to indexed-chunk
    # facts: when a hit carries `value.chunk` (an indexed slice of a
    # document), the chunk body IS the answer material, and stripping it
    # leaves Decision unable to synthesise — it sees that chunks exist but
    # cannot read them, so it loops on `search_knowledge`. We render a
    # short chunk preview here so Decision can answer directly from the
    # memory-hit list when search_knowledge has already populated it.
    if not hits:
        return "  (none)"
    out = []
    for h in hits[:10]:
        line = f"  - [{h.kind}] {h.descriptor}"
        val = h.value or {}
        if val:
            raw = val.get("raw")
            chunk = val.get("chunk")
            if isinstance(raw, str) and raw.strip():
                line += f"\n      raw: {raw[:200]}"
            elif isinstance(chunk, str) and chunk.strip():
                src = val.get("source") or ""
                preview = chunk[:600].replace("\n", " ")
                more = "…" if len(chunk) > 600 else ""
                line += f"\n      chunk ({src}): {preview}{more}"
            else:
                compact = {
                    k: v for k, v in val.items()
                    if k != "chunk" and not (isinstance(v, str) and len(v) > 200)
                }
                if compact:
                    line += f"\n      value: {json.dumps(compact)[:240]}"
        out.append(line)
    return "\n".join(out)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "  (empty)"
    lines = []
    for h in history[-6:]:
        kind = h.get("kind", "?")
        if kind == "answer":
            lines.append(f"  - iter {h.get('iter')}: ANSWER → {(h.get('text') or '')[:140]}")
        elif kind == "action":
            tool = h.get("tool")
            # NOTES_RUNS §6 (1): agent7.py already clips result_descriptor at
            # 300 chars; clipping again at 140 here was hiding the tail of
            # multi-entry tool outputs like list_dir's file list, and Decision
            # was confidently treating partial views as complete. Match the
            # 300-char ceiling so the model sees everything that was stored.
            desc = h.get("result_descriptor", "")[:300]
            art = f" (artifact {h['artifact_id']})" if h.get("artifact_id") else ""
            lines.append(f"  - iter {h.get('iter')}: {tool}{art} → {desc}")
        else:
            lines.append(f"  - iter {h.get('iter')}: {kind} {h}")
    return "\n".join(lines)


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return ""
    parts = ["\n\nATTACHED ARTIFACTS:"]
    for art_id, data in attached:
        text = data.decode("utf-8", errors="replace")
        if len(text) > ATTACH_HEAD + ATTACH_TAIL + 50:
            text = (
                text[:ATTACH_HEAD]
                + f"\n\n...[truncated; full size {len(data)} bytes]...\n\n"
                + text[-ATTACH_TAIL:]
            )
        parts.append(f"--- {art_id} ---\n{text}")
    return "\n".join(parts)


# ── public API ────────────────────────────────────────────────────────────


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    ensure_gateway()

    prompt = (
        f"GOAL:\n  {goal.text}\n\n"
        f"MEMORY HITS:\n{_format_hits(hits)}\n\n"
        f"RECENT HISTORY:\n{_format_history(history)}"
        f"{_format_attached(attached)}"
    )

    reply = LLM().chat(
        prompt=prompt,
        system=_SYSTEM_PROMPT,
        cache_system=True,
        tools=mcp_tools,
        tool_choice="auto",
        auto_route="decision",
        temperature=0,
        max_tokens=2048,
    )

    tcs = reply.get("tool_calls") or []
    if tcs:
        tc = tcs[0]
        return DecisionOutput(
            tool_call=ToolCall(
                name=tc["name"],
                arguments=tc.get("arguments") or {},
            )
        )
    return DecisionOutput(answer=(reply.get("text") or "").strip())
