"""
build_corpus.py — Fetch and index 50+ pages across 4+ domains.

Domains:
  1. Wikipedia       — ML/AI reference articles
  2. Python Docs     — asyncio library pages
  3. Python PEPs     — async-related PEPs
  4. arXiv abstracts — key ML papers
  5. Real Python     — asyncio tutorials

Run:
    uv run python scripts/build_corpus.py
    uv run python scripts/build_corpus.py --dry-run   # list URLs, no indexing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx

ROOT    = Path(__file__).resolve().parent.parent
RAG_URL = "http://127.0.0.1:8108"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Corpus: 55 pages across 5 domains ────────────────────────────────────────
#
# Each entry: (url, title, domain, topics[])

CORPUS: list[tuple[str, str, str, list[str]]] = [

    # ── Domain 1: Wikipedia — ML/AI (22 pages) ──────────────────────────────

    ("https://en.wikipedia.org/wiki/Backpropagation",
     "Backpropagation", "wikipedia",
     ["neural-network-training", "gradient"]),

    ("https://en.wikipedia.org/wiki/Gradient_descent",
     "Gradient Descent", "wikipedia",
     ["neural-network-training", "gradient"]),

    ("https://en.wikipedia.org/wiki/Loss_function",
     "Loss Function", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Vanishing_gradient_problem",
     "Vanishing Gradient Problem", "wikipedia",
     ["gradient", "neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Stochastic_gradient_descent",
     "Stochastic Gradient Descent", "wikipedia",
     ["gradient", "neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
     "Transformer Architecture", "wikipedia",
     ["transformer", "attention"]),

    ("https://en.wikipedia.org/wiki/Attention_(machine_learning)",
     "Attention Mechanism", "wikipedia",
     ["attention", "transformer"]),

    ("https://en.wikipedia.org/wiki/BERT_(language_model)",
     "BERT Language Model", "wikipedia",
     ["transformer", "attention"]),

    ("https://en.wikipedia.org/wiki/Long_short-term_memory",
     "Long Short-Term Memory (LSTM)", "wikipedia",
     ["neural-network-training", "gradient"]),

    ("https://en.wikipedia.org/wiki/Recurrent_neural_network",
     "Recurrent Neural Network", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Convolutional_neural_network",
     "Convolutional Neural Network", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Dropout_(neural_networks)",
     "Dropout (Neural Networks)", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Batch_normalization",
     "Batch Normalization", "wikipedia",
     ["neural-network-training", "gradient"]),

    ("https://en.wikipedia.org/wiki/Transfer_learning",
     "Transfer Learning", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Large_language_model",
     "Large Language Model", "wikipedia",
     ["transformer", "attention"]),

    ("https://en.wikipedia.org/wiki/Softmax_function",
     "Softmax Function", "wikipedia",
     ["neural-network-training", "attention"]),

    ("https://en.wikipedia.org/wiki/Cross-entropy",
     "Cross-Entropy", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Word_embedding",
     "Word Embedding", "wikipedia",
     ["transformer"]),

    ("https://en.wikipedia.org/wiki/Word2vec",
     "Word2Vec", "wikipedia",
     ["transformer"]),

    ("https://en.wikipedia.org/wiki/Regularization_(mathematics)",
     "Regularization", "wikipedia",
     ["neural-network-training", "gradient"]),

    ("https://en.wikipedia.org/wiki/Generative_adversarial_network",
     "Generative Adversarial Network", "wikipedia",
     ["neural-network-training"]),

    ("https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
     "Reinforcement Learning from Human Feedback", "wikipedia",
     ["transformer", "gradient"]),

    # ── Domain 2: Python Docs — asyncio (10 pages) ───────────────────────────

    ("https://docs.python.org/3/library/asyncio.html",
     "asyncio — Asynchronous I/O", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-task.html",
     "asyncio — Coroutines and Tasks", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-eventloop.html",
     "asyncio — Event Loop", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-sync.html",
     "asyncio — Synchronisation Primitives", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-queue.html",
     "asyncio — Queues", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-stream.html",
     "asyncio — Streams", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-exceptions.html",
     "asyncio — Exceptions", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-dev.html",
     "asyncio — Developing with asyncio", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-subprocess.html",
     "asyncio — Subprocesses", "python-docs",
     ["python-async"]),

    ("https://docs.python.org/3/library/asyncio-protocol.html",
     "asyncio — Transports and Protocols", "python-docs",
     ["python-async"]),

    # ── Domain 3: Python PEPs — async (5 pages) ──────────────────────────────

    ("https://peps.python.org/pep-0492/",
     "PEP 492 — Coroutines with async and await", "peps",
     ["python-async"]),

    ("https://peps.python.org/pep-0525/",
     "PEP 525 — Asynchronous Generators", "peps",
     ["python-async"]),

    ("https://peps.python.org/pep-0530/",
     "PEP 530 — Asynchronous Comprehensions", "peps",
     ["python-async"]),

    ("https://peps.python.org/pep-3156/",
     "PEP 3156 — Asynchronous IO Support", "peps",
     ["python-async"]),

    ("https://peps.python.org/pep-0567/",
     "PEP 567 — Context Variables", "peps",
     ["python-async"]),

    # ── Domain 4: arXiv abstracts — key ML papers (12 pages) ─────────────────

    ("https://arxiv.org/abs/1706.03762",
     "Attention Is All You Need", "arxiv",
     ["transformer", "attention"]),

    ("https://arxiv.org/abs/1810.04805",
     "BERT: Pre-training of Deep Bidirectional Transformers", "arxiv",
     ["transformer", "attention"]),

    ("https://arxiv.org/abs/2205.01068",
     "Chain-of-Thought Prompting Elicits Reasoning", "arxiv",
     ["transformer"]),

    ("https://arxiv.org/abs/2210.11610",
     "ReAct: Synergizing Reasoning and Acting in LLMs", "arxiv",
     ["transformer"]),

    ("https://arxiv.org/abs/2106.09685",
     "LoRA: Low-Rank Adaptation of Large Language Models", "arxiv",
     ["transformer", "gradient"]),

    ("https://arxiv.org/abs/2305.18290",
     "Direct Preference Optimization (DPO)", "arxiv",
     ["gradient", "neural-network-training"]),

    ("https://arxiv.org/abs/2302.13971",
     "LLaMA: Open and Efficient Foundation Language Models", "arxiv",
     ["transformer"]),

    ("https://arxiv.org/abs/2005.14165",
     "GPT-3: Language Models are Few-Shot Learners", "arxiv",
     ["transformer", "attention"]),

    ("https://arxiv.org/abs/1512.03385",
     "Deep Residual Learning for Image Recognition (ResNet)", "arxiv",
     ["neural-network-training", "gradient"]),

    ("https://arxiv.org/abs/1409.0473",
     "Neural Machine Translation by Jointly Learning to Align and Translate", "arxiv",
     ["attention", "transformer"]),

    ("https://arxiv.org/abs/1412.6980",
     "Adam: A Method for Stochastic Optimization", "arxiv",
     ["gradient", "neural-network-training"]),

    ("https://arxiv.org/abs/1607.06450",
     "Layer Normalization", "arxiv",
     ["neural-network-training", "transformer"]),

    # ── Domain 5: Real Python — async tutorials (6 pages) ────────────────────

    ("https://realpython.com/async-io-python/",
     "Async IO in Python: A Complete Walkthrough", "realpython",
     ["python-async"]),

    ("https://realpython.com/python-concurrency/",
     "Speed Up Your Python Program With Concurrency", "realpython",
     ["python-async"]),

    ("https://realpython.com/python-async-features/",
     "Getting Started With Async Features in Python", "realpython",
     ["python-async"]),

    ("https://realpython.com/python-gil/",
     "What Is the Python Global Interpreter Lock (GIL)?", "realpython",
     ["python-async"]),

    ("https://realpython.com/learning-paths/python-concurrency-parallel-programming/",
     "Python Concurrency and Parallel Programming", "realpython",
     ["python-async"]),

    ("https://realpython.com/python-sleep/",
     "Python sleep(): How to Add Time Delays to Your Code", "realpython",
     ["python-async"]),
]

assert len(CORPUS) >= 50, f"Corpus too small: {len(CORPUS)}"


# ── HTML text extractor ───────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    _SKIP = frozenset([
        "script", "style", "noscript", "nav", "header", "footer",
        "aside", "iframe", "svg", "canvas", "figure", "form",
        "button", "select", "textarea",
    ])

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list) -> None:
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = " ".join(self._parts)
        return re.sub(r"\s+", " ", raw).strip()


def extract_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass
    return p.text()


# ── HTTP fetch with retry ─────────────────────────────────────────────────────

def fetch(url: str, *, timeout: int = 20, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            resp = httpx.get(url, headers=HEADERS, timeout=timeout,
                             follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            print(f"    HTTP {resp.status_code}")
            return None
        except Exception as exc:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"    fetch failed: {exc!r:.80}")
                return None
    return None


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_page(url: str, title: str, text: str) -> bool:
    try:
        r = httpx.post(
            f"{RAG_URL}/index",
            json={"text": text, "url": url, "title": title},
            timeout=60,
        )
        if r.status_code == 200:
            chunks = r.json().get("chunks_indexed", 0)
            print(f"    indexed {chunks} chunk(s)")
            return True
        print(f"    /index returned {r.status_code}: {r.text[:120]}")
        return False
    except Exception as exc:
        print(f"    /index failed: {exc!r:.80}")
        return False


# ── Manifest generation ───────────────────────────────────────────────────────

def write_manifest(indexed: list[dict], failed: list[dict]) -> Path:
    r = httpx.get(f"{RAG_URL}/status", timeout=5).json()

    lines = [
        "# Corpus Manifest",
        "",
        f"Generated by `scripts/build_corpus.py`  ",
        f"**page_count**: {r.get('page_count')}  ",
        f"**chunk_count**: {r.get('chunk_count')}  ",
        f"**index_size_bytes**: {r.get('index_size_bytes', 0):,}  ",
        "",
        f"**Pages indexed**: {len(indexed)} / {len(CORPUS)}  ",
        f"**Failed / skipped**: {len(failed)}  ",
        "",
        "## Indexed Pages",
        "",
        "| # | URL | Title | Domain | Topics |",
        "|---|-----|-------|--------|--------|",
    ]

    for i, entry in enumerate(indexed, 1):
        topics = ", ".join(entry["topics"])
        url_md = f"[link]({entry['url']})"
        lines.append(
            f"| {i} | {url_md} | {entry['title']} | {entry['domain']} | {topics} |"
        )

    # Domain summary
    domain_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}
    for e in indexed:
        domain_counts[e["domain"]] = domain_counts.get(e["domain"], 0) + 1
        for t in e["topics"]:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    lines += [
        "",
        "## Domain Breakdown",
        "",
        "| Domain | Pages |",
        "|--------|-------|",
    ]
    for d, n in sorted(domain_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {d} | {n} |")

    lines += [
        "",
        "## Topic Coverage (CRP-3)",
        "",
        "| Topic | Pages |",
        "|-------|-------|",
    ]
    topic_labels = {
        "neural-network-training": "Neural network training",
        "transformer":             "Transformer architecture",
        "attention":               "Attention mechanisms",
        "python-async":            "Python async patterns",
        "gradient":                "Gradient-related topics",
    }
    for tag, label in topic_labels.items():
        lines.append(f"| {label} | {topic_counts.get(tag, 0)} |")

    if failed:
        lines += [
            "",
            "## Failed / Skipped",
            "",
            "| URL | Reason |",
            "|-----|--------|",
        ]
        for e in failed:
            lines.append(f"| {e['url']} | {e['reason']} |")

    out = ROOT / "docs" / "corpus_manifest.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> int:
    print("=" * 72)
    print(f"CORPUS BUILDER — {len(CORPUS)} pages across 5 domains")
    print("=" * 72)

    if dry_run:
        domains: dict[str, int] = {}
        for url, title, domain, topics in CORPUS:
            domains[domain] = domains.get(domain, 0) + 1
            print(f"  [{domain:12}] {title}")
        print(f"\n  Domains: {dict(sorted(domains.items()))}")
        print(f"  Total:   {len(CORPUS)} pages")
        return 0

    # Check rag_server is up
    try:
        httpx.get(f"{RAG_URL}/health", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"ERROR: rag_server not reachable at {RAG_URL}: {exc}")
        return 1

    indexed: list[dict] = []
    failed:  list[dict] = []

    for i, (url, title, domain, topics) in enumerate(CORPUS, 1):
        print(f"\n[{i:02d}/{len(CORPUS)}] {title}")
        print(f"         {url}")

        html = fetch(url)
        if html is None:
            failed.append({"url": url, "title": title, "domain": domain,
                           "topics": topics, "reason": "fetch failed"})
            continue

        text = extract_text(html)
        word_count = len(text.split())
        print(f"    extracted {word_count} words")

        if word_count < 200:
            failed.append({"url": url, "title": title, "domain": domain,
                           "topics": topics, "reason": f"only {word_count} words"})
            print("    SKIP — below 200-word threshold")
            continue

        ok = index_page(url, title, text)
        if ok:
            indexed.append({"url": url, "title": title, "domain": domain,
                            "topics": topics})
        else:
            failed.append({"url": url, "title": title, "domain": domain,
                           "topics": topics, "reason": "index failed"})

        time.sleep(0.4)   # gentle rate limiting

    # Final status
    print("\n" + "=" * 72)
    try:
        status = httpx.get(f"{RAG_URL}/status", timeout=5).json()
        print(f"GET /status → page_count={status.get('page_count')}, "
              f"chunk_count={status.get('chunk_count')}")
    except Exception:
        status = {}

    page_count = status.get("page_count", len(indexed))
    pass_count = page_count >= 50

    domains_represented = len({e["domain"] for e in indexed})
    pass_domains = domains_represented >= 4

    print(f"Pages indexed this run : {len(indexed)} / {len(CORPUS)}")
    print(f"Failed / skipped       : {len(failed)}")
    print(f"page_count ≥ 50        : {'✓' if pass_count else '✗'}  ({page_count})")
    print(f"≥ 4 distinct domains   : {'✓' if pass_domains else '✗'}  ({domains_represented})")

    manifest_path = write_manifest(indexed, failed)
    print(f"Corpus manifest        : {manifest_path}")

    overall = pass_count and pass_domains
    print(f"\nRESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="List URLs without fetching or indexing")
    args = ap.parse_args()
    sys.exit(main(dry_run=args.dry_run))
