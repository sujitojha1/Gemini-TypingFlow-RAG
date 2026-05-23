"""
Test Query B — Tokyo activities and weather (multi-goal, memory carryover)

Expected flow (~8 iterations):
  Goal 1: Find 3 family-friendly things to do in Tokyo this weekend
          Decision calls web_search → results in memory (keyword + vector)
  Goal 2: Fetch Saturday's weather forecast for Tokyo
          Decision calls web_search or fetch_url → result in memory
          Memory records tool_outcome with weather data
  Goal 3: Select the most appropriate activity given the weather
          Memory.read carries Goal 2's weather tool_outcome into Decision
          Decision answers with a recommendation

Key S7 behaviours under test:
  - Perception decomposes the query into ≥2 distinct goals
  - Memory carryover: Goal 2's weather outcome is retrievable when Goal 3 runs
  - Vector path active: tool_outcome items written with embeddings
  - Final answer integrates activities + weather + recommendation

Run:
    uv run python tests/test_query_b.py
    uv run python tests/test_query_b.py --no-clean   # keep existing state
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY = (
    "Find 3 family-friendly things to do in Tokyo this weekend. "
    "Check Saturday's weather forecast there and tell me which one "
    "is most appropriate."
)

STATE_DIR   = ROOT / "state"
MEMORY_FILE = STATE_DIR / "memory.json"

# ── answer content checks ─────────────────────────────────────────────────────

# Each sub-list = one family-friendly Tokyo activity area.
# At least 3 distinct groups must appear in the answer.
ACTIVITY_GROUPS = [
    ["disneyland", "disney", "disneysea"],
    ["teamlab", "team lab", "digital art", "planets"],
    ["ueno", "ueno zoo", "ueno park"],
    ["skytree", "sky tree", "tokyo skytree"],
    ["shinjuku gyoen", "gyoen", "shinjuku garden"],
    ["senso-ji", "sensoji", "asakusa temple", "asakusa"],
    ["ghibli", "studio ghibli", "miyazaki"],
    ["odaiba", "palette town"],
    ["sumida aquarium", "aquarium", "tokyo aquarium"],
    ["joypolis", "amusement park", "theme park"],
    ["sumo", "baseball", "sport"],
    ["national museum", "edo museum", "science museum"],
    ["park", "garden", "botanical"],          # generic outdoor
    ["museum", "gallery", "exhibition"],       # generic indoor
]
MIN_ACTIVITY_GROUPS = 3

# Weather keywords — at least 2 must appear in the answer or memory records,
# confirming the forecast was actually fetched and used.
WEATHER_KEYWORDS = [
    "rain", "rainy", "sunny", "cloud", "cloudy", "clear sky",
    "temperature", "forecast", "weather", "humid", "wind",
    "degree", "celsius", "fahrenheit", "shower", "overcast",
    "precipitation", "partly", "mm", "°c", "°f",
]
MIN_WEATHER_HITS = 2

# Recommendation signals — at least 1 must link an activity to the weather.
RECOMMENDATION_SIGNALS = [
    "recommend", "suggest", "most appropriate", "best choice",
    "ideal", "perfect for", "suitable", "given the weather",
    "because of", "considering the", "since it", "therefore",
    "indoor", "outdoor", "if it rains", "in case of rain",
    "on a sunny", "due to",
]


# ── state helpers ─────────────────────────────────────────────────────────────

def clean_state() -> None:
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
        print(f"[test] cleared {STATE_DIR}")
    STATE_DIR.mkdir(parents=True)


def _memory_items() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text())


def _memory_full_text() -> str:
    parts: list[str] = []
    for m in _memory_items():
        parts.append(m.get("descriptor", ""))
        val = m.get("value", {})
        if isinstance(val, dict):
            parts.extend(str(v) for v in val.values())
    return " ".join(parts).lower()


# ── checks ────────────────────────────────────────────────────────────────────

def check_activities(answer: str) -> tuple[bool, list[str]]:
    """At least MIN_ACTIVITY_GROUPS distinct Tokyo activity areas in the answer."""
    low     = answer.lower()
    matched = [
        i for i, group in enumerate(ACTIVITY_GROUPS, 1)
        if any(signal in low for signal in group)
    ]
    if len(matched) < MIN_ACTIVITY_GROUPS:
        missing = [i for i in range(1, len(ACTIVITY_GROUPS) + 1) if i not in matched]
        return False, [
            f"only {len(matched)} activity group(s) matched in answer "
            f"(need ≥{MIN_ACTIVITY_GROUPS}); "
            f"missing group(s): {missing}; "
            f"signals: {[ACTIVITY_GROUPS[i - 1][:3] for i in missing]}"
        ]
    return True, []


def check_weather(answer: str) -> tuple[bool, list[str]]:
    """Weather keywords must appear in the answer or memory records."""
    low      = answer.lower()
    mem_text = _memory_full_text()
    combined = low + " " + mem_text
    hits     = [kw for kw in WEATHER_KEYWORDS if kw in combined]
    if len(hits) < MIN_WEATHER_HITS:
        return False, [
            f"only {len(hits)} weather keyword(s) found (need ≥{MIN_WEATHER_HITS}): "
            f"found={hits}; checked against answer + memory"
        ]
    return True, []


def check_recommendation(answer: str) -> tuple[bool, list[str]]:
    """Answer must contain at least one recommendation signal linking activity to weather."""
    low = answer.lower()
    hit = next((s for s in RECOMMENDATION_SIGNALS if s in low), None)
    if not hit:
        return False, [
            "no recommendation signal found in answer; "
            f"signals checked: {RECOMMENDATION_SIGNALS[:8]}..."
        ]
    return True, []


def check_multi_goal_memory() -> tuple[bool, list[str]]:
    """Memory must reference ≥2 distinct goal_ids — Perception decomposed the query."""
    items    = _memory_items()
    if not items:
        return False, ["memory.json is empty or missing"]
    goal_ids = {m.get("goal_id") for m in items if m.get("goal_id")}
    if len(goal_ids) < 2:
        return False, [
            f"only {len(goal_ids)} distinct goal_id(s) in memory "
            f"(expected ≥2); ids found: {goal_ids}"
        ]
    return True, []


def check_weather_in_memory() -> tuple[bool, list[str]]:
    """A tool_outcome record containing weather keywords must exist (carryover source)."""
    items = _memory_items()
    tool_outcomes = [m for m in items if m.get("kind") == "tool_outcome"]
    if not tool_outcomes:
        return False, ["no tool_outcome records in memory.json"]
    mem_text = _memory_full_text()
    hits     = [kw for kw in WEATHER_KEYWORDS if kw in mem_text]
    if len(hits) < MIN_WEATHER_HITS:
        return False, [
            "tool_outcome records exist but contain no weather keywords — "
            "weather fetch result was not persisted to memory; "
            f"found: {hits}"
        ]
    return True, []


def check_vector_path() -> tuple[bool, list[str]]:
    """At least some memory items must carry embeddings (S7 vector path active)."""
    items         = _memory_items()
    with_embedding = [m for m in items if m.get("embedding") is not None]
    if not with_embedding:
        return False, [
            "no memory items have embeddings — "
            "gateway V7 embed endpoint was not called or vector path is broken"
        ]
    return True, []


# ── main ──────────────────────────────────────────────────────────────────────

async def main(clean: bool) -> int:
    if clean:
        clean_state()

    import agent7

    print("=" * 78)
    print("TEST QUERY B — Tokyo activities + weather (multi-goal, memory carryover)")
    print("=" * 78)

    answer = await agent7.run(QUERY)

    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    low      = answer.lower()
    mem_text = _memory_full_text()

    act_pass,  act_failures  = check_activities(answer)
    wx_pass,   wx_failures   = check_weather(answer)
    rec_pass,  rec_failures  = check_recommendation(answer)
    mg_pass,   mg_failures   = check_multi_goal_memory()
    wxm_pass,  wxm_failures  = check_weather_in_memory()
    vec_pass,  vec_failures  = check_vector_path()

    # ── Activity groups ──────────────────────────────────────────────────────
    matched_acts = 0
    for i, group in enumerate(ACTIVITY_GROUPS, 1):
        hit = next((s for s in group if s in low), None)
        if hit:
            matched_acts += 1
        print(f"  activity group {i:02d}                     : {'✓' if hit else '✗'}"
              + (f"  ('{hit}')" if hit else f"  (signals: {group[:3]}...)"))
    print(f"  ≥{MIN_ACTIVITY_GROUPS} activity groups matched        : "
          f"{'✓' if matched_acts >= MIN_ACTIVITY_GROUPS else '✗'}  ({matched_acts} matched)")

    # ── Weather keywords ─────────────────────────────────────────────────────
    combined    = low + " " + mem_text
    wx_hits     = [kw for kw in WEATHER_KEYWORDS if kw in combined]
    wx_in_ans   = [kw for kw in WEATHER_KEYWORDS if kw in low]
    wx_in_mem   = [kw for kw in WEATHER_KEYWORDS if kw in mem_text]
    print(f"  weather keywords in answer           : {wx_in_ans or '—'}")
    print(f"  weather keywords in memory           : {wx_in_mem or '—'}")
    print(f"  ≥{MIN_WEATHER_HITS} weather keywords found          : "
          f"{'✓' if len(wx_hits) >= MIN_WEATHER_HITS else '✗'}  ({len(wx_hits)} found: {wx_hits[:5]})")

    # ── Recommendation ───────────────────────────────────────────────────────
    rec_hit = next((s for s in RECOMMENDATION_SIGNALS if s in low), None)
    print(f"  recommendation signal in answer      : {'✓' if rec_hit else '✗'}"
          + (f"  ('{rec_hit}')" if rec_hit else ""))

    # ── Multi-goal decomposition ─────────────────────────────────────────────
    items    = _memory_items()
    goal_ids = {m.get("goal_id") for m in items if m.get("goal_id")}
    print(f"  ≥2 distinct goal_ids in memory       : {'✓' if mg_pass else '✗'}"
          + f"  ({len(goal_ids)} goal(s): {list(goal_ids)[:4]})")

    # ── Weather in memory (carryover source) ─────────────────────────────────
    tool_names = [m.get("value", {}).get("tool", "?")
                  for m in items if m.get("kind") == "tool_outcome"]
    print(f"  weather keywords in memory records   : {'✓' if wxm_pass else '✗'}"
          + f"  (tool_outcomes from: {tool_names})")

    # ── Vector path ──────────────────────────────────────────────────────────
    with_emb = [m for m in items if m.get("embedding") is not None]
    print(f"  embeddings in memory (vector active) : {'✓' if vec_pass else '✗'}"
          + f"  ({len(with_emb)}/{len(items)} items have embeddings)")

    # ── Failures ─────────────────────────────────────────────────────────────
    all_failures = (act_failures + wx_failures + rec_failures
                    + mg_failures + wxm_failures + vec_failures)
    if all_failures:
        print("\n  FAILURES:")
        for f in all_failures:
            print(f"    ✗ {f}")

    overall = act_pass and wx_pass and rec_pass and mg_pass and wxm_pass and vec_pass
    print(f"\n  RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    return 0 if overall else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="Skip clearing state/ before the run")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(clean=args.clean)))
