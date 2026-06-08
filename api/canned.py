"""
Canned clips + queries. Each query maps to a pinned program + replay cache. A CLIP now also
carries the data the free-text path needs: a per-clip replay `cache` to execute against and a
codegen `hint` (timing/event window) injected into the prompt (Q2 -> Alt C) so the hero clip
keeps grounding without a hardcoded constant in SYSTEM_PROMPT. The pinned QUERIES stay the
known-good fallback for the canned path; free text has NO pinned fallback by design.
"""
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
EX = API_DIR / "examples"

CLIPS = {
    "single-goal": {
        "id": "single-goal",
        "label": "Messi vs Mexico — World Cup 2022",
        "width": 1872, "height": 1042, "duration_ms": 5067,
        # Per-clip replay cache the free-text path replays against (the committed hero cache).
        "cache": str(EX / "hero_cache.json"),
        # Per-clip timing/event-window hint injected into the codegen prompt (Q2 -> Alt C). The
        # hero timing used to be a SYSTEM_PROMPT constant; it now lives here as DATA so the prompt
        # stays clip-agnostic while the one demo keeps grounding. The committed cache only carries
        # detect + a pinned OCR around ~4.0-4.6s, so a free-text program only grounds when codegen
        # reproduces this window.
        "hint": "goal ~3500-5000ms; detect class 'person', crop jersey, read_text, fps 8",
    },
    "best-goal": {
        "id": "best-goal",
        "label": "Free kick #7 — Best Goals 2025",
        # Dims + duration from ffprobe of best-goals-2025-30s.mov (registry wins at cache-build).
        "width": 2636, "height": 1474, "duration_ms": 30086,
        # The second clip's replay cache, DERIVED (Alt C) by run_precompute over the checked-in
        # detect fixture (api/tests/fixtures/fake_detect_best-goal.json) — not hand-authored.
        "cache": str(EX / "best-goal_cache.json"),
        # Per-clip timing/event-window hint injected into the codegen prompt. Genuinely DIFFERS
        # from the hero (a different goal window + a #7 scorer), proving the prompt is clip-agnostic
        # and the second clip supplies its own data. The derived cache carries 5 person dets per
        # sampled frame + a pinned #7 read around ~8.5-9.0s.
        "hint": "goal ~8000-13000ms; detect class 'person', crop jersey, read_text, fps 8 (scorer #7 ~9000ms)",
    },
}

QUERIES = [
    {
        "id": "hero-10-first-goal",
        "text": "Does #10 score the first goal?",
        "clip": "single-goal",
        "program": str(EX / "hero_program.json"),
        "cache": str(EX / "hero_cache.json"),
    },
    # ---- multi-clip catalog: 4 new queries on the best-goal clip stressing op-paths the hero
    #      chain never hits (count / filter-on-a-non-hero-value / out-of-bounds / read_text readout).
    #      All bind to the best-goal clip's Alt-C fixture-derived cache (new-clip-only, resolved).
    {
        "id": "best-goal-count-players",
        "text": "How many players are visible?",
        "clip": "best-goal",
        "program": str(EX / "best-goal_count_program.json"),
        "cache": str(EX / "best-goal_cache.json"),
    },
    {
        "id": "best-goal-seven-first-goal",
        "text": "Does #7 score the first goal?",
        "clip": "best-goal",
        "program": str(EX / "best-goal_filter-seven_program.json"),
        "cache": str(EX / "best-goal_cache.json"),
    },
    {
        "id": "best-goal-ninetynine-first-goal",
        "text": "Does #99 score the first goal?",
        "clip": "best-goal",
        "program": str(EX / "best-goal_out-of-bounds_program.json"),
        "cache": str(EX / "best-goal_cache.json"),
    },
    {
        "id": "best-goal-scorer-number",
        "text": "What number does the scorer wear?",
        "clip": "best-goal",
        "program": str(EX / "best-goal_readout_program.json"),
        "cache": str(EX / "best-goal_cache.json"),
    },
]


def by_id(query_id: str) -> dict | None:
    return next((q for q in QUERIES if q["id"] == query_id), None)


def clip_by_id(clip_id: str) -> dict | None:
    """Resolve a clip by id (mirrors by_id for QUERIES). Returns the clip dict — which carries a
    loadable `cache` path and a codegen `hint` — or None for an unknown id, so the route owns the
    one 404 site (CLIPS is already a dict, but a named resolver keeps a single lookup point)."""
    return CLIPS.get(clip_id)
