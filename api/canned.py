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
    "bernabeu-counter": {
        "id": "bernabeu-counter",
        "label": "Real Madrid counter — Champions League 2025",
        "width": 2636, "height": 1474, "duration_ms": 30086,
        # Second clip's replay cache (Alt C: DERIVED from clips/bernabeu-counter/manifest.json +
        # api/tests/fixtures/fake_detect_bernabeu.json by scripts/precompute.py, not hand-authored).
        "cache": str(EX / "bernabeu-counter_cache.json"),
        # Per-clip codegen hint — a DIFFERENT window/event than the hero's, so codegen-repeatability
        # across clips is genuinely exercised (proven by a mocked free-text-path prompt test).
        # goal ~14250ms (the real goal moment, verified against the source video — the prior
        # 9000-11000/10500 window was derived from a FAKE fixture, not the actual clip); the
        # analyzed window widens to 9000-14800ms to cover it. This invalidates every committed
        # bernabeu-counter eval fixture's prompt_version (D6 STALE). TODO(recapture): the
        # single-goal eval fixtures were ALSO deliberately re-stamped stale (see
        # eval/fixtures/single-goal-*.json prompt_version) so the D6 gate reads uniform-all-stale
        # (advisory, exit 0) instead of partial staleness (hard FAIL) until the live re-capture
        # lands and restamps everything for real.
        "hint": "goal ~14250ms; analyzed window 9000-14800ms; detect class 'person', crop jersey, read_text, fps 8; scorer wears #7",
    },
}

QUERIES = [
    {
        "id": "hero-10-first-goal",
        "text": "Does #10 score the goal?",
        "clip": "single-goal",
        "program": str(EX / "hero_program.json"),
        "cache": str(EX / "hero_cache.json"),
    },
    {
        # scene-description: sample_frames -> describe_scene -> answer, grounding to the pinned
        # whole-scene caption (the describe_scene primitive — a new question shape the detect/
        # read_text surface couldn't express). Anchors the bank's scene cell canonical question.
        "id": "single-goal-scene",
        "text": "Describe the scene in this clip.",
        "clip": "single-goal",
        "program": str(EX / "single-goal_scene_program.json"),
        "cache": str(EX / "hero_cache.json"),
    },
    # ---- bernabeu-counter: 4 query shapes the hero chain never hits (multi-clip / multi-query) ----
    {
        # count -> answer (no crop/read_text/filter/temporal_order). Grounds to a number.
        "id": "bernabeu-count-players",
        "text": "How many players are visible?",
        "clip": "bernabeu-counter",
        "program": str(EX / "bernabeu-counter_count_program.json"),
        "cache": str(EX / "bernabeu-counter_cache.json"),
    },
    {
        # filter on a NON-hero jersey value (#7, not #10) -> grounds Yes naming #7 (subject derived).
        "id": "bernabeu-7-first-goal",
        "text": "Does #7 score the goal?",
        "clip": "bernabeu-counter",
        "program": str(EX / "bernabeu-counter_first-goal-7_program.json"),
        "cache": str(EX / "bernabeu-counter_cache.json"),
    },
    {
        # out-of-bounds: filter on a value NOT in the cache (#23) -> honest ungrounded (grounded=false).
        "id": "bernabeu-23-first-goal",
        "text": "Does #23 score the goal?",
        "clip": "bernabeu-counter",
        "program": str(EX / "bernabeu-counter_first-goal-23_program.json"),
        "cache": str(EX / "bernabeu-counter_cache.json"),
    },
    {
        # read_text -> answer readout: reads back the scorer's pinned jersey number ('7').
        "id": "bernabeu-scorer-number",
        "text": "What number does the scorer wear?",
        "clip": "bernabeu-counter",
        "program": str(EX / "bernabeu-counter_scorer-number_program.json"),
        "cache": str(EX / "bernabeu-counter_cache.json"),
    },
]


def by_id(query_id: str) -> dict | None:
    return next((q for q in QUERIES if q["id"] == query_id), None)


def clip_by_id(clip_id: str) -> dict | None:
    """Resolve a clip by id (mirrors by_id for QUERIES). Returns the clip dict — which carries a
    loadable `cache` path and a codegen `hint` — or None for an unknown id, so the route owns the
    one 404 site (CLIPS is already a dict, but a named resolver keeps a single lookup point)."""
    return CLIPS.get(clip_id)
