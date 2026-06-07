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
}

QUERIES = [
    {
        "id": "hero-10-first-goal",
        "text": "Does #10 score the first goal?",
        "clip": "single-goal",
        "program": str(EX / "hero_program.json"),
        "cache": str(EX / "hero_cache.json"),
    },
]


def by_id(query_id: str) -> dict | None:
    return next((q for q in QUERIES if q["id"] == query_id), None)


def clip_by_id(clip_id: str) -> dict | None:
    """Resolve a clip by id (mirrors by_id for QUERIES). Returns the clip dict — which carries a
    loadable `cache` path and a codegen `hint` — or None for an unknown id, so the route owns the
    one 404 site (CLIPS is already a dict, but a named resolver keeps a single lookup point)."""
    return CLIPS.get(clip_id)
