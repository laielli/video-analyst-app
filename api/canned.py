"""
Canned clips + queries for v1. Each canned query maps to a pinned program + replay cache.
Free text (the ViperGPT thesis, end to end) compiles a TYPED question into a live program
against a clip's replay `cache`, with NO pinned fallback — so each clip also carries:
  - `cache`: the replay cache the free-text path executes against (same artifact the canned
    query points at; replay is free + deterministic).
  - `hint`: the per-clip timing/event-window the codegen prompt injects as DATA (Q2 -> Alt C),
    so the hero timing survives without being hardcoded in codegen.SYSTEM_PROMPT.
"""
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
EX = API_DIR / "examples"

CLIPS = {
    "single-goal": {
        "id": "single-goal",
        "label": "Messi vs Mexico — World Cup 2022",
        "width": 1872, "height": 1042, "duration_ms": 5067,
        # The replay cache the free-text path executes the generated program against.
        "cache": str(EX / "hero_cache.json"),
        # Q2 -> Alt C: timing/event window for THIS clip, injected into the codegen prompt as
        # data (not a SYSTEM_PROMPT constant). The committed cache only carries detect around
        # the goal window + the pinned #10 OCR, so this hint is what lets a typed question
        # reproduce the grounded window. A new clip supplies its own hint.
        "hint": "goal moment ~4000ms; legible scorer jersey ~4625ms; sample ~3500-5000ms, "
                "detect person, fps 8",
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
    """Resolve a clip by id (mirrors by_id over QUERIES; one 404 site for the free-text route).
    Returns the clip dict carrying its `cache` path + codegen `hint`, or None if unknown."""
    return CLIPS.get(clip_id)
