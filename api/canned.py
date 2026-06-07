"""
Canned clips + queries for v1 (no free-text input — design Premise 1/6). Each query maps
to a pinned program + replay cache. When Azure OpenAI codegen lands, the `program` becomes
live-generated and validated, but this registry (and the SSE contract) stay the same.
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
        # Per-clip timing/event-window hint (Q2 -> Alt C): the grounding signal that used to be a
        # hardcoded constant in codegen.SYSTEM_PROMPT. Injected into the codegen prompt as DATA so
        # the prompt stays clip-agnostic and a free-text program can still reproduce the hero
        # window that the committed cache actually covers (detect @ ~4.6s, goal @ ~4.0s).
        "hint": "goal at ~4000ms; the legible scorer jersey is ~4625ms; sample roughly "
                "start_ms 3500, end_ms 5000, fps 8; detect class \"person\".",
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
    """Resolve a clip by id (mirrors by_id over QUERIES). Returns the CLIPS entry — carrying a
    loadable `cache` path and the per-clip codegen `hint` — or None for an unknown id. One 404
    site for the free-text path; the route raises HTTP 404 when this returns None."""
    return CLIPS.get(clip_id)
