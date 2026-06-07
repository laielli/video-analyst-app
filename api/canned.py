"""
Canned clips + queries for v1. Canned queries map to a pinned program + replay cache. The
free-text path (typed questions) reuses the per-clip `cache` (the cache to replay against) and
`hint` (the timing/event-window injected into the codegen prompt) — the SSE contract stays the
same. When Azure OpenAI codegen lands, the `program` becomes live-generated and validated.
"""
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
EX = API_DIR / "examples"

CLIPS = {
    "single-goal": {
        "id": "single-goal",
        "label": "Messi vs Mexico — World Cup 2022",
        "width": 1872, "height": 1042, "duration_ms": 5067,
        # The free-text path replays generated programs against this committed hero cache.
        "cache": str(EX / "hero_cache.json"),
        # Per-clip timing/event-window hint (Q2 -> Alt C): the grounding signal that used to be
        # hardcoded in codegen.SYSTEM_PROMPT, now carried as DATA and injected into the prompt.
        # The committed cache only holds detect + a pinned OCR around this window, so a free-text
        # program only grounds when codegen reproduces it.
        "hint": "goal ~3500-5000ms (scorer legible ~4600ms); detect person, crop jersey, fps 8",
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
    """Resolve a clip by id (mirrors `by_id` for queries). The free-text path reads `cache`
    to replay against and `hint` to inject into the codegen prompt. Unknown id -> None (the
    route raises the 404)."""
    return CLIPS.get(clip_id)
