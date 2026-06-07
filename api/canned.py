"""
Canned clips + queries for v1. Each query maps to a pinned program + replay cache; the free-text
path (D-DR4) reuses the same per-clip `cache` to replay against and the per-clip `hint` to inject
into the codegen prompt. When Azure OpenAI codegen lands, the `program` becomes live-generated and
validated, but this registry (and the SSE contract) stay the same.
"""
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
EX = API_DIR / "examples"

CLIPS = {
    "single-goal": {
        "id": "single-goal",
        "label": "Messi vs Mexico — World Cup 2022",
        "width": 1872, "height": 1042, "duration_ms": 5067,
        # Per-clip replay cache the free-text path loads (Q2 → Alt C: data, not a server twin).
        "cache": str(EX / "hero_cache.json"),
        # Timing/event-window hint injected into the codegen prompt (Q2 → Alt C). This is the hero
        # window that used to live as a hardcoded constant in codegen.SYSTEM_PROMPT — it survives
        # here as per-clip data so the prompt stays clip-agnostic and the demo keeps grounding.
        "hint": "goal at ~4000ms; the #10 scorer is legible ~4600ms. To ground a question about "
                "the goal, sample frames roughly start_ms 3500, end_ms 5000, fps 8, detect person.",
    },
}

# Fields safe to expose via /api/catalog (the `cache` path stays server-internal).
_CLIP_PUBLIC_FIELDS = ("id", "label", "width", "height", "duration_ms", "hint")

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
    """Resolve a clip by id (mirrors by_id over CLIPS). Returns the full clip dict — including the
    server-internal `cache` path and the codegen `hint` — or None for an unknown clip. The route
    is responsible for raising 404 on None (CLIPS.get returns None, it does not raise)."""
    return CLIPS.get(clip_id)


def public_clip(clip: dict) -> dict:
    """A clip dict with only the fields safe to send to the browser (drops the `cache` path)."""
    return {k: clip[k] for k in _CLIP_PUBLIC_FIELDS if k in clip}
