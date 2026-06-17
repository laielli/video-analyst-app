"""Gallery curation parity — the web gallery's static curation (web/lib/curated.json) is the ONLY
data source the runs front-door reads (no runtime fetch), so it MUST never silently drift off the
canonical registries. This read-only test parses the committed JSON and asserts it stays pinned to
canned.QUERIES + canned.CLIPS labels + the eval bank's canonical shapes.

Creds-free; reads committed files only. Mirrors test_eval_bank.py's file-reading + EVAL_DIR-on-path
idiom (the eval bank carries the canonical shape per canonical question)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
EVAL_DIR = API_DIR / "eval"
CURATED = REPO_ROOT / "web" / "lib" / "curated.json"

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import canned  # noqa: E402


def _curation() -> list[dict]:
    return json.loads(CURATED.read_text())


def _canonical_shapes() -> dict[tuple[str, str], str]:
    """(clip_id, question) -> shape, over the eval bank's canonical-distance cases (the single
    source of truth for a question's machine shape label)."""
    bank = json.loads((EVAL_DIR / "question_bank.json").read_text())
    out: dict[tuple[str, str], str] = {}
    for c in bank["cases"]:
        if c.get("distance") == "canonical" or c.get("phrasing_class") == "canonical":
            out[(c["clip_id"], c["question"])] = c["shape"]
    return out


def test_curated_file_exists_and_is_nonempty():
    assert CURATED.exists(), f"missing static curation at {CURATED}"
    data = _curation()
    assert isinstance(data, list) and data, "curated.json must be a non-empty array"


def test_curated_id_clip_set_matches_canned_queries():
    """(a) The {(id, clip)} set exactly equals canned.QUERIES's — no extra/missing curated entry."""
    curated = {(e["id"], e["clip"]) for e in _curation()}
    canned_set = {(q["id"], q["clip"]) for q in canned.QUERIES}
    assert curated == canned_set
    # And one curated entry per canned query (no duplicate ids).
    ids = [e["id"] for e in _curation()]
    assert len(ids) == len(set(ids)) == len(canned.QUERIES)


def test_curated_clip_labels_match_canned_clips():
    """(b) Each entry's clipLabel equals canned.CLIPS[clip]['label']."""
    for e in _curation():
        assert e["clipLabel"] == canned.CLIPS[e["clip"]]["label"], (
            f"clipLabel drift for {e['id']!r} (clip {e['clip']!r})"
        )


def test_curated_shapes_match_eval_bank_canonical_shapes():
    """(c) Each entry's shape equals the eval bank's canonical shape for that clip+question."""
    canonical = _canonical_shapes()
    by_id = {q["id"]: q for q in canned.QUERIES}
    for e in _curation():
        q = by_id[e["id"]]  # id parity already proven above
        key = (q["clip"], q["text"])
        assert key in canonical, f"no canonical bank shape for {key!r}"
        assert e["shape"] == canonical[key], (
            f"shape drift for {e['id']!r}: curated {e['shape']!r} != bank {canonical[key]!r}"
        )


def test_curated_text_matches_canned_query_text():
    """The displayed query text must also match the canned query verbatim (the gallery links into
    /?query=<id>, but the card surfaces the text — keep it true to the registry)."""
    by_id = {q["id"]: q for q in canned.QUERIES}
    for e in _curation():
        assert e["text"] == by_id[e["id"]]["text"], f"text drift for {e['id']!r}"
