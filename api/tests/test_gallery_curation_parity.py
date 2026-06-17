"""Backend parity guard for the web gallery's static curation (web/lib/curated.json).

The gallery front door is a credential-less, no-network surface: it renders from a committed
curation file build-imported by web/lib/gallery.ts (Resolved #4 — option C). That file duplicates
data already owned by api/canned.py and api/eval/question_bank.json, so it CAN drift. This test is
the anti-drift gate: it parses curated.json and asserts it exactly mirrors the canonical registries.

Mirrors test_sse_stream.py's file-reading posture: read the committed JSON, assert against canned /
the bank. No live Azure, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import canned

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
CURATED = REPO_ROOT / "web" / "lib" / "curated.json"
QUESTION_BANK = API_DIR / "eval" / "question_bank.json"


def _load_curation() -> list[dict]:
    assert CURATED.exists(), f"missing static curation: {CURATED}"
    data = json.loads(CURATED.read_text())
    assert isinstance(data, list) and data, "curated.json must be a non-empty array"
    return data


def _bank_shape_by_question() -> dict[str, str]:
    """Map a canonical question string -> its shape, from the eval bank."""
    bank = json.loads(QUESTION_BANK.read_text())
    out: dict[str, str] = {}
    for case in bank["cases"]:
        out.setdefault(case["question"], case["shape"])
    return out


def test_curation_id_clip_set_matches_canned_queries():
    """(a) The (id, clip) set in curated.json exactly equals canned.QUERIES's — no missing, no
    extra, no clip mismatch. Card-per-query, so it's one curation entry per canned query."""
    curation = _load_curation()
    curated_pairs = {(e["id"], e["clip"]) for e in curation}
    canned_pairs = {(q["id"], q["clip"]) for q in canned.QUERIES}
    assert curated_pairs == canned_pairs
    # One entry per query (no dupes).
    assert len(curation) == len(canned.QUERIES)


def test_curation_clip_label_matches_canned_clips():
    """(b) Each entry's clipLabel equals canned.CLIPS[clip]['label']."""
    for e in _load_curation():
        assert e["clipLabel"] == canned.CLIPS[e["clip"]]["label"], (
            f"{e['id']}: clipLabel drifted from canned.CLIPS[{e['clip']!r}].label"
        )


def test_curation_shape_matches_question_bank():
    """(c) Each entry's shape equals the canonical shape for its question in the eval bank. The
    entry's text must itself match the canned query's text (so the lookup is honest)."""
    shape_by_question = _bank_shape_by_question()
    canned_text = {q["id"]: q["text"] for q in canned.QUERIES}
    for e in _load_curation():
        # The curation's text must mirror the canned query's text exactly.
        assert e["text"] == canned_text[e["id"]], f"{e['id']}: text drifted from canned.QUERIES"
        # ...and the question must exist in the bank with the curation's shape.
        assert e["text"] in shape_by_question, f"{e['id']}: question not in eval bank"
        assert e["shape"] == shape_by_question[e["text"]], (
            f"{e['id']}: shape {e['shape']!r} != bank shape {shape_by_question[e['text']]!r}"
        )
