"""
Backend parity guard for the gallery's static curation (web/lib/curated.json).

The gallery / front door reads a committed static curation so it renders with ZERO runtime API
dependency. To keep that committed data from silently drifting off the canonical registries, this
test asserts the curation EXACTLY matches:
  - (a) its {(id, clip)} set == canned.QUERIES's,
  - (b) each entry's clipLabel == canned.CLIPS[clip]["label"],
  - (c) each entry's shape == the canonical shape for that question in api/eval/question_bank.json
        (matched by clip_id + exact question text — the same key the bank uses).

Mirrors test_sse_stream.py's file-reading posture (read the committed JSON, assert against the
in-process registries). If a query/clip/label/shape is added or renamed in the registries, this
test fails until web/lib/curated.json is updated to match.
"""
from __future__ import annotations

import json
from pathlib import Path

import canned

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
CURATED = REPO_ROOT / "web" / "lib" / "curated.json"
BANK = API_DIR / "eval" / "question_bank.json"


def _load_curation() -> list[dict]:
    assert CURATED.exists(), f"static curation missing: {CURATED}"
    data = json.loads(CURATED.read_text())
    assert isinstance(data, list) and data, "curated.json must be a non-empty array"
    return data


def _canonical_shape_by_question() -> dict[tuple[str, str], str]:
    """Map (clip_id, question text) -> shape from the eval bank. Each canned question matches
    exactly one bank case by this key, so the shape is unambiguous (single source of truth)."""
    bank = json.loads(BANK.read_text())
    out: dict[tuple[str, str], str] = {}
    for case in bank["cases"]:
        out[(case["clip_id"], case["question"])] = case["shape"]
    return out


def test_curated_file_exists_and_nonempty():
    data = _load_curation()
    # every entry carries the fields the gallery's view-model assembly + parity checks need.
    for entry in data:
        for k in ("id", "text", "clip", "clipLabel", "shape"):
            assert k in entry, f"curated entry missing {k}: {entry}"


def test_curated_id_clip_set_matches_canned_queries_exactly():
    data = _load_curation()
    curated_pairs = {(e["id"], e["clip"]) for e in data}
    canned_pairs = {(q["id"], q["clip"]) for q in canned.QUERIES}
    assert curated_pairs == canned_pairs, (
        f"curation drifted off canned.QUERIES.\n"
        f"only in curation: {curated_pairs - canned_pairs}\n"
        f"only in canned:   {canned_pairs - curated_pairs}"
    )
    # no duplicate ids, and the count matches (card-per-query).
    ids = [e["id"] for e in data]
    assert len(ids) == len(set(ids)) == len(canned.QUERIES)


def test_curated_text_matches_canned_query_text():
    data = _load_curation()
    canned_text = {q["id"]: q["text"] for q in canned.QUERIES}
    for e in data:
        assert e["text"] == canned_text[e["id"]], (
            f"{e['id']}: curated text {e['text']!r} != canned {canned_text[e['id']]!r}"
        )


def test_curated_clip_label_matches_canned_clips():
    data = _load_curation()
    for e in data:
        expected = canned.CLIPS[e["clip"]]["label"]
        assert e["clipLabel"] == expected, (
            f"{e['id']}: clipLabel {e['clipLabel']!r} != canned.CLIPS[{e['clip']!r}].label {expected!r}"
        )


def test_curated_shape_matches_eval_bank_canonical_shape():
    data = _load_curation()
    shape_of = _canonical_shape_by_question()
    for e in data:
        key = (e["clip"], e["text"])
        assert key in shape_of, f"{e['id']}: no eval-bank case for {key}"
        assert e["shape"] == shape_of[key], (
            f"{e['id']}: shape {e['shape']!r} != eval-bank canonical {shape_of[key]!r}"
        )
