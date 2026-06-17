"""
Backend parity test for the runs-gallery static curation (web/lib/curated.json).

The gallery has ZERO runtime API dependency: it renders from a small committed JSON curation
build-imported into the static export. To keep that curation from silently drifting off the
canonical registries, this test parses web/lib/curated.json and asserts:
  (a) its {(id, clip)} set EXACTLY equals canned.QUERIES's {(id, clip)};
  (b) each entry's clipLabel equals canned.CLIPS[clip]["label"];
  (c) each entry's shape equals the canonical shape for that question in
      api/eval/question_bank.json (matched on the verbatim canned query text).

Mirrors test_sse_stream.py's file-reading posture (read a committed JSON file, assert against the
in-process registries). Read-only: no backend code changes.
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
    assert CURATED.exists(), f"static curation missing: {CURATED}"
    data = json.loads(CURATED.read_text())
    assert isinstance(data, list) and data, "curated.json must be a non-empty array"
    return data


def _canonical_shape_by_question() -> dict[str, str]:
    """Map a canonical question string -> its eval-bank shape (canonical phrasing_class only, so a
    paraphrase never shadows the canonical shape)."""
    bank = json.loads(QUESTION_BANK.read_text())
    out: dict[str, str] = {}
    for case in bank["cases"]:
        if case.get("phrasing_class") == "canonical":
            out[case["question"]] = case["shape"]
    return out


def test_curation_id_clip_set_matches_canned_queries():
    curation = _load_curation()
    curated_pairs = {(e["id"], e["clip"]) for e in curation}
    canned_pairs = {(q["id"], q["clip"]) for q in canned.QUERIES}
    assert curated_pairs == canned_pairs, (
        "curated.json {(id, clip)} drifted off canned.QUERIES: "
        f"only-in-curation={curated_pairs - canned_pairs}, "
        f"only-in-canned={canned_pairs - curated_pairs}"
    )


def test_curation_clip_labels_match_canned_clips():
    for e in _load_curation():
        expected = canned.CLIPS[e["clip"]]["label"]
        assert e["clipLabel"] == expected, (
            f"clipLabel drift for {e['id']}: curated={e['clipLabel']!r} != canned={expected!r}"
        )


def test_curation_shapes_match_eval_bank_canonical_shapes():
    shape_by_question = _canonical_shape_by_question()
    for e in _load_curation():
        # The curated entry's text is the verbatim canned query text; the eval bank carries the
        # canonical question -> shape. The shapes must agree.
        canned_query = canned.by_id(e["id"])
        assert canned_query is not None, f"curated id {e['id']!r} not in canned.QUERIES"
        question = canned_query["text"]
        assert question in shape_by_question, (
            f"no canonical eval-bank case for curated query text {question!r}"
        )
        assert e["shape"] == shape_by_question[question], (
            f"shape drift for {e['id']}: curated={e['shape']!r} != "
            f"eval-bank canonical={shape_by_question[question]!r}"
        )


def test_curation_text_matches_canned_query_text():
    # Defensive: the parity matcher above keys on canned text, so a stale curated `text` would
    # silently fall through. Pin it directly.
    for e in _load_curation():
        canned_query = canned.by_id(e["id"])
        assert canned_query is not None
        assert e["text"] == canned_query["text"], (
            f"text drift for {e['id']}: curated={e['text']!r} != canned={canned_query['text']!r}"
        )
