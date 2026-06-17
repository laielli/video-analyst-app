"""
Backend parity test for the gallery's static curation (web/lib/curated.json).

The gallery front door reads a committed static curation with ZERO runtime API dependency
(Resolved #4). To keep that curation from silently drifting off the canonical registries, this
test parses web/lib/curated.json and asserts it stays in lockstep with:

  - canned.QUERIES         — the (id, clip) set must match EXACTLY (no extra/missing entries),
  - canned.CLIPS[clip].label — each entry's clipLabel must equal the human clip label,
  - api/eval/question_bank.json — each entry's shape must equal the canonical shape for that
    question (the bank case whose (question, clip_id) match the curated entry).

File-reading pattern mirrors test_sse_stream.py / test_catalog_multiclip.py (Path off the repo
root + json.load). No web build step; the JSON is the shared contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import canned

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
CURATED = REPO_ROOT / "web" / "lib" / "curated.json"
BANK = API_DIR / "eval" / "question_bank.json"


def _load_curated() -> list[dict]:
    assert CURATED.exists(), f"static curation missing at {CURATED}"
    data = json.loads(CURATED.read_text())
    assert isinstance(data, list) and data, "curated.json must be a non-empty array"
    return data


def _bank_shape_by_question() -> dict[tuple[str, str], str]:
    """Map (question, clip_id) -> shape from the eval bank. The (question, clip_id) pair is unique
    per shape across the canned canonical questions, so it pins the canonical shape unambiguously."""
    bank = json.loads(BANK.read_text())
    out: dict[tuple[str, str], str] = {}
    for case in bank["cases"]:
        out[(case["question"], case["clip_id"])] = case["shape"]
    return out


def test_curation_file_exists_and_nonempty():
    curated = _load_curated()
    # one curated card per canned query (card-per-query, not per-clip).
    assert len(curated) == len(canned.QUERIES)


def test_curation_id_clip_set_matches_canned_queries_exactly():
    curated = _load_curated()
    curated_pairs = {(c["id"], c["clip"]) for c in curated}
    canned_pairs = {(q["id"], q["clip"]) for q in canned.QUERIES}
    assert curated_pairs == canned_pairs, (
        f"curated.json drifted off canned.QUERIES: "
        f"only-in-curated={curated_pairs - canned_pairs}, only-in-canned={canned_pairs - curated_pairs}"
    )


def test_curation_clip_label_matches_canned_clips():
    curated = _load_curated()
    for c in curated:
        expected = canned.CLIPS[c["clip"]]["label"]
        assert c["clipLabel"] == expected, (
            f"clipLabel drift for {c['id']}: curated={c['clipLabel']!r} != canned={expected!r}"
        )


def test_curation_shape_matches_canonical_eval_bank_shape():
    curated = _load_curated()
    shape_by_q = _bank_shape_by_question()
    for c in curated:
        # resolve the canonical question text from canned.QUERIES (the curated text mirrors it).
        canned_q = canned.by_id(c["id"])
        assert canned_q is not None, f"curated id {c['id']!r} not in canned.QUERIES"
        key = (canned_q["text"], c["clip"])
        assert key in shape_by_q, f"no eval-bank case for {key}"
        assert c["shape"] == shape_by_q[key], (
            f"shape drift for {c['id']}: curated={c['shape']!r} != bank canonical={shape_by_q[key]!r}"
        )


def test_curation_text_mirrors_canned_query_text():
    # defensive: the curated card text should match the canonical canned query text verbatim.
    curated = _load_curated()
    for c in curated:
        canned_q = canned.by_id(c["id"])
        assert canned_q is not None
        assert c["text"] == canned_q["text"], (
            f"text drift for {c['id']}: curated={c['text']!r} != canned={canned_q['text']!r}"
        )
