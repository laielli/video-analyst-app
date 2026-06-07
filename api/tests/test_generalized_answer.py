"""
Phase 0 (from /codex consult): the interpreter's answer path is generalized so a free-text
question is answered honestly per the SHAPE of the binding op_answer consumes — never the
hardcoded '#10 scored the first goal' verdict. These tests pin:
  - a counting question answers the count, and NEVER emits the hero verdict (the regression bug),
  - a temporal question derives its subject (#N) from the filter, not a literal '#10',
  - an answer over an unsupported / empty binding grounds out honestly (grounded:false, Q1→A),
  - a text-readout question returns the read value.

Replay-mode over the committed hero cache; no Azure, no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from interpreter import Cache, Interpreter

API_DIR = Path(__file__).resolve().parent.parent
HERO_CACHE = API_DIR / "examples" / "hero_cache.json"

HERO_VERDICT = "Yes — #10 scored the first goal"


@pytest.fixture
def cache():
    return Cache.load(HERO_CACHE)


def _frames_people(start=3500, end=5000, fps=8):
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": start, "end_ms": end, "fps": fps}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]


def test_count_question_answers_count_not_hero_verdict(cache):
    """count -> answer('how many players?') answers the count and never the #10 hero verdict."""
    program = _frames_people() + [
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players are visible?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert HERO_VERDICT not in (f["verdict"] or "")
    # The answer is about the count: the cache has 5 people @ 4625ms.
    assert f["answer"] == "5"
    assert "5" in (f["verdict"] or "")


def test_temporal_answer_derives_subject_from_filter(cache):
    """A 'did #7 score first?' program names #7 (derived from the filter), not a literal #10."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        # Filter on text == 7. The hero cache only pins #10, so this yields an empty subject — the
        # point of the test is the SUBJECT FRAMING, not whether #7 actually scored.
        {"id": "sevens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "7"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "sevens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Did #7 score the first goal?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    # #7 didn't score (only #10 is pinned), so the verdict is a 'No' that names #7 — NOT #10.
    assert "#7" in (f["verdict"] or "")
    assert "#10" not in (f["verdict"] or "")


def test_temporal_answer_for_ten_still_names_ten(cache):
    """Sanity: the hero #10 question still names #10 (derived from the filter on text == 10)."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score the first goal?"}},
    ]
    doc = Interpreter(cache).run(program)
    assert doc["findings"]["verdict"] == HERO_VERDICT


def test_answer_on_unsupported_kind_is_ungrounded(cache):
    """answer handed a binding kind it can't synthesize (a count of 0) grounds out honestly."""
    # A window that misses the only cached frame (4625) -> 0 people -> count 0 -> ungrounded.
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["answer"] is None and f["verdict"] is None
    assert HERO_VERDICT not in (f["verdict"] or "")


def test_text_readout_question(cache):
    """read_text -> answer('what number is the scorer wearing?') returns the read value (10)."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "result", "op": "answer", "args": {"from": "numbers", "question": "what number is the scorer wearing?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert "10" in (f["answer"] or "")
    assert HERO_VERDICT not in (f["verdict"] or "")


def test_generalized_findings_validate_against_schema(cache):
    """Both a grounded count answer and an ungrounded answer validate against run_doc.schema.json."""
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    V = jsonschema.Draft202012Validator(rd_schema)

    grounded = Interpreter(cache).run(_frames_people() + [
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ])
    ungrounded = Interpreter(cache).run([
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ])
    assert not list(V.iter_errors(grounded))
    assert not list(V.iter_errors(ungrounded))
