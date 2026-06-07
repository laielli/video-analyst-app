"""
Phase 0 (from /codex): the interpreter's answer path is generalized to answer the question that
was ACTUALLY asked — a counting question reports a count, a temporal question names the subject it
was given, a text-readout question reads the value, and an unsynthesizable binding grounds out
honestly (grounded:false) instead of emitting the hardcoded #10 verdict.

These are pure interpreter tests (no Azure, no fastapi): build a small program, run it over the
committed hero cache, and assert the answer shape. Pattern mirrors test_precompute.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from interpreter import Cache, Interpreter

API_DIR = Path(__file__).resolve().parent.parent
HERO_CACHE = API_DIR / "examples" / "hero_cache.json"

HERO_VERDICT = "#10 scored the first goal"


def _cache():
    return Cache.load(HERO_CACHE)


def _frames_people(start=3500, end=5000, fps=8):
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": start, "end_ms": end, "fps": fps}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]


def test_count_question_answers_count_not_hero_verdict():
    """count -> answer('how many players are visible?') answers about the count and NEVER emits the
    hardcoded '#10 scored the first goal' string. This is the regression for the hardcoded-verdict bug."""
    program = _frames_people() + [
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players are visible?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    # 5 people detected at the cached frame.
    assert f["answer"] == "5", f
    assert "5 found" in f["verdict"]
    # the dispositive assertion: the count question must not borrow the #10 scorer verdict.
    assert HERO_VERDICT not in (f["verdict"] or "")
    assert "#10" not in (f["answer"] or "")


def test_temporal_answer_derives_subject_from_filter():
    """A 'did #10 score first?' program yields a verdict naming the subject DERIVED from the
    filter's matched value, not a literal baked into op_answer."""
    program = _frames_people() + [
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score the first goal?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["verdict"] == "Yes — #10 scored the first goal"

    # And the subject is genuinely derived: filter on a DIFFERENT number gives a different framing
    # (and grounds out, since #99 isn't the scorer / isn't present).
    program99 = _frames_people() + [
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "99"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #99 score the first goal?"}},
    ]
    doc99 = Interpreter(_cache()).run(program99)
    f99 = doc99["findings"]
    # #99 isn't the scorer -> honest "No", and the subject label is NOT the literal #10.
    assert f99["verdict"].startswith("No")
    assert "#10" not in f99["verdict"]


def test_text_readout_question():
    """A 'what number is the scorer wearing?' program ending read_text -> answer returns the read value."""
    program = _frames_people() + [
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "result", "op": "answer", "args": {"from": "numbers", "question": "what number is the scorer wearing?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert "10" in f["answer"]
    assert "Read:" in f["verdict"]
    assert HERO_VERDICT not in f["verdict"]


def test_existence_question_answers_presence():
    """A detect -> answer('is there a person?') program returns a presence verdict, not the #10 string."""
    program = _frames_people() + [
        {"id": "result", "op": "answer", "args": {"from": "people", "question": "is there a person in frame?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["answer"] == "Yes"
    assert "present" in f["verdict"]
    assert HERO_VERDICT not in f["verdict"]


def test_answer_on_unsupported_kind_is_ungrounded():
    """answer handed a binding it can't synthesize from (an empty count == 0) emits grounded:false
    (ties to Q1=A), not a fabricated verdict."""
    # sample a window with no cached detections -> 0 people -> count 0 -> ungrounded.
    program = _frames_people(start=0, end=200) + [
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["answer"] is None and f["verdict"] is None
    assert HERO_VERDICT not in (f["verdict"] or "")
