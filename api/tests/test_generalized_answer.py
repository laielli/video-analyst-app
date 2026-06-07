"""
Phase 0 (from /codex): the generalized op_answer. Before this, op_answer hardcoded the #10
first-goal verdict regardless of the binding it consumed — a counting question would compile,
validate, execute, and STILL emit "Yes/No — #10 scored the first goal". These tests pin the
honest, kind-dispatched behavior: the answer reflects the question actually asked, or grounds
out honestly. Mirror pattern: test_precompute.py (replay over the committed hero cache).
"""
from __future__ import annotations

from pathlib import Path

from interpreter import Cache, Interpreter

API_DIR = Path(__file__).resolve().parent.parent
HERO_CACHE = API_DIR / "examples" / "hero_cache.json"

HERO_VERDICT = "#10 scored the first goal"


def _cache():
    return Cache.load(HERO_CACHE)


def _detect_people():
    # The hero detect window: sample around 4625ms (the only ts with detections in the cache).
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4500, "end_ms": 4700, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]


def test_count_question_answers_count_not_hero_verdict():
    """`count -> answer("how many players are visible?")` answers the count and NEVER emits the
    hardcoded #10 first-goal verdict. This is the regression for the hardcoded-verdict bug."""
    program = _detect_people() + [
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players are visible?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    # the answer is a count, not the #10 verdict.
    assert HERO_VERDICT not in (f["verdict"] or "")
    # there is exactly one person detected in the cache window (p_messi at 4625).
    n = next(t for t in doc["trace"] if t["op"] == "count")
    assert f["answer"] == n["output_label"]
    assert str(f["answer"]).isdigit()


def test_temporal_answer_derives_subject_from_filter():
    """A "did #7 score first?" program yields a verdict naming #7 (derived from the filter's
    matched value), NOT a literal #10. Since #7 didn't score, the verdict is a grounded 'No'."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4500, "end_ms": 4700, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "sevens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "7"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "sevens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Did #7 score the first goal?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    # #7 is not in the cache, so the filter is empty -> #7 didn't score -> grounded No, naming #7.
    assert f["grounded"] is True
    assert f["answer"] == "No"
    # the verdict must NOT claim #10; with no matched value it falls back to a neutral subject.
    assert "#10" not in (f["verdict"] or "")


def test_temporal_answer_hero_still_says_ten():
    """The hero `filter text == 10 -> temporal_order -> answer` still derives the #10 verdict —
    from the filter's matched value, not a literal. Guards the generalization didn't break hero."""
    from validate_program import strip_comments
    import json
    program = strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))["program"]
    doc = Interpreter(_cache()).run(program)
    assert doc["findings"]["verdict"] == "Yes — #10 scored the first goal"
    assert doc["findings"]["grounded"] is True


def test_answer_on_unsupported_kind_is_ungrounded():
    """answer handed a binding kind it can't synthesize -> grounded:false (ties to Q1=A), not a
    fabricated verdict. We inject an unsupported binding directly into the env."""
    from interpreter.primitives import op_answer
    cache = _cache()
    env = {"weird": {"kind": "banana", "value": 1}}
    step = {"id": "result", "op": "answer", "args": {"from": "weird", "question": "what?"}}
    binding, result = op_answer(step, env, cache)
    assert binding["value"]["grounded"] is False
    assert binding["value"]["answer"] is None
    assert binding["value"]["verdict"] is None
    assert HERO_VERDICT not in str(binding["value"])
    assert result["status"] == "empty"


def test_count_zero_is_ungrounded():
    """A count that resolves to 0 has no groundable subject -> honest ungrounded, not '0 matches'
    masquerading as a confident answer."""
    from interpreter.primitives import op_answer
    cache = _cache()
    env = {"n": {"kind": "number", "value": 0}}
    step = {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}}
    binding, _ = op_answer(step, env, cache)
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_text_readout_question():
    """A "what number is the scorer wearing?" program ending read_text -> answer returns the read
    value (10, the pinned p_messi OCR), not the temporal verdict."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4500, "end_ms": 4700, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "result", "op": "answer", "args": {"from": "numbers", "question": "What number is the scorer wearing?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["answer"] == "10"
    assert "10" in (f["verdict"] or "")
    assert HERO_VERDICT not in (f["verdict"] or "")


def test_presence_question_existence():
    """An existence program (detect -> answer over the detections collection) answers presence."""
    program = _detect_people() + [
        {"id": "result", "op": "answer", "args": {"from": "people", "question": "Is there a player on screen?"}},
    ]
    doc = Interpreter(_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["answer"] == "Yes"
    assert HERO_VERDICT not in (f["verdict"] or "")
