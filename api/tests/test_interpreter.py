"""
Interpreter-level orchestration + Cache loaders. The interpreter walks a (pre-validated) program,
maintains the binding env id->value, and assembles the run-doc (trace + findings). These tests
cover the error guards (unknown op, undefined binding ref), the grounded-vs-ungrounded findings
split run() assembles, and Cache.load() round-tripping a JSON file with its accessors.
"""
from __future__ import annotations

import json

import pytest

from interpreter import Cache, Interpreter


def make_cache(detect=None, read_text=None, events=None):
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {}, "read_text": read_text or {}, "events": events or [],
    })


# ---------------------------------------------------------------------------------------
# run() error guards
# ---------------------------------------------------------------------------------------

def test_run_raises_on_unknown_op():
    program = [{"id": "x", "op": "not_an_op", "args": {}}]
    with pytest.raises(ValueError, match="unknown op"):
        Interpreter(make_cache()).run(program)


def test_run_raises_on_undefined_binding_reference():
    # `detect` references frames binding "ghost" that was never produced.
    program = [{"id": "people", "op": "detect", "args": {"frames": "ghost", "classes": ["person"]}}]
    with pytest.raises(ValueError, match="undefined binding 'ghost'"):
        Interpreter(make_cache()).run(program)


# ---------------------------------------------------------------------------------------
# findings: grounded vs ungrounded split
# ---------------------------------------------------------------------------------------

def test_run_findings_grounded_split():
    # sample_frames @ fps 8 over 0-200ms has stride 125 -> sampled ts {0, 125}; pin the cached
    # detection on a sampled ts (0) so detect actually finds it.
    cache = make_cache(
        detect={"0": [{"det_id": "p", "cls": "person", "confidence": 0.9, "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]},
        read_text={"p": {"text": "10", "confidence": 0.9, "source": "pinned"}},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p"}],
    )
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 200, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score first?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["partial"] is False
    assert f["reason"] is None
    assert f["answer"] == "Yes"
    # supporting_step prefers the temporal_order step id.
    assert f["supporting_step"] == "ordered"
    # the trace carries one entry per program step.
    assert len(doc["trace"]) == len(program)
    assert doc["query"] == "Does #10 score first?"


def test_run_findings_ungrounded_when_answer_cannot_ground():
    # count over an empty detect set -> answer can't ground -> findings ungrounded.
    cache = make_cache()  # empty cache, no detections anywhere
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["partial"] is True
    assert f["reason"] == "no-grounded-answer"
    assert f["answer"] is None and f["verdict"] is None


def test_run_findings_no_answer_step():
    # A program with no answer step -> the no-answer-step ungrounded reason.
    cache = make_cache()
    program = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["reason"] == "no-answer-step"


def test_run_query_defaults_to_answer_question_when_none():
    cache = make_cache()
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "n", "op": "count", "args": {"items": "frames"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "the asked question"}},
    ]
    doc = Interpreter(cache).run(program)  # no explicit query
    assert doc["query"] == "the asked question"


# ---------------------------------------------------------------------------------------
# Cache.load + accessors
# ---------------------------------------------------------------------------------------

def test_cache_load_roundtrips_json(tmp_path):
    data = {
        "clip": {"id": "c1", "width": 10, "height": 20, "duration_ms": 3000},
        "detect": {"500": [{"det_id": "d0", "cls": "person", "confidence": 0.9, "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]},
        "read_text": {"d0": {"text": "10", "confidence": 0.8, "source": "pinned"}},
        "events": [
            {"ts_ms": 2000, "type": "goal", "scorer_det": "d0"},
            {"ts_ms": 1000, "type": "kickoff"},
            {"ts_ms": 1500, "type": "goal", "scorer_det": "d0"},
        ],
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(data))
    cache = Cache.load(p)
    assert cache.clip["id"] == "c1"
    # analyzed_frames returns the int ts of cached detect frames.
    assert cache.analyzed_frames() == [500]
    # detections_at takes an int and reads the str-keyed dict.
    assert cache.detections_at(500)[0]["det_id"] == "d0"
    assert cache.detections_at(999) == []
    # read_text_for slices the OCR map.
    assert cache.read_text_for("d0")["text"] == "10"
    assert cache.read_text_for("nope") is None
    # goal_events filters to goal-type and sorts by ts (kickoff dropped, goals sorted).
    goals = cache.goal_events()
    assert [g["ts_ms"] for g in goals] == [1500, 2000]
    assert all(g["type"] == "goal" for g in goals)
