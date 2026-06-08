"""
Interpreter-level orchestration + Cache loader tests.

Covers Interpreter.run's contract beyond the per-op tests:
  - unknown op name -> ValueError (interpreter.py op lookup)
  - undefined binding reference -> ValueError (the _refs pre-check)
  - the grounded-vs-ungrounded findings split run() assembles (_findings)
  - Cache.load round-trips a JSON file; read_text_for / goal_events / analyzed_frames slice it.
"""
from __future__ import annotations

import json

import pytest

from interpreter import Cache, Interpreter


def make_cache(detect=None, read_text=None, events=None):
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {},
        "read_text": read_text or {},
        "events": events or [],
    })


# ----------------------------------------------------------------------------------------
# Interpreter.run — error paths
# ----------------------------------------------------------------------------------------

def test_run_raises_on_unknown_op():
    program = [{"id": "x", "op": "teleport", "args": {}}]
    with pytest.raises(ValueError, match="unknown op 'teleport'"):
        Interpreter(make_cache()).run(program)


def test_run_raises_on_undefined_binding_reference():
    # count references an id ("ghost") that was never produced by an earlier step.
    program = [
        {"id": "n", "op": "count", "args": {"items": "ghost"}},
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    with pytest.raises(ValueError, match="references undefined binding 'ghost'"):
        Interpreter(make_cache()).run(program)


def test_run_literal_args_are_not_treated_as_refs():
    # region/where/classes are literals, not binding refs — the _refs filter must skip them.
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "c", "op": "crop", "args": {"detections": "p", "region": "jersey"}},
        {"id": "r", "op": "answer", "args": {"from": "c", "question": "q"}},
    ]
    # Should not raise: 'jersey'/'person' are literals, not undefined bindings.
    doc = Interpreter(make_cache()).run(program)
    assert len(doc["trace"]) == 4


# ----------------------------------------------------------------------------------------
# Interpreter.run — findings assembly (grounded vs ungrounded split)
# ----------------------------------------------------------------------------------------

def test_run_assembles_grounded_findings_with_supporting_step():
    cache = make_cache(
        detect={"4000": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                          "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}}]},
        read_text={"p_messi": {"text": "10", "source": "pinned"}},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi"}],
    )
    # Window starts exactly at 4000 so the stride (1000/8 -> 125) lands the first sampled frame
    # on the cached detect key "4000"; otherwise detect would find nothing and never ground.
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
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
    # supporting_step prefers the temporal_order step.
    assert f["supporting_step"] == "ordered"
    assert doc["query"] == "Does #10 score first?"
    assert doc["clip"]["id"] == "t"


def test_run_ungrounded_findings_when_answer_cannot_ground():
    # detect window misses the cache -> 0 people -> count 0 -> answer can't ground.
    cache = make_cache(detect={"4000": [{"det_id": "a", "cls": "person", "confidence": 0.5,
                                         "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]})
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["partial"] is True
    assert f["answer"] is None and f["verdict"] is None
    assert f["reason"] == "no-grounded-answer"


def test_run_ungrounded_findings_when_no_answer_step():
    program = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    doc = Interpreter(make_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["reason"] == "no-answer-step"


def test_run_query_defaults_to_answer_question_when_none():
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "r", "op": "answer", "args": {"from": "f", "question": "derived?"}},
    ]
    doc = Interpreter(make_cache()).run(program, query=None)
    assert doc["query"] == "derived?"


# ----------------------------------------------------------------------------------------
# Cache loaders + accessors
# ----------------------------------------------------------------------------------------

def test_cache_load_round_trips_json_file(tmp_path):
    data = {
        "clip": {"id": "z", "width": 10, "height": 20, "duration_ms": 1000},
        "detect": {"4625": [{"det_id": "p", "cls": "person", "confidence": 0.5, "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]},
        "read_text": {"p": {"text": "10", "source": "pinned"}},
        "events": [{"ts_ms": 4000, "type": "goal", "scorer_det": "p"},
                   {"ts_ms": 4500, "type": "kickoff"}],
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(data))
    cache = Cache.load(path)
    assert cache.clip == data["clip"]
    # analyzed_frames returns int keys, sorted.
    assert cache.analyzed_frames() == [4625]
    # detections_at takes an int and reads the str key.
    assert cache.detections_at(4625)[0]["det_id"] == "p"
    assert cache.detections_at(9999) == []
    # read_text_for slices by det_id.
    assert cache.read_text_for("p")["text"] == "10"
    assert cache.read_text_for("nope") is None
    # goal_events filters type == goal and sorts by ts.
    goals = cache.goal_events()
    assert len(goals) == 1
    assert goals[0]["ts_ms"] == 4000


def test_cache_goal_events_sorted_by_ts():
    cache = make_cache(events=[
        {"ts_ms": 5000, "type": "goal"},
        {"ts_ms": 3000, "type": "goal"},
        {"ts_ms": 4000, "type": "kickoff"},
    ])
    assert [e["ts_ms"] for e in cache.goal_events()] == [3000, 5000]
