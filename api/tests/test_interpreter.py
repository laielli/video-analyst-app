"""
Interpreter-level orchestration + Cache loaders.

Covers:
  (1) run() raises on an unknown op name,
  (2) run() raises on an undefined binding reference,
  (3) the grounded-vs-ungrounded `findings` split run() assembles,
  (4) Cache.load() round-trips a JSON file and read_text_for/goal_events/detections_at/
      analyzed_frames accessors return the right slices.
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


# ---------------------------------------------------------------- error paths

def test_run_raises_on_unknown_op():
    interp = Interpreter(make_cache())
    program = [{"id": "x", "op": "no_such_op", "args": {}}]
    with pytest.raises(ValueError, match="unknown op 'no_such_op'"):
        interp.run(program)


def test_run_raises_on_undefined_binding_reference():
    interp = Interpreter(make_cache())
    # `detect` references frames binding 'missing' that was never produced.
    program = [{"id": "people", "op": "detect",
                "args": {"frames": "missing", "classes": ["person"]}}]
    with pytest.raises(ValueError, match="references undefined binding 'missing'"):
        interp.run(program)


def test_refs_excludes_literal_args():
    # region/by/where/classes are literals, not binding refs -> must not be checked as bindings.
    step = {"id": "j", "op": "crop", "args": {"detections": "people", "region": "jersey"}}
    assert Interpreter._refs(step) == ["people"]
    step2 = {"id": "f", "op": "sample_frames",
             "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}
    assert Interpreter._refs(step2) == []


# ---------------------------------------------------------------- findings split

def _grounded_program():
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score first?"}},
    ]


def _grounded_cache():
    dets = [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
             "box": {"x": 0.03, "y": 0.19, "w": 0.22, "h": 0.63}}]
    return make_cache(
        detect={"4625": dets},
        read_text={"p_messi": {"text": "10", "confidence": None, "source": "pinned"}},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi",
                 "box": {"x": 0.86, "y": 0.34, "w": 0.1, "h": 0.34}}],
    )


def test_run_assembles_grounded_findings():
    doc = Interpreter(_grounded_cache()).run(_grounded_program())
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["answer"] == "Yes"
    assert f["verdict"] == "Yes — #10 scored the first goal"
    assert f["reason"] is None
    assert f["partial"] is False
    # supporting_step prefers the temporal_order step id.
    assert f["supporting_step"] == "ordered"
    # full run-doc shape.
    assert doc["schema_version"] == "1"
    assert doc["query"] == "Does #10 score first?"
    assert doc["clip"]["id"] == "t"
    assert len(doc["trace"]) == len(_grounded_program())


def test_run_query_falls_back_to_answer_question():
    # No explicit query -> derived from the answer step's question.
    doc = Interpreter(_grounded_cache()).run(_grounded_program())
    assert doc["query"] == "Does #10 score first?"


def test_run_explicit_query_overrides():
    doc = Interpreter(_grounded_cache()).run(_grounded_program(), query="custom?")
    assert doc["query"] == "custom?"


def test_run_assembles_ungrounded_findings_when_no_subject():
    # Goal exists, but no jersey reads "10" -> the temporal subject is empty -> ungrounded.
    cache = make_cache(
        detect={"4625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                          "box": {"x": 0.0, "y": 0.0, "w": 0.2, "h": 0.6}}]},
        read_text={},  # nothing legible
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi", "box": None}],
    )
    doc = Interpreter(cache).run(_grounded_program())
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["answer"] is None
    assert f["verdict"] is None
    assert f["partial"] is True
    assert f["reason"] == "no-grounded-answer"


def test_run_ungrounded_when_no_answer_step():
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "n", "op": "count", "args": {"items": "frames"}},
    ]
    doc = Interpreter(make_cache()).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["reason"] == "no-answer-step"
    assert f["partial"] is True


# ---------------------------------------------------------------- Cache loaders

def test_cache_load_round_trips_json(tmp_path):
    data = {
        "clip": {"id": "c", "width": 10, "height": 20, "duration_ms": 1000},
        "detect": {"100": [{"det_id": "d0", "cls": "person", "confidence": 0.5,
                            "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]},
        "read_text": {"d0": {"text": "9", "confidence": 0.8, "source": "cached"}},
        "events": [{"ts_ms": 200, "type": "goal", "scorer_det": "d0"},
                   {"ts_ms": 50, "type": "kickoff"}],
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(data))
    cache = Cache.load(p)
    assert cache.clip["id"] == "c"
    assert cache.analyzed_frames() == [100]
    assert cache.detections_at(100)[0]["det_id"] == "d0"
    assert cache.detections_at(999) == []  # missing ts -> empty list
    assert cache.read_text_for("d0") == {"text": "9", "confidence": 0.8, "source": "cached"}
    assert cache.read_text_for("nope") is None
    # goal_events filters to type=goal and sorts by ts.
    goals = cache.goal_events()
    assert len(goals) == 1
    assert goals[0]["ts_ms"] == 200


def test_cache_goal_events_sorted():
    cache = make_cache(events=[
        {"ts_ms": 500, "type": "goal", "scorer_det": "b"},
        {"ts_ms": 100, "type": "goal", "scorer_det": "a"},
        {"ts_ms": 300, "type": "kickoff"},
    ])
    goals = cache.goal_events()
    assert [g["ts_ms"] for g in goals] == [100, 500]


def test_cache_analyzed_frames_sorted_ints():
    cache = make_cache(detect={"4625": [], "100": [], "2000": []})
    assert cache.analyzed_frames() == [100, 2000, 4625]
