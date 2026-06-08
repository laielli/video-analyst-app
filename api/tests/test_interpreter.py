"""
Interpreter-level orchestration + cache loaders.

Covers Interpreter.run()'s error paths (unknown op, undefined binding reference), the
grounded-vs-ungrounded findings split it assembles, and the Cache loader/accessors
(Cache.load round-trips a JSON file; read_text_for/goal_events return the right slices).
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


# --------------------------------------------------------------------------------------
# Interpreter.run error paths
# --------------------------------------------------------------------------------------

def test_run_raises_on_unknown_op():
    interp = Interpreter(make_cache())
    program = [{"id": "x", "op": "not_a_real_op", "args": {}}]
    with pytest.raises(ValueError, match="unknown op 'not_a_real_op'"):
        interp.run(program)


def test_run_raises_on_undefined_binding_reference():
    interp = Interpreter(make_cache())
    # `detect` references frames binding "missing" which was never produced.
    program = [{"id": "people", "op": "detect", "args": {"frames": "missing", "classes": ["person"]}}]
    with pytest.raises(ValueError, match="references undefined binding 'missing'"):
        interp.run(program)


# --------------------------------------------------------------------------------------
# Interpreter.run findings split
# --------------------------------------------------------------------------------------

def _hero_program(q="Does #10 score the first goal?"):
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": q}},
    ]


def test_run_assembles_grounded_findings():
    cache = make_cache(
        detect={"3625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                          "box": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}}]},
        read_text={"p_messi": {"text": "10", "source": "pinned"}},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi"}],
    )
    doc = Interpreter(cache).run(_hero_program())
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["reason"] is None
    assert f["partial"] is False
    assert f["answer"] == "Yes"
    # supporting_step points at the temporal_order step
    assert f["supporting_step"] == "ordered"
    # query falls through to the answer step's question when not passed
    assert doc["query"] == "Does #10 score the first goal?"
    assert doc["clip"] == cache.clip


def test_run_assembles_ungrounded_findings_when_subject_absent():
    # Goal exists but no jersey reads -> filter empties -> subject absent -> ungrounded.
    cache = make_cache(
        detect={"3625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                          "box": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5}}]},
        read_text={},  # nothing legible
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi"}],
    )
    doc = Interpreter(cache).run(_hero_program())
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["partial"] is True
    assert f["answer"] is None and f["verdict"] is None
    assert f["reason"] == "no-grounded-answer"


def test_run_ungrounded_when_no_answer_step():
    cache = make_cache()
    program = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    doc = Interpreter(cache).run(program, query="just sampling")
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["reason"] == "no-answer-step"
    assert doc["query"] == "just sampling"


def test_run_explicit_query_overrides_answer_question():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p"}])
    program = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    doc = Interpreter(cache).run(program, query="my custom query")
    assert doc["query"] == "my custom query"
    assert doc["program"] is program  # program passed through
    assert isinstance(doc["trace"], list) and len(doc["trace"]) == 1


# --------------------------------------------------------------------------------------
# Cache loader + accessors
# --------------------------------------------------------------------------------------

def test_cache_load_round_trips_json_file(tmp_path):
    payload = {
        "clip": {"id": "c", "width": 10, "height": 20, "duration_ms": 1000, "fps": 60},
        "detect": {"100": [{"det_id": "p0", "cls": "person", "confidence": 0.9,
                            "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]},
        "read_text": {"p0": {"text": "10", "source": "pinned"}},
        "events": [
            {"ts_ms": 200, "type": "goal", "scorer_det": "p0"},
            {"ts_ms": 100, "type": "goal", "scorer_det": "p1"},
            {"ts_ms": 50, "type": "kickoff"},
        ],
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(payload))
    cache = Cache.load(p)
    assert cache.clip == payload["clip"]
    # analyzed_frames -> int-sorted timestamps with cached detections
    assert cache.analyzed_frames() == [100]
    # detections_at keys are string ms
    assert cache.detections_at(100)[0]["det_id"] == "p0"
    assert cache.detections_at(999) == []  # absent -> empty
    # read_text_for slices the read dict
    assert cache.read_text_for("p0")["text"] == "10"
    assert cache.read_text_for("nope") is None
    # goal_events filters type==goal and sorts by ts_ms (kickoff dropped)
    goals = cache.goal_events()
    assert [g["ts_ms"] for g in goals] == [100, 200]
    assert all(g["type"] == "goal" for g in goals)


def test_cache_load_round_trips_committed_hero_cache():
    from pathlib import Path
    api_dir = Path(__file__).resolve().parent.parent
    cache = Cache.load(api_dir / "examples" / "hero_cache.json")
    assert cache.clip["id"] == "single-goal"
    assert cache.analyzed_frames() == [4625]
    assert cache.read_text_for("p_messi")["text"] == "10"
    assert cache.goal_events()[0]["scorer_det"] == "p_messi"
