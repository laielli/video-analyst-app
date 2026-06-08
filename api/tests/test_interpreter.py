"""
Interpreter-level orchestration + Cache loader tests. The interpreter walks a (pre-validated)
DSL program over a Cache, maintaining a binding environment and emitting a run-doc. These tests
cover the orchestration contracts the per-op tests can't: unknown-op rejection, undefined-binding
rejection, and the grounded-vs-ungrounded findings split run() assembles. Plus the Cache loader
round-trips a JSON file and its accessors return the right slices.
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


# ---- run() error paths ------------------------------------------------------------------

def test_run_raises_on_unknown_op():
    program = [{"id": "x", "op": "teleport", "args": {}}]
    with pytest.raises(ValueError, match="unknown op 'teleport'"):
        Interpreter(make_cache()).run(program)


def test_run_raises_on_undefined_binding_reference():
    # `detect` references frames binding "ghost" which is never produced.
    program = [{"id": "people", "op": "detect", "args": {"frames": "ghost", "classes": ["person"]}}]
    with pytest.raises(ValueError, match="references undefined binding 'ghost'"):
        Interpreter(make_cache()).run(program)


# ---- run() findings split ---------------------------------------------------------------

def test_run_assembles_grounded_findings():
    cache = make_cache(
        detect={"4625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                          "box": {"x": 0.03, "y": 0.19, "w": 0.22, "h": 0.63}}]},
        read_text={"p_messi": {"text": "10", "confidence": None, "source": "pinned"}},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi",
                 "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}}],
    )
    program = [
        # start at the cached frame ts (4625) so the sampled stride lands exactly on it.
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4625, "end_ms": 4700, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score the first goal?"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["partial"] is False
    assert f["answer"] == "Yes"
    assert f["reason"] is None
    # supporting_step prefers the temporal_order step.
    assert f["supporting_step"] == "ordered"
    # query falls back to the answer step's question when not passed.
    assert doc["query"] == "Does #10 score the first goal?"
    assert doc["clip"] == cache.clip
    assert len(doc["trace"]) == len(program)


def test_run_assembles_ungrounded_findings_when_answer_cannot_ground():
    # count over an empty detect set -> answer can't ground -> ungrounded with the op's reason.
    cache = make_cache(detect={})
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
    assert f["answer"] is None and f["verdict"] is None
    assert f["reason"] == "no-grounded-answer"


def test_run_ungrounded_no_answer_step():
    # program with no answer op at all -> findings reason 'no-answer-step'.
    cache = make_cache(detect={})
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "n", "op": "count", "args": {"items": "frames"}},
    ]
    doc = Interpreter(cache).run(program)
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["reason"] == "no-answer-step"
    # query defaults to "" when no answer step supplies a question.
    assert doc["query"] == ""


def test_run_explicit_query_overrides_answer_question():
    cache = make_cache(detect={})
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "n", "op": "count", "args": {"items": "frames"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "inner question"}},
    ]
    doc = Interpreter(cache).run(program, query="outer question")
    assert doc["query"] == "outer question"


# ---- Cache loader + accessors -----------------------------------------------------------

def test_cache_load_roundtrips_and_accessors(tmp_path):
    data = {
        "clip": {"id": "c1", "width": 200, "height": 100, "duration_ms": 6000},
        "detect": {"4625": [{"det_id": "p0", "cls": "person", "confidence": 0.9,
                             "box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}],
                   "100": [{"det_id": "q0", "cls": "ball", "confidence": 0.5,
                            "box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}]},
        "read_text": {"p0": {"text": "10", "confidence": None, "source": "pinned"}},
        "events": [
            {"ts_ms": 5000, "type": "goal", "scorer_det": "p0"},
            {"ts_ms": 4000, "type": "goal", "scorer_det": "p1"},
            {"ts_ms": 1000, "type": "kickoff"},
        ],
    }
    p = tmp_path / "cache.json"
    p.write_text(json.dumps(data))
    cache = Cache.load(p)

    assert cache.clip == data["clip"]
    # analyzed_frames sorts keys as ints.
    assert cache.analyzed_frames() == [100, 4625]
    # detections_at keys on str(ts).
    assert cache.detections_at(4625)[0]["det_id"] == "p0"
    assert cache.detections_at(999) == []
    # read_text_for slices by det_id.
    assert cache.read_text_for("p0")["text"] == "10"
    assert cache.read_text_for("missing") is None
    # goal_events filters type==goal and sorts by ts.
    goals = cache.goal_events()
    assert [g["ts_ms"] for g in goals] == [4000, 5000]
    assert all(g["type"] == "goal" for g in goals)


def test_cache_load_path_accepts_string(tmp_path):
    data = {"clip": {"id": "c"}, "detect": {}, "read_text": {}, "events": []}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    cache = Cache.load(str(p))  # str path also works
    assert cache.clip["id"] == "c"
