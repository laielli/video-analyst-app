"""
Dedicated unit tests for the 8 ops in interpreter/primitives.py.

Each op has the uniform contract fn(step, env, cache) -> (binding, step_result):
  - binding: the value stored under step["id"] for later steps.
  - step_result: the run-doc trace entry (status/source/output_label/evidence...).

detect/read_text/temporal_order read from the Cache; crop/filter/count are pure compute
over `env`; op_answer dispatches on the `from` binding's kind and (for the grounded temporal
branch only) re-reads the cache via analyzed_frames()/detections_at() to build scorer overlays.

We build a minimal in-memory Cache directly (cache.py wraps the dict; it reads data["clip"], so
the fixture MUST carry a "clip" key). Tests assert on BOTH the returned binding and the run-doc
result entry, so they bite on either side of the contract.
"""
from __future__ import annotations

from interpreter import Cache
from interpreter.primitives import (
    op_sample_frames, op_detect, op_crop, op_read_text,
    op_filter, op_count, op_temporal_order, op_answer, OPS,
    _subject_label,
)


def make_cache(detect=None, read_text=None, events=None):
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {},
        "read_text": read_text or {},
        "events": events or [],
    })


def step(id, op, args):
    return {"id": id, "op": op, "args": args}


# ----------------------------------------------------------------------------------------
# op_sample_frames
# ----------------------------------------------------------------------------------------

def test_sample_frames_stride_and_count():
    s = step("frames", "sample_frames", {"start_ms": 0, "end_ms": 1000, "fps": 8})
    binding, result = op_sample_frames(s, {}, make_cache())
    # stride = max(1, round(1000/8)) = 125 -> 0,125,...,1000 = 9 frames
    assert binding["kind"] == "frames"
    ts = [it["frame_ts_ms"] for it in binding["items"]]
    assert ts == [0, 125, 250, 375, 500, 625, 750, 875, 1000]
    assert result["status"] == "done"
    assert result["op"] == "sample_frames"
    assert result["output_label"] == "9 frames @ 8fps"
    assert result["input_label"] == "clip 0-1000ms"
    # evidence focuses the middle frame
    assert result["evidence"]["frame_ts_ms"] == ts[len(ts) // 2]


def test_sample_frames_high_fps_collapses_stride_to_one():
    # fps high enough that round(1000/fps) -> 0, clamped to 1 -> one frame per ms.
    s = step("frames", "sample_frames", {"start_ms": 10, "end_ms": 13, "fps": 3000})
    binding, _ = op_sample_frames(s, {}, make_cache())
    assert [it["frame_ts_ms"] for it in binding["items"]] == [10, 11, 12, 13]


# ----------------------------------------------------------------------------------------
# op_detect
# ----------------------------------------------------------------------------------------

def _detect_cache():
    return make_cache(detect={
        "100": [
            {"det_id": "a", "cls": "Person", "confidence": 0.9, "box": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.3}},
            {"det_id": "b", "cls": "ball", "confidence": 0.4, "box": {"x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05}},
        ],
        "200": [
            {"det_id": "c", "cls": "PERSON", "confidence": 0.7, "box": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.3}},
        ],
    })


def test_detect_case_insensitive_match_and_status_done():
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}, {"frame_ts_ms": 200}]}}
    s = step("people", "detect", {"frames": "frames", "classes": ["person"]})
    binding, result = op_detect(s, env, _detect_cache())
    # case-insensitive match: "Person" and "PERSON" both match "person"; ball excluded.
    ids = [d["det_id"] for d in binding["items"]]
    assert ids == ["a", "c"]
    assert result["status"] == "done"
    assert result["output_label"] == "2 people"
    assert result["confidence"] == 0.9  # max confidence
    # overlays only on focus_ts (last det's frame = 200), so only det "c" overlays.
    assert result["evidence"]["frame_ts_ms"] == 200
    assert len(result["evidence"]["overlays"]) == 1
    assert result["evidence"]["overlays"][0]["tone"] == "azure"


def test_detect_non_person_noun_is_objects():
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    s = step("balls", "detect", {"frames": "frames", "classes": ["ball"]})
    _, result = op_detect(s, env, _detect_cache())
    assert result["output_label"] == "1 objects"


def test_detect_empty_status_and_note():
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    s = step("cars", "detect", {"frames": "frames", "classes": ["car"]})
    binding, result = op_detect(s, env, _detect_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["note"] == "no detections in the sampled frames"
    # focus falls back to the last frame's ts (no dets).
    assert result["evidence"]["frame_ts_ms"] == 100
    assert result["confidence"] is None


# ----------------------------------------------------------------------------------------
# op_crop
# ----------------------------------------------------------------------------------------

def _crop_env():
    return {"people": {"kind": "detections", "items": [
        {"det_id": "a", "confidence": 0.9, "box": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}, "frame_ts_ms": 100},
        {"det_id": "b", "confidence": 0.5, "box": {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.6}, "frame_ts_ms": 200},
    ]}}


def test_crop_jersey_region_box_math():
    s = step("jerseys", "crop", {"detections": "people", "region": "jersey"})
    binding, result = op_crop(s, _crop_env(), make_cache())
    c0 = binding["items"][0]
    # jersey offsets 0.22/0.10/0.56/0.22 over box (0,0,1,1)
    assert c0["box"] == {"x": 0.22, "y": 0.10, "w": 0.56, "h": 0.22}
    assert c0["crop_id"] == "c0"
    assert c0["det_id"] == "a"
    assert c0["person_box"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    assert c0["person_conf"] == 0.9
    assert binding["items"][1]["crop_id"] == "c1"
    assert result["status"] == "done"
    assert result["output_label"] == "2 jersey crops"
    assert result["producer"] == "crop_jersey"
    # overlays only on focus_ts (last crop's frame = 200).
    assert result["evidence"]["frame_ts_ms"] == 200
    assert result["evidence"]["overlays"][0]["label"] == "crop_jersey"


def test_crop_full_region_passthrough():
    s = step("jerseys", "crop", {"detections": "people", "region": "full"})
    binding, result = op_crop(s, _crop_env(), make_cache())
    # full region copies the person box verbatim.
    assert binding["items"][0]["box"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    assert result["evidence"]["overlays"][0]["label"] == "crop_full"


def test_crop_empty_when_no_detections():
    env = {"people": {"kind": "detections", "items": []}}
    s = step("jerseys", "crop", {"detections": "people", "region": "jersey"})
    binding, result = op_crop(s, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"


# ----------------------------------------------------------------------------------------
# op_read_text
# ----------------------------------------------------------------------------------------

def _read_env():
    return {"jerseys": {"kind": "crops", "items": [
        {"det_id": "a", "box": {"x": 0, "y": 0, "w": 1, "h": 1}, "person_box": {"x": 0, "y": 0, "w": 1, "h": 1}, "frame_ts_ms": 100},
        {"det_id": "b", "box": {"x": 0, "y": 0, "w": 1, "h": 1}, "person_box": {"x": 0, "y": 0, "w": 1, "h": 1}, "frame_ts_ms": 200},
    ]}}


def test_read_text_source_precedence_pinned_wins():
    cache = make_cache(read_text={
        "a": {"text": "10", "confidence": 0.8, "source": "cached"},
        "b": {"text": "7", "confidence": 0.9, "source": "pinned"},
    })
    s = step("numbers", "read_text", {"crops": "jerseys"})
    binding, result = op_read_text(s, _read_env(), cache)
    assert {t["text"] for t in binding["items"]} == {"10", "7"}
    # pinned present -> step_source is "pinned" even though "cached" also present.
    assert result["source"] == "pinned"
    # output_label is sorted unique texts.
    assert result["output_label"] == "10, 7"


def test_read_text_cached_over_live():
    cache = make_cache(read_text={
        "a": {"text": "5", "source": "live"},
        "b": {"text": "9", "source": "cached"},
    })
    s = step("numbers", "read_text", {"crops": "jerseys"})
    _, result = op_read_text(s, _read_env(), cache)
    assert result["source"] == "cached"


def test_read_text_note_passthrough_and_confidence():
    cache = make_cache(read_text={
        "a": {"text": "10", "confidence": None, "source": "pinned", "note": "hand-verified"},
        "b": {"text": "10", "confidence": 0.42, "source": "pinned"},
    })
    s = step("numbers", "read_text", {"crops": "jerseys"})
    _, result = op_read_text(s, _read_env(), cache)
    assert result["note"] == "hand-verified"
    # first non-None confidence is carried.
    assert result["confidence"] == 0.42
    # duplicate texts collapse in the sorted-unique label.
    assert result["output_label"] == "10"


def test_read_text_nothing_legible_note():
    cache = make_cache(read_text={})  # no reads for any det
    s = step("numbers", "read_text", {"crops": "jerseys"})
    binding, result = op_read_text(s, _read_env(), cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "(none legible)"
    assert result["note"] == "no legible text in the jersey crops (read returned nothing)"


# ----------------------------------------------------------------------------------------
# op_filter
# ----------------------------------------------------------------------------------------

def test_filter_string_coerced_equality_keeps_matches():
    env = {"numbers": {"kind": "texts", "items": [
        {"text": "10", "person_box": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, "frame_ts_ms": 100},
        {"text": 10, "frame_ts_ms": 150},   # int 10 == "10" via str coercion
        {"text": "7", "frame_ts_ms": 200},
    ]}}
    s = step("tens", "filter", {"items": "numbers", "where": {"field": "text", "equals": "10"}})
    binding, result = op_filter(s, env, make_cache())
    assert len(binding["items"]) == 2
    assert binding["kind"] == "texts"  # kind preserved
    assert result["status"] == "done"
    assert result["output_label"] == "2 match · #10"
    assert result["input_label"] == "texts where text == 10"
    # overlay uses person_box (or box) of matched items.
    assert result["evidence"]["overlays"][0]["label"] == "#10"


def test_filter_empty_match_note():
    env = {"numbers": {"kind": "texts", "items": [{"text": "7"}]}}
    s = step("tens", "filter", {"items": "numbers", "where": {"field": "text", "equals": "10"}})
    binding, result = op_filter(s, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "0 match"
    assert result["note"] == "no items with text == 10"


# ----------------------------------------------------------------------------------------
# op_count
# ----------------------------------------------------------------------------------------

def test_count_value_and_output_label():
    env = {"people": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}, {"det_id": "c"}]}}
    s = step("n", "count", {"items": "people"})
    binding, result = op_count(s, env, make_cache())
    assert binding == {"kind": "number", "value": 3}
    assert result["output_label"] == "3"
    assert result["input_label"] == "detections"
    assert result["status"] == "done"


def test_count_zero():
    env = {"people": {"kind": "detections", "items": []}}
    s = step("n", "count", {"items": "people"})
    binding, result = op_count(s, env, make_cache())
    assert binding["value"] == 0
    assert result["output_label"] == "0"


# ----------------------------------------------------------------------------------------
# _subject_label helper
# ----------------------------------------------------------------------------------------

def test_subject_label_derivation():
    assert _subject_label([{"text": "10"}]) == "#10"
    assert _subject_label([{"text": ""}, {"text": "7"}]) == "#7"
    assert _subject_label([{"text": None}, {"foo": "bar"}]) is None
    assert _subject_label([]) is None
    assert _subject_label(None) is None


# ----------------------------------------------------------------------------------------
# op_temporal_order
# ----------------------------------------------------------------------------------------

def _goal_cache():
    return make_cache(events=[
        {"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi", "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}},
        {"ts_ms": 4500, "type": "goal", "scorer_det": "p_other"},
        {"ts_ms": 1000, "type": "kickoff"},  # non-goal ignored
    ])


def test_temporal_order_subject_is_first_scorer_true():
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p_messi", "text": "10"}]}}
    s = step("ordered", "temporal_order", {"events": "tens", "by": "timestamp"})
    binding, result = op_temporal_order(s, env, _goal_cache())
    v = binding["value"]
    assert v["first"]["ts_ms"] == 4000  # goals sorted by ts; kickoff excluded
    assert v["subject_label"] == "#10"
    assert v["subject_is_first_scorer"] is True
    assert v["subject_dets"] == ["p_messi"]
    assert result["status"] == "done"
    assert result["output_label"] == "goal @ 4000ms (first)"
    # green goal overlay at the first goal ts.
    assert result["evidence"]["frame_ts_ms"] == 4000
    assert result["evidence"]["overlays"][0]["tone"] == "green"


def test_temporal_order_subject_is_first_scorer_false():
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p_other", "text": "7"}]}}
    s = step("ordered", "temporal_order", {"events": "tens", "by": "timestamp"})
    binding, _ = op_temporal_order(s, env, _goal_cache())
    # p_other scored the 4500 goal, not the first (4000) -> not first scorer.
    assert binding["value"]["subject_is_first_scorer"] is False
    assert binding["value"]["subject_label"] == "#7"


def test_temporal_order_empty_when_no_goal_events():
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p_messi", "text": "10"}]}}
    s = step("ordered", "temporal_order", {"events": "tens", "by": "timestamp"})
    binding, result = op_temporal_order(s, env, make_cache(events=[]))
    assert binding["value"]["first"] is None
    assert binding["value"]["subject_is_first_scorer"] is False
    assert result["status"] == "empty"
    assert result["output_label"] == "no goal events"


# ----------------------------------------------------------------------------------------
# op_answer — highest-value op; dispatch by `from` binding kind + ungrounded discriminator
# ----------------------------------------------------------------------------------------

def test_answer_ordered_grounded_yes_with_scorer_overlay():
    # The grounded temporal branch re-reads the cache (analyzed_frames/detections_at) to overlay
    # the scorer box, so the cache MUST carry a `detect` entry with the scorer det_id.
    cache = make_cache(
        detect={"4000": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                          "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}}]},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi"}],
    )
    env = {"ordered": {"kind": "ordered", "value": {
        "first": {"ts_ms": 4000, "scorer_det": "p_messi"},
        "subject_label": "#10", "subject_is_first_scorer": True,
    }}}
    s = step("result", "answer", {"from": "ordered", "question": "Does #10 score the first goal?"})
    binding, result = op_answer(s, env, cache)
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "Yes"
    assert binding["value"]["yes"] is True
    assert binding["value"]["verdict"] == "Yes — #10 scored the first goal"
    assert result["status"] == "done"
    assert result["output_label"] == "Yes"
    # the scorer-overlay re-read path fired: a green box on the scorer at ts 4000.
    assert result["evidence"]["frame_ts_ms"] == 4000
    ov = result["evidence"]["overlays"]
    assert ov and ov[0]["label"] == "#10 scored" and ov[0]["tone"] == "green"


def test_answer_ordered_grounded_no():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi"}])
    env = {"ordered": {"kind": "ordered", "value": {
        "first": {"ts_ms": 4000, "scorer_det": "p_messi"},
        "subject_label": "#7", "subject_is_first_scorer": False,
    }}}
    s = step("result", "answer", {"from": "ordered", "question": "Does #7 score the first goal?"})
    binding, result = op_answer(s, env, cache)
    assert binding["value"]["answer"] == "No"
    assert binding["value"]["yes"] is False
    assert binding["value"]["verdict"] == "No — the first goal was not scored by #7"
    assert result["output_label"] == "No"


def test_answer_ordered_ungrounded_no_first():
    env = {"ordered": {"kind": "ordered", "value": {"first": None, "subject_label": "#10"}}}
    s = step("result", "answer", {"from": "ordered", "question": "q"})
    binding, result = op_answer(s, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"
    assert binding["value"]["answer"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "(couldn't ground)"


def test_answer_ordered_ungrounded_no_subject():
    # A first goal exists but the subject filter matched nobody -> ground out, don't overclaim "No".
    env = {"ordered": {"kind": "ordered", "value": {
        "first": {"ts_ms": 4000, "scorer_det": "p_messi"}, "subject_label": None,
        "subject_is_first_scorer": False,
    }}}
    s = step("result", "answer", {"from": "ordered", "question": "q"})
    binding, _ = op_answer(s, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_count_grounded_and_ungrounded():
    env = {"n": {"kind": "number", "value": 5}}
    s = step("result", "answer", {"from": "n", "question": "how many?"})
    binding, result = op_answer(s, env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "5"
    assert binding["value"]["verdict"] == "5 found"
    assert result["output_label"] == "5"

    env0 = {"n": {"kind": "number", "value": 0}}
    s0 = step("result", "answer", {"from": "n", "question": "how many?"})
    b0, _ = op_answer(s0, env0, make_cache())
    assert b0["value"]["grounded"] is False
    assert b0["value"]["reason"] == "no-grounded-answer"


def test_answer_texts_grounded_and_ungrounded():
    env = {"t": {"kind": "texts", "items": [{"text": "10"}, {"text": "7"}, {"text": "10"}]}}
    s = step("result", "answer", {"from": "t", "question": "what numbers?"})
    binding, result = op_answer(s, env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "10, 7"  # sorted unique
    assert binding["value"]["verdict"] == "Read: 10, 7"
    assert result["output_label"] == "10, 7"

    env_empty = {"t": {"kind": "texts", "items": [{"text": ""}, {"foo": "bar"}]}}
    b, _ = op_answer(step("r", "answer", {"from": "t", "question": "q"}), env_empty, make_cache())
    assert b["value"]["grounded"] is False
    assert b["value"]["reason"] == "no-grounded-answer"


def test_answer_presence_grounded_and_ungrounded():
    env = {"d": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}]}}
    s = step("result", "answer", {"from": "d", "question": "is there a player?"})
    binding, result = op_answer(s, env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "Yes"
    assert binding["value"]["yes"] is True
    assert binding["value"]["verdict"] == "Yes — found 2"
    assert result["output_label"] == "Yes"

    env_empty = {"d": {"kind": "crops", "items": []}}
    b, _ = op_answer(step("r", "answer", {"from": "d", "question": "q"}), env_empty, make_cache())
    assert b["value"]["grounded"] is False
    assert b["value"]["reason"] == "no-grounded-answer"


def test_answer_presence_over_frames_kind():
    env = {"f": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    b, r = op_answer(step("r", "answer", {"from": "f", "question": "q"}), env, make_cache())
    assert b["value"]["grounded"] is True
    assert b["value"]["answer"] == "Yes"


def test_answer_unsupported_kind_is_ungrounded():
    env = {"x": {"kind": "answer", "value": {"grounded": True}}}  # answer kind isn't dispatchable
    b, r = op_answer(step("r", "answer", {"from": "x", "question": "q"}), env, make_cache())
    assert b["value"]["grounded"] is False
    assert b["value"]["reason"] == "unsupported-kind"
    assert "unsupported-kind" in r["note"]


# ----------------------------------------------------------------------------------------
# OPS registry — all 8 ops registered
# ----------------------------------------------------------------------------------------

def test_ops_registry_has_eight_ops():
    assert set(OPS) == {
        "sample_frames", "detect", "crop", "read_text",
        "filter", "count", "temporal_order", "answer",
    }
    assert all(callable(fn) for fn in OPS.values())
