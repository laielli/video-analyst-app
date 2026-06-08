"""
Dedicated unit tests for the 8 ops in interpreter/primitives.py.

Each op has the uniform contract fn(step, env, cache) -> (binding, step_result):
  - binding: the value later steps reference (stored under step["id"]).
  - step_result: the run-doc trace entry (status, provenance, EVIDENCE overlays).

detect/read_text/temporal_order (and op_answer's grounded temporal branch) read from the
Cache; crop/filter/count are pure compute over `env`. We build a tiny in-memory Cache that
mirrors the real hero_cache.json shape — note the "clip" key is REQUIRED (cache.py reads
data["clip"]) and the grounded `ordered` answer path re-reads `detect` to draw the scorer
overlay, so that fixture carries detect entries.
"""
from __future__ import annotations

from interpreter import Cache
from interpreter.primitives import (
    op_sample_frames, op_detect, op_crop, op_read_text,
    op_filter, op_count, op_temporal_order, op_answer, OPS, _subject_label,
)


def make_cache(detect=None, read_text=None, events=None):
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {},
        "read_text": read_text or {},
        "events": events or [],
    })


def box(x=0.1, y=0.1, w=0.2, h=0.4):
    return {"x": x, "y": y, "w": w, "h": h}


# ---------------------------------------------------------------- op_sample_frames

def test_sample_frames_count_and_label():
    step = {"id": "frames", "op": "sample_frames",
            "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}
    binding, result = op_sample_frames(step, {}, make_cache())
    # stride = max(1, round(1000/8)) = round(125) = 125; range(0, 1001, 125) -> 9 frames.
    assert binding["kind"] == "frames"
    assert len(binding["items"]) == 9
    assert binding["items"][0] == {"frame_ts_ms": 0}
    assert result["op"] == "sample_frames"
    assert result["status"] == "done"
    assert result["output_label"] == "9 frames @ 8fps"
    assert result["input_label"] == "clip 0-1000ms"
    assert result["source"] == "live"


def test_sample_frames_stride_collapses_to_one():
    # fps high enough that 1000/fps rounds below 1 -> stride clamps to 1.
    step = {"id": "f", "op": "sample_frames",
            "args": {"start_ms": 0, "end_ms": 3, "fps": 2000}}
    binding, result = op_sample_frames(step, {}, make_cache())
    assert len(binding["items"]) == 4  # range(0, 4, 1)
    assert [it["frame_ts_ms"] for it in binding["items"]] == [0, 1, 2, 3]


def test_sample_frames_evidence_is_middle_frame():
    step = {"id": "f", "op": "sample_frames",
            "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}
    _, result = op_sample_frames(step, {}, make_cache())
    # 9 frames -> middle index 4 -> ts 500.
    assert result["evidence"]["frame_ts_ms"] == 500
    assert result["evidence"]["overlays"] == []


# ---------------------------------------------------------------- op_detect

def _detect_env_cache():
    dets = [
        {"det_id": "p0", "cls": "Person", "confidence": 0.9, "box": box()},
        {"det_id": "b0", "cls": "ball", "confidence": 0.4, "box": box(0.5, 0.5, 0.05, 0.05)},
    ]
    cache = make_cache(detect={"100": dets})
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    return env, cache


def test_detect_case_insensitive_match_and_status_done():
    env, cache = _detect_env_cache()
    step = {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["PERSON"]}}
    binding, result = op_detect(step, env, cache)
    assert binding["kind"] == "detections"
    assert len(binding["items"]) == 1
    assert binding["items"][0]["det_id"] == "p0"
    assert binding["items"][0]["frame_ts_ms"] == 100
    assert result["status"] == "done"
    assert result["output_label"] == "1 people"
    assert result["confidence"] == 0.9
    assert result["source"] == "cached"
    assert "note" not in result


def test_detect_empty_status_and_note():
    env, cache = _detect_env_cache()
    step = {"id": "cars", "op": "detect", "args": {"frames": "frames", "classes": ["car"]}}
    binding, result = op_detect(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "0 objects"  # not "person" -> "objects" noun
    assert result["confidence"] is None
    assert result["note"] == "no detections in the sampled frames"


def test_detect_overlays_only_on_focus_ts():
    dets_a = [{"det_id": "p0", "cls": "person", "confidence": 0.8, "box": box()}]
    dets_b = [{"det_id": "p1", "cls": "person", "confidence": 0.7, "box": box(0.2, 0.2, 0.1, 0.1)}]
    cache = make_cache(detect={"100": dets_a, "200": dets_b})
    env = {"frames": {"kind": "frames",
                      "items": [{"frame_ts_ms": 100}, {"frame_ts_ms": 200}]}}
    step = {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}}
    binding, result = op_detect(step, env, cache)
    assert len(binding["items"]) == 2
    # focus_ts is the LAST detection's frame (200); overlays only for that frame.
    assert result["evidence"]["frame_ts_ms"] == 200
    assert len(result["evidence"]["overlays"]) == 1
    assert result["evidence"]["overlays"][0]["label"] == "person 0.70"
    assert result["evidence"]["overlays"][0]["tone"] == "azure"


# ---------------------------------------------------------------- op_crop

def _crop_env():
    dets = [
        {"det_id": "p0", "confidence": 0.9, "box": box(0.0, 0.0, 1.0, 1.0), "frame_ts_ms": 100},
        {"det_id": "p1", "confidence": 0.8, "box": box(0.1, 0.2, 0.4, 0.5), "frame_ts_ms": 200},
    ]
    return {"people": {"kind": "detections", "items": dets}}


def test_crop_jersey_box_math():
    env = _crop_env()
    step = {"id": "j", "op": "crop", "args": {"detections": "people", "region": "jersey"}}
    binding, result = op_crop(step, env, make_cache())
    assert binding["kind"] == "crops"
    assert len(binding["items"]) == 2
    c0 = binding["items"][0]
    # box (0,0,1,1): x = 0 + 0.22*1 = 0.22; y = 0 + 0.10*1 = 0.10; w = 0.56; h = 0.22
    assert c0["box"] == {"x": 0.22, "y": 0.10, "w": 0.56, "h": 0.22}
    assert c0["crop_id"] == "c0"
    assert c0["det_id"] == "p0"
    assert c0["person_box"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    assert c0["person_conf"] == 0.9
    assert c0["frame_ts_ms"] == 100
    assert binding["items"][1]["crop_id"] == "c1"
    assert result["output_label"] == "2 jersey crops"
    assert result["status"] == "done"


def test_crop_full_region_passthrough():
    env = _crop_env()
    step = {"id": "f", "op": "crop", "args": {"detections": "people", "region": "full"}}
    binding, result = op_crop(step, env, make_cache())
    c0 = binding["items"][0]
    assert c0["box"] == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}  # copy of person box
    # box is a distinct dict (dict(b)), not the same object.
    assert c0["box"] is not env["people"]["items"][0]["box"]
    assert result["producer"] == "crop_full"


def test_crop_empty():
    env = {"people": {"kind": "detections", "items": []}}
    step = {"id": "j", "op": "crop", "args": {"detections": "people", "region": "jersey"}}
    binding, result = op_crop(step, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"


# ---------------------------------------------------------------- op_read_text

def _read_env():
    crops = [
        {"crop_id": "c0", "det_id": "p0", "box": box(), "person_box": box(), "frame_ts_ms": 100},
        {"crop_id": "c1", "det_id": "p1", "box": box(0.3, 0.3, 0.1, 0.1),
         "person_box": box(0.3, 0.3, 0.1, 0.1), "frame_ts_ms": 200},
    ]
    return {"jerseys": {"kind": "crops", "items": crops}}


def test_read_text_source_precedence_pinned_wins():
    env = _read_env()
    cache = make_cache(read_text={
        "p0": {"text": "10", "confidence": 0.99, "source": "cached"},
        "p1": {"text": "7", "confidence": 0.5, "source": "pinned"},
    })
    step = {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}}
    binding, result = op_read_text(step, env, cache)
    assert binding["kind"] == "texts"
    assert len(binding["items"]) == 2
    # pinned present in sources -> step source is "pinned".
    assert result["source"] == "pinned"
    # output_label is sorted unique texts.
    assert result["output_label"] == "10, 7"
    assert result["status"] == "done"


def test_read_text_source_cached_when_no_pinned():
    env = _read_env()
    cache = make_cache(read_text={
        "p0": {"text": "10", "confidence": 0.99, "source": "cached"},
    })
    step = {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}}
    _, result = op_read_text(step, env, cache)
    assert result["source"] == "cached"
    assert result["output_label"] == "10"


def test_read_text_none_legible_note():
    env = _read_env()
    cache = make_cache(read_text={})  # nothing legible
    step = {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}}
    binding, result = op_read_text(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "(none legible)"
    assert result["note"] == "no legible text in the jersey crops (read returned nothing)"
    assert result["source"] == "live"  # no pinned/cached -> live


def test_read_text_carries_first_note():
    env = _read_env()
    cache = make_cache(read_text={
        "p0": {"text": "10", "confidence": None, "source": "pinned", "note": "hand-verified"},
    })
    step = {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}}
    _, result = op_read_text(step, env, cache)
    assert result["note"] == "hand-verified"


# ---------------------------------------------------------------- op_filter

def test_filter_string_coerced_equality():
    items = [
        {"det_id": "a", "text": 10, "person_box": box(), "frame_ts_ms": 100},
        {"det_id": "b", "text": "10", "person_box": box(), "frame_ts_ms": 200},
        {"det_id": "c", "text": "7", "person_box": box(), "frame_ts_ms": 300},
    ]
    env = {"numbers": {"kind": "texts", "items": items}}
    step = {"id": "tens", "op": "filter",
            "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}}
    binding, result = op_filter(step, env, make_cache())
    # str(10) == str("10") -> both the int and string match.
    assert len(binding["items"]) == 2
    assert {it["det_id"] for it in binding["items"]} == {"a", "b"}
    assert binding["kind"] == "texts"  # kind preserved
    assert result["status"] == "done"
    assert result["output_label"] == "2 match · #10"
    assert result["input_label"] == "texts where text == 10"


def test_filter_empty_match_note():
    items = [{"det_id": "a", "text": "7", "person_box": box(), "frame_ts_ms": 100}]
    env = {"numbers": {"kind": "texts", "items": items}}
    step = {"id": "tens", "op": "filter",
            "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}}
    binding, result = op_filter(step, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "0 match"
    assert result["note"] == "no items with text == 10"


def test_filter_overlay_uses_person_box_and_focus_ts():
    items = [{"det_id": "a", "text": "10", "person_box": box(0.4, 0.4, 0.2, 0.2),
              "frame_ts_ms": 222}]
    env = {"numbers": {"kind": "texts", "items": items}}
    step = {"id": "tens", "op": "filter",
            "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}}
    _, result = op_filter(step, env, make_cache())
    ov = result["evidence"]["overlays"]
    assert len(ov) == 1
    assert ov[0]["box"] == {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}
    assert ov[0]["label"] == "#10"
    assert result["evidence"]["frame_ts_ms"] == 222


# ---------------------------------------------------------------- op_count

def test_count_value_and_label():
    env = {"people": {"kind": "detections", "items": [{}, {}, {}]}}
    step = {"id": "n", "op": "count", "args": {"items": "people"}}
    binding, result = op_count(step, env, make_cache())
    assert binding == {"kind": "number", "value": 3}
    assert result["output_label"] == "3"
    assert result["input_label"] == "detections"
    assert result["status"] == "done"


def test_count_zero():
    env = {"people": {"kind": "detections", "items": []}}
    step = {"id": "n", "op": "count", "args": {"items": "people"}}
    binding, result = op_count(step, env, make_cache())
    assert binding["value"] == 0
    assert result["output_label"] == "0"


# ---------------------------------------------------------------- op_temporal_order

def test_subject_label_derivation_helper():
    assert _subject_label([{"text": "10"}]) == "#10"
    assert _subject_label([{"text": ""}, {"text": "7"}]) == "#7"
    assert _subject_label([{"text": None}]) is None
    assert _subject_label([]) is None


def test_temporal_order_subject_is_first_scorer_true():
    events = [{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi",
               "box": box(0.86, 0.34, 0.1, 0.34)}]
    cache = make_cache(events=events)
    subj_items = [{"det_id": "p_messi", "text": "10", "frame_ts_ms": 4625}]
    env = {"tens": {"kind": "texts", "items": subj_items}}
    step = {"id": "ordered", "op": "temporal_order",
            "args": {"events": "tens", "by": "timestamp"}}
    binding, result = op_temporal_order(step, env, cache)
    v = binding["value"]
    assert v["subject_label"] == "#10"
    assert v["subject_is_first_scorer"] is True
    assert v["subject_dets"] == ["p_messi"]
    assert v["first"]["ts_ms"] == 4000
    assert result["status"] == "done"
    assert result["output_label"] == "goal @ 4000ms (first)"
    assert len(result["evidence"]["overlays"]) == 1
    assert result["evidence"]["overlays"][0]["tone"] == "green"


def test_temporal_order_subject_is_first_scorer_false():
    events = [{"ts_ms": 4000, "type": "goal", "scorer_det": "p_other", "box": box()}]
    cache = make_cache(events=events)
    subj_items = [{"det_id": "p_messi", "text": "10", "frame_ts_ms": 4625}]
    env = {"tens": {"kind": "texts", "items": subj_items}}
    step = {"id": "ordered", "op": "temporal_order",
            "args": {"events": "tens", "by": "timestamp"}}
    binding, _ = op_temporal_order(step, env, cache)
    assert binding["value"]["subject_is_first_scorer"] is False


def test_temporal_order_empty_when_no_goal_events():
    cache = make_cache(events=[{"ts_ms": 100, "type": "kickoff"}])  # no goal
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p", "text": "10"}]}}
    step = {"id": "ordered", "op": "temporal_order",
            "args": {"events": "tens", "by": "timestamp"}}
    binding, result = op_temporal_order(step, env, cache)
    assert binding["value"]["first"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "no goal events"


# ---------------------------------------------------------------- op_answer

def _ordered_binding(first=True, subject_label="#10", subject_is_first=True,
                     scorer_det="p_messi"):
    first_ev = ({"ts_ms": 4000, "scorer_det": scorer_det, "box": box()} if first else None)
    return {"kind": "ordered", "value": {
        "events": [first_ev] if first_ev else [],
        "first": first_ev,
        "subject_dets": ["p_messi"] if scorer_det else [],
        "subject_label": subject_label,
        "subject_is_first_scorer": subject_is_first,
    }}


def test_answer_ordered_grounded_yes_with_scorer_overlay():
    # The grounded ordered branch re-reads the cache, so detect MUST carry the scorer det.
    dets = [{"det_id": "p_messi", "cls": "person", "confidence": 0.7, "box": box()}]
    cache = make_cache(detect={"4625": dets})
    env = {"ordered": _ordered_binding(subject_is_first=True)}
    step = {"id": "result", "op": "answer",
            "args": {"from": "ordered", "question": "Does #10 score first?"}}
    binding, result = op_answer(step, env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "Yes"
    assert v["yes"] is True
    assert v["verdict"] == "Yes — #10 scored the first goal"
    assert result["status"] == "done"
    assert result["output_label"] == "Yes"
    # scorer overlay drawn from the cache re-read.
    ov = result["evidence"]["overlays"]
    assert len(ov) == 1
    assert ov[0]["label"] == "#10 scored"
    assert ov[0]["tone"] == "green"
    assert result["evidence"]["frame_ts_ms"] == 4625


def test_answer_ordered_grounded_no():
    cache = make_cache(detect={"4625": [{"det_id": "p_other", "cls": "person",
                                         "confidence": 0.6, "box": box()}]})
    env = {"ordered": _ordered_binding(subject_is_first=False, scorer_det="p_other")}
    step = {"id": "result", "op": "answer",
            "args": {"from": "ordered", "question": "Does #10 score first?"}}
    binding, result = op_answer(step, env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "No"
    assert v["yes"] is False
    assert v["verdict"] == "No — the first goal was not scored by #10"
    assert result["output_label"] == "No"


def test_answer_ordered_ungrounded_no_goal():
    cache = make_cache()
    env = {"ordered": _ordered_binding(first=False)}
    step = {"id": "result", "op": "answer",
            "args": {"from": "ordered", "question": "Does #10 score first?"}}
    binding, result = op_answer(step, env, cache)
    v = binding["value"]
    assert v["grounded"] is False
    assert v["reason"] == "no-grounded-answer"
    assert v["answer"] is None and v["verdict"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "(couldn't ground)"


def test_answer_ordered_ungrounded_no_subject():
    # A first goal exists but the subject filter matched nobody -> ground out (don't overclaim).
    cache = make_cache(detect={"4625": [{"det_id": "p_messi", "cls": "person",
                                         "confidence": 0.7, "box": box()}]})
    env = {"ordered": _ordered_binding(subject_label=None)}
    step = {"id": "result", "op": "answer",
            "args": {"from": "ordered", "question": "Does #10 score first?"}}
    binding, _ = op_answer(step, env, cache)
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_number_grounded():
    env = {"n": {"kind": "number", "value": 4}}
    step = {"id": "result", "op": "answer",
            "args": {"from": "n", "question": "How many?"}}
    binding, result = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "4"
    assert binding["value"]["verdict"] == "4 found"
    assert result["output_label"] == "4"


def test_answer_number_ungrounded_when_zero():
    env = {"n": {"kind": "number", "value": 0}}
    step = {"id": "result", "op": "answer",
            "args": {"from": "n", "question": "How many?"}}
    binding, _ = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_texts_grounded():
    env = {"t": {"kind": "texts", "items": [
        {"text": "10"}, {"text": "7"}, {"text": "10"}, {"text": ""}]}}
    step = {"id": "result", "op": "answer",
            "args": {"from": "t", "question": "What numbers?"}}
    binding, result = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "10, 7"  # sorted unique
    assert binding["value"]["verdict"] == "Read: 10, 7"
    assert result["output_label"] == "10, 7"


def test_answer_texts_ungrounded_when_nothing_legible():
    env = {"t": {"kind": "texts", "items": [{"text": ""}, {"text": None}]}}
    step = {"id": "result", "op": "answer",
            "args": {"from": "t", "question": "What numbers?"}}
    binding, _ = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_presence_grounded_detections():
    env = {"d": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}]}}
    step = {"id": "result", "op": "answer",
            "args": {"from": "d", "question": "Is there a player?"}}
    binding, result = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "Yes"
    assert binding["value"]["yes"] is True
    assert binding["value"]["verdict"] == "Yes — found 2"
    assert result["output_label"] == "Yes"


def test_answer_presence_ungrounded_when_empty():
    for kind in ("detections", "crops", "frames"):
        env = {"x": {"kind": kind, "items": []}}
        step = {"id": "result", "op": "answer",
                "args": {"from": "x", "question": "Is there one?"}}
        binding, _ = op_answer(step, env, make_cache())
        assert binding["value"]["grounded"] is False, kind
        assert binding["value"]["reason"] == "no-grounded-answer", kind


def test_answer_unsupported_kind_discriminator():
    env = {"a": {"kind": "answer", "value": {}}}  # answer-of-answer: unsupported
    step = {"id": "result", "op": "answer",
            "args": {"from": "a", "question": "?"}}
    binding, result = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "unsupported-kind"
    assert "unsupported-kind" in result["note"]


def test_ops_table_has_all_eight():
    assert set(OPS) == {
        "sample_frames", "detect", "crop", "read_text",
        "filter", "count", "temporal_order", "answer",
    }
