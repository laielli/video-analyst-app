"""
Dedicated unit tests for the 8 ops in interpreter/primitives.py.

The ops have a uniform contract: fn(step, env, cache) -> (binding, step_result).
detect/read_text/temporal_order read from `cache`; crop/filter/count are pure compute over
`env`. op_answer is NOT fully pure — its grounded temporal branch re-reads the cache via
analyzed_frames()/detections_at() to build scorer overlays.

These tests assert on BOTH the returned binding AND the run-doc trace entry. They are built on
a tiny in-memory Cache (cache.py just wraps the dict — but it reads data["clip"], so the
fixture MUST include "clip").
"""
from __future__ import annotations

from interpreter.cache import Cache
from interpreter.primitives import (
    op_sample_frames,
    op_detect,
    op_crop,
    op_read_text,
    op_filter,
    op_count,
    op_temporal_order,
    op_answer,
    OPS,
)


def make_cache(detect=None, read_text=None, events=None):
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {},
        "read_text": read_text or {},
        "events": events or [],
    })


def _box(x=0.1, y=0.1, w=0.2, h=0.4):
    return {"x": x, "y": y, "w": w, "h": h}


# --------------------------------------------------------------------------------------
# op_sample_frames
# --------------------------------------------------------------------------------------

def test_sample_frames_stride_and_count():
    cache = make_cache()
    step = {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}
    binding, result = op_sample_frames(step, {}, cache)
    # stride = max(1, round(1000/8)) = 125; range(0, 1001, 125) -> 0,125,...,1000 (9 frames)
    assert binding["kind"] == "frames"
    assert len(binding["items"]) == 9
    assert binding["items"][0] == {"frame_ts_ms": 0}
    assert binding["items"][-1] == {"frame_ts_ms": 1000}
    assert result["op"] == "sample_frames"
    assert result["status"] == "done"
    assert result["output_label"] == "9 frames @ 8fps"
    assert result["evidence"]["frame_ts_ms"] == binding["items"][len(binding["items"]) // 2]["frame_ts_ms"]


def test_sample_frames_high_fps_stride_collapses_to_one():
    cache = make_cache()
    # fps high enough that round(1000/fps) == 0 -> stride collapses to max(1, 0) == 1
    step = {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 5, "fps": 5000}}
    binding, result = op_sample_frames(step, {}, cache)
    assert [it["frame_ts_ms"] for it in binding["items"]] == [0, 1, 2, 3, 4, 5]
    assert result["output_label"] == "6 frames @ 5000fps"


# --------------------------------------------------------------------------------------
# op_detect
# --------------------------------------------------------------------------------------

def test_detect_case_insensitive_match_and_overlays_on_focus():
    # Two frames; the focus is the LAST detection's ts. Class match is case-insensitive.
    cache = make_cache(detect={
        "100": [{"det_id": "a", "cls": "Person", "confidence": 0.9, "box": _box()}],
        "200": [{"det_id": "b", "cls": "PERSON", "confidence": 0.8, "box": _box(0.5)}],
    })
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}, {"frame_ts_ms": 200}]}}
    step = {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}}
    binding, result = op_detect(step, env, cache)
    assert binding["kind"] == "detections"
    assert len(binding["items"]) == 2
    assert result["status"] == "done"
    assert result["output_label"] == "2 people"
    assert result["confidence"] == 0.9  # max confidence
    # focus_ts == last det's frame_ts_ms == 200; overlays only on focus frame
    assert result["evidence"]["frame_ts_ms"] == 200
    assert len(result["evidence"]["overlays"]) == 1
    assert result["evidence"]["overlays"][0]["tone"] == "azure"
    assert "note" not in result


def test_detect_objects_noun_when_not_just_person():
    cache = make_cache(detect={"100": [{"det_id": "z", "cls": "ball", "confidence": 0.4, "box": _box()}]})
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    step = {"id": "d", "op": "detect", "args": {"frames": "frames", "classes": ["ball"]}}
    _, result = op_detect(step, env, cache)
    assert result["output_label"] == "1 objects"


def test_detect_empty_status_and_note():
    cache = make_cache(detect={"100": [{"det_id": "x", "cls": "ball", "confidence": 0.4, "box": _box()}]})
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    step = {"id": "d", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}}
    binding, result = op_detect(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["confidence"] is None
    assert result["note"] == "no detections in the sampled frames"
    # with no dets, focus falls back to the last frame's ts
    assert result["evidence"]["frame_ts_ms"] == 100
    assert result["evidence"]["overlays"] == []


# --------------------------------------------------------------------------------------
# op_crop
# --------------------------------------------------------------------------------------

def test_crop_jersey_region_box_math_and_carry():
    b = {"x": 0.10, "y": 0.20, "w": 1.0, "h": 1.0}
    env = {"dets": {"kind": "detections", "items": [
        {"det_id": "p0", "box": b, "confidence": 0.75, "frame_ts_ms": 4625},
    ]}}
    step = {"id": "jerseys", "op": "crop", "args": {"detections": "dets", "region": "jersey"}}
    binding, result = op_crop(step, env, cache=make_cache())
    crop = binding["items"][0]
    # jersey offsets: x + 0.22w, y + 0.10h, 0.56w, 0.22h
    assert crop["box"] == {"x": round(0.10 + 0.22, 4), "y": round(0.20 + 0.10, 4),
                           "w": round(0.56, 4), "h": round(0.22, 4)}
    assert crop["crop_id"] == "c0"
    assert crop["det_id"] == "p0"
    assert crop["person_box"] == b  # original carried through
    assert crop["person_conf"] == 0.75
    assert result["status"] == "done"
    assert result["output_label"] == "1 jersey crops"
    assert result["evidence"]["overlays"][0]["label"] == "crop_jersey"


def test_crop_full_region_passthrough_and_id_minting():
    b0, b1 = _box(0.0), _box(0.5)
    env = {"dets": {"kind": "detections", "items": [
        {"det_id": "p0", "box": b0, "frame_ts_ms": 10},
        {"det_id": "p1", "box": b1, "frame_ts_ms": 20},
    ]}}
    step = {"id": "c", "op": "crop", "args": {"detections": "dets", "region": "full"}}
    binding, result = op_crop(step, env, cache=make_cache())
    assert [c["crop_id"] for c in binding["items"]] == ["c0", "c1"]
    # full region -> box is a copy of the person box (not the original object)
    assert binding["items"][0]["box"] == b0
    assert binding["items"][0]["box"] is not b0
    assert binding["items"][1]["box"] == b1
    assert result["evidence"]["overlays"][0]["label"] == "crop_full"


def test_crop_empty_when_no_detections():
    env = {"dets": {"kind": "detections", "items": []}}
    step = {"id": "c", "op": "crop", "args": {"detections": "dets", "region": "jersey"}}
    binding, result = op_crop(step, env, cache=make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"


# --------------------------------------------------------------------------------------
# op_read_text
# --------------------------------------------------------------------------------------

def _crop(det_id, ts=10, box=None, person_box=None):
    return {"crop_id": f"c-{det_id}", "det_id": det_id, "box": box or _box(),
            "person_box": person_box or _box(), "frame_ts_ms": ts}


def test_read_text_source_precedence_pinned_over_cached_over_live():
    cache = make_cache(read_text={
        "a": {"text": "10", "confidence": 0.9, "source": "live"},
        "b": {"text": "7", "confidence": 0.8, "source": "cached"},
        "c": {"text": "22", "confidence": None, "source": "pinned"},
    })
    env = {"crops": {"kind": "crops", "items": [_crop("a"), _crop("b"), _crop("c")]}}
    step = {"id": "numbers", "op": "read_text", "args": {"crops": "crops"}}
    binding, result = op_read_text(step, env, cache)
    assert binding["kind"] == "texts"
    assert len(binding["items"]) == 3
    # pinned present anywhere -> step_source == pinned
    assert result["source"] == "pinned"
    # output_label is sorted unique texts
    assert result["output_label"] == "10, 22, 7"
    # confidence is the first non-None text confidence
    assert result["confidence"] == 0.9


def test_read_text_cached_precedence_when_no_pinned():
    cache = make_cache(read_text={
        "a": {"text": "10", "confidence": 0.9, "source": "live"},
        "b": {"text": "7", "confidence": 0.8, "source": "cached"},
    })
    env = {"crops": {"kind": "crops", "items": [_crop("a"), _crop("b")]}}
    step = {"id": "n", "op": "read_text", "args": {"crops": "crops"}}
    _, result = op_read_text(step, env, cache)
    assert result["source"] == "cached"


def test_read_text_live_when_only_live_source():
    cache = make_cache(read_text={"a": {"text": "10", "confidence": 0.9, "source": "live"}})
    env = {"crops": {"kind": "crops", "items": [_crop("a")]}}
    step = {"id": "n", "op": "read_text", "args": {"crops": "crops"}}
    _, result = op_read_text(step, env, cache)
    assert result["source"] == "live"


def test_read_text_note_when_nothing_legible():
    cache = make_cache(read_text={})  # no reads
    env = {"crops": {"kind": "crops", "items": [_crop("a"), _crop("b")]}}
    step = {"id": "n", "op": "read_text", "args": {"crops": "crops"}}
    binding, result = op_read_text(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "(none legible)"
    assert result["note"] == "no legible text in the jersey crops (read returned nothing)"


def test_read_text_carries_pin_note():
    cache = make_cache(read_text={
        "a": {"text": "10", "confidence": None, "source": "pinned", "note": "hand-verified"},
    })
    env = {"crops": {"kind": "crops", "items": [_crop("a")]}}
    step = {"id": "n", "op": "read_text", "args": {"crops": "crops"}}
    _, result = op_read_text(step, env, cache)
    assert result["note"] == "hand-verified"


# --------------------------------------------------------------------------------------
# op_filter
# --------------------------------------------------------------------------------------

def test_filter_string_coerced_equality_keeps_kind():
    env = {"src": {"kind": "texts", "items": [
        {"text": "10", "person_box": _box(), "frame_ts_ms": 4625},
        {"text": "7", "person_box": _box(0.5), "frame_ts_ms": 4625},
    ]}}
    # equals value is an int 10; field value is str "10" -> str-coerced match
    step = {"id": "tens", "op": "filter", "args": {"items": "src", "where": {"field": "text", "equals": 10}}}
    binding, result = op_filter(step, env, cache=make_cache())
    assert binding["kind"] == "texts"  # kind preserved
    assert len(binding["items"]) == 1
    assert binding["items"][0]["text"] == "10"
    assert result["status"] == "done"
    assert result["output_label"] == "1 match · #10"
    assert result["evidence"]["overlays"][0]["label"] == "#10"


def test_filter_empty_match_note():
    env = {"src": {"kind": "texts", "items": [{"text": "7", "box": _box()}]}}
    step = {"id": "f", "op": "filter", "args": {"items": "src", "where": {"field": "text", "equals": "10"}}}
    binding, result = op_filter(step, env, cache=make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "0 match"
    assert result["note"] == "no items with text == 10"


# --------------------------------------------------------------------------------------
# op_count
# --------------------------------------------------------------------------------------

def test_count_value_and_label():
    env = {"src": {"kind": "detections", "items": [{"a": 1}, {"a": 2}, {"a": 3}]}}
    step = {"id": "n", "op": "count", "args": {"items": "src"}}
    binding, result = op_count(step, env, cache=make_cache())
    assert binding == {"kind": "number", "value": 3}
    assert result["status"] == "done"
    assert result["output_label"] == "3"
    assert result["input_label"] == "detections"


def test_count_zero():
    env = {"src": {"kind": "frames", "items": []}}
    step = {"id": "n", "op": "count", "args": {"items": "src"}}
    binding, result = op_count(step, env, cache=make_cache())
    assert binding["value"] == 0
    assert result["output_label"] == "0"


# --------------------------------------------------------------------------------------
# op_temporal_order
# --------------------------------------------------------------------------------------

def test_temporal_order_subject_is_first_scorer_true():
    cache = make_cache(events=[
        {"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi", "box": _box()},
        {"ts_ms": 6000, "type": "goal", "scorer_det": "p_other"},
    ])
    # subject filtered to det p_messi with legible text "10"
    env = {"subj": {"kind": "texts", "items": [{"det_id": "p_messi", "text": "10"}]}}
    step = {"id": "ordered", "op": "temporal_order", "args": {"events": "subj", "by": "timestamp"}}
    binding, result = op_temporal_order(step, env, cache)
    v = binding["value"]
    assert binding["kind"] == "ordered"
    assert v["first"]["ts_ms"] == 4000  # sorted by ts
    assert v["subject_label"] == "#10"  # derived, not literal
    assert v["subject_dets"] == ["p_messi"]
    assert v["subject_is_first_scorer"] is True
    assert result["status"] == "done"
    assert result["output_label"] == "goal @ 4000ms (first)"
    assert result["evidence"]["frame_ts_ms"] == 4000
    assert result["evidence"]["overlays"][0]["tone"] == "green"


def test_temporal_order_subject_is_first_scorer_false():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_other"}])
    env = {"subj": {"kind": "texts", "items": [{"det_id": "p_messi", "text": "10"}]}}
    step = {"id": "o", "op": "temporal_order", "args": {"events": "subj"}}
    binding, _ = op_temporal_order(step, env, cache)
    assert binding["value"]["subject_is_first_scorer"] is False
    assert binding["value"]["subject_label"] == "#10"


def test_temporal_order_empty_when_no_goal_events():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "kickoff"}])  # no goal events
    env = {"subj": {"kind": "texts", "items": [{"det_id": "p", "text": "10"}]}}
    step = {"id": "o", "op": "temporal_order", "args": {"events": "subj"}}
    binding, result = op_temporal_order(step, env, cache)
    assert binding["value"]["first"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "no goal events"


def test_temporal_order_subject_label_none_when_no_legible_text():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p"}])
    env = {"subj": {"kind": "texts", "items": [{"det_id": "p", "text": ""}]}}  # no legible subject
    step = {"id": "o", "op": "temporal_order", "args": {"events": "subj"}}
    binding, _ = op_temporal_order(step, env, cache)
    assert binding["value"]["subject_label"] is None


# --------------------------------------------------------------------------------------
# op_answer — the highest-value op. Kind dispatch + grounded/ungrounded discriminator.
# --------------------------------------------------------------------------------------

def _answer_step(from_id, q="Q?"):
    return {"id": "result", "op": "answer", "args": {"from": from_id, "question": q}}


def test_answer_ordered_grounded_yes_rereads_cache_for_overlay():
    # Grounded temporal: subject IS the first scorer. The grounded path re-reads the cache
    # via analyzed_frames()/detections_at() to build the scorer overlay -> fixture needs detect.
    cache = make_cache(
        detect={"4625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75, "box": _box()}]},
        events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi"}],
    )
    env = {"ordered": {"kind": "ordered", "value": {
        "events": cache.goal_events(), "first": {"ts_ms": 4000, "scorer_det": "p_messi"},
        "subject_dets": ["p_messi"], "subject_label": "#10", "subject_is_first_scorer": True,
    }}}
    binding, result = op_answer(_answer_step("ordered", "Does #10 score first?"), env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "Yes"
    assert v["verdict"] == "Yes — #10 scored the first goal"
    assert v["yes"] is True
    assert result["status"] == "done"
    assert result["output_label"] == "Yes"
    # scorer overlay re-read from cache: at ts 4625 the p_messi det matches
    assert result["evidence"]["frame_ts_ms"] == 4625
    assert result["evidence"]["overlays"][0]["label"] == "#10 scored"


def test_answer_ordered_grounded_no():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_other"}])
    env = {"ordered": {"kind": "ordered", "value": {
        "events": [], "first": {"ts_ms": 4000, "scorer_det": "p_other"},
        "subject_dets": ["p_messi"], "subject_label": "#10", "subject_is_first_scorer": False,
    }}}
    binding, result = op_answer(_answer_step("ordered"), env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "No"
    assert v["verdict"] == "No — the first goal was not scored by #10"
    assert v["yes"] is False


def test_answer_count_grounded():
    env = {"n": {"kind": "number", "value": 5}}
    binding, result = op_answer(_answer_step("n", "how many?"), env, cache=make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "5"
    assert v["verdict"] == "5 found"
    assert result["output_label"] == "5"


def test_answer_texts_grounded_sorted_unique():
    env = {"t": {"kind": "texts", "items": [
        {"text": "10"}, {"text": "7"}, {"text": "10"}, {"text": ""},
    ]}}
    binding, result = op_answer(_answer_step("t", "what number?"), env, cache=make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "10, 7"  # sorted("10","7") -> ["10","7"]
    assert v["verdict"] == "Read: 10, 7"
    assert result["output_label"] == "10, 7"


def test_answer_presence_detections_grounded():
    env = {"d": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}]}}
    binding, result = op_answer(_answer_step("d", "is there a player?"), env, cache=make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "Yes"
    assert v["verdict"] == "Yes — found 2"
    assert v["yes"] is True
    assert result["output_label"] == "Yes"


def test_answer_presence_crops_and_frames_grounded():
    for kind in ("crops", "frames"):
        env = {"x": {"kind": kind, "items": [{"id": 1}]}}
        binding, _ = op_answer(_answer_step("x"), env, cache=make_cache())
        assert binding["value"]["grounded"] is True, kind
        assert binding["value"]["answer"] == "Yes"


def test_answer_ungrounded_reasons():
    """The ungrounded discriminator: each empty/unsupported binding -> grounded=False + reason.
    This is the over-claim guard (primitives.py): a missing answer must NOT fabricate a verdict.
    """
    cases = [
        # (binding, expected reason)
        ({"kind": "ordered", "value": {"first": None}}, "no-grounded-answer"),  # no goal events
        ({"kind": "ordered", "value": {"first": {"ts_ms": 1}, "subject_label": None,
                                        "subject_is_first_scorer": False}}, "no-grounded-answer"),  # subject not found
        ({"kind": "number", "value": 0}, "no-grounded-answer"),  # nothing to count
        ({"kind": "texts", "items": []}, "no-grounded-answer"),  # nothing legible
        ({"kind": "detections", "items": []}, "no-grounded-answer"),  # empty presence
        ({"kind": "weird", "value": 1}, "unsupported-kind"),  # unsupported kind
    ]
    for src, reason in cases:
        env = {"src": src}
        binding, result = op_answer(_answer_step("src"), env, cache=make_cache())
        v = binding["value"]
        assert v["grounded"] is False, src
        assert v["reason"] == reason, src
        assert v["answer"] is None and v["verdict"] is None and v["yes"] is None
        assert result["status"] == "empty"
        assert result["output_label"] == "(couldn't ground)"
        assert reason in result["note"]


def test_answer_ungrounded_subject_not_found_does_not_overclaim():
    """A first goal exists but the subject filter matched nobody -> ground out, never a fake 'No'.
    (Asserts the absent-subject guard in _answer_temporal.)"""
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "p_other"}])
    env = {"ordered": {"kind": "ordered", "value": {
        "first": {"ts_ms": 4000, "scorer_det": "p_other"}, "subject_label": None,
        "subject_is_first_scorer": False,
    }}}
    binding, _ = op_answer(_answer_step("ordered"), env, cache)
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


# --------------------------------------------------------------------------------------
# OPS registry
# --------------------------------------------------------------------------------------

def test_ops_registry_covers_all_eight():
    assert set(OPS) == {
        "sample_frames", "detect", "crop", "read_text",
        "filter", "count", "temporal_order", "answer",
    }
