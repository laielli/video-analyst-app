"""
Dedicated unit tests for the 8 ops in interpreter/primitives.py. Each op has the uniform
contract fn(step, env, cache) -> (binding, step_result). We assert on BOTH halves: the binding
(what later steps reference) and the run-doc trace entry (status/provenance/labels/overlays).

detect/read_text/temporal_order/op_answer(grounded temporal) read from the Cache; crop/filter/
count are pure compute over `env`. We build a minimal in-memory Cache directly — the constructor
just wraps the dict — so these stay isolated from the committed hero cache file.
"""
from __future__ import annotations

from interpreter import Cache
from interpreter.primitives import (
    op_sample_frames, op_detect, op_describe_scene, op_crop, op_read_text,
    op_filter, op_count, op_temporal_order, op_answer, _subject_label,
)
from limits import MAX_CAPTION_LEN


def make_cache(detect=None, read_text=None, events=None, captions=None):
    """Minimal Cache. MUST include the 'clip' key — Cache.__init__ reads data['clip']."""
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {},
        "read_text": read_text or {},
        "events": events or [],
        "captions": captions or {},
    })


def _box(x=0.1, y=0.1, w=0.2, h=0.4):
    return {"x": x, "y": y, "w": w, "h": h}


# ---------------------------------------------------------------------------------------
# op_sample_frames
# ---------------------------------------------------------------------------------------

def test_sample_frames_stride_and_count():
    step = {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}
    binding, result = op_sample_frames(step, {}, make_cache())
    # stride = max(1, round(1000/8)) = round(125) = 125 -> 0,125,...,1000 = 9 frames
    assert binding["kind"] == "frames"
    assert [f["frame_ts_ms"] for f in binding["items"]] == list(range(0, 1001, 125))
    assert len(binding["items"]) == 9
    assert result["status"] == "done"
    assert result["op"] == "sample_frames"
    assert result["output_label"] == "9 frames @ 8fps"
    assert result["input_label"] == "clip 0-1000ms"
    # evidence frame is the middle sampled ts.
    assert result["evidence"]["frame_ts_ms"] == binding["items"][len(binding["items"]) // 2]["frame_ts_ms"]


def test_sample_frames_stride_collapses_to_one():
    # fps high enough that round(1000/fps) < 1 -> stride clamps to 1.
    step = {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 5, "fps": 1000}}
    binding, result = op_sample_frames(step, {}, make_cache())
    # round(1000/1000)=1 -> stride 1 -> 0..5 inclusive = 6 frames.
    assert len(binding["items"]) == 6
    assert [f["frame_ts_ms"] for f in binding["items"]] == [0, 1, 2, 3, 4, 5]


def test_sample_frames_stride_clamps_when_round_is_zero():
    # fps = 3000 -> round(1000/3000) = round(0.333) = 0 -> max(1,0) = 1.
    step = {"id": "f", "op": "sample_frames", "args": {"start_ms": 10, "end_ms": 12, "fps": 3000}}
    binding, _ = op_sample_frames(step, {}, make_cache())
    assert [f["frame_ts_ms"] for f in binding["items"]] == [10, 11, 12]


# ---------------------------------------------------------------------------------------
# op_detect
# ---------------------------------------------------------------------------------------

def test_detect_case_insensitive_match_and_overlays_on_focus_only():
    cache = make_cache(detect={
        "100": [{"det_id": "d0", "cls": "Person", "confidence": 0.9, "box": _box()}],
        "200": [{"det_id": "d1", "cls": "PERSON", "confidence": 0.5, "box": _box(0.5)}],
    })
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}, {"frame_ts_ms": 200}]}}
    step = {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}}
    binding, result = op_detect(step, env, cache)
    # both frames match despite mixed case in the cache.
    assert len(binding["items"]) == 2
    assert result["status"] == "done"
    assert result["op"] == "detect"
    assert result["output_label"] == "2 people"
    assert result["producer"] == "Azure AI Vision · detect"
    assert result["source"] == "cached"
    # focus_ts is the LAST detection's frame -> overlays only for that frame (one box).
    assert result["evidence"]["frame_ts_ms"] == 200
    assert len(result["evidence"]["overlays"]) == 1
    # confidence is the max across detections.
    assert result["confidence"] == 0.9


def test_detect_objects_noun_when_class_not_person():
    cache = make_cache(detect={"100": [{"det_id": "b0", "cls": "ball", "confidence": 0.4, "box": _box()}]})
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 100}]}}
    step = {"id": "balls", "op": "detect", "args": {"frames": "frames", "classes": ["ball"]}}
    _, result = op_detect(step, env, cache)
    assert result["output_label"] == "1 objects"


def test_detect_empty_carries_note_and_empty_status():
    cache = make_cache(detect={"100": [{"det_id": "d0", "cls": "person", "confidence": 0.9, "box": _box()}]})
    # frame with NO cached detections.
    env = {"frames": {"kind": "frames", "items": [{"frame_ts_ms": 999}]}}
    step = {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}}
    binding, result = op_detect(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["note"] == "no detections in the sampled frames"
    assert result["confidence"] is None
    # focus falls back to the last frame ts, no overlays.
    assert result["evidence"]["frame_ts_ms"] == 999
    assert result["evidence"]["overlays"] == []


# ---------------------------------------------------------------------------------------
# op_crop
# ---------------------------------------------------------------------------------------

def test_crop_jersey_region_box_math():
    b = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    env = {"people": {"kind": "detections", "items": [
        {"det_id": "d0", "cls": "person", "confidence": 0.8, "box": b, "frame_ts_ms": 300},
    ]}}
    step = {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}}
    binding, result = op_crop(step, env, make_cache())
    c = binding["items"][0]
    # jersey offsets: x+0.22w, y+0.10h, 0.56w, 0.22h.
    assert c["box"] == {"x": 0.22, "y": 0.10, "w": 0.56, "h": 0.22}
    assert c["crop_id"] == "c0"
    assert c["det_id"] == "d0"
    assert c["person_box"] == b
    assert c["person_conf"] == 0.8
    assert result["status"] == "done"
    assert result["producer"] == "crop_jersey"
    assert result["output_label"] == "1 jersey crops"
    # overlay labelled crop_jersey on the focus frame.
    assert result["evidence"]["overlays"][0]["label"] == "crop_jersey"


def test_crop_full_region_passthrough_and_crop_id_minting():
    b0, b1 = _box(0.1), _box(0.6)
    env = {"people": {"kind": "detections", "items": [
        {"det_id": "d0", "box": b0, "confidence": 0.7, "frame_ts_ms": 100},
        {"det_id": "d1", "box": b1, "confidence": 0.6, "frame_ts_ms": 100},
    ]}}
    step = {"id": "crops", "op": "crop", "args": {"detections": "people", "region": "full"}}
    binding, result = op_crop(step, env, make_cache())
    # full region -> box is a copy of the person box, distinct ids minted by index.
    assert binding["items"][0]["box"] == b0
    assert binding["items"][1]["box"] == b1
    assert [c["crop_id"] for c in binding["items"]] == ["c0", "c1"]
    assert result["producer"] == "crop_full"
    assert result["evidence"]["overlays"][0]["label"] == "crop_full"


def test_crop_empty_detections():
    env = {"people": {"kind": "detections", "items": []}}
    step = {"id": "crops", "op": "crop", "args": {"detections": "people", "region": "jersey"}}
    binding, result = op_crop(step, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"


# ---------------------------------------------------------------------------------------
# op_read_text
# ---------------------------------------------------------------------------------------

def _crops_env(*det_ids):
    return {"jerseys": {"kind": "crops", "items": [
        {"crop_id": f"c{i}", "det_id": d, "box": _box(i * 0.1), "person_box": _box(), "frame_ts_ms": 400 + i}
        for i, d in enumerate(det_ids)
    ]}}


def test_read_text_source_precedence_pinned_wins():
    cache = make_cache(read_text={
        "d0": {"text": "10", "confidence": 0.9, "source": "cached"},
        "d1": {"text": "7", "confidence": 0.8, "source": "pinned"},
    })
    env = _crops_env("d0", "d1")
    step = {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}}
    binding, result = op_read_text(step, env, cache)
    assert len(binding["items"]) == 2
    # pinned present -> step_source is pinned (highest precedence).
    assert result["source"] == "pinned"
    assert result["status"] == "done"
    # output_label is the sorted unique texts.
    assert result["output_label"] == "10, 7"


def test_read_text_cached_beats_live():
    cache = make_cache(read_text={
        "d0": {"text": "9", "confidence": 0.7, "source": "live"},
        "d1": {"text": "5", "confidence": 0.6, "source": "cached"},
    })
    binding, result = op_read_text(
        {"id": "n", "op": "read_text", "args": {"crops": "jerseys"}}, _crops_env("d0", "d1"), cache)
    assert result["source"] == "cached"
    # first non-None confidence is carried.
    assert result["confidence"] == 0.7
    assert len(binding["items"]) == 2


def test_read_text_note_when_nothing_legible():
    cache = make_cache(read_text={})  # no reads for any det_id
    binding, result = op_read_text(
        {"id": "n", "op": "read_text", "args": {"crops": "jerseys"}}, _crops_env("d0"), cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "(none legible)"
    assert result["note"] == "no legible text in the jersey crops (read returned nothing)"


def test_read_text_carries_cache_note_first():
    cache = make_cache(read_text={"d0": {"text": "10", "confidence": None, "source": "pinned", "note": "hand-verified"}})
    _, result = op_read_text(
        {"id": "n", "op": "read_text", "args": {"crops": "jerseys"}}, _crops_env("d0"), cache)
    assert result["note"] == "hand-verified"


# ---------------------------------------------------------------------------------------
# op_filter
# ---------------------------------------------------------------------------------------

def test_filter_string_coerced_equality_keeps_kind():
    env = {"numbers": {"kind": "texts", "items": [
        {"text": "10", "person_box": _box(), "frame_ts_ms": 100},
        {"text": 10, "person_box": _box(0.3), "frame_ts_ms": 200},  # int 10 coerces to "10"
        {"text": "7", "person_box": _box(0.6), "frame_ts_ms": 300},
    ]}}
    step = {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}}
    binding, result = op_filter(step, env, make_cache())
    # both "10" and int 10 match via str() coercion.
    assert len(binding["items"]) == 2
    assert binding["kind"] == "texts"  # kind preserved from source
    assert result["status"] == "done"
    assert "1 match" not in result["output_label"]
    assert result["output_label"] == "2 match · #10"
    assert result["input_label"] == "texts where text == 10"


def test_filter_empty_match_carries_note():
    env = {"numbers": {"kind": "texts", "items": [{"text": "7", "person_box": _box(), "frame_ts_ms": 100}]}}
    step = {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}}
    binding, result = op_filter(step, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["note"] == "no items with text == 10"
    assert result["output_label"] == "0 match"


# ---------------------------------------------------------------------------------------
# op_count
# ---------------------------------------------------------------------------------------

def test_count_value_equals_len():
    env = {"people": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}, {"det_id": "c"}]}}
    step = {"id": "n", "op": "count", "args": {"items": "people"}}
    binding, result = op_count(step, env, make_cache())
    assert binding["kind"] == "number"
    assert binding["value"] == 3
    assert result["output_label"] == "3"
    assert result["status"] == "done"
    assert result["input_label"] == "detections"


def test_count_zero():
    env = {"people": {"kind": "detections", "items": []}}
    binding, result = op_count({"id": "n", "op": "count", "args": {"items": "people"}}, env, make_cache())
    assert binding["value"] == 0
    assert result["output_label"] == "0"


# ---------------------------------------------------------------------------------------
# op_temporal_order
# ---------------------------------------------------------------------------------------

def test_subject_label_helper():
    assert _subject_label([{"text": "10"}]) == "#10"
    assert _subject_label([{"text": None}, {"text": "7"}]) == "#7"
    assert _subject_label([{"text": ""}]) is None
    assert _subject_label([]) is None
    assert _subject_label(None) is None


def test_temporal_order_subject_is_first_scorer_true():
    cache = make_cache(events=[
        {"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi", "box": _box()},
        {"ts_ms": 4500, "type": "goal", "scorer_det": "other"},
    ])
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p_messi", "text": "10"}]}}
    step = {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}}
    binding, result = op_temporal_order(step, env, cache)
    v = binding["value"]
    assert v["first"]["ts_ms"] == 4000
    assert v["subject_label"] == "#10"
    assert v["subject_is_first_scorer"] is True
    assert v["subject_dets"] == ["p_messi"]
    assert result["status"] == "done"
    assert result["output_label"] == "goal @ 4000ms (first)"
    # overlay drawn on the first goal frame.
    assert result["evidence"]["frame_ts_ms"] == 4000
    assert result["evidence"]["overlays"][0]["kind"] == "goal"


def test_temporal_order_subject_is_first_scorer_false():
    cache = make_cache(events=[{"ts_ms": 4000, "type": "goal", "scorer_det": "someone_else"}])
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p_messi", "text": "10"}]}}
    binding, _ = op_temporal_order(
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}}, env, cache)
    assert binding["value"]["subject_is_first_scorer"] is False
    assert binding["value"]["subject_label"] == "#10"


def test_temporal_order_empty_when_no_goal_events():
    cache = make_cache(events=[{"ts_ms": 100, "type": "kickoff"}])  # no goal-type events
    env = {"tens": {"kind": "texts", "items": [{"det_id": "p", "text": "10"}]}}
    binding, result = op_temporal_order(
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}}, env, cache)
    assert binding["value"]["first"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "no goal events"


# ---------------------------------------------------------------------------------------
# op_answer — kind dispatch + grounded/ungrounded discrimination (highest-value op)
# ---------------------------------------------------------------------------------------

def _ordered_env(subject_label="#10", first_scorer="p_messi", subject_dets=("p_messi",), has_goal=True):
    first = {"ts_ms": 4000, "type": "goal", "scorer_det": first_scorer} if has_goal else None
    return {"ordered": {"kind": "ordered", "value": {
        "events": [first] if first else [], "first": first,
        "subject_dets": list(subject_dets), "subject_label": subject_label,
        "subject_is_first_scorer": bool(first and first.get("scorer_det") in subject_dets),
    }}}


def test_answer_ordered_grounded_yes():
    # The grounded ordered path re-reads the cache -> fixture MUST carry detect entries so the
    # scorer-overlay path runs (not silently skipped).
    cache = make_cache(detect={"4625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75, "box": _box()}]})
    env = _ordered_env(first_scorer="p_messi", subject_dets=("p_messi",))
    step = {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score the first goal?"}}
    binding, result = op_answer(step, env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["yes"] is True
    assert v["answer"] == "Yes"
    assert v["verdict"] == "Yes — #10 scored the first goal"
    assert result["status"] == "done"
    assert result["output_label"] == "Yes"
    # scorer overlay drawn from the re-read cache.
    assert result["evidence"]["frame_ts_ms"] == 4625
    assert result["evidence"]["overlays"][0]["label"] == "#10 scored"


def test_answer_ordered_grounded_no():
    cache = make_cache(detect={"4625": [{"det_id": "rival", "cls": "person", "confidence": 0.6, "box": _box()}]})
    env = _ordered_env(subject_label="#10", first_scorer="rival", subject_dets=("p_messi",))
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "ordered", "question": "q"}}, env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["yes"] is False
    assert v["answer"] == "No"
    assert v["verdict"] == "No — the first goal was not scored by #10"
    assert result["output_label"] == "No"


def test_answer_number():
    env = {"n": {"kind": "number", "value": 5}}
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "how many?"}}, env, make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "5"
    assert v["verdict"] == "5 found"
    assert result["output_label"] == "5"


def test_answer_texts():
    env = {"t": {"kind": "texts", "items": [{"text": "10"}, {"text": "7"}, {"text": "10"}]}}
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "t", "question": "what numbers?"}}, env, make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    # sorted unique.
    assert v["answer"] == "10, 7"
    assert v["verdict"] == "Read: 10, 7"
    assert result["output_label"] == "10, 7"


def test_answer_presence_detections():
    env = {"d": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}]}}
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "d", "question": "is there a player?"}}, env, make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "Yes"
    assert v["yes"] is True
    assert v["verdict"] == "Yes — found 2"
    assert result["output_label"] == "Yes"


# --- ungrounded discriminators (grounded=False + reason; never a fabricated verdict) ---

def test_answer_ungrounded_no_goal_events():
    env = _ordered_env(has_goal=False)
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "ordered", "question": "q"}}, env, make_cache())
    v = binding["value"]
    assert v["grounded"] is False
    assert v["reason"] == "no-grounded-answer"
    assert v["answer"] is None and v["verdict"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "(couldn't ground)"
    assert "no-grounded-answer" in result["note"]


def test_answer_ungrounded_subject_not_found():
    # A first goal exists, but the subject filter matched nobody (subject_label None).
    cache = make_cache(detect={"4625": [{"det_id": "rival", "cls": "person", "confidence": 0.6, "box": _box()}]})
    env = _ordered_env(subject_label=None, first_scorer="rival", subject_dets=())
    binding, _ = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "ordered", "question": "q"}}, env, cache)
    v = binding["value"]
    assert v["grounded"] is False
    assert v["reason"] == "no-grounded-answer"


def test_answer_ungrounded_count_zero():
    env = {"n": {"kind": "number", "value": 0}}
    binding, _ = op_answer({"id": "r", "op": "answer", "args": {"from": "n", "question": "q"}}, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_ungrounded_empty_texts():
    env = {"t": {"kind": "texts", "items": [{"text": ""}, {"text": None}]}}
    binding, _ = op_answer({"id": "r", "op": "answer", "args": {"from": "t", "question": "q"}}, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_ungrounded_empty_presence():
    env = {"d": {"kind": "detections", "items": []}}
    binding, _ = op_answer({"id": "r", "op": "answer", "args": {"from": "d", "question": "q"}}, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_ungrounded_unsupported_kind():
    # An answer-kind binding (or any kind not in the dispatch table) -> unsupported-kind reason.
    env = {"x": {"kind": "answer", "value": {}}}
    binding, _ = op_answer({"id": "r", "op": "answer", "args": {"from": "x", "question": "q"}}, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "unsupported-kind"


def test_answer_ordered_grounded_skips_overlay_when_no_cached_detection():
    # Grounded temporal answer whose scorer det_id has no cached detection -> still grounded,
    # but no scorer overlay (the re-read finds nothing). Proves the overlay path is conditional.
    cache = make_cache(detect={})  # no detect entries at all
    env = _ordered_env(first_scorer="p_messi", subject_dets=("p_messi",))
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "ordered", "question": "q"}}, env, cache)
    assert binding["value"]["grounded"] is True
    assert result["evidence"]["overlays"] == []
    assert result["evidence"]["frame_ts_ms"] == 0


# ---------------------------------------------------------------------------------------
# op_describe_scene (new primitive) + the captions answer branch
# ---------------------------------------------------------------------------------------

def _frames_env(*ts):
    return {"frames": {"kind": "frames", "items": [{"frame_ts_ms": t} for t in ts]}}


def test_describe_scene_binding_and_trace():
    cache = make_cache(captions={"4625": [
        {"text": "A player runs with the ball.", "confidence": 0.81, "box": _box(0, 0, 1, 1)}]})
    step = {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}}
    binding, result = op_describe_scene(step, _frames_env(4625), cache)
    assert binding["kind"] == "captions"
    assert binding["items"][0]["text"] == "A player runs with the ball."
    assert binding["items"][0]["frame_ts_ms"] == 4625
    assert result["status"] == "done"
    assert result["output_label"] == "A player runs with the ball."
    # one overlay (the full-frame caption box) at the focus frame.
    assert len(result["evidence"]["overlays"]) == 1
    assert result["evidence"]["overlays"][0]["label"] == "A player runs with the ball."


def test_describe_scene_empty_is_diagnostic():
    # No cached captions in the sampled window -> status:empty + a diagnostic note (D-DR5).
    cache = make_cache(captions={})
    step = {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}}
    binding, result = op_describe_scene(step, _frames_env(4625), cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["note"] and "searched" in result["note"]  # WHAT was searched + WHY


def test_describe_scene_caption_text_bounded():
    # Adversarial/security: a cached caption longer than MAX_CAPTION_LEN is truncated before it
    # enters the binding (model free text is bounded at the boundary).
    long_text = "x" * (MAX_CAPTION_LEN + 100)
    cache = make_cache(captions={"4625": [
        {"text": long_text, "confidence": 0.5, "box": _box(0, 0, 1, 1)}]})
    step = {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}}
    binding, _ = op_describe_scene(step, _frames_env(4625), cache)
    assert len(binding["items"][0]["text"]) == MAX_CAPTION_LEN


def test_describe_scene_respects_max_captions():
    # max_captions caps how many captions the binding carries across frames.
    cache = make_cache(captions={
        "100": [{"text": "a", "confidence": 0.5, "box": _box(0, 0, 1, 1)}],
        "200": [{"text": "b", "confidence": 0.5, "box": _box(0, 0, 1, 1)}],
    })
    step = {"id": "scene", "op": "describe_scene", "args": {"frames": "frames", "max_captions": 1}}
    binding, _ = op_describe_scene(step, _frames_env(100, 200), cache)
    assert len(binding["items"]) == 1


def test_answer_caption_grounds_to_caption_text():
    env = {"scene": {"kind": "captions", "items": [
        {"text": "A goal is scored.", "confidence": 0.9, "box": _box(0, 0, 1, 1), "frame_ts_ms": 4625}]}}
    binding, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "what is happening?"}},
        env, make_cache())
    assert binding["value"]["grounded"] is True
    assert binding["value"]["answer"] == "A goal is scored."
    # codex P1: the grounded captions answer renders the readout in output_label, NOT "No".
    assert result["output_label"] == "A goal is scored."


def test_answer_caption_empty_grounds_out():
    env = {"scene": {"kind": "captions", "items": []}}
    binding, _ = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "q"}}, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "no-grounded-answer"


def test_answer_caption_output_label_is_readout():
    # codex P1 regression guard: a grounded captions answer must NOT fall into the Yes/No branch.
    env = {"scene": {"kind": "captions", "items": [
        {"text": "Players celebrate near the goal.", "confidence": 0.7,
         "box": _box(0, 0, 1, 1), "frame_ts_ms": 4625}]}}
    _, result = op_answer(
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "q"}}, env, make_cache())
    assert result["output_label"] == "Players celebrate near the goal."
    assert result["output_label"] not in ("Yes", "No")
