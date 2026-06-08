"""
Dedicated unit tests for the 8 ops in interpreter/primitives.py — the vision-API surface the
generated program composes. Each op is fn(step, env, cache) -> (binding, step_result); we assert
on BOTH the binding (what later steps reference) and the run-doc trace entry (status/provenance/
EVIDENCE overlays the UI renders). detect/read_text/temporal_order read from the Cache; crop/
filter/count are pure compute over env; op_answer dispatches on the `from` binding's kind, and its
grounded temporal branch re-reads the cache to build scorer overlays.
"""
from __future__ import annotations

from interpreter import Cache
from interpreter.primitives import (
    OPS,
    op_sample_frames,
    op_detect,
    op_crop,
    op_read_text,
    op_filter,
    op_count,
    op_temporal_order,
    op_answer,
    _subject_label,
)


def make_cache(detect=None, read_text=None, events=None):
    # The Cache constructor reads data["clip"] eagerly — it MUST be present or it KeyErrors.
    return Cache({
        "clip": {"id": "t", "width": 100, "height": 100, "duration_ms": 5000},
        "detect": detect or {},
        "read_text": read_text or {},
        "events": events or [],
    })


def _step(id, op, args):
    return {"id": id, "op": op, "args": args}


# ---- op_sample_frames -------------------------------------------------------------------

def test_sample_frames_stride_and_count():
    cache = make_cache()
    # fps 8 -> stride = max(1, round(1000/8)) = round(125) = 125.
    step = _step("frames", "sample_frames", {"start_ms": 0, "end_ms": 500, "fps": 8})
    binding, result = op_sample_frames(step, {}, cache)
    expected_ts = list(range(0, 501, 125))  # [0,125,250,375,500]
    assert [it["frame_ts_ms"] for it in binding["items"]] == expected_ts
    assert binding["kind"] == "frames"
    assert result["output_label"] == f"{len(expected_ts)} frames @ 8fps"
    assert result["status"] == "done"
    assert result["op"] == "sample_frames"
    # evidence focuses on the middle frame.
    assert result["evidence"]["frame_ts_ms"] == expected_ts[len(expected_ts) // 2]


def test_sample_frames_stride_collapses_to_one():
    cache = make_cache()
    # fps 1000 -> stride = max(1, round(1000/1000)) = 1, so every ms is sampled.
    step = _step("frames", "sample_frames", {"start_ms": 0, "end_ms": 3, "fps": 1000})
    binding, _ = op_sample_frames(step, {}, cache)
    assert [it["frame_ts_ms"] for it in binding["items"]] == [0, 1, 2, 3]


# ---- op_detect --------------------------------------------------------------------------

def _frames_env(*ts):
    return {"frames": {"kind": "frames", "items": [{"frame_ts_ms": t} for t in ts]}}


def test_detect_matches_classes_case_insensitive_and_overlays_focus_only():
    cache = make_cache(detect={
        "100": [{"det_id": "a", "cls": "Person", "confidence": 0.9, "box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}],
        "200": [{"det_id": "b", "cls": "PERSON", "confidence": 0.8, "box": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1}},
                {"det_id": "c", "cls": "ball", "confidence": 0.5, "box": {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}}],
    })
    env = _frames_env(100, 200)
    step = _step("people", "detect", {"frames": "frames", "classes": ["person"]})
    binding, result = op_detect(step, env, cache)
    # case-insensitive match: a and b match, the ball does not.
    dets = binding["items"]
    assert {d["det_id"] for d in dets} == {"a", "b"}
    assert binding["kind"] == "detections"
    assert result["status"] == "done"
    assert result["output_label"] == "2 people"
    assert result["confidence"] == 0.9  # max confidence
    # focus_ts is the last detection's frame (200); overlays ONLY on that frame.
    assert result["evidence"]["frame_ts_ms"] == 200
    assert len(result["evidence"]["overlays"]) == 1
    assert result["evidence"]["overlays"][0]["tone"] == "azure"


def test_detect_empty_status_and_note():
    cache = make_cache(detect={})  # no cached detections anywhere
    env = _frames_env(100, 200)
    step = _step("people", "detect", {"frames": "frames", "classes": ["person"]})
    binding, result = op_detect(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["note"] == "no detections in the sampled frames"
    assert result["output_label"] == "0 people"
    # with no dets, focus_ts falls back to the last frame.
    assert result["evidence"]["frame_ts_ms"] == 200


def test_detect_noun_is_objects_for_non_person():
    cache = make_cache(detect={
        "100": [{"det_id": "c", "cls": "ball", "confidence": 0.5, "box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}}],
    })
    env = _frames_env(100)
    step = _step("balls", "detect", {"frames": "frames", "classes": ["ball"]})
    _, result = op_detect(step, env, cache)
    assert result["output_label"] == "1 objects"


# ---- op_crop ----------------------------------------------------------------------------

def _dets_env(*dets):
    return {"people": {"kind": "detections", "items": list(dets)}}


def test_crop_jersey_region_box_math():
    b = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4}
    env = _dets_env({"det_id": "d0", "box": b, "confidence": 0.77, "frame_ts_ms": 300})
    step = _step("jerseys", "crop", {"detections": "people", "region": "jersey"})
    binding, result = op_crop(step, env, make_cache())
    crop = binding["items"][0]
    assert crop["box"] == {
        "x": round(0.1 + 0.22 * 0.5, 4),
        "y": round(0.2 + 0.10 * 0.4, 4),
        "w": round(0.56 * 0.5, 4),
        "h": round(0.22 * 0.4, 4),
    }
    assert crop["crop_id"] == "c0"
    assert crop["det_id"] == "d0"
    assert crop["person_box"] == b  # original carried as person_box
    assert crop["person_conf"] == 0.77
    assert binding["kind"] == "crops"
    assert result["output_label"] == "1 jersey crops"
    assert result["evidence"]["overlays"][0]["label"] == "crop_jersey"


def test_crop_full_region_passthrough_and_crop_id_minting():
    b0 = {"x": 0.0, "y": 0.0, "w": 0.2, "h": 0.2}
    b1 = {"x": 0.5, "y": 0.5, "w": 0.3, "h": 0.3}
    env = _dets_env(
        {"det_id": "d0", "box": b0, "confidence": 0.5, "frame_ts_ms": 100},
        {"det_id": "d1", "box": b1, "confidence": 0.6, "frame_ts_ms": 200},
    )
    step = _step("crops", "crop", {"detections": "people", "region": "full"})
    binding, result = op_crop(step, env, make_cache())
    crops = binding["items"]
    assert [c["crop_id"] for c in crops] == ["c0", "c1"]  # minted by index
    assert crops[0]["box"] == b0 and crops[1]["box"] == b1  # full -> passthrough (copy)
    assert crops[0]["box"] is not b0  # dict(b) makes a copy
    assert result["evidence"]["overlays"][0]["label"] == "crop_full"


def test_crop_empty_when_no_detections():
    env = _dets_env()
    step = _step("crops", "crop", {"detections": "people", "region": "jersey"})
    binding, result = op_crop(step, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"


# ---- op_read_text -----------------------------------------------------------------------

def _crops_env(*crops):
    return {"jerseys": {"kind": "crops", "items": list(crops)}}


def _crop(det_id, ts=400):
    return {"crop_id": "c0", "det_id": det_id, "box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1},
            "person_box": {"x": 0, "y": 0, "w": 0.2, "h": 0.2}, "frame_ts_ms": ts}


def test_read_text_source_precedence_pinned_over_cached_over_live():
    cache = make_cache(read_text={
        "dl": {"text": "7", "confidence": 0.9, "source": "live"},
        "dc": {"text": "9", "confidence": 0.8, "source": "cached"},
        "dp": {"text": "10", "confidence": None, "source": "pinned"},
    })
    env = _crops_env(_crop("dl"), _crop("dc"), _crop("dp"))
    step = _step("numbers", "read_text", {"crops": "jerseys"})
    _, result = op_read_text(step, env, cache)
    # pinned wins the step-level source whenever ANY pin is present.
    assert result["source"] == "pinned"
    # output_label is sorted unique texts.
    assert result["output_label"] == "10, 7, 9"
    assert result["status"] == "done"


def test_read_text_cached_when_no_pin():
    cache = make_cache(read_text={
        "dl": {"text": "7", "confidence": 0.9, "source": "live"},
        "dc": {"text": "9", "confidence": 0.8, "source": "cached"},
    })
    env = _crops_env(_crop("dl"), _crop("dc"))
    step = _step("numbers", "read_text", {"crops": "jerseys"})
    _, result = op_read_text(step, env, cache)
    assert result["source"] == "cached"


def test_read_text_note_when_nothing_legible():
    cache = make_cache(read_text={})  # no reads at all
    env = _crops_env(_crop("dx"))
    step = _step("numbers", "read_text", {"crops": "jerseys"})
    binding, result = op_read_text(step, env, cache)
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["output_label"] == "(none legible)"
    assert result["note"] == "no legible text in the jersey crops (read returned nothing)"


def test_read_text_carries_cache_note_when_present():
    cache = make_cache(read_text={
        "dp": {"text": "10", "confidence": None, "source": "pinned", "note": "hand-verified"},
    })
    env = _crops_env(_crop("dp"))
    step = _step("numbers", "read_text", {"crops": "jerseys"})
    _, result = op_read_text(step, env, cache)
    assert result["note"] == "hand-verified"


# ---- op_filter --------------------------------------------------------------------------

def test_filter_string_coerced_equality_and_kind_preserved():
    src = {"kind": "texts", "items": [
        {"text": "10", "person_box": {"x": 0, "y": 0, "w": 0.1, "h": 0.1}, "frame_ts_ms": 400},
        {"text": "7", "person_box": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1}, "frame_ts_ms": 410},
    ]}
    env = {"numbers": src}
    # equals is the INT 10 — string-coerced equality must still match the str "10".
    step = _step("tens", "filter", {"items": "numbers", "where": {"field": "text", "equals": 10}})
    binding, result = op_filter(step, env, make_cache())
    assert len(binding["items"]) == 1
    assert binding["items"][0]["text"] == "10"
    assert binding["kind"] == "texts"  # kind preserved
    assert result["status"] == "done"
    assert result["output_label"] == "1 match · #10"
    # overlay drawn over the matched person_box.
    assert result["evidence"]["overlays"][0]["label"] == "#10"


def test_filter_empty_match_note():
    src = {"kind": "texts", "items": [{"text": "7", "frame_ts_ms": 1}]}
    env = {"numbers": src}
    step = _step("tens", "filter", {"items": "numbers", "where": {"field": "text", "equals": "10"}})
    binding, result = op_filter(step, env, make_cache())
    assert binding["items"] == []
    assert result["status"] == "empty"
    assert result["note"] == "no items with text == 10"
    assert result["output_label"] == "0 match"


# ---- op_count ---------------------------------------------------------------------------

def test_count_value_and_output_label():
    src = {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}, {"det_id": "c"}]}
    env = {"people": src}
    step = _step("n", "count", {"items": "people"})
    binding, result = op_count(step, env, make_cache())
    assert binding == {"kind": "number", "value": 3}
    assert result["output_label"] == "3"
    assert result["input_label"] == "detections"
    assert result["status"] == "done"


def test_count_zero():
    env = {"people": {"kind": "detections", "items": []}}
    step = _step("n", "count", {"items": "people"})
    binding, result = op_count(step, env, make_cache())
    assert binding["value"] == 0
    assert result["output_label"] == "0"


# ---- _subject_label helper --------------------------------------------------------------

def test_subject_label_derivation():
    assert _subject_label([{"text": "10"}]) == "#10"
    assert _subject_label([{"text": ""}, {"text": "7"}]) == "#7"  # skips empty
    assert _subject_label([]) is None
    assert _subject_label([{"text": None}]) is None


# ---- op_temporal_order ------------------------------------------------------------------

def _events_env(*items):
    return {"tens": {"kind": "texts", "items": list(items)}}


def test_temporal_order_subject_is_first_scorer_true():
    cache = make_cache(events=[
        {"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi", "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}},
    ])
    env = _events_env({"det_id": "p_messi", "text": "10", "frame_ts_ms": 4625})
    step = _step("ordered", "temporal_order", {"events": "tens", "by": "timestamp"})
    binding, result = op_temporal_order(step, env, cache)
    v = binding["value"]
    assert v["subject_label"] == "#10"
    assert v["subject_is_first_scorer"] is True
    assert v["first"]["ts_ms"] == 4000
    assert binding["kind"] == "ordered"
    assert result["status"] == "done"
    assert result["output_label"] == "goal @ 4000ms (first)"
    # green goal overlay drawn.
    assert result["evidence"]["overlays"][0]["tone"] == "green"


def test_temporal_order_subject_is_first_scorer_false():
    cache = make_cache(events=[
        {"ts_ms": 4000, "type": "goal", "scorer_det": "someone_else", "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}},
    ])
    env = _events_env({"det_id": "p_messi", "text": "10", "frame_ts_ms": 4625})
    step = _step("ordered", "temporal_order", {"events": "tens", "by": "timestamp"})
    binding, _ = op_temporal_order(step, env, cache)
    assert binding["value"]["subject_is_first_scorer"] is False
    assert binding["value"]["subject_label"] == "#10"


def test_temporal_order_empty_when_no_goal_events():
    cache = make_cache(events=[])  # no goal events
    env = _events_env({"det_id": "p_messi", "text": "10", "frame_ts_ms": 4625})
    step = _step("ordered", "temporal_order", {"events": "tens", "by": "timestamp"})
    binding, result = op_temporal_order(step, env, cache)
    assert binding["value"]["first"] is None
    assert result["status"] == "empty"
    assert result["output_label"] == "no goal events"


# ---- op_answer --------------------------------------------------------------------------

def _ordered_binding(first=None, subject_label="#10", subject_is_first=True, subject_dets=None):
    return {"kind": "ordered", "value": {
        "events": [first] if first else [],
        "first": first,
        "subject_dets": subject_dets or ["p_messi"],
        "subject_label": subject_label,
        "subject_is_first_scorer": subject_is_first,
    }}


def test_answer_ordered_grounded_yes_rereads_cache_for_overlay():
    # The grounded `ordered` branch re-reads the cache via analyzed_frames()/detections_at() to
    # build the scorer overlay; the fixture MUST include matching detect entries.
    cache = make_cache(detect={
        "4625": [{"det_id": "p_messi", "cls": "person", "confidence": 0.75,
                  "box": {"x": 0.03, "y": 0.19, "w": 0.22, "h": 0.63}}],
    })
    first = {"ts_ms": 4000, "type": "goal", "scorer_det": "p_messi", "box": {"x": 0.8, "y": 0.3, "w": 0.1, "h": 0.3}}
    env = {"ordered": _ordered_binding(first=first, subject_is_first=True)}
    step = _step("result", "answer", {"from": "ordered", "question": "Does #10 score the first goal?"})
    binding, result = op_answer(step, env, cache)
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "Yes"
    assert v["verdict"] == "Yes — #10 scored the first goal"
    assert v["yes"] is True
    assert result["status"] == "done"
    assert result["output_label"] == "Yes"
    # the scorer overlay was rebuilt from the cache (overlay path NOT silently skipped).
    overlays = result["evidence"]["overlays"]
    assert overlays and overlays[0]["label"] == "#10 scored"
    assert result["evidence"]["frame_ts_ms"] == 4625


def test_answer_ordered_grounded_no():
    cache = make_cache(detect={})
    first = {"ts_ms": 4000, "type": "goal", "scorer_det": "other", "box": None}
    env = {"ordered": _ordered_binding(first=first, subject_is_first=False)}
    step = _step("result", "answer", {"from": "ordered", "question": "q"})
    binding, result = op_answer(step, env, cache)
    v = binding["value"]
    assert v["answer"] == "No"
    assert v["verdict"] == "No — the first goal was not scored by #10"
    assert v["yes"] is False
    assert result["output_label"] == "No"


def test_answer_count_grounded():
    env = {"n": {"kind": "number", "value": 5}}
    step = _step("result", "answer", {"from": "n", "question": "how many?"})
    binding, result = op_answer(step, env, make_cache())
    v = binding["value"]
    assert v["grounded"] is True
    assert v["answer"] == "5"
    assert v["verdict"] == "5 found"
    assert result["output_label"] == "5"


def test_answer_texts_grounded():
    env = {"numbers": {"kind": "texts", "items": [{"text": "10"}, {"text": "7"}, {"text": "10"}]}}
    step = _step("result", "answer", {"from": "numbers", "question": "what numbers?"})
    binding, result = op_answer(step, env, make_cache())
    v = binding["value"]
    assert v["answer"] == "10, 7"  # sorted unique: "10" < "7"
    assert v["verdict"] == "Read: 10, 7"
    assert result["output_label"] == "10, 7"


def test_answer_presence_grounded_detections():
    env = {"people": {"kind": "detections", "items": [{"det_id": "a"}, {"det_id": "b"}]}}
    step = _step("result", "answer", {"from": "people", "question": "is there a person?"})
    binding, result = op_answer(step, env, make_cache())
    v = binding["value"]
    assert v["answer"] == "Yes"
    assert v["verdict"] == "Yes — found 2"
    assert v["yes"] is True
    assert result["output_label"] == "Yes"


def test_answer_ungrounded_reasons():
    """The ungrounded discriminator (grounded=False + reason) — never a fabricated verdict.
    Reverting the op_answer over-claim guards should turn one of these red."""
    c = make_cache()
    # ordered with no first goal -> no-grounded-answer
    env = {"ordered": _ordered_binding(first=None)}
    b, r = op_answer(_step("a", "answer", {"from": "ordered", "question": "q"}), env, c)
    assert b["value"]["grounded"] is False and b["value"]["reason"] == "no-grounded-answer"
    assert b["value"]["answer"] is None and b["value"]["verdict"] is None
    assert r["status"] == "empty" and r["output_label"] == "(couldn't ground)"

    # ordered with a first goal but NO subject label (subject filter matched nobody) -> ungrounded.
    first = {"ts_ms": 4000, "scorer_det": "x"}
    env = {"ordered": _ordered_binding(first=first, subject_label=None)}
    b, _ = op_answer(_step("a", "answer", {"from": "ordered", "question": "q"}), env, c)
    assert b["value"]["grounded"] is False and b["value"]["reason"] == "no-grounded-answer"

    # count == 0 -> ungrounded
    env = {"n": {"kind": "number", "value": 0}}
    b, _ = op_answer(_step("a", "answer", {"from": "n", "question": "q"}), env, c)
    assert b["value"]["grounded"] is False and b["value"]["reason"] == "no-grounded-answer"

    # empty texts -> ungrounded
    env = {"t": {"kind": "texts", "items": []}}
    b, _ = op_answer(_step("a", "answer", {"from": "t", "question": "q"}), env, c)
    assert b["value"]["grounded"] is False and b["value"]["reason"] == "no-grounded-answer"

    # empty presence -> ungrounded (NOT a fake "No")
    env = {"d": {"kind": "detections", "items": []}}
    b, _ = op_answer(_step("a", "answer", {"from": "d", "question": "q"}), env, c)
    assert b["value"]["grounded"] is False and b["value"]["reason"] == "no-grounded-answer"


def test_answer_unsupported_kind():
    env = {"weird": {"kind": "frob", "items": []}}
    step = _step("result", "answer", {"from": "weird", "question": "q"})
    binding, result = op_answer(step, env, make_cache())
    assert binding["value"]["grounded"] is False
    assert binding["value"]["reason"] == "unsupported-kind"
    assert "frob" in result["note"]


# ---- OPS registry -----------------------------------------------------------------------

def test_ops_registry_covers_all_eight():
    assert set(OPS) == {
        "sample_frames", "detect", "crop", "read_text",
        "filter", "count", "temporal_order", "answer",
    }
    assert OPS["sample_frames"] is op_sample_frames
    assert OPS["answer"] is op_answer
