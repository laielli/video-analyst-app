"""
Focused unit tests for the trust-boundary validator (validate_program.semantic_errors +
structural_errors + the new resource caps) and the limits<->schema sync guard. There was no
dedicated unit test for semantic_errors before this entry — it was exercised only indirectly via
test_free_text_run.py + the CLI. These pin the new arg-domain / cap helpers and the regression
guard that the defaults are not set below any real committed program.
"""
from __future__ import annotations

import json
from pathlib import Path

import canned
from validate_program import (
    SCHEMA_PATH,
    semantic_errors,
    sampled_frame_count,
    strip_comments,
    structural_errors,
    validation_errors,
)
from limits import (
    MAX_DETECT_CLASSES,
    MAX_SAMPLED_FRAMES,
    MAX_SCENE_CAPTIONS,
    MAX_STEPS,
    MAX_STR_ARG_LEN,
    MAX_TIME_MS,
)

API_DIR = Path(__file__).resolve().parent.parent
EX = API_DIR / "examples"

# Every committed program + the clip it runs against (so duration-bounded checks are exercised).
PINNED = [
    (EX / "hero_program.json", "single-goal"),
    (EX / "bernabeu-counter_count_program.json", "bernabeu-counter"),
    (EX / "bernabeu-counter_first-goal-7_program.json", "bernabeu-counter"),
    (EX / "bernabeu-counter_first-goal-23_program.json", "bernabeu-counter"),
    (EX / "bernabeu-counter_scorer-number_program.json", "bernabeu-counter"),
]


def test_pinned_programs_pass_tightened_validator():
    # Regression guard: defaults must not be set below any real committed program.
    for path, clip_id in PINNED:
        raw = json.loads(path.read_text())
        clip = canned.clip_by_id(clip_id)
        errs = validation_errors(raw, clip=clip)
        assert errs == [], f"{path.name} should pass the tightened validator, got: {errs}"


def test_limit_constants_match_schema():
    # Mandatory sync guard: the schema's static limits MUST equal the limits.py constants so they
    # can never silently drift (the schema is duplicated guidance; limits.py is the source).
    schema = json.loads(SCHEMA_PATH.read_text())
    defs = schema["$defs"]

    assert schema["properties"]["program"]["maxItems"] == MAX_STEPS

    classes = defs["detect"]["properties"]["args"]["properties"]["classes"]
    assert classes["minItems"] == 1
    assert classes["maxItems"] == MAX_DETECT_CLASSES
    assert classes["items"]["maxLength"] == MAX_STR_ARG_LEN

    sf = defs["sample_frames"]["properties"]["args"]["properties"]
    assert sf["fps"]["minimum"] == 1 and sf["fps"]["maximum"] == 30
    assert sf["start_ms"]["minimum"] == 0 and sf["start_ms"]["maximum"] == MAX_TIME_MS
    assert sf["end_ms"]["minimum"] == 0 and sf["end_ms"]["maximum"] == MAX_TIME_MS

    # describe_scene.args.max_captions caps to MAX_SCENE_CAPTIONS (kept in sync with limits.py).
    ds = defs["describe_scene"]["properties"]["args"]["properties"]
    assert ds["max_captions"]["minimum"] == 1
    assert ds["max_captions"]["maximum"] == MAX_SCENE_CAPTIONS

    # string-arg caps on id (every op), binding, filter.where.field/equals.
    assert defs["binding"]["maxLength"] == MAX_STR_ARG_LEN
    where = defs["filter"]["properties"]["args"]["properties"]["where"]["properties"]
    assert where["field"]["maxLength"] == MAX_STR_ARG_LEN
    assert where["equals"]["maxLength"] == MAX_STR_ARG_LEN
    for op in ("sample_frames", "detect", "describe_scene", "crop", "read_text", "filter", "count",
               "temporal_order", "answer"):
        assert defs[op]["properties"]["id"]["maxLength"] == MAX_STR_ARG_LEN


def test_schema_maxitems_rejects_independently():
    # A MAX_STEPS + 1 program fails the schema-level structural gate ALONE (even if semantic_errors
    # regressed). Proves the static limit is a real second line of defense.
    schema = json.loads(SCHEMA_PATH.read_text())
    steps = [{"id": f"s{k}", "op": "count", "args": {"items": "s0"}} for k in range(MAX_STEPS + 1)]
    struct = structural_errors({"program": steps}, schema)
    assert struct, "schema maxItems must reject an over-length program structurally"


def test_legit_full_clip_sweep_passes():
    # A sample_frames over the whole bernabeu clip (0..30086ms) @ fps8 and @ fps30 both pass the
    # cumulative cap (the envelope sits above the largest legitimate sweep). duration_ms=30086 is
    # in canned.py, not the clip manifest.
    clip = canned.clip_by_id("bernabeu-counter")
    dur = clip["duration_ms"]
    assert sampled_frame_count(0, dur, 8) <= MAX_SAMPLED_FRAMES
    assert sampled_frame_count(0, dur, 30) <= MAX_SAMPLED_FRAMES
    for fps in (8, 30):
        prog = {"program": [
            {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": dur, "fps": fps}},
            {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "d"}},
            {"id": "r", "op": "answer", "args": {"from": "n", "question": "how many?"}},
        ]}
        assert validation_errors(prog, clip=clip) == [], f"full-clip sweep @ fps{fps} should pass"


def test_step_cap_boundary_exact():
    # Exactly MAX_STEPS passes the step cap; MAX_STEPS + 1 fails it. (Boundary correctness.)
    def chain(n):
        steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
        # n total steps: 1 sample + (n-2) counts + 1 answer
        for k in range(n - 2):
            steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
        steps.append({"id": "r", "op": "answer", "args": {"from": "f", "question": "q"}})
        return steps

    at_cap = chain(MAX_STEPS)
    assert len(at_cap) == MAX_STEPS
    assert not any("max is" in e and "steps" in e for e in semantic_errors(at_cap))

    over = chain(MAX_STEPS + 1)
    assert len(over) == MAX_STEPS + 1
    assert any("steps; max is" in e for e in semantic_errors(over))


def test_non_string_classes_element_rejected():
    # semantic-level guard: a non-string classes element (structural anyOf would also reject, but
    # the semantic check is the gate when validation is called directly with structure bypassed).
    sem = semantic_errors([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": [123]}},
        {"id": "r", "op": "answer", "args": {"from": "d", "question": "q"}},
    ])
    assert any("not a string" in e for e in sem)


def test_banned_invisible_chars_rejected_everywhere():
    # Trust-boundary guard (2026-07-19): zero-width / RTL-override / control chars in ANY
    # free-string arg are rejected — a question rendered with a bidi override would display
    # differently than it executes, and a noise-salted query must ground out (noise-precedence),
    # not be answered as if clean. Mirrors the live adv-unicode-rtl-noise / adv-zero-width-
    # count-bait fixtures, which replay through this exact rejection to an honest ungrounded.
    def _prog(question="how many players are visible?", cls="person", equals=None):
        steps = [
            {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
            {"id": "d", "op": "detect", "args": {"frames": "f", "classes": [cls]}},
        ]
        if equals is not None:
            steps.append({"id": "g", "op": "filter",
                          "args": {"items": "d", "where": {"field": "cls", "equals": equals}}})
        steps.append({"id": "r", "op": "answer",
                      "args": {"from": steps[-1]["id"], "question": question}})
        return steps

    # RTL override in answer.question (the adv-unicode-rtl-noise shape).
    sem = semantic_errors(_prog(question="How many players ‮erehtsi‬ are visible?"))
    assert any("invisible/bidi-control" in e and "(answer)" in e for e in sem)
    # Zero-width space in answer.question (the adv-zero-width-count-bait shape).
    sem = semantic_errors(_prog(question="How​ many​ players?"))
    assert any("invisible/bidi-control" in e for e in sem)
    # C0 control smuggled into a detect class.
    sem = semantic_errors(_prog(cls="per\x07son"))
    assert any("invisible/bidi-control" in e and "(detect)" in e for e in sem)
    # Bidi isolate smuggled into filter.where.equals.
    sem = semantic_errors(_prog(equals="⁦person⁩"))
    assert any("invisible/bidi-control" in e and "(filter)" in e for e in sem)


def test_legit_unicode_survives_banned_char_guard():
    # No overfitting: real-world typography (№, em-dash, curly quotes, accents) must pass —
    # the ban covers only invisible/control/bidi code points, never visible characters.
    sem = semantic_errors([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "r", "op": "answer",
         "args": {"from": "d", "question": "Does №10 — the 'capitán' — score?"}},
    ])
    assert not any("invisible/bidi-control" in e for e in sem)


def test_unserializable_doc_fails_closed():
    # A doc that can't be json-serialized is treated as oversized (fails closed), not a crash.
    class _Unserializable:
        pass

    errs = validation_errors({"program": [{"x": _Unserializable()}]})
    assert errs and errs[0].startswith("size:")


def test_arg_domain_leaves_short_strings_alone():
    # The arg-domain checks must NOT reject legitimate short strings (no overfitting).
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person", "ball"]}},
        {"id": "g", "op": "filter", "args": {"items": "d", "where": {"field": "cls", "equals": "person"}}},
        {"id": "r", "op": "answer", "args": {"from": "g", "question": "is there a person?"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []


# --------------------------------------------------------------------------------------
# describe_scene (Phase 1) — kind flow, arg-domain, adversarial rejects, temporal_order close-off
# --------------------------------------------------------------------------------------

def _scene_prog(scene_args=None, extra=None):
    scene_args = scene_args if scene_args is not None else {"frames": "frames"}
    steps = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 2000, "end_ms": 4000, "fps": 4}},
        {"id": "scene", "op": "describe_scene", "args": scene_args},
    ]
    if extra:
        steps.extend(extra)
    steps.append({"id": "result", "op": "answer", "args": {"from": "scene", "question": "what is happening?"}})
    return {"program": steps}


def test_describe_scene_kind_flows_to_answer():
    # sample_frames -> describe_scene -> answer is a valid chain (captions is answerable).
    clip = {"duration_ms": 8000}
    assert validation_errors(_scene_prog(), clip=clip) == []


def test_describe_scene_rejects_non_frames_input():
    # describe_scene over a `detections` binding is a kind-flow error (frames-only input).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 2000, "end_ms": 4000, "fps": 4}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip={"duration_ms": 8000})
    assert any("expects kind 'frames'" in e and "describe_scene" in e for e in errs)


def test_describe_scene_max_captions_overlong_rejected():
    # adversarial: max_captions over the cap is rejected (structurally by the schema maximum).
    errs = validation_errors(_scene_prog({"frames": "frames", "max_captions": MAX_SCENE_CAPTIONS + 1}),
                             clip={"duration_ms": 8000})
    assert errs


def test_describe_scene_max_captions_in_range_ok():
    # a max_captions within bounds validates.
    assert validation_errors(_scene_prog({"frames": "frames", "max_captions": MAX_SCENE_CAPTIONS}),
                             clip={"duration_ms": 8000}) == []


def test_describe_scene_unknown_op_still_rejected():
    # adversarial: a typo'd op ('caption_scene') is rejected structurally (the op enum).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 2000, "end_ms": 4000, "fps": 4}},
        {"id": "scene", "op": "caption_scene", "args": {"frames": "frames"}},
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip={"duration_ms": 8000})
    assert errs and any(e.startswith("structural:") for e in errs)


def test_temporal_order_rejects_captions():
    # codex P1: temporal_order(captions) is interpreter-meaningless (op_temporal_order keys on
    # det_id) and MUST NOT validate, even though captions is a COLLECTION_KIND.
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 2000, "end_ms": 4000, "fps": 4}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "scene", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip={"duration_ms": 8000})
    assert any("temporal_order" in e and "scene" in e for e in errs), errs


def test_filter_and_count_accept_captions():
    # captions is a full COLLECTION_KIND: filter/count compose over it (composability headroom).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 2000, "end_ms": 4000, "fps": 4}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}},
        {"id": "n", "op": "count", "args": {"items": "scene"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    assert validation_errors(prog, clip={"duration_ms": 8000}) == []


# --------------------------------------------------------------------------------------
# detect.on_pitch (optional bool arg) — validator acceptance/rejection
# --------------------------------------------------------------------------------------

def _detect_prog(detect_args):
    return {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", **detect_args}},
        {"id": "result", "op": "answer", "args": {"from": "people", "question": "is there a person?"}},
    ]}


def test_on_pitch_absent_is_valid():
    # unset (default false, unchanged behavior) must still validate.
    prog = _detect_prog({"classes": ["person"]})
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []


def test_on_pitch_true_is_valid():
    prog = _detect_prog({"classes": ["person"], "on_pitch": True})
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []


def test_on_pitch_false_is_valid():
    prog = _detect_prog({"classes": ["person"], "on_pitch": False})
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []


def test_on_pitch_non_bool_rejected_structurally():
    # the schema types on_pitch as boolean; a string/number/object must fail structurally.
    for bad in ("true", 1, {"x": 1}, [True]):
        prog = _detect_prog({"classes": ["person"], "on_pitch": bad})
        errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
        assert errs and all(e.startswith("structural:") for e in errs), (bad, errs)


def test_on_pitch_non_bool_rejected_semantically_when_structure_bypassed():
    # the semantic-level arg-domain check fires independently (mirrors test_detect_classes_*
    # in test_program_limits.py): a caller that only runs semantic_errors must still catch it.
    prog = _detect_prog({"classes": ["person"], "on_pitch": "yes"})
    sem = semantic_errors(prog["program"], clip=canned.clip_by_id("single-goal"))
    assert any("on_pitch must be a boolean" in e for e in sem)


def test_pinned_bernabeu_first_goal_programs_pass_with_widened_window():
    # regression guard: the widened 9000-14800ms windows (containing the real ~14250ms goal
    # moment) + on_pitch:true still validate against the committed clip duration.
    for name in ("bernabeu-counter_first-goal-7_program.json", "bernabeu-counter_first-goal-23_program.json"):
        raw = json.loads((EX / name).read_text())
        assert validation_errors(raw, clip=canned.clip_by_id("bernabeu-counter")) == [], name


def test_scene_pinned_program_validates():
    # the committed pinned scene program passes the real validator over its clip.
    raw = json.loads((EX / "single-goal_scene_program.json").read_text())
    assert validation_errors(raw, clip=canned.clip_by_id("single-goal")) == []
