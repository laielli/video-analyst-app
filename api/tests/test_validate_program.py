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

    # describe_scene.max_captions bound mirrors MAX_SCENE_CAPTIONS (the count cap).
    ds = defs["describe_scene"]["properties"]["args"]["properties"]
    assert ds["max_captions"]["minimum"] == 1
    assert ds["max_captions"]["maximum"] == MAX_SCENE_CAPTIONS

    # string-arg caps on id (every op), binding, filter.where.field/equals.
    assert defs["binding"]["maxLength"] == MAX_STR_ARG_LEN
    where = defs["filter"]["properties"]["args"]["properties"]["where"]["properties"]
    assert where["field"]["maxLength"] == MAX_STR_ARG_LEN
    assert where["equals"]["maxLength"] == MAX_STR_ARG_LEN
    for op in ("sample_frames", "detect", "describe_scene", "crop", "read_text", "filter",
               "count", "temporal_order", "answer"):
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


# ---------------------------------------------------------------------------------------
# describe_scene — the new captions primitive's validator surface (Phase 1)
# ---------------------------------------------------------------------------------------

def _scene_program(args=None, with_max=None):
    a = {"frames": "f"}
    if with_max is not None:
        a["max_captions"] = with_max
    return {"program": [
        {"id": "f", "op": "sample_frames", "args": args or {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": a},
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "what is happening?"}},
    ]}


def test_describe_scene_kind_flows_to_answer():
    # sample_frames -> describe_scene -> answer validates: the captions kind is answerable.
    assert validation_errors(_scene_program(), clip=canned.clip_by_id("single-goal")) == []
    # with an explicit (valid) max_captions arg too.
    assert validation_errors(_scene_program(with_max=4), clip=canned.clip_by_id("single-goal")) == []


def test_describe_scene_rejects_non_frames_input():
    # describe_scene fed a non-frames binding (detections) -> a semantic kind-flow error.
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "d"}},
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert any("describe_scene" in e and "frames" in e for e in errs)


def test_describe_scene_max_captions_overlong_rejected():
    # max_captions over the cap is rejected by _arg_domain_errors (adversarial; mirrors
    # test_detect_classes_overlong_rejected).
    over = _scene_program(with_max=MAX_SCENE_CAPTIONS + 1)
    assert validation_errors(over, clip=canned.clip_by_id("single-goal"))
    huge = _scene_program(with_max=100000)
    assert validation_errors(huge, clip=canned.clip_by_id("single-goal"))
    # zero / negative also rejected (below the floor).
    assert validation_errors(_scene_program(with_max=0), clip=canned.clip_by_id("single-goal"))


def test_describe_scene_unknown_op_still_rejected():
    # The whitelist stays closed: a 'caption_scene' typo is rejected (no fuzzy op match).
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "scene", "op": "caption_scene", "args": {"frames": "f"}},
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal"))


def test_temporal_order_rejects_captions():
    # captions joining COLLECTION_KINDS must NOT silently legalize temporal_order over captions
    # (op_temporal_order keys on det_id). The explicit kind allow-list closes that off.
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "f"}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "scene", "by": "timestamp"}},
        {"id": "r", "op": "answer", "args": {"from": "ordered", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert any("temporal_order" in e and "events" in e for e in errs), \
        f"temporal_order(captions) must be a kind-flow error, got: {errs}"


def test_captions_compose_with_count_and_filter():
    # The default knob: captions are a full COLLECTION_KIND, so count/filter over them validate.
    count_prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "f"}},
        {"id": "n", "op": "count", "args": {"items": "scene"}},
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    assert validation_errors(count_prog, clip=canned.clip_by_id("single-goal")) == []
