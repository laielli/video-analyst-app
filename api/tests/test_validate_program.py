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

    # describe_scene.args.max_captions cap is kept in sync with MAX_SCENE_CAPTIONS.
    mc = defs["describe_scene"]["properties"]["args"]["properties"]["max_captions"]
    assert mc["minimum"] == 1 and mc["maximum"] == MAX_SCENE_CAPTIONS

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
# describe_scene (new primitive) — kind flow + arg-domain + adversarial rejects
# ---------------------------------------------------------------------------------------

def _scene(frames="frames", **args):
    return {"id": "scene", "op": "describe_scene", "args": {"frames": frames, **args}}


def test_describe_scene_kind_flows_to_answer():
    # sample_frames -> describe_scene -> answer is a clean, valid chain (captions is answerable).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        _scene(),
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "What is happening?"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []


def test_describe_scene_rejects_non_frames_input():
    # describe_scene over a detections binding (not frames) -> kind-flow error.
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        _scene(frames="d"),
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert any("expects kind 'frames'" in e for e in errs)


def test_describe_scene_max_captions_overlong_rejected():
    # max_captions = MAX_SCENE_CAPTIONS + 1 is rejected (structurally by the schema maximum).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        _scene(max_captions=MAX_SCENE_CAPTIONS + 1),
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    # the SEMANTIC-level domain check also fires when structural is bypassed.
    sem = semantic_errors(prog["program"], clip=canned.clip_by_id("single-goal"))
    assert any("max_captions" in e and "out of" in e for e in sem)


def test_describe_scene_unknown_op_still_rejected():
    # a 'caption_scene' typo is not whitelisted -> rejected before execution.
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        {"id": "scene", "op": "caption_scene", "args": {"frames": "frames"}},
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal"))


def test_temporal_order_rejects_captions():
    # temporal_order(describe_scene(...)) -> kind-flow error: op_temporal_order keys on det_id
    # and cannot order a captions binding (codex P1).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        _scene(),
        {"id": "ord", "op": "temporal_order", "args": {"events": "scene", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ord", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert any("temporal_order" in e and "captions" in e for e in errs)


def test_captions_compose_with_filter_and_count():
    # captions is a FULL collection kind: filter/count accept it (composability headroom).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        _scene(),
        {"id": "n", "op": "count", "args": {"items": "scene"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []
