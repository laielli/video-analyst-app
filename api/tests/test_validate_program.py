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

    # describe_scene.args.max_captions bound matches MAX_SCENE_CAPTIONS (1..cap).
    mc = defs["describe_scene"]["properties"]["args"]["properties"]["max_captions"]
    assert mc["minimum"] == 1
    assert mc["maximum"] == MAX_SCENE_CAPTIONS

    sf = defs["sample_frames"]["properties"]["args"]["properties"]
    assert sf["fps"]["minimum"] == 1 and sf["fps"]["maximum"] == 30
    assert sf["start_ms"]["minimum"] == 0 and sf["start_ms"]["maximum"] == MAX_TIME_MS
    assert sf["end_ms"]["minimum"] == 0 and sf["end_ms"]["maximum"] == MAX_TIME_MS

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


# ---- describe_scene (layer-2 trust boundary) --------------------------------------------

def _scene_prog(scene_args, *, from_id="scene"):
    return {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": scene_args},
        {"id": "r", "op": "answer", "args": {"from": from_id, "question": "what is happening?"}},
    ]}


def test_describe_scene_kind_flows_to_answer():
    # sample_frames -> describe_scene -> answer validates (captions is answerable).
    assert validation_errors(_scene_prog({"frames": "frames"}),
                             clip=canned.clip_by_id("single-goal")) == []


def test_describe_scene_max_captions_within_bounds_ok():
    assert validation_errors(_scene_prog({"frames": "frames", "max_captions": MAX_SCENE_CAPTIONS}),
                             clip=canned.clip_by_id("single-goal")) == []


def test_describe_scene_rejects_non_frames_input():
    # describe_scene over a `detections` binding is a kind-flow error.
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "people"}},
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert any("describe_scene" in e and "frames" in e for e in errs)


def test_describe_scene_max_captions_overlong_rejected():
    # max_captions over the cap is rejected (adversarial). Structural schema rejects it too.
    errs = validation_errors(_scene_prog({"frames": "frames", "max_captions": MAX_SCENE_CAPTIONS + 1}),
                             clip=canned.clip_by_id("single-goal"))
    assert errs, "max_captions over MAX_SCENE_CAPTIONS must be rejected"
    # Semantic-layer guard fires too (validation directly, structure bypassed).
    sem = semantic_errors(_scene_prog({"frames": "frames", "max_captions": 999})["program"])
    assert any("max_captions" in e for e in sem)


def test_describe_scene_unknown_op_still_rejected():
    # A typo op `caption_scene` is rejected structurally (the schema op enum).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "scene", "op": "caption_scene", "args": {"frames": "frames"}},
        {"id": "r", "op": "answer", "args": {"from": "scene", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert errs and any(e.startswith("structural:") for e in errs)


def test_temporal_order_rejects_captions():
    # codex P1: captions are a COLLECTION_KIND (filter/count compose), but temporal_order keys on
    # det_id and CANNOT order captions — it must reject them via its explicit kind allow-list.
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}},
        {"id": "ord", "op": "temporal_order", "args": {"events": "scene", "by": "timestamp"}},
        {"id": "r", "op": "answer", "args": {"from": "ord", "question": "q"}},
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id("single-goal"))
    assert any("temporal_order" in e and "captions" in e for e in errs)


def test_captions_compose_with_filter_and_count():
    # captions IS a COLLECTION_KIND, so filter/count over captions DO validate (composability).
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}},
        {"id": "n", "op": "count", "args": {"items": "scene"}},
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id("single-goal")) == []


def test_pinned_scene_program_passes_validator():
    raw = json.loads((EX / "single-goal_scene_program.json").read_text())
    errs = validation_errors(raw, clip=canned.clip_by_id("single-goal"))
    assert errs == [], errs
