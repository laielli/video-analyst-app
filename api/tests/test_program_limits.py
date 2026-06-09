"""
Trust-boundary hardening: the adversarial suite. Feeds malformed / oversized / hostile programs
through the validator and the free-text server path and proves they are REJECTED BEFORE ANY OP
EXECUTES, plus runtime-guard tests that prove the caps still fire if the validator is bypassed.

Zero-execution discipline (the suite's whole point): every reject is asserted with a spy on
`Interpreter.run` (the proven class-method monkeypatch from test_free_text_run.py:75-80) showing
the spy was never called. We do NOT patch primitives module attrs — `OPS` holds function refs in
a module dict, so a primitive-level spy must `monkeypatch.setitem(OPS, ...)` (used only where a
primitive-level spy is genuinely needed; the OOM tests assert at Interpreter.run instead).

Also folds in the validator UNIT tests (DESIGN-ROOM: one file, not a separate
test_validate_program.py): pinned-programs-pass, schema/limits drift guard, legit-full-sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import canned
import codegen
import server
import validate_program
from interpreter import Cache, Interpreter, ProgramLimitExceeded
from limits import (
    MAX_DETECT_CLASSES,
    MAX_PROGRAM_BYTES,
    MAX_SAMPLED_FRAMES,
    MAX_STEPS,
    MAX_STR_ARG_LEN,
)
from validate_program import semantic_errors, structural_errors, validation_errors

API_DIR = Path(__file__).resolve().parent.parent
CLIP = canned.clip_by_id("single-goal")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _answer(from_id="x", q="q"):
    return {"id": "result", "op": "answer", "args": {"from": from_id, "question": q}}


def _spy_run(monkeypatch):
    """Install the class-method Interpreter.run spy and return its call counter."""
    called = {"n": 0}
    real = Interpreter.run

    def spy(self, program, query=None):
        called["n"] += 1
        return real(self, program, query=query)

    monkeypatch.setattr(Interpreter, "run", spy)
    return called


def _free_text_reject(mock_codegen, monkeypatch, program, reason="invalid-program"):
    """Drive build_free_text_run_doc with a mocked codegen program and assert it is rejected to
    an ungrounded doc with `reason`, with ZERO Interpreter.run calls."""
    mock_codegen(program=program)
    called = _spy_run(monkeypatch)
    doc = server.build_free_text_run_doc("anything hostile", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == reason
    assert called["n"] == 0, "rejected program must never reach Interpreter.run (trust boundary)"
    return doc


# --------------------------------------------------------------------------------------
# step / frame caps
# --------------------------------------------------------------------------------------

def test_oversized_program_step_bomb_rejected(mock_codegen, monkeypatch):
    # MAX_STEPS + 1 trivial (no-op-arg) count steps + an answer. Schema maxItems OR semantic cap
    # rejects it; either way the program never executes.
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "s0"}} for i in range(MAX_STEPS + 1)]
    steps.append(_answer())
    assert validation_errors({"program": steps})
    _free_text_reject(mock_codegen, monkeypatch, steps)


def test_frame_count_bomb_rejected_without_clip():
    # The server path always resolves a clip, so clip=None is a direct/CLI call. A single huge
    # window with no clip is closed by the duration-INDEPENDENT cumulative cap. The count is O(1)
    # arithmetic — no list is materialized by the validator.
    bomb = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
        _answer(from_id="f"),
    ]}
    errs = validation_errors(bomb, clip=None)
    assert errs
    assert any("frames" in e for e in errs)


def test_frame_count_bomb_rejected_with_clip(mock_codegen, monkeypatch):
    # Many small in-bounds windows summing > MAX_SAMPLED_FRAMES (the cumulative cap). Each window
    # fits the clip duration individually, so only the SUM trips the cap. fps high enough that one
    # window is dense. With clip's duration (5067ms) @ fps30, stride 33 => ~154 frames/window;
    # ceil(MAX/154)+1 windows guarantees the sum exceeds the cap.
    per = len(range(0, CLIP["duration_ms"] + 1, max(1, round(1000 / 30))))
    nwin = MAX_SAMPLED_FRAMES // per + 2
    steps = [
        {"id": f"f{i}", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": CLIP["duration_ms"], "fps": 30}}
        for i in range(nwin)
    ]
    # keep step count under MAX_STEPS so the FRAME cap (not the step cap) is what fires.
    assert len(steps) + 1 <= MAX_STEPS, "test must isolate the frame cap, not trip the step cap"
    steps.append(_answer(from_id="f0"))
    errs = validation_errors({"program": steps}, clip=CLIP)
    assert any("samples" in e and "frames" in e for e in errs)
    _free_text_reject(mock_codegen, monkeypatch, steps)


# --------------------------------------------------------------------------------------
# arg-domain caps
# --------------------------------------------------------------------------------------

def test_detect_classes_empty_rejected(mock_codegen, monkeypatch):
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": []}},
        _answer(from_id="d"),
    ]
    assert validation_errors({"program": prog}, clip=CLIP)
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_detect_classes_overlong_rejected(mock_codegen, monkeypatch):
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect",
         "args": {"frames": "f", "classes": [f"c{i}" for i in range(MAX_DETECT_CLASSES + 1)]}},
        _answer(from_id="d"),
    ]
    assert validation_errors({"program": prog}, clip=CLIP)
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_huge_string_arg_rejected(mock_codegen, monkeypatch):
    # giant filter.where.equals AND a giant classes[0] — both string-length caps.
    big = "x" * (MAX_STR_ARG_LEN + 1)
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": [big]}},
        {"id": "c", "op": "crop", "args": {"detections": "d", "region": "jersey"}},
        {"id": "t", "op": "read_text", "args": {"crops": "c"}},
        {"id": "flt", "op": "filter", "args": {"items": "t", "where": {"field": "text", "equals": big}}},
        _answer(from_id="flt"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert errs
    _free_text_reject(mock_codegen, monkeypatch, prog)


# --------------------------------------------------------------------------------------
# structural / semantic reject classes (regression: each still short-circuits with zero exec)
# --------------------------------------------------------------------------------------

def test_deeply_nested_args_rejected(mock_codegen, monkeypatch):
    # An extra/unknown deeply-nested arg key trips additionalProperties:false (structural) before
    # semantics ever runs.
    prog = [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "evil": {"a": {"b": {"c": [1, 2, 3]}}}}},
        _answer(from_id="f"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert errs and errs[0].startswith("structural:")
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_dangling_ref_rejected(mock_codegen, monkeypatch):
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "nope", "classes": ["person"]}},
        _answer(from_id="d"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert any("not a prior step id" in e for e in errs)
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_forward_or_self_ref_rejected(mock_codegen, monkeypatch):
    # detect references a LATER step id (declaration-order violation; a true cycle is impossible
    # in a forward-only ref model). Pins the declaration-order class, distinct from dangling-ref.
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "later", "classes": ["person"]}},
        {"id": "later", "op": "count", "args": {"items": "f"}},
        _answer(from_id="d"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert any("not a prior step id" in e for e in errs)
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_duplicate_ids_rejected(mock_codegen, monkeypatch):
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "f", "op": "count", "args": {"items": "f"}},
        _answer(from_id="f"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert any("duplicate id" in e for e in errs)
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_unknown_op_rejected_before_execution(mock_codegen, monkeypatch):
    prog = [
        {"id": "x", "op": "rm_rf", "args": {}},
        _answer(from_id="x"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert errs and errs[0].startswith("structural:")
    _free_text_reject(mock_codegen, monkeypatch, prog)
    # If the unknown op EVER reached the interpreter (it must not on the validated path) it would
    # raise — proven independently here.
    with pytest.raises(ValueError, match="unknown op"):
        Interpreter(_empty_cache()).run([{"id": "x", "op": "rm_rf", "args": {}}])


def test_wrong_arg_types_rejected(mock_codegen, monkeypatch):
    prog = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 1.5, "end_ms": 1000, "fps": "8"}},
        _answer(from_id="f"),
    ]
    errs = validation_errors({"program": prog}, clip=CLIP)
    assert errs and errs[0].startswith("structural:")
    _free_text_reject(mock_codegen, monkeypatch, prog)


def test_cross_clip_cache_reference_rejected():
    # Invariant test: there is NO DSL arg that names a cache/file/clip path — the cache is selected
    # by clip_id at the server (server.build_free_text_run_doc loads clip["cache"]), never by the
    # program. So a crafted program CANNOT redirect the cache. Assert the boundary is absent by
    # design: no op's required/optional args include a path-like binding.
    schema = json.loads((API_DIR / "schema" / "dsl.schema.json").read_text())
    forbidden = {"cache", "path", "file", "clip", "clip_id", "url", "src", "source"}
    for op_name, op_def in schema["$defs"].items():
        if op_name == "binding":
            continue
        arg_props = op_def["properties"]["args"]["properties"]
        assert not (set(arg_props) & forbidden), f"{op_name} exposes a path-like arg: {arg_props}"


# --------------------------------------------------------------------------------------
# runtime guards (validator BYPASSED) — both checks fire
# --------------------------------------------------------------------------------------

def _empty_cache():
    return Cache({"clip": {"id": "t", "width": 10, "height": 10, "duration_ms": 5000},
                  "detect": {}, "read_text": {}, "events": []})


def test_runtime_cap_fires_when_validation_bypassed():
    interp = Interpreter(_empty_cache())

    # (1) frame bomb JUST OVER the cap (NOT end=2e9 — that would materialize 60M ints if the cap
    # regressed). Pick a window that yields exactly MAX_SAMPLED_FRAMES + 1 frames at fps 30.
    stride = max(1, round(1000 / 30))
    end = MAX_SAMPLED_FRAMES * stride  # range(0, end+1, stride) length == MAX_SAMPLED_FRAMES + 1
    assert len(range(0, end + 1, stride)) == MAX_SAMPLED_FRAMES + 1
    over = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": end, "fps": 30}}]
    with pytest.raises(ProgramLimitExceeded, match="materialize"):
        interp.run(over)

    # (2) step bomb: MAX_STEPS + 1 steps -> the check at the TOP of run() fires before any op.
    bomb = [{"id": f"s{i}", "op": "count", "args": {"items": "s0"}} for i in range(MAX_STEPS + 1)]
    with pytest.raises(ProgramLimitExceeded, match="steps"):
        interp.run(bomb)


def test_pre_materialization_cap_does_not_build_the_bomb():
    # Prove the cap is PRE-materialization: a genuinely huge window (~60M frames) raises
    # IMMEDIATELY rather than OOMing/hanging. If the guard were a post-hoc len() on a built list
    # this call would materialize 60M ints (multi-GB) before any check — so completing in well
    # under a second via a raised ProgramLimitExceeded is the observable proof the cap sits before
    # list(range(...)).
    import time
    interp = Interpreter(_empty_cache())
    huge = [{"id": "f", "op": "sample_frames",
             "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}}]
    t0 = time.monotonic()
    with pytest.raises(ProgramLimitExceeded):
        interp.run(huge)
    assert time.monotonic() - t0 < 1.0, "cap must fire before materializing the frame list"


# --------------------------------------------------------------------------------------
# codegen + validator size cap (before parse)
# --------------------------------------------------------------------------------------

def test_oversized_codegen_output_rejected_before_parse(monkeypatch):
    # (a) validation_errors rejects an oversized doc BEFORE the structural pass: spy on
    # structural_errors to prove it is never invoked.
    called = {"struct": 0}
    real_struct = validate_program.structural_errors

    def struct_spy(*a, **k):
        called["struct"] += 1
        return real_struct(*a, **k)
    monkeypatch.setattr(validate_program, "structural_errors", struct_spy)

    huge_doc = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "_pad": "z" * (MAX_PROGRAM_BYTES + 10)}},
        _answer(from_id="f"),
    ]}
    errs = validation_errors(huge_doc)
    assert errs and errs[0].startswith("size:")
    assert called["struct"] == 0, "oversized doc must be rejected before the structural pass"

    # (b) codegen.generate's byte cap rejects an oversized completion before parse.
    class _Resp:
        class _Choice:
            class _Msg:
                content = '{"program": [' + '0' * (MAX_PROGRAM_BYTES + 100) + "]}"
            message = _Msg()
        choices = [_Choice()]

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    assert kwargs.get("max_tokens"), "codegen must pass max_tokens (cost/DoS guard)"
                    return _Resp()

    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    cg._client = _Client()
    cg._deployment = "gpt-4o"
    with pytest.raises(ValueError, match="exceeds"):
        cg.generate("anything")


# --------------------------------------------------------------------------------------
# reject path leaves no partial trace
# --------------------------------------------------------------------------------------

def test_reject_produces_no_partial_trace(mock_codegen, monkeypatch):
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "s0"}} for i in range(MAX_STEPS + 1)]
    steps.append(_answer())
    doc = _free_text_reject(mock_codegen, monkeypatch, steps)
    assert doc["trace"] == [], "a validation reject must carry NO partial trace"
    assert doc["program"] == [], "a validation reject must carry NO program"


# --------------------------------------------------------------------------------------
# SSE reject contract
# --------------------------------------------------------------------------------------

def test_sse_reject_contract(mock_codegen):
    from fastapi.testclient import TestClient
    from test_sse_stream import events  # reuse the SSE parser

    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "s0"}} for i in range(MAX_STEPS + 1)]
    steps.append(_answer())
    mock_codegen(program=steps)
    client = TestClient(server.app)
    resp = client.get("/api/run", params={"query_text": "step bomb please", "pace_ms": 0})
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    # rejected program -> empty trace -> meta, findings, done (no step events).
    assert names == ["meta", "findings", "done"]
    assert evs[0][1]["program_source"] == "ungrounded"
    findings = evs[1][1]
    assert findings["grounded"] is False
    assert findings["reason"] == "invalid-program"


# --------------------------------------------------------------------------------------
# canned live path: step bomb -> falls back to pinned (D-DR6 holds under the new caps)
# --------------------------------------------------------------------------------------

def test_canned_live_step_bomb_falls_back_to_pinned(mock_codegen, monkeypatch):
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "s0"}} for i in range(MAX_STEPS + 1)]
    steps.append(_answer())
    mock_codegen(program=steps)
    called = _spy_run(monkeypatch)
    doc = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    # live program rejected at validation -> route serves the PINNED replay.
    assert doc["program_source"] == "pinned"
    # the rejected (oversized) program was never executed; only the pinned 7-step program ran.
    assert called["n"] == 1
    assert len(doc["program"]) <= MAX_STEPS


# --------------------------------------------------------------------------------------
# validator unit tests (folded in — DESIGN-ROOM: one file)
# --------------------------------------------------------------------------------------

def _all_pinned_programs():
    """(program_doc, clip) for every committed program — hero + the four bernabeu shapes."""
    out = []
    for q in canned.QUERIES:
        doc = json.loads(Path(q["program"]).read_text())
        clip = canned.clip_by_id(q["clip"])
        out.append((q["id"], doc, clip))
    return out


@pytest.mark.parametrize("qid,doc,clip", _all_pinned_programs())
def test_pinned_programs_pass_tightened_validator(qid, doc, clip):
    errs = validation_errors(doc, clip=clip)
    assert errs == [], f"pinned program '{qid}' must pass the tightened validator: {errs}"


def test_limit_constants_match_schema():
    # Mandatory drift guard: the schema re-encodes the statically-expressible limits; assert they
    # equal the limits.py constants so the two sources can never silently diverge.
    schema = json.loads((API_DIR / "schema" / "dsl.schema.json").read_text())
    assert schema["properties"]["program"]["maxItems"] == MAX_STEPS
    detect_classes = schema["$defs"]["detect"]["properties"]["args"]["properties"]["classes"]
    assert detect_classes["maxItems"] == MAX_DETECT_CLASSES
    assert detect_classes["minItems"] == 1
    assert detect_classes["items"]["maxLength"] == MAX_STR_ARG_LEN
    where = schema["$defs"]["filter"]["properties"]["args"]["properties"]["where"]["properties"]
    assert where["field"]["maxLength"] == MAX_STR_ARG_LEN
    assert where["equals"]["maxLength"] == MAX_STR_ARG_LEN
    # id maxLength present on every op def (representative check across all 8).
    for op_name, op_def in schema["$defs"].items():
        if op_name == "binding":
            continue
        assert op_def["properties"]["id"]["maxLength"] == MAX_STR_ARG_LEN
    fps = schema["$defs"]["sample_frames"]["properties"]["args"]["properties"]["fps"]
    assert (fps["minimum"], fps["maximum"]) == (validate_program.FPS_MIN, validate_program.FPS_MAX)


def test_schema_maxitems_rejects_independently():
    # A MAX_STEPS + 1 program fails structural_errors ALONE (the schema gate works even if
    # semantic_errors regressed).
    schema = json.loads((API_DIR / "schema" / "dsl.schema.json").read_text())
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "s0"}} for i in range(MAX_STEPS + 1)]
    steps.append(_answer())
    errs = structural_errors({"program": steps}, schema)
    assert errs, "schema maxItems must reject an over-length program on its own"


def test_legit_full_clip_sweep_passes():
    # A sample_frames over the WHOLE bernabeu clip (0..30086ms) passes the cumulative cap at both
    # fps8 (~241 frames) and fps30 (~912 frames) — the envelope sits above the largest legit sweep.
    bclip = canned.clip_by_id("bernabeu-counter")
    dur = bclip["duration_ms"]
    for fps in (8, 30):
        prog = {"program": [
            {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": dur, "fps": fps}},
            {"id": "n", "op": "count", "args": {"items": "f"}},
            _answer(from_id="n"),
        ]}
        errs = validation_errors(prog, clip=bclip)
        assert errs == [], f"legit full-clip sweep @ fps{fps} must pass: {errs}"
    # sanity: the fps30 sweep is the documented ~912, comfortably under MAX_SAMPLED_FRAMES.
    cnt = validate_program._sampled_frame_count(
        {"start_ms": 0, "end_ms": dur, "fps": 30}, bclip)
    assert 800 < cnt < MAX_SAMPLED_FRAMES
