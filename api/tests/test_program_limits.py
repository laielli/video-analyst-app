"""
Trust-boundary hardening — adversarial + unit suite for the resource/semantic guards on the
generated-program executor (plan: trust-boundary-hardening-20260609).

Two halves in one file (DESIGN-ROOM default: a single test_program_limits.py):

  (A) ADVERSARIAL — feed malformed / oversized / hostile programs through the free-text path and
      prove they are REJECTED BEFORE ANY OP EXECUTES. The zero-execution assertion uses the proven
      pattern from test_free_text_run.py:75-80: monkeypatch.setattr(Interpreter, "run", spy) on the
      CLASS (NOT a primitives module attr — OPS holds function references, so patching
      primitives.op_sample_frames never rebinds the dict entry). Where a primitive-level spy is
      genuinely needed, the dict entry itself is patched (monkeypatch.setitem(OPS, ...)).

  (B) VALIDATOR UNIT — focused tests for the new caps / arg-domain helpers, the schema<->limits
      sync guard, and the pinned-programs-still-pass regression.

All runs are mocked codegen / replay cache (no creds, no billing).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
import canned
from interpreter import Cache, Interpreter, ProgramLimitExceeded
from interpreter.primitives import OPS
from validate_program import (
    structural_errors,
    validation_errors,
)
from limits import (
    MAX_DETECT_CLASSES,
    MAX_PROGRAM_BYTES,
    MAX_SAMPLED_FRAMES,
    MAX_STEPS,
    MAX_STR_ARG_LEN,
)

API_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = API_DIR / "schema" / "dsl.schema.json"
DSL_SCHEMA = json.loads(SCHEMA_PATH.read_text())
CLIP = canned.clip_by_id("single-goal")


# ---------------------------------------------------------------------------------------
# shared spy: assert Interpreter.run is NEVER called (the trust boundary, class-method spy)
# ---------------------------------------------------------------------------------------

def _install_run_spy(monkeypatch):
    """Install a class-method spy on Interpreter.run; returns a counter dict. Mirrors the proven
    pattern in test_free_text_run.py — patch the CLASS so OPS resolution is irrelevant."""
    called = {"n": 0}
    real_run = Interpreter.run

    def spy(self, program, query=None):
        called["n"] += 1
        return real_run(self, program, query=query)

    monkeypatch.setattr(Interpreter, "run", spy)
    return called


def _answer(_from="x", q="q"):
    return {"id": "result", "op": "answer", "args": {"from": _from, "question": q}}


# =======================================================================================
# (A) ADVERSARIAL SUITE — every reject asserted BEFORE ANY OP EXECUTES
# =======================================================================================

def test_oversized_program_step_bomb_rejected(mock_codegen, monkeypatch):
    # MAX_STEPS + 1 trivial count steps over one frames binding, then answer.
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}]
    for k in range(MAX_STEPS):  # pushes total over MAX_STEPS
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(_from="c0"))
    assert len(steps) > MAX_STEPS
    assert validation_errors({"program": steps}, clip=CLIP)  # rejected by the step cap

    called = _install_run_spy(monkeypatch)
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("step bomb", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["n"] == 0


def test_frame_count_bomb_rejected_without_clip():
    # Direct unit call with clip=None (the server path always resolves a clip; clip=None is the
    # CLI/direct case). The count cap is duration-independent, so the no-clip OOM gap is closed.
    program = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
        _answer(_from="f"),
    ]}
    errors = validation_errors(program, clip=None)
    assert errors  # rejected even with no clip
    assert any("frames" in e for e in errors)


def test_frame_count_bomb_rejected_with_clip(mock_codegen, monkeypatch):
    # Several in-bounds windows whose CUMULATIVE frame count exceeds MAX_SAMPLED_FRAMES while the
    # STEP count stays under MAX_STEPS — so ONLY the cumulative-frame cap (not the step cap) can
    # catch it. Each 0-5000ms @ fps30 window is the full clip-window (~152 frames); enough windows
    # sum over the cap within the step budget. Each window is individually in-bounds (end < dur).
    end_ms = min(5000, CLIP["duration_ms"])
    per_window = len(range(0, end_ms + 1, max(1, round(1000 / 30))))
    n_windows = (MAX_SAMPLED_FRAMES // per_window) + 1
    steps = []
    for k in range(n_windows):
        steps.append({"id": f"f{k}", "op": "sample_frames",
                      "args": {"start_ms": 0, "end_ms": end_ms, "fps": 30}})
    steps.append({"id": "c", "op": "count", "args": {"items": "f0"}})
    steps.append(_answer(_from="c"))
    # keep the step count under MAX_STEPS so the FRAME cap (not the step cap) is what fires.
    assert len(steps) <= MAX_STEPS, "test must isolate the cumulative-frame cap from the step cap"
    assert n_windows * per_window > MAX_SAMPLED_FRAMES
    errors = validation_errors({"program": steps}, clip=CLIP)
    assert any("frames" in e and "max" in e for e in errors)

    called = _install_run_spy(monkeypatch)
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("frame bomb", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["n"] == 0


def test_detect_classes_empty_rejected():
    # classes: [] — empty (schema minItems also catches this; the local validator is the gate).
    program = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": []}},
        {"id": "c", "op": "count", "args": {"items": "p"}},
        _answer(_from="c"),
    ]}
    assert validation_errors(program, clip=CLIP)


def test_detect_classes_overlong_rejected():
    program = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "p", "op": "detect",
         "args": {"frames": "f", "classes": [f"cls{k}" for k in range(MAX_DETECT_CLASSES + 1)]}},
        {"id": "c", "op": "count", "args": {"items": "p"}},
        _answer(_from="c"),
    ]}
    assert validation_errors(program, clip=CLIP)


def test_huge_string_arg_rejected():
    # giant filter.where.equals AND a giant classes[0] — both must be rejected.
    giant = "x" * (MAX_STR_ARG_LEN + 1)
    huge_equals = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "j", "op": "crop", "args": {"detections": "p", "region": "jersey"}},
        {"id": "n", "op": "read_text", "args": {"crops": "j"}},
        {"id": "flt", "op": "filter", "args": {"items": "n", "where": {"field": "text", "equals": giant}}},
        _answer(_from="flt"),
    ]}
    assert validation_errors(huge_equals, clip=CLIP)

    huge_class = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": [giant]}},
        {"id": "c", "op": "count", "args": {"items": "p"}},
        _answer(_from="c"),
    ]}
    assert validation_errors(huge_class, clip=CLIP)


def test_deeply_nested_args_rejected():
    # Pathological extra/nested arg objects -> rejected by additionalProperties:false (structural)
    # BEFORE semantics, with zero execution.
    nested = {"a": {"b": {"c": {"d": [1, 2, 3]}}}}
    program = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "evil": nested}},
        _answer(_from="f"),
    ]}
    errors = validation_errors(program, clip=CLIP)
    assert errors
    assert any(e.startswith("structural:") for e in errors)


def test_dangling_ref_rejected():
    # detect.frames references an id never produced.
    program = {"program": [
        {"id": "p", "op": "detect", "args": {"frames": "ghost", "classes": ["person"]}},
        {"id": "c", "op": "count", "args": {"items": "p"}},
        _answer(_from="c"),
    ]}
    errors = validation_errors(program, clip=CLIP)
    assert any("not a prior step id" in e for e in errors)


def test_forward_or_self_ref_rejected():
    # A step references a LATER id (forward ref) — rejected by the declaration-order check.
    program = {"program": [
        {"id": "c", "op": "count", "args": {"items": "later"}},   # 'later' declared after
        {"id": "later", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        _answer(_from="c"),
    ]}
    errors = validation_errors(program, clip=CLIP)
    assert any("not a prior step id" in e for e in errors)


def test_duplicate_ids_rejected():
    program = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "f", "op": "count", "args": {"items": "f"}},  # duplicate id 'f'
        _answer(_from="f"),
    ]}
    errors = validation_errors(program, clip=CLIP)
    assert any("duplicate id" in e for e in errors)


def test_unknown_op_rejected_before_execution(monkeypatch):
    # op "rm_rf" -> structural anyOf failure; the validator rejects it. And, if Interpreter.run
    # ever saw it (it must not, on the validated path), it would raise "unknown op".
    program = {"program": [
        {"id": "x", "op": "rm_rf", "args": {}},
        _answer(_from="x"),
    ]}
    assert validation_errors(program, clip=CLIP)
    # interpreter-level: an unknown op raises ValueError if ever reached directly.
    with pytest.raises(ValueError):
        Interpreter(Cache.load(CLIP["cache"])).run(program["program"])


def test_wrong_arg_types_rejected():
    bad_fps = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": "8"}},
        _answer(_from="f"),
    ]}
    assert validation_errors(bad_fps, clip=CLIP)
    bad_start = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 1.5, "end_ms": 1000, "fps": 8}},
        _answer(_from="f"),
    ]}
    assert validation_errors(bad_start, clip=CLIP)


def test_cross_clip_cache_reference_rejected():
    # Invariant test: NO DSL op accepts a path/clip/cache arg — the cache is selected by clip_id at
    # the server (build_free_text_run_doc loads clip["cache"]), never by the program. A program
    # cannot redirect the cache. Document the boundary by proving no op's arg schema names a
    # file/clip/cache field.
    forbidden = {"path", "cache", "clip", "clip_id", "file", "filename", "src", "url"}
    for op_def in DSL_SCHEMA["$defs"].values():
        props = op_def.get("properties", {})
        args = props.get("args", {})
        arg_props = set(args.get("properties", {}).keys()) if isinstance(args, dict) else set()
        assert not (arg_props & forbidden), f"DSL op exposes a cache/path arg: {arg_props & forbidden}"
    # And a crafted program carrying such a key is rejected by additionalProperties:false.
    crafted = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "cache": "../other_clip_cache.json"}},
        _answer(_from="f"),
    ]}
    assert validation_errors(crafted, clip=CLIP)


def test_runtime_cap_fires_when_validation_bypassed():
    # Call Interpreter.run DIRECTLY (bypassing the validator) to prove BOTH runtime guards.
    cache = Cache.load(CLIP["cache"])
    interp = Interpreter(cache)

    # (1) frame-bomb JUST OVER the cap (NOT end=2e9 — that would build 60M ints if the cap
    # regresses). A window yielding MAX_SAMPLED_FRAMES + 1 frames at fps30.
    fps = 30
    stride = max(1, round(1000 / fps))
    end = MAX_SAMPLED_FRAMES * stride  # len(range(0, end+1, stride)) == MAX_SAMPLED_FRAMES + 1
    assert len(range(0, end + 1, stride)) == MAX_SAMPLED_FRAMES + 1
    over_cap = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": end, "fps": fps}},
        _answer(_from="f"),
    ]
    with pytest.raises(ProgramLimitExceeded):
        interp.run(over_cap)

    # (2) step-bomb (MAX_STEPS + 1) -> ProgramLimitExceeded from the check at the top of run.
    step_bomb = [{"id": f"s{k}", "op": "count", "args": {"items": "s0"}}
                 for k in range(MAX_STEPS + 2)]
    with pytest.raises(ProgramLimitExceeded):
        interp.run(step_bomb)


def test_oversized_codegen_output_rejected_before_parse(monkeypatch):
    # A doc whose serialized size exceeds MAX_PROGRAM_BYTES -> rejected by the size gate BEFORE the
    # structural/jsonschema pass. Prove jsonschema is never invoked by spying on structural_errors.
    import validate_program

    structural_called = {"n": 0}
    real_structural = validate_program.structural_errors

    def spy_structural(doc, schema):
        structural_called["n"] += 1
        return real_structural(doc, schema)

    monkeypatch.setattr(validate_program, "structural_errors", spy_structural)

    blob = "z" * (MAX_PROGRAM_BYTES + 10)
    oversized = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "_pad": blob}},
        _answer(_from="f"),
    ]}
    errors = validation_errors(oversized, clip=CLIP)
    assert errors
    assert any(e.startswith("size:") for e in errors)
    assert structural_called["n"] == 0, "structural/jsonschema must NOT run on an oversized blob"


def test_codegen_generate_byte_cap_rejects_oversized():
    # codegen.AzureCodegen.generate's byte cap rejects an oversized completion (a fake client whose
    # message.content exceeds MAX_PROGRAM_BYTES) rather than returning the blob.
    import codegen

    blob = "z" * (MAX_PROGRAM_BYTES + 50)
    content = json.dumps({"program": [{"id": "f", "op": "sample_frames",
                                       "args": {"start_ms": 0, "end_ms": 1, "fps": 1, "_pad": blob}}]})

    class _Msg:
        def __init__(self, c): self.message = type("M", (), {"content": c})()

    class _Resp:
        def __init__(self, c): self.choices = [_Msg(c)]

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _Resp(content)

    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    cg._client = _FakeClient()
    cg._deployment = "gpt-4o"
    with pytest.raises(ValueError):
        cg.generate("anything")


def test_reject_produces_no_partial_trace(mock_codegen):
    # An oversized/step-bomb free-text program -> the _ungrounded doc has trace == [] and
    # program == [] (the reject path passes neither program nor trace).
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}]
    for k in range(MAX_STEPS):
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(_from="c0"))
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("step bomb", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert doc["trace"] == []
    assert doc["program"] == []


def test_sse_reject_contract(mock_codegen):
    # Drive GET /api/run (SSE) with a mocked step-bomb program -> the stream delivers
    # meta -> findings -> done with reason "invalid-program" (no step events; empty trace).
    from fastapi.testclient import TestClient

    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}}]
    for k in range(MAX_STEPS):
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(_from="c0"))
    mock_codegen(program=steps)

    client = TestClient(server.app)
    resp = client.get("/api/run", params={"query_text": "step bomb please", "pace_ms": 0})
    assert resp.status_code == 200
    names, findings = [], None
    for block in resp.text.strip().split("\n\n"):
        nm, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                nm = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if nm:
            names.append(nm)
        if nm == "findings":
            findings = data
    assert names == ["meta", "findings", "done"]  # no step events on a reject
    assert findings["grounded"] is False
    assert findings["reason"] == "invalid-program"


def test_canned_live_step_bomb_falls_back_to_pinned(mock_codegen, monkeypatch):
    # On the CANNED route, a mocked live-codegen step-bomb -> _live_run_doc returns None and the
    # route serves the pinned replay (program_source == "pinned"). Pins the limits-specific D-DR6
    # fallback. Spy proves the over-cap program never reaches a full Interpreter.run replay.
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}}]
    for k in range(MAX_STEPS):
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(_from="c0"))

    real_run = Interpreter.run
    seen = []

    def spy(self, program, *a, **kw):
        seen.append(len(program))
        return real_run(self, program, *a, **kw)

    monkeypatch.setattr(Interpreter, "run", spy)
    mock_codegen(program=steps)
    doc = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    assert doc["program_source"] == "pinned"
    # the over-cap program (len > MAX_STEPS) never reached a replay run.
    assert all(n <= MAX_STEPS for n in seen)


# =======================================================================================
# (B) VALIDATOR UNIT TESTS
# =======================================================================================

def _committed_programs():
    """(program_steps, clip) for every committed pinned program."""
    from validate_program import strip_comments
    out = []
    for q in canned.QUERIES:
        steps = strip_comments(json.loads(Path(q["program"]).read_text()))["program"]
        out.append((q["id"], steps, canned.clip_by_id(q["clip"])))
    return out


def test_pinned_programs_pass_tightened_validator():
    # Regression guard: defaults must not be set below any real program.
    for qid, steps, clip in _committed_programs():
        errors = validation_errors({"program": steps}, clip=clip)
        assert errors == [], f"pinned program '{qid}' must pass the tightened validator: {errors}"


def test_limit_constants_match_schema():
    # Mandatory sync guard — the schema duplicates limits in maxItems/maxLength/bounds, so assert
    # they equal the limits.py constants. They cannot silently drift.
    assert DSL_SCHEMA["properties"]["program"]["maxItems"] == MAX_STEPS
    classes = DSL_SCHEMA["$defs"]["detect"]["properties"]["args"]["properties"]["classes"]
    assert classes["maxItems"] == MAX_DETECT_CLASSES
    assert classes["minItems"] == 1
    assert classes["items"]["maxLength"] == MAX_STR_ARG_LEN
    # id maxLength on every op def, and filter.where field/equals maxLength.
    for name, op_def in DSL_SCHEMA["$defs"].items():
        if name == "binding":
            continue
        assert op_def["properties"]["id"]["maxLength"] == MAX_STR_ARG_LEN, name
    where = DSL_SCHEMA["$defs"]["filter"]["properties"]["args"]["properties"]["where"]["properties"]
    assert where["field"]["maxLength"] == MAX_STR_ARG_LEN
    assert where["equals"]["maxLength"] == MAX_STR_ARG_LEN
    # fps bounds.
    fps = DSL_SCHEMA["$defs"]["sample_frames"]["properties"]["args"]["properties"]["fps"]
    assert fps["minimum"] == 1 and fps["maximum"] == 30


def test_schema_maxitems_rejects_independently():
    # A MAX_STEPS + 1 program fails structural_errors ALONE (schema gate works even if
    # semantic_errors regresses).
    steps = [{"id": f"c{k}", "op": "count", "args": {"items": "c0"}} for k in range(MAX_STEPS + 1)]
    errors = structural_errors({"program": steps}, DSL_SCHEMA)
    assert errors, "schema maxItems must reject a too-long program independently of semantics"


def test_legit_full_clip_sweep_passes():
    # A sample_frames over the whole bernabeu clip passes MAX_SAMPLED_FRAMES at fps8 AND fps30
    # (the envelope sits above the largest legitimate sweep). duration_ms=30086 (canned.py).
    bclip = canned.clip_by_id("bernabeu-counter")
    dur = bclip["duration_ms"]
    for fps in (8, 30):
        sweep = {"program": [
            {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": dur, "fps": fps}},
            {"id": "p", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
            {"id": "c", "op": "count", "args": {"items": "p"}},
            _answer(_from="c"),
        ]}
        errors = validation_errors(sweep, clip=bclip)
        assert errors == [], f"full-clip sweep @ fps{fps} must pass: {errors}"


def test_sampled_frame_count_matches_interpreter_stride():
    # The validator's frame count must be byte-identical to what op_sample_frames materializes.
    from validate_program import _sampled_frame_count
    for start, end, fps in [(0, 1000, 8), (3500, 5000, 8), (0, 30086, 30), (10000, 10001, 1)]:
        stride = max(1, round(1000 / fps))
        expected = len(range(start, end + 1, stride))
        got = _sampled_frame_count({"start_ms": start, "end_ms": end, "fps": fps}, None)
        assert got == expected, (start, end, fps, got, expected)
