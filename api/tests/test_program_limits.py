"""
Trust-boundary hardening — adversarial suite + validator unit tests.

The headline contract (CLAUDE.md layer 2): generated code runs at the executor, so it is
isolated and validated. Every malformed / oversized / hostile program is REJECTED at the
pre-execution gate — `validation_errors` runs BEFORE `Interpreter.run` — so no op ever executes
on a rejected program. The zero-execution assertions use the PROVEN spy pattern from
`test_free_text_run.py:75-80`: `monkeypatch.setattr(Interpreter, "run", spy)` on the CLASS (NOT a
module-level primitive — `OPS` holds function references and would never see the patch).

Two runtime guards are also proven INDEPENDENTLY of the validator (a future entry point could
call `Interpreter.run` without validating): the pre-materialization frame cap inside
`op_sample_frames` and the `MAX_STEPS` check at the top of `Interpreter.run`. Both raise
`ProgramLimitExceeded`.

DESIGN-ROOM: this is the single suite (no separate `test_validate_program.py`) — the adversarial
cases and the validator unit tests share the same fixtures and limit imports, so one file keeps
the trust boundary's coverage in one place.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
import canned
import codegen
import limits
import validate_program
from validate_program import validation_errors, semantic_errors, strip_comments
from interpreter import Cache, Interpreter, ProgramLimitExceeded

API_DIR = Path(__file__).resolve().parent.parent
DSL_SCHEMA = json.loads((API_DIR / "schema" / "dsl.schema.json").read_text())
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
CLIP_ID = "single-goal"


# ---- helpers ----------------------------------------------------------------------------

def _answer(from_id: str = "f", q: str = "q") -> dict:
    return {"id": "result", "op": "answer", "args": {"from": from_id, "question": q}}


def _doc(steps: list[dict]) -> dict:
    return {"program": steps}


def _run_spy(monkeypatch):
    """Install the proven class-method spy on Interpreter.run and return a counter dict whose
    ['n'] is the number of times run was invoked. Calls through to the real run so a regression
    (a rejected program reaching run) is observable, not masked."""
    called = {"n": 0}
    real_run = Interpreter.run

    def spy(self, program, query=None):
        called["n"] += 1
        return real_run(self, program, query=query)

    monkeypatch.setattr(Interpreter, "run", spy)
    return called


def _structural_spy(monkeypatch):
    """Spy on validate_program.structural_errors so we can prove the oversize cap short-circuits
    BEFORE the (superlinear) jsonschema structural pass is ever entered."""
    called = {"n": 0}
    real = validate_program.structural_errors

    def spy(doc, schema):
        called["n"] += 1
        return real(doc, schema)

    monkeypatch.setattr(validate_program, "structural_errors", spy)
    return called


# =========================================================================================
# Adversarial suite — every reject asserts ZERO op execution via the Interpreter.run spy.
# =========================================================================================

def test_oversized_program_step_bomb_rejected(mock_codegen, monkeypatch):
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(limits.MAX_STEPS + 1)]
    steps.append(_answer())
    # direct validator: rejected.
    assert validation_errors(_doc(steps))
    # via the free-text path: ungrounded, invalid-program, zero execution.
    spy = _run_spy(monkeypatch)
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert spy["n"] == 0


def test_frame_count_bomb_rejected_without_clip():
    # Direct unit call with clip=None (the server path always resolves a clip; clip=None only
    # happens on direct/CLI calls). The cumulative count cap is duration-independent, so the
    # no-clip gap is closed. The count is O(1) arithmetic — no list is built by the validator.
    bomb = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
        _answer(),
    ])
    errors = validation_errors(bomb, clip=None)
    assert errors
    assert any("frames" in e for e in errors)


def test_frame_count_bomb_rejected_with_clip(mock_codegen, monkeypatch):
    # Many small windows summing > MAX_SAMPLED_FRAMES (cumulative cap). Each window is small, so
    # no single _sample_frames_bounds check fires; the SUM is what rejects.
    clip = canned.clip_by_id(CLIP_ID)
    # at fps1, stride=1000, each window 0..999000 -> 1000 frames; three of them = 3000 > 2000.
    win = {"start_ms": 0, "end_ms": 999_000, "fps": 1}
    steps = [{"id": f"w{i}", "op": "sample_frames", "args": dict(win)} for i in range(3)]
    steps.append(_answer(from_id="w0"))
    # (end_ms exceeds clip duration so _sample_frames_bounds also fires, but the cumulative cap is
    # the invariant under test; assert the cumulative message is present.)
    errors = semantic_errors([s for s in steps], clip=None)  # no clip -> only the count cap bites
    assert any("frames; max is" in e for e in errors)
    spy = _run_spy(monkeypatch)
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert spy["n"] == 0


def test_detect_classes_empty_rejected():
    bad = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": []}},
        _answer(from_id="d"),
    ])
    # rejected at the full gate (the schema's minItems:1 fires structurally; the generic anyOf
    # message is fine — the program is rejected).
    assert validation_errors(bad)
    # and the SEMANTIC layer catches it independently (the real gate under strict:false): isolate
    # semantic_errors so a schema regression doesn't silently drop the empty-classes check.
    sem = semantic_errors(bad["program"])
    assert any("classes is empty" in e for e in sem)


def test_detect_classes_overlong_rejected():
    overlong = ["c"] * (limits.MAX_DETECT_CLASSES + 1)
    bad = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": overlong}},
        _answer(from_id="d"),
    ])
    assert validation_errors(bad)
    # semantic layer also catches it (re-check under strict:false): isolate semantic_errors.
    sem = semantic_errors(bad["program"])
    assert any("classes has" in e for e in sem)


def test_huge_string_arg_rejected():
    huge = "x" * (limits.MAX_STR_ARG_LEN + 1)
    # filter.where.equals overlong.
    bad_filter = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "flt", "op": "filter", "args": {"items": "f", "where": {"field": "text", "equals": huge}}},
        _answer(from_id="flt"),
    ])
    assert validation_errors(bad_filter)
    # a giant classes[0] element.
    bad_class = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": [huge]}},
        _answer(from_id="d"),
    ])
    assert validation_errors(bad_class)


def test_deeply_nested_args_rejected():
    # extra/nested arg objects -> additionalProperties:false structural failure, zero execution.
    bad = _doc([
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "evil": {"a": {"b": {"c": 1}}}}},
        _answer(),
    ])
    errors = validation_errors(bad)
    assert errors
    assert all(e.startswith("structural:") or e.startswith("oversize:") for e in errors)


def test_dangling_ref_rejected():
    bad = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "nope", "classes": ["person"]}},
        _answer(from_id="d"),
    ])
    sem = semantic_errors(bad["program"])
    assert any("not a prior step id" in e for e in sem)


def test_forward_or_self_ref_rejected():
    # 'd' references a LATER step 'later' (forward ref) -> declaration-order failure.
    bad = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "later", "classes": ["person"]}},
        {"id": "later", "op": "count", "args": {"items": "d"}},
        _answer(from_id="later"),
    ])
    sem = semantic_errors(bad["program"])
    assert any("not a prior step id" in e for e in sem)


def test_duplicate_ids_rejected():
    bad = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "f", "op": "count", "args": {"items": "f"}},
        _answer(from_id="f"),
    ])
    sem = semantic_errors(bad["program"])
    assert any("duplicate id" in e for e in sem)


def test_unknown_op_rejected_before_execution():
    bad = _doc([{"id": "x", "op": "rm_rf", "args": {}}, _answer(from_id="x")])
    assert validation_errors(bad)  # structural anyOf failure
    # interpreter-level: if ever reached (it must NOT be on the validated path), it raises.
    cache = Cache.load(canned.clip_by_id(CLIP_ID)["cache"])
    with pytest.raises(ValueError):
        Interpreter(cache).run([{"id": "x", "op": "rm_rf", "args": {}}])


def test_wrong_arg_types_rejected():
    bad_str = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": "8"}},
        _answer(),
    ])
    assert validation_errors(bad_str)
    bad_float = _doc([
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 1.5, "end_ms": 1000, "fps": 8}},
        _answer(),
    ])
    assert validation_errors(bad_float)


def test_cross_clip_cache_reference_rejected():
    # Invariant test: NO DSL op accepts a cache/file/clip path arg. The cache is selected by
    # clip_id (server.py:220 loads clip["cache"]), not the program — so a crafted program cannot
    # redirect the cache. Prove the attack surface is absent by construction: no op's args schema
    # names a path/cache/clip field.
    forbidden = {"cache", "path", "file", "clip", "clip_id", "url"}
    for op_def in DSL_SCHEMA["$defs"].values():
        args = op_def.get("properties", {}).get("args", {})
        props = set(args.get("properties", {}).keys())
        assert not (props & forbidden), f"a DSL op exposes a cache/path arg: {props & forbidden}"


def test_runtime_cap_fires_when_validation_bypassed():
    # Call Interpreter.run DIRECTLY (bypassing the validator) to prove BOTH runtime guards.
    cache = Cache.load(canned.clip_by_id(CLIP_ID)["cache"])
    interp = Interpreter(cache)

    # (1) frame bomb just OVER the cap (NOT end=2e9 — that would materialize 60M ints in CI if the
    # cap regressed). At fps1, stride=1000, len(range(0, end+1, 1000)) = end//1000 + 1; choose end
    # so the count is MAX_SAMPLED_FRAMES + 1.
    end = limits.MAX_SAMPLED_FRAMES * 1000  # -> MAX+1 frames
    over = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": end, "fps": 1}},
        _answer(),
    ]
    with pytest.raises(ProgramLimitExceeded) as ei:
        interp.run(over)
    assert "sample_frames" in str(ei.value)

    # (2) step bomb -> the MAX_STEPS check at the top of run, before any op executes.
    step_bomb = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(limits.MAX_STEPS + 1)]
    with pytest.raises(ProgramLimitExceeded) as ei2:
        interp.run(step_bomb)
    assert "steps" in str(ei2.value)


def test_oversized_codegen_output_rejected_before_parse(monkeypatch):
    # A doc whose serialized size exceeds MAX_PROGRAM_BYTES is rejected BEFORE the structural pass.
    spy = _structural_spy(monkeypatch)
    blob = "y" * (limits.MAX_PROGRAM_BYTES + 1)
    big = _doc([{"id": "f", "op": "filter", "args": {"items": "x", "where": {"field": "t", "equals": blob}}}])
    errors = validation_errors(big)
    assert errors and errors[0].startswith("oversize:")
    assert spy["n"] == 0, "oversize cap must short-circuit before structural_errors"

    # codegen.generate's byte-cap: an oversized completion raises (caught by callers as a miss).
    class _Resp:
        class _Choice:
            class _Msg:
                content = json.dumps({"program": [{"id": "f", "op": "filter",
                                                   "args": {"items": "x", "where": {"field": "t", "equals": blob}}}]})
            message = _Msg()
        choices = [_Choice()]

    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    cg._deployment = "gpt-4o"

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _Resp()
    cg._client = _Client()
    with pytest.raises(ValueError):
        cg.generate("anything")


def test_reject_produces_no_partial_trace(mock_codegen):
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(limits.MAX_STEPS + 1)]
    steps.append(_answer())
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    # the reject path passes neither program nor trace -> both empty (no partial run leaked).
    assert doc["trace"] == []
    assert doc["program"] == []


def test_sse_reject_contract(mock_codegen):
    from fastapi.testclient import TestClient

    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(limits.MAX_STEPS + 1)]
    steps.append(_answer())
    mock_codegen(program=steps)
    client = TestClient(server.app)
    resp = client.get("/api/run", params={"query_text": "anything", "pace_ms": 0})
    assert resp.status_code == 200
    # parse the SSE body: meta -> findings -> done (empty trace -> no step events).
    names, datas = [], []
    for block in resp.text.strip().split("\n\n"):
        nm = da = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                nm = line[len("event: "):]
            elif line.startswith("data: "):
                da = json.loads(line[len("data: "):])
        names.append(nm)
        datas.append(da)
    assert names == ["meta", "findings", "done"]
    assert datas[0]["program_source"] == "ungrounded"
    assert datas[1]["reason"] == "invalid-program"


def test_canned_live_step_bomb_falls_back_to_pinned(mock_codegen, monkeypatch):
    # On the canned route, a mocked live-codegen step-bomb -> _live_run_doc returns None and the
    # route serves the pinned replay. (D-DR6 safe live->pinned downgrade for the limit case.)
    spy = _run_spy(monkeypatch)
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(limits.MAX_STEPS + 1)]
    steps.append(_answer())
    mock_codegen(program=steps)
    doc = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    assert doc["program_source"] == "pinned"
    # the step-bomb was rejected at validation; only the pinned program executed (1 run call).
    assert spy["n"] == 1
    # and the executed program was the pinned hero chain, not the bomb.
    assert doc["program"][0]["id"] == "frames"


# =========================================================================================
# Validator unit tests — pinned programs pass, schema/limits stay in sync.
# =========================================================================================

def _committed_programs():
    """(program_doc, clip) for every committed program: hero + the four bernabeu programs."""
    out = []
    hero = strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))
    out.append((hero, canned.clip_by_id("single-goal")))
    bclip = canned.clip_by_id("bernabeu-counter")
    for q in canned.QUERIES:
        if q["clip"] == "bernabeu-counter":
            prog = strip_comments(json.loads(Path(q["program"]).read_text()))
            out.append((prog, bclip))
    return out


def test_pinned_programs_pass_tightened_validator():
    progs = _committed_programs()
    assert len(progs) == 5, "expected hero + 4 bernabeu committed programs"
    for prog, clip in progs:
        assert validation_errors(prog, clip=clip) == [], f"a committed program failed: {prog['program'][0]['id']}"


def test_limit_constants_match_schema():
    # MANDATORY sync guard: the schema's static mirrors must equal the limits.py constants so they
    # cannot silently drift.
    props = DSL_SCHEMA["properties"]
    defs = DSL_SCHEMA["$defs"]
    assert props["program"]["maxItems"] == limits.MAX_STEPS
    classes = defs["detect"]["properties"]["args"]["properties"]["classes"]
    assert classes["maxItems"] == limits.MAX_DETECT_CLASSES
    assert classes["minItems"] == 1
    assert classes["items"]["maxLength"] == limits.MAX_STR_ARG_LEN
    where = defs["filter"]["properties"]["args"]["properties"]["where"]["properties"]
    assert where["field"]["maxLength"] == limits.MAX_STR_ARG_LEN
    assert where["equals"]["maxLength"] == limits.MAX_STR_ARG_LEN
    sf = defs["sample_frames"]["properties"]["args"]["properties"]
    assert sf["fps"]["minimum"] == validate_program.FPS_MIN
    assert sf["fps"]["maximum"] == validate_program.FPS_MAX
    assert sf["start_ms"]["minimum"] == 0
    # the per-op id maxLength mirrors MAX_STR_ARG_LEN.
    assert defs["sample_frames"]["properties"]["id"]["maxLength"] == limits.MAX_STR_ARG_LEN


def test_schema_maxitems_rejects_independently():
    # A MAX_STEPS+1 program fails structural_errors ALONE (the schema gate works even if
    # semantic_errors regresses).
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(limits.MAX_STEPS + 1)]
    steps.append(_answer())
    struct = validate_program.structural_errors(_doc(steps), DSL_SCHEMA)
    assert struct, "schema maxItems must reject an over-length program on its own"


def test_legit_full_clip_sweep_passes():
    # Whole bernabeu clip (duration_ms=30086 in canned.py) at fps8 (~241 frames) AND fps30
    # (~912 frames) both pass MAX_SAMPLED_FRAMES (the envelope is above the largest legit sweep).
    bclip = canned.clip_by_id("bernabeu-counter")
    for fps in (8, 30):
        sweep = _doc([
            {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 30086, "fps": fps}},
            {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
            {"id": "c", "op": "count", "args": {"items": "d"}},
            _answer(from_id="c"),
        ])
        assert validation_errors(sweep, clip=bclip) == [], f"full-clip fps{fps} sweep should pass"
