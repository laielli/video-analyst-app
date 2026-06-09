"""
Adversarial suite for the trust-boundary hardening (resource + semantic guards on the
generated-program executor). Every reject is asserted to happen BEFORE any op executes, via the
proven zero-execution spy from test_free_text_run.py:75-80 — `monkeypatch.setattr(Interpreter,
"run", spy)` on the CLASS (NOT module-level primitives, which OPS would never see). The headline
contract: a malformed / oversized / hostile generated program resolves to an `_ungrounded`
run-doc with `reason: "invalid-program"` (or `execution-error` for the runtime cap), with the
interpreter never touched on the validator-reject path.

The runtime-guard tests bypass the validator and call Interpreter.run / op_sample_frames directly
to prove the two residual guards (pre-materialization frame cap in op_sample_frames, MAX_STEPS at
the top of run) fire independently — using a window JUST over the cap, never end=2e9 (which would
materialize 60M ints in CI if the cap ever regressed).
"""
from __future__ import annotations

import json

import pytest

import server
import canned
import codegen
from interpreter import Cache, Interpreter
from interpreter.primitives import op_sample_frames
from validate_program import validation_errors
from limits import (
    MAX_DETECT_CLASSES,
    MAX_PROGRAM_BYTES,
    MAX_SAMPLED_FRAMES,
    MAX_STEPS,
    MAX_STR_ARG_LEN,
    ProgramLimitExceeded,
    sampled_frame_count,
)


CLIP_ID = "single-goal"


@pytest.fixture
def run_spy(monkeypatch):
    """Install the class-method spy on Interpreter.run and return a {n} counter dict. Asserting
    counter['n'] == 0 proves zero execution on a reject path (the trust boundary)."""
    counter = {"n": 0}
    real_run = Interpreter.run

    def spy(self, program, query=None):
        counter["n"] += 1
        return real_run(self, program, query=query)

    monkeypatch.setattr(Interpreter, "run", spy)
    return counter


def _answer(from_id="x", q="q"):
    return {"id": "result", "op": "answer", "args": {"from": from_id, "question": q}}


# =======================================================================================
# Validator-reject classes (asserted via direct validation_errors AND the free-text path)
# =======================================================================================

def test_oversized_program_step_bomb_rejected(mock_codegen, run_spy):
    # MAX_STEPS + 1 trivial count steps over a tiny frame window, last step answer.
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
    for k in range(MAX_STEPS):  # pushes total to MAX_STEPS + 1
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(from_id="f"))
    assert len(steps) > MAX_STEPS
    assert validation_errors({"program": steps}, clip=canned.clip_by_id(CLIP_ID))

    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert run_spy["n"] == 0, "oversized program must never reach Interpreter.run"


def test_frame_count_bomb_rejected_without_clip():
    # Direct unit call with clip=None (the server path always resolves a clip; clip=None occurs
    # only on direct/CLI calls). The count cap is duration-independent so the no-clip gap closes.
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
        _answer(from_id="f"),
    ]}
    errs = validation_errors(prog, clip=None)
    assert errs, "an unbounded sample_frames window must be rejected even with no clip"


def test_frame_count_bomb_rejected_with_clip(run_spy, mock_codegen):
    # Many small in-bounds windows whose CUMULATIVE frame count exceeds MAX_SAMPLED_FRAMES.
    clip = canned.clip_by_id(CLIP_ID)
    # each window 0..1000ms @ fps30 ~= 31 frames; ~70 of them blow the 2000 cap.
    per = sampled_frame_count(0, 1000, 30)
    n_windows = (MAX_SAMPLED_FRAMES // per) + 2
    steps = []
    for k in range(n_windows):
        steps.append({"id": f"f{k}", "op": "sample_frames",
                      "args": {"start_ms": 0, "end_ms": 1000, "fps": 30}})
    steps.append(_answer(from_id="f0"))
    assert n_windows * per > MAX_SAMPLED_FRAMES
    assert validation_errors({"program": steps}, clip=clip)

    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    assert run_spy["n"] == 0


def test_detect_classes_empty_rejected():
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": []}},
        _answer(from_id="d"),
    ]}
    # rejected structurally (schema minItems:1 makes the detect anyOf branch fail).
    errs = validation_errors(prog, clip=canned.clip_by_id(CLIP_ID))
    assert errs and all(e.startswith("structural:") for e in errs)
    # and the SEMANTIC-level empty check fires independently when structural is bypassed (a future
    # caller validating only semantics, or a schema regression, is still covered).
    from validate_program import semantic_errors
    sem = semantic_errors(prog["program"], clip=canned.clip_by_id(CLIP_ID))
    assert any("classes is empty" in e for e in sem)


def test_detect_classes_overlong_rejected():
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect",
         "args": {"frames": "f", "classes": [f"c{k}" for k in range(MAX_DETECT_CLASSES + 1)]}},
        _answer(from_id="d"),
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id(CLIP_ID))


def test_huge_string_arg_rejected():
    big = "x" * (MAX_STR_ARG_LEN + 1)
    # giant filter.where.equals
    prog1 = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "g", "op": "filter", "args": {"items": "d", "where": {"field": "cls", "equals": big}}},
        _answer(from_id="g"),
    ]}
    assert validation_errors(prog1, clip=canned.clip_by_id(CLIP_ID))
    # giant classes[0]
    prog2 = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": [big]}},
        _answer(from_id="d"),
    ]}
    assert validation_errors(prog2, clip=canned.clip_by_id(CLIP_ID))


def test_deeply_nested_args_rejected(run_spy, mock_codegen):
    # extra/unknown nested arg objects are rejected structurally (additionalProperties:false),
    # short-circuiting semantics — a regression that lets garbage args through would fail here.
    prog = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 100, "fps": 8, "evil": {"a": {"b": {"c": 1}}}}},
        _answer(from_id="f"),
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id(CLIP_ID))
    assert errs and all(e.startswith("structural:") for e in errs)


def test_dangling_ref_rejected(run_spy, mock_codegen):
    prog = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "nope", "classes": ["person"]}},
        _answer(from_id="d"),
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id(CLIP_ID))
    assert any("not a prior step id" in e for e in errs)

    mock_codegen(program=prog["program"])
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    assert run_spy["n"] == 0


def test_forward_or_self_ref_rejected():
    # self-ref: detect references its own id; forward-ref: detect references a LATER step.
    self_ref = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "d", "classes": ["person"]}},
        _answer(from_id="d"),
    ]}
    assert validation_errors(self_ref, clip=canned.clip_by_id(CLIP_ID))
    forward_ref = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "later", "classes": ["person"]}},
        {"id": "later", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        _answer(from_id="d"),
    ]}
    assert validation_errors(forward_ref, clip=canned.clip_by_id(CLIP_ID))


def test_duplicate_ids_rejected():
    prog = {"program": [
        {"id": "dup", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "dup", "op": "detect", "args": {"frames": "dup", "classes": ["person"]}},
        _answer(from_id="dup"),
    ]}
    errs = validation_errors(prog, clip=canned.clip_by_id(CLIP_ID))
    assert any("duplicate id" in e for e in errs)


def test_unknown_op_rejected_before_execution():
    prog = {"program": [
        {"id": "x", "op": "rm_rf", "args": {"path": "/"}},
        _answer(from_id="x"),
    ]}
    assert validation_errors(prog, clip=canned.clip_by_id(CLIP_ID))
    # and the interpreter (were the validator ever bypassed) raises on an unknown op.
    with pytest.raises(ValueError, match="unknown op"):
        Interpreter(Cache.load(canned.clip_by_id(CLIP_ID)["cache"])).run([{"id": "x", "op": "rm_rf", "args": {}}])


def test_wrong_arg_types_rejected():
    # fps as a string, start_ms as a float — both structural type failures.
    bad_fps = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": "8"}},
        _answer(from_id="f"),
    ]}
    assert validation_errors(bad_fps, clip=canned.clip_by_id(CLIP_ID))
    bad_start = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 1.5, "end_ms": 100, "fps": 8}},
        _answer(from_id="f"),
    ]}
    assert validation_errors(bad_start, clip=canned.clip_by_id(CLIP_ID))


def test_cross_clip_cache_reference_rejected():
    # Invariant test: NO DSL op accepts a path/clip/cache arg — the cache is selected by clip_id
    # at the route (server.build_free_text_run_doc loads clip["cache"]), never by the program. A
    # crafted program therefore cannot redirect which cache it runs against. We assert the
    # attack surface is absent by construction: any path-like arg fails additionalProperties:false.
    schema = json.loads((canned.API_DIR / "schema" / "dsl.schema.json").read_text())
    blob = json.dumps(schema)
    for forbidden in ("cache", "path", "file", "clip_id", "clip"):
        # No op arg is named after a cache/file/clip selector.
        assert f'"{forbidden}"' not in blob, f"DSL must not expose a '{forbidden}' arg"
    # And a program that tries to smuggle one is rejected structurally.
    smuggle = {"program": [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 100, "fps": 8, "cache": "../other_clip_cache.json"}},
        _answer(from_id="f"),
    ]}
    errs = validation_errors(smuggle, clip=canned.clip_by_id(CLIP_ID))
    assert errs and all(e.startswith("structural:") for e in errs)


# =======================================================================================
# Runtime guards — bypass the validator, call the interpreter/op directly
# =======================================================================================

def test_runtime_cap_fires_when_validation_bypassed():
    cache = Cache.load(canned.clip_by_id(CLIP_ID)["cache"])
    # (1) a frame-bomb JUST over the cap (NOT end=2e9 — that would materialize 60M ints if the
    # cap regressed). Find a window yielding MAX_SAMPLED_FRAMES + 1 frames at fps1 (stride 1000ms).
    over_end = (MAX_SAMPLED_FRAMES) * 1000  # fps1 -> stride 1000; count = (end-0)//1000 + 1
    assert sampled_frame_count(0, over_end, 1) > MAX_SAMPLED_FRAMES
    step = {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": over_end, "fps": 1}}
    with pytest.raises(ProgramLimitExceeded):
        op_sample_frames(step, {}, cache)

    # (2) a MAX_STEPS + 1 step-bomb -> ProgramLimitExceeded from the check at the top of run().
    big_program = [{"id": f"s{k}", "op": "count", "args": {"items": "s0"}} for k in range(MAX_STEPS + 1)]
    with pytest.raises(ProgramLimitExceeded):
        Interpreter(cache).run(big_program)


def test_runtime_frame_cap_does_not_materialize(monkeypatch):
    # Prove the pre-materialization cap is O(1): if it built the list first, this would OOM. We
    # assert it raises without ever calling list(range(...)) by trapping `range` would be fragile;
    # instead we use a window so large that materialization would be catastrophic and rely on the
    # raise happening near-instantly. end=2e9 @ fps30 -> ~60M; the cap must reject before list().
    cache = Cache.load(canned.clip_by_id(CLIP_ID)["cache"])
    step = {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}}
    with pytest.raises(ProgramLimitExceeded):
        op_sample_frames(step, {}, cache)


def test_oversized_codegen_output_rejected_before_parse(monkeypatch):
    # (a) validation_errors rejects an oversized doc BEFORE the structural/jsonschema pass.
    import validate_program
    called = {"struct": 0}
    real_struct = validate_program.structural_errors

    def spy_struct(doc, schema):
        called["struct"] += 1
        return real_struct(doc, schema)

    monkeypatch.setattr(validate_program, "structural_errors", spy_struct)
    # build a doc whose serialized size exceeds MAX_PROGRAM_BYTES (a giant unused string arg).
    huge = "z" * (MAX_PROGRAM_BYTES + 100)
    oversized = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8, "x": huge}},
    ]}
    errs = validation_errors(oversized)
    assert errs and errs[0].startswith("size:")
    assert called["struct"] == 0, "oversized doc must be rejected BEFORE the jsonschema walk"

    # (b) codegen.generate's byte-cap path rejects the blob rather than returning it.
    class _Choice:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    # honor max_tokens being passed (the cap on the API call).
                    assert "max_tokens" in kwargs and kwargs["max_tokens"] > 0
                    return _Resp(json.dumps({"program": [{"junk": huge}]}))

    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    cg._client = _FakeClient()
    cg._deployment = "gpt-4o"
    with pytest.raises(ValueError):
        cg.generate("anything")


# =======================================================================================
# End-to-end reject contract on the free-text + SSE paths
# =======================================================================================

def test_reject_produces_no_partial_trace(mock_codegen, run_spy):
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
    for k in range(MAX_STEPS):
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(from_id="f"))
    mock_codegen(program=steps)
    doc = server.build_free_text_run_doc("anything", CLIP_ID)
    assert doc["program_source"] == "ungrounded"
    assert doc["program"] == [], "validator-reject path must not leak a partial program"
    assert doc["trace"] == [], "validator-reject path must not leak a partial trace"
    assert run_spy["n"] == 0


def test_sse_reject_contract(mock_codegen):
    from fastapi.testclient import TestClient

    def parse(text):
        out = []
        for block in text.strip().split("\n\n"):
            name = data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    name = line[len("event: "):]
                elif line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
            out.append((name, data))
        return out

    # a mocked step-bomb program on the free-text path -> meta -> findings -> done with
    # reason invalid-program and an empty trace (no step events).
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
    for k in range(MAX_STEPS):
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(from_id="f"))
    mock_codegen(program=steps)

    client = TestClient(server.app)
    resp = client.get("/api/run", params={"query_text": "anything", "pace_ms": 0})
    assert resp.status_code == 200
    evs = parse(resp.text)
    names = [n for n, _ in evs]
    assert names == ["meta", "findings", "done"]
    assert evs[0][1]["program_source"] == "ungrounded"
    findings = evs[1][1]
    assert findings["grounded"] is False
    assert findings["reason"] == "invalid-program"


def test_canned_live_step_bomb_falls_back_to_pinned(mock_codegen, run_spy):
    # On the CANNED route, a mocked live-codegen step-bomb -> _live_run_doc returns None ->
    # the route serves the pinned replay (program_source == "pinned"). The D-DR6 fallback holds
    # for limit rejections, never a 500.
    steps = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
    for k in range(MAX_STEPS):
        steps.append({"id": f"c{k}", "op": "count", "args": {"items": "f"}})
    steps.append(_answer(from_id="f"))
    mock_codegen(program=steps)
    doc = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    assert doc["program_source"] == "pinned"
    # the oversized live program never reached Interpreter.run; only the pinned (7-step) program did.
    assert run_spy["n"] == 1
