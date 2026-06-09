"""
Adversarial trust-boundary suite for the LLM-generated-program executor (the ViperGPT layer-2
trust boundary; see CLAUDE.md + plan trust-boundary-hardening). Every reject is asserted to
happen BEFORE ANY OP EXECUTES — the zero-execution spy monkeypatches `Interpreter.run` on the
CLASS (the proven pattern from test_free_text_run.py:75-80), never a primitives module attr
(OPS holds function references, so a module-attr patch would never be seen).

What this file proves:
  - oversized / step-bomb / frame-bomb / bad-arg-domain programs are REJECTED at validation
    (the free-text path resolves them to a `program_source: "ungrounded"` doc with
    `reason: "invalid-program"`, and `Interpreter.run` is never called);
  - the pre-parse byte cap rejects an oversized doc before the jsonschema walk;
  - the TWO runtime guards (the pre-materialization cap in op_sample_frames + the MAX_STEPS
    check at the top of Interpreter.run) fire when the validator is bypassed;
  - the committed pinned programs all still pass the tightened validator (no false reject);
  - the schema's static limits stay in sync with limits.py.

Codegen is mocked via the conftest `mock_codegen` fixture; replay is over the committed caches.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
import canned
import codegen
import limits
from interpreter import Cache, Interpreter
from limits import (
    MAX_DETECT_CLASSES,
    MAX_PROGRAM_BYTES,
    MAX_SAMPLED_FRAMES,
    MAX_STEPS,
    MAX_STR_ARG_LEN,
    ProgramLimitExceeded,
)
import validate_program
from validate_program import strip_comments, validation_errors

API_DIR = Path(__file__).resolve().parent.parent
EX = API_DIR / "examples"
DSL_SCHEMA = json.loads((API_DIR / "schema" / "dsl.schema.json").read_text())

CLIP = "single-goal"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _answer(from_id="f", q="q"):
    return {"id": "result", "op": "answer", "args": {"from": from_id, "question": q}}


def _trivial_count_program(n_filler_steps: int) -> list[dict]:
    """A structurally + semantically VALID program of (n_filler_steps + 3) steps: one
    sample_frames, then n_filler_steps no-op filters chained off it, then count -> answer. Used to
    build a step-bomb that passes every check EXCEPT the step cap, so the step cap is what rejects
    it (not a side effect of a malformed ref)."""
    prog = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    prev = "f"
    for i in range(n_filler_steps):
        sid = f"keep{i}"
        prog.append({"id": sid, "op": "filter", "args": {"items": prev,
                     "where": {"field": "frame_ts_ms", "equals": "x"}}})
        prev = sid
    prog.append({"id": "n", "op": "count", "args": {"items": prev}})
    prog.append({"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}})
    return prog


def _install_run_spy(monkeypatch):
    """Spy on Interpreter.run (CLASS method — the proven zero-execution pattern). Returns a dict
    whose 'n' counts calls; the real run still executes so valid programs behave normally."""
    called = {"n": 0}
    real_run = Interpreter.run

    def spy(self, program, query=None):
        called["n"] += 1
        return real_run(self, program, query=query)

    monkeypatch.setattr(Interpreter, "run", spy)
    return called


# --------------------------------------------------------------------------------------
# step-count bomb
# --------------------------------------------------------------------------------------

def test_oversized_program_step_bomb_rejected(mock_codegen, monkeypatch):
    program = _trivial_count_program(MAX_STEPS)  # MAX_STEPS + 3 total steps > cap
    assert len(program) > MAX_STEPS
    # direct validator: rejected.
    assert validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    # via the free-text path: ungrounded + zero execution.
    mock_codegen(program=program)
    called = _install_run_spy(monkeypatch)
    doc = server.build_free_text_run_doc("anything", CLIP)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["n"] == 0, "step-bomb must never reach Interpreter.run"


def test_schema_maxitems_rejects_independently():
    """A MAX_STEPS + 1 program fails the STRUCTURAL (schema) gate alone — the schema-level
    maxItems works even if semantic_errors regressed."""
    program = _trivial_count_program(MAX_STEPS)
    struct = validate_program.structural_errors(strip_comments({"program": program}), DSL_SCHEMA)
    assert struct, "schema maxItems should reject a > MAX_STEPS program on its own"


# --------------------------------------------------------------------------------------
# frame-count bombs (cumulative cap, duration-independent)
# --------------------------------------------------------------------------------------

def test_frame_count_bomb_rejected_without_clip():
    # A single sample_frames whose window would materialize ~60M frames. clip=None (only the
    # direct/CLI call path; the server always resolves a clip). The count cap is O(1) arithmetic,
    # so NO list is built by the validator even on this bomb.
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
        _answer(),
    ]
    errs = validation_errors({"program": program}, clip=None)
    assert errs, "frame-count bomb must be rejected even without a clip"
    assert any("frames" in e for e in errs)


def test_frame_count_bomb_rejected_with_clip(monkeypatch, mock_codegen):
    # Many small in-bounds windows that individually pass but SUM over the cumulative cap. Each
    # window samples ~9 frames (1000ms @ fps8, stride 125 -> 9); enough of them exceeds 2000.
    clip = canned.clip_by_id(CLIP)
    dur = clip["duration_ms"]  # 5067ms; each 1000ms window @ fps8 -> 9 frames
    per_window = len(range(0, 1001, max(1, round(1000 / 8))))
    n_windows = (MAX_SAMPLED_FRAMES // per_window) + 5  # comfortably over the cap
    program = []
    for i in range(n_windows):
        program.append({"id": f"f{i}", "op": "sample_frames",
                        "args": {"start_ms": 0, "end_ms": min(1000, dur), "fps": 8}})
    # tie them into a valid-enough tail (count over the LAST frames binding) so only the
    # cumulative-frame cap is the rejecter, plus the step cap may also fire — both are rejects.
    program.append({"id": "n", "op": "count", "args": {"items": f"f{n_windows - 1}"}})
    program.append({"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}})
    errs = validation_errors({"program": program}, clip=clip)
    assert errs, "cumulative frame bomb must be rejected"

    mock_codegen(program=program)
    called = _install_run_spy(monkeypatch)
    doc = server.build_free_text_run_doc("anything", CLIP)
    assert doc["program_source"] == "ungrounded"
    assert called["n"] == 0


# --------------------------------------------------------------------------------------
# detect.classes domain
# --------------------------------------------------------------------------------------

def test_detect_classes_empty_rejected():
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": []}},
        {"id": "n", "op": "count", "args": {"items": "d"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs, "empty detect.classes must be rejected (schema minItems + semantic check)"


def test_detect_classes_overlong_rejected():
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f",
         "classes": [f"c{i}" for i in range(MAX_DETECT_CLASSES + 1)]}},
        {"id": "n", "op": "count", "args": {"items": "d"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs, "detect.classes over MAX_DETECT_CLASSES must be rejected"


# --------------------------------------------------------------------------------------
# huge string args
# --------------------------------------------------------------------------------------

def test_huge_string_arg_rejected():
    big = "x" * (MAX_STR_ARG_LEN + 1)
    # giant filter.where.equals
    prog_equals = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "k", "op": "filter", "args": {"items": "d", "where": {"field": "cls", "equals": big}}},
        {"id": "n", "op": "count", "args": {"items": "k"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    assert validation_errors({"program": prog_equals}, clip=canned.clip_by_id(CLIP))
    # giant classes[0]
    prog_class = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": [big]}},
        {"id": "n", "op": "count", "args": {"items": "d"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    assert validation_errors({"program": prog_class}, clip=canned.clip_by_id(CLIP))


# --------------------------------------------------------------------------------------
# structural rejects (anyOf / additionalProperties / types) short-circuit semantics
# --------------------------------------------------------------------------------------

def test_deeply_nested_args_rejected():
    # extra/unknown nested arg keys -> additionalProperties:false structural failure.
    program = [
        {"id": "f", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": 1000, "fps": 8, "extra": {"deep": {"deeper": [1, 2, 3]}}}},
        _answer(),
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and any(e.startswith("structural:") for e in errs)


def test_dangling_ref_rejected(monkeypatch, mock_codegen):
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "ghost", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "d"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and any("not a prior step id" in e for e in errs)
    mock_codegen(program=program)
    called = _install_run_spy(monkeypatch)
    server.build_free_text_run_doc("anything", CLIP)
    assert called["n"] == 0


def test_forward_or_self_ref_rejected():
    # references a LATER step id (declaration-order violation) — distinct from the dangling class.
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "n", "op": "count", "args": {"items": "later"}},        # 'later' declared after
        {"id": "later", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and any("not a prior step id" in e for e in errs)


def test_duplicate_ids_rejected():
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "f", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},  # dup id
        {"id": "n", "op": "count", "args": {"items": "f"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and any("duplicate id" in e for e in errs)


def test_unknown_op_rejected_before_execution():
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "x", "op": "rm_rf", "args": {"path": "/"}},  # not a whitelisted op
        _answer(),
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and any(e.startswith("structural:") for e in errs)
    # belt-and-suspenders: if such a program ever reached the interpreter it raises (it must not).
    with pytest.raises(ValueError):
        Interpreter(Cache.load(canned.clip_by_id(CLIP)["cache"])).run(program)


def test_wrong_arg_types_rejected():
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 1.5, "end_ms": 1000, "fps": "8"}},
        _answer(),
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and any(e.startswith("structural:") for e in errs)


def test_cross_clip_cache_reference_rejected():
    """Invariant test: NO DSL op accepts a path / cache / clip arg — the cache is selected by
    clip_id at the server (build_free_text_run_doc loads clip['cache']), never by the program. A
    crafted program cannot redirect the interpreter to another clip's cache. We prove the attack
    surface is absent: every op's arg properties are a fixed whitelist with additionalProperties
    false, and none names a path/cache/clip/file."""
    forbidden = {"path", "cache", "clip", "clip_id", "file", "filename", "url", "src", "source"}
    for name, d in DSL_SCHEMA["$defs"].items():
        if name == "binding":
            continue
        arg_props = d["properties"]["args"].get("properties", {})
        assert d["properties"]["args"]["additionalProperties"] is False
        assert not (set(arg_props) & forbidden), f"op '{name}' exposes a cache/path arg"


# --------------------------------------------------------------------------------------
# runtime guards (validator bypassed) — both checks
# --------------------------------------------------------------------------------------

def test_runtime_cap_fires_when_validation_bypassed():
    cache = Cache.load(canned.clip_by_id(CLIP)["cache"])
    interp = Interpreter(cache)

    # (1) frame-bomb JUST over the cap (not 2e9 — that would build 60M ints if the cap regressed).
    # start=0, end=66000, fps=30 -> stride 33 -> 2001 frames (= MAX_SAMPLED_FRAMES + 1).
    stride = max(1, round(1000 / 30))
    end = MAX_SAMPLED_FRAMES * stride  # yields MAX_SAMPLED_FRAMES + 1 frames
    assert len(range(0, end + 1, stride)) == MAX_SAMPLED_FRAMES + 1
    bomb = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": end, "fps": 30}},
        {"id": "result", "op": "answer", "args": {"from": "f", "question": "q"}},
    ]
    with pytest.raises(ProgramLimitExceeded):
        interp.run(bomb)

    # (2) step-bomb: MAX_STEPS + 1 trivial steps -> the top-of-run() guard fires before any op.
    steps = [{"id": f"s{i}", "op": "count", "args": {"items": "x"}} for i in range(MAX_STEPS + 1)]
    with pytest.raises(ProgramLimitExceeded):
        interp.run(steps)


def test_runtime_frame_cap_builds_no_list_on_bomb(monkeypatch):
    """The pre-materialization cap must reject WITHOUT building list(range(...)). Patch the
    builtin `list` inside primitives to a tripwire that fails the test if it is ever called with a
    range during op_sample_frames — the cap raises before that line."""
    import interpreter.primitives as prims
    cache = Cache.load(canned.clip_by_id(CLIP)["cache"])
    interp = Interpreter(cache)
    stride = max(1, round(1000 / 30))
    end = MAX_SAMPLED_FRAMES * stride

    real_list = list
    tripped = {"hit": False}

    def tripwire(x=()):
        if isinstance(x, range) and len(x) > MAX_SAMPLED_FRAMES:
            tripped["hit"] = True
        return real_list(x)

    monkeypatch.setattr(prims, "list", tripwire, raising=False)
    bomb = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": end, "fps": 30}},
        {"id": "result", "op": "answer", "args": {"from": "f", "question": "q"}},
    ]
    with pytest.raises(ProgramLimitExceeded):
        interp.run(bomb)
    assert tripped["hit"] is False, "over-cap range must never be materialized"


# --------------------------------------------------------------------------------------
# oversized codegen output — byte cap before parse
# --------------------------------------------------------------------------------------

def test_oversized_codegen_output_rejected_before_parse(monkeypatch):
    # A doc whose serialized size exceeds MAX_PROGRAM_BYTES must be rejected by validation_errors
    # BEFORE structural_errors / jsonschema is invoked. Spy on structural_errors to assert it is
    # never reached.
    reached = {"n": 0}
    real_struct = validate_program.structural_errors

    def struct_spy(doc, schema):
        reached["n"] += 1
        return real_struct(doc, schema)

    monkeypatch.setattr(validate_program, "structural_errors", struct_spy)

    huge_q = "y" * (MAX_PROGRAM_BYTES + 1000)
    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "result", "op": "answer", "args": {"from": "f", "question": huge_q}},
    ]
    errs = validation_errors({"program": program}, clip=canned.clip_by_id(CLIP))
    assert errs and errs[0].startswith("size:")
    assert reached["n"] == 0, "oversized doc must be rejected before the jsonschema walk"


def test_codegen_generate_byte_cap_returns_rejection(monkeypatch):
    # codegen.AzureCodegen.generate must reject an oversized completion (raise) rather than hand
    # the blob downstream. Drive _client.chat.completions.create to return a giant content blob.
    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    class _Resp:
        def __init__(self, content):
            self.choices = [_Msg(content)]

    blob = '{"program": [' + ("0," * (MAX_PROGRAM_BYTES // 2)) + "0]}"
    assert len(blob.encode("utf-8")) > MAX_PROGRAM_BYTES

    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    cg._deployment = "gpt-4o"

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs.get("max_tokens") == codegen.MAX_COMPLETION_TOKENS
                    return _Resp(blob)

    cg._client = _FakeClient()
    with pytest.raises(ValueError) as ei:
        cg.generate("anything")
    assert "bytes" in str(ei.value)


# --------------------------------------------------------------------------------------
# reject leaks no partial trace; SSE + canned contracts
# --------------------------------------------------------------------------------------

def test_reject_produces_no_partial_trace(mock_codegen, monkeypatch):
    program = _trivial_count_program(MAX_STEPS)  # step-bomb -> rejected at validation
    mock_codegen(program=program)
    called = _install_run_spy(monkeypatch)
    doc = server.build_free_text_run_doc("anything", CLIP)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert doc["trace"] == [], "a validation reject must carry NO trace (no partial run leaked)"
    assert doc["program"] == [], "a validation reject must carry NO program"
    assert called["n"] == 0


def test_sse_reject_contract(mock_codegen):
    from fastapi.testclient import TestClient
    program = _trivial_count_program(MAX_STEPS)
    mock_codegen(program=program)
    client = TestClient(server.app)
    resp = client.get("/api/run", params={"query_text": "anything", "pace_ms": 0})
    assert resp.status_code == 200
    names, findings = [], None
    for block in resp.text.strip().split("\n\n"):
        if not block.strip():
            continue
        nm = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                nm = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        names.append(nm)
        if nm == "findings":
            findings = data
    # empty trace -> meta -> findings -> done.
    assert names == ["meta", "findings", "done"]
    assert findings["grounded"] is False
    assert findings["reason"] == "invalid-program"


def test_canned_live_step_bomb_falls_back_to_pinned(mock_codegen):
    # On the CANNED route, a live-codegen step-bomb is rejected by _live_run_doc -> the route
    # serves the pinned replay (program_source == 'pinned'). Pins the limits-specific D-DR6 fallback.
    program = _trivial_count_program(MAX_STEPS)
    mock_codegen(program=program)
    doc = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    assert doc["program_source"] == "pinned"
    assert doc["findings"]["grounded"] is True  # the pinned program grounds normally


# --------------------------------------------------------------------------------------
# no false rejects: pinned programs + legit sweeps pass; schema/limits stay in sync
# --------------------------------------------------------------------------------------

PINNED = [
    ("hero_program.json", "single-goal"),
    ("bernabeu-counter_count_program.json", "bernabeu-counter"),
    ("bernabeu-counter_first-goal-7_program.json", "bernabeu-counter"),
    ("bernabeu-counter_first-goal-23_program.json", "bernabeu-counter"),
    ("bernabeu-counter_scorer-number_program.json", "bernabeu-counter"),
]


@pytest.mark.parametrize("fname,clip_id", PINNED)
def test_pinned_programs_pass_tightened_validator(fname, clip_id):
    raw = json.loads((EX / fname).read_text())
    doc = strip_comments(raw)
    clip = canned.clip_by_id(clip_id)
    assert validation_errors({"program": doc["program"]}, clip=clip) == [], fname


def test_limit_constants_match_schema():
    # The schema duplicates the limits; assert they can't silently drift.
    assert DSL_SCHEMA["properties"]["program"]["maxItems"] == MAX_STEPS
    classes = DSL_SCHEMA["$defs"]["detect"]["properties"]["args"]["properties"]["classes"]
    assert classes["maxItems"] == MAX_DETECT_CLASSES
    assert classes["minItems"] == 1
    assert classes["items"]["maxLength"] == MAX_STR_ARG_LEN
    where = DSL_SCHEMA["$defs"]["filter"]["properties"]["args"]["properties"]["where"]["properties"]
    assert where["field"]["maxLength"] == MAX_STR_ARG_LEN
    assert where["equals"]["maxLength"] == MAX_STR_ARG_LEN
    assert DSL_SCHEMA["$defs"]["binding"]["maxLength"] == MAX_STR_ARG_LEN
    fps = DSL_SCHEMA["$defs"]["sample_frames"]["properties"]["args"]["properties"]["fps"]
    assert (fps["minimum"], fps["maximum"]) == (validate_program.FPS_MIN, validate_program.FPS_MAX)


def test_legit_full_clip_sweep_passes():
    clip = canned.clip_by_id("bernabeu-counter")  # duration_ms 30086
    dur = clip["duration_ms"]
    # whole clip @ fps8 (~241 frames) passes.
    p8 = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": dur, "fps": 8}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "d"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    assert validation_errors(p8, clip=clip) == []
    # whole clip @ fps30 (~912 frames) ALSO passes (envelope above the largest legitimate sweep).
    p30 = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": dur, "fps": 30}},
        {"id": "d", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "d"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    assert validation_errors(p30, clip=clip) == []
