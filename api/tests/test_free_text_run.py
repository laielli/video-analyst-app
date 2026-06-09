"""
Phase 2 + security tests for the free-text run path (codegen mocked, replay over the committed
hero cache). The headline contract: a typed question runs the ViperGPT loop with NO pinned
fallback, so any miss is an explicit `program_source: "ungrounded"` run-doc — never a hero-program
substitution, never a 500.

Pattern mirror: test_baseline_harness.py (import server + canned), test_precompute.py (replay).
The codegen seam is the conftest `mock_codegen` fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
import canned
import codegen
import run_cache
from interpreter import Cache, Interpreter
from validate_program import validation_errors
from conftest import FakeCodegen

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


def _schema_errors(doc):
    import jsonschema
    return list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


# --------------------------------------------------------------------------------------
# test_free_text_grounded_returns_live_source
# --------------------------------------------------------------------------------------

def test_free_text_grounded_returns_live_source(mock_codegen, hero_program):
    mock_codegen(program=hero_program)  # codegen enabled, returns the hero program
    doc = server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    assert doc["program_source"] == "live"
    assert doc["findings"]["grounded"] is True
    assert doc["findings"]["answer"]  # non-empty
    # A grounded free-text run now carries a permalink `run_id` (the content-address). It is a
    # meta-level field the routes surface in the envelope/meta and strip from the streamed doc,
    # so the run-doc body stays schema-pure once run_id is removed.
    assert run_cache.is_valid_key(doc["run_id"])
    body = {k: v for k, v in doc.items() if k != "run_id"}
    assert not _schema_errors(body)


# --------------------------------------------------------------------------------------
# test_free_text_codegen_disabled_is_ungrounded — the no-fallback contract
# --------------------------------------------------------------------------------------

def test_free_text_codegen_disabled_is_ungrounded(mock_codegen):
    mock_codegen(enabled=False)
    doc = server.build_free_text_run_doc("how many players?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False
    assert doc["findings"]["reason"] == "codegen-disabled"
    assert doc["findings"]["answer"] is None and doc["findings"]["verdict"] is None
    # and it does NOT equal the pinned hero run-doc (no substitution).
    pinned = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    assert doc["findings"] != pinned["findings"]
    assert doc["program_source"] != pinned["program_source"]
    assert not _schema_errors(doc)


# --------------------------------------------------------------------------------------
# test_free_text_invalid_program_is_ungrounded_not_substituted — trust boundary
# --------------------------------------------------------------------------------------

def test_free_text_invalid_program_is_ungrounded_not_substituted(mock_codegen, monkeypatch):
    # codegen returns a structurally/semantically invalid program (last step is not `answer`).
    invalid = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    mock_codegen(program=invalid)

    # spy on Interpreter.run to assert it is NEVER called for an invalid program.
    called = {"n": 0}
    real_run = Interpreter.run

    def spy(self, program, query=None):
        called["n"] += 1
        return real_run(self, program, query=query)
    monkeypatch.setattr(Interpreter, "run", spy)

    doc = server.build_free_text_run_doc("anything", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["n"] == 0, "unvalidated program must never reach Interpreter.run (trust boundary)"


# --------------------------------------------------------------------------------------
# test_free_text_executes_but_does_not_ground — validated program runs but misses cache
# --------------------------------------------------------------------------------------

def test_free_text_executes_but_does_not_ground(mock_codegen):
    # A VALID program whose sampled window (0-100ms) misses every cached frame (cache @ 4625) ->
    # detect empty -> count == 0 -> the answer can't ground -> ungrounded with reason
    # 'no-grounded-answer', but the validated program's trace IS streamed so the UI can show
    # WHERE it ran dry (D-DR5). (A temporal program would still find the cache's goal event and
    # ground a "No"; a count over an empty detect set is the genuine no-grounded-answer path.)
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    mock_codegen(program=program)
    doc = server.build_free_text_run_doc("how many players?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False
    assert doc["findings"]["reason"] == "no-grounded-answer"
    # the validated program's trace is present so the failure is legible (D-DR5).
    assert doc["trace"], "ungrounded-after-execution must carry the trace"
    assert len(doc["program"]) == len(program)
    assert not _schema_errors(doc)


# --------------------------------------------------------------------------------------
# test_free_text_codegen_error_is_ungrounded — generation failure -> ungrounded (no leak)
# --------------------------------------------------------------------------------------

def test_free_text_codegen_error_is_ungrounded(mock_codegen):
    mock_codegen(raise_exc=RuntimeError("azure 500 at /secret/path"))
    doc = server.build_free_text_run_doc("anything", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "codegen-error"
    # the raw exception text never leaks into the findings.
    assert "secret" not in json.dumps(doc)


# --------------------------------------------------------------------------------------
# test_run_route_requires_exactly_one_of_query_or_query_text (plain-helper unit test)
# --------------------------------------------------------------------------------------

def test_run_route_requires_exactly_one_of_query_or_query_text():
    from fastapi import HTTPException

    def status(fn):
        try:
            fn()
            return 200
        except HTTPException as e:
            return e.status_code

    # neither -> 400
    assert status(lambda: server._resolve_run_request(None, None, None)) == 400
    # both -> 400
    assert status(lambda: server._resolve_run_request("hero-10-first-goal", "how many?", None)) == 400
    # unknown query -> 404
    assert status(lambda: server._resolve_run_request("nope", None, None)) == 404
    # unknown clip -> 404
    assert status(lambda: server._resolve_run_request(None, "how many?", "no-such-clip")) == 404
    # valid canned / free
    assert server._resolve_run_request("hero-10-first-goal", None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None)[0] == "free"


# --------------------------------------------------------------------------------------
# security: query-text cap + numeric-bound validation
# --------------------------------------------------------------------------------------

def test_free_text_rejects_overlong_query_text(mock_codegen):
    from fastapi import HTTPException
    fake = mock_codegen(program=[])  # would be called if the cap didn't short-circuit
    long = "x" * (server.MAX_QUERY_TEXT_LEN + 1)
    with pytest.raises(HTTPException) as ei:
        server._resolve_run_request(None, long, None)
    assert ei.value.status_code == 400
    # codegen was NOT called (cap is enforced before codegen).
    assert FakeCodegen.total_calls == 0


def test_validation_rejects_out_of_range_sample_frames():
    clip = canned.clip_by_id("single-goal")
    dur = clip["duration_ms"]
    # end beyond clip duration -> rejected.
    bad_end = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": dur + 5000, "fps": 8}},
        {"id": "r", "op": "answer", "args": {"from": "f", "question": "q"}},
    ]}
    assert validation_errors(bad_end, clip=clip)
    # fps out of [1,30] -> rejected.
    bad_fps = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 9999}},
        {"id": "r", "op": "answer", "args": {"from": "f", "question": "q"}},
    ]}
    assert validation_errors(bad_fps, clip=clip)
    # start >= end -> rejected even without a clip.
    bad_order = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 2000, "end_ms": 1000, "fps": 8}},
        {"id": "r", "op": "answer", "args": {"from": "f", "question": "q"}},
    ]}
    assert validation_errors(bad_order)
    # a well-formed in-bounds window passes the bounds gate (still needs answer-is-last etc).
    good = {"program": [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "c", "op": "count", "args": {"items": "f"}},
        {"id": "r", "op": "answer", "args": {"from": "c", "question": "q"}},
    ]}
    assert validation_errors(good, clip=clip) == []


# --------------------------------------------------------------------------------------
# test_canned_clip_by_id_resolves_cache
# --------------------------------------------------------------------------------------

def test_canned_clip_by_id_resolves_cache():
    clip = canned.clip_by_id("single-goal")
    assert clip is not None
    cache = Cache.load(clip["cache"])  # loadable
    assert cache.clip["id"] == "single-goal"
    assert canned.clip_by_id("does-not-exist") is None


# --------------------------------------------------------------------------------------
# test_canned_live_codegen_rejects_out_of_range_program — the canned path shares the
# free-text path's numeric-bounds trust boundary (regression for the post-merge [P1] gap:
# _live_run_doc previously called validation_errors WITHOUT the clip, so an out-of-range
# generated program could reach Interpreter.run on the canned route).
# --------------------------------------------------------------------------------------

def test_canned_live_codegen_rejects_out_of_range_program(mock_codegen, monkeypatch):
    real_run = Interpreter.run
    executed = []

    def spy(self, program, *args, **kwargs):
        executed.append(program)
        # Never actually replay an out-of-range window (range would be enormous). _live_run_doc
        # catches the exception and falls back, so a regression is caught here without hanging.
        if any(s.get("args", {}).get("end_ms", 0) > clip_duration for s in program):
            raise RuntimeError("out-of-range program reached Interpreter.run (trust boundary breached)")
        return real_run(self, program, *args, **kwargs)

    clip_duration = canned.clip_by_id("single-goal")["duration_ms"]
    monkeypatch.setattr(Interpreter, "run", spy)

    oob = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 999_999_999, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    mock_codegen(program=oob)  # canned query whose live codegen returns an out-of-range program
    doc = server.build_run_doc(canned.by_id("hero-10-first-goal"))
    # The live program is rejected at validation -> the route falls back to the pinned program.
    assert doc["program_source"] == "pinned"
    # The out-of-range program was NEVER executed (rejected before Interpreter.run).
    assert all(
        not any(s.get("args", {}).get("end_ms", 0) > clip_duration for s in p)
        for p in executed
    )
