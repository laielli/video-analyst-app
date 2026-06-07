"""
Phase 2: server-level tests for the free-text branch. Codegen is mocked (conftest.FakeCodegen via
the enable_codegen / disable_codegen fixtures); execution replays over the committed hero cache.
No network, no creds. These import `server` (FastAPI) so they run wherever fastapi + pytest both
resolve (api/.venv with pytest installed) — the param-validation logic is the extracted plain
helper `resolve_run_params`, unit-tested directly (no TestClient), per the plan BLOCKER.

Mirrors test_baseline_harness.py (import via the module) + test_precompute.py (cache replay).
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import server
import canned

API_DIR = Path(__file__).resolve().parent.parent
RUNDOC_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
HERO_RUN_DOC = json.loads((API_DIR / "examples" / "hero_run_doc.json").read_text())


def _schema_valid(doc):
    return list(jsonschema.Draft202012Validator(RUNDOC_SCHEMA).iter_errors(doc))


# ---- grounded happy path ----------------------------------------------------------------

def test_free_text_grounded_returns_live_source(enable_codegen, hero_program):
    """Codegen mocked to return the committed hero program + hero cache loaded ->
    build_free_text_run_doc returns program_source 'live', a non-empty grounded answer, and a
    schema-valid run-doc. Mirrors test_hero_cache_replays_to_grounded_answer."""
    enable_codegen(program=hero_program)
    doc = server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    assert doc["program_source"] == "live"
    assert doc["findings"]["answer"]
    assert doc["findings"].get("grounded") is not False
    assert doc["findings"]["verdict"] == "Yes — #10 scored the first goal"
    assert not _schema_valid(doc)


# ---- no-fallback contract: every miss is ungrounded, never the hero program -------------

def test_free_text_codegen_disabled_is_ungrounded(disable_codegen):
    """codegen.enabled() False -> ungrounded (program_source 'ungrounded', grounded False,
    reason 'codegen-disabled'), and the doc does NOT equal the pinned hero run-doc."""
    doc = server.build_free_text_run_doc("anything at all", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False
    assert doc["findings"]["reason"] == "codegen-disabled"
    assert doc["program"] == [] and doc["trace"] == []
    # no hero substitution.
    assert doc["program"] != HERO_RUN_DOC["program"]
    assert doc["findings"]["verdict"] is None
    assert not _schema_valid(doc)


def test_free_text_invalid_program_is_ungrounded_not_substituted(enable_codegen, monkeypatch):
    """Codegen returns a structurally/semantically invalid program (answer not last) -> ungrounded
    with reason 'invalid-program', and Interpreter.run is NEVER called (trust boundary)."""
    invalid = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    enable_codegen(program=invalid)

    calls = {"run": 0}
    orig = server.Interpreter.run

    def spy(self, program, query=None):
        calls["run"] += 1
        return orig(self, program, query=query)

    monkeypatch.setattr(server.Interpreter, "run", spy)

    doc = server.build_free_text_run_doc("x", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert calls["run"] == 0, "unvalidated program must never reach Interpreter.run"
    assert not _schema_valid(doc)


def test_free_text_executes_but_does_not_ground(enable_codegen):
    """A valid program whose sampled window misses the cached frames -> count 0 -> ungrounded with
    reason 'no-grounded-answer', and the (validated) program's trace IS present so the UI can show
    where it ran dry (extends D-DR5). Relates to test_degraded_sibling_carries_partial_signal."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 200, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    enable_codegen(program=program)
    doc = server.build_free_text_run_doc("how many players?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "no-grounded-answer"
    assert len(doc["trace"]) > 0, "trace should stream so the failure is legible"
    assert not _schema_valid(doc)


def test_free_text_codegen_error_is_ungrounded_no_leak(enable_codegen):
    """A codegen exception -> ungrounded reason 'codegen-error' (a fixed enum), never the raw
    exception text in the payload (info-leakage guard)."""
    enable_codegen(raise_exc=ValueError("secret Azure endpoint https://internal/x failed"))
    doc = server.build_free_text_run_doc("x", "single-goal")
    assert doc["findings"]["reason"] == "codegen-error"
    assert "secret" not in json.dumps(doc)


# ---- route param validation (extracted plain helper; no TestClient) ---------------------

def test_run_route_requires_exactly_one_of_query_or_query_text():
    from fastapi import HTTPException

    # neither -> 400
    with pytest.raises(HTTPException) as e1:
        server.resolve_run_params(None, None, None)
    assert e1.value.status_code == 400

    # both -> 400
    with pytest.raises(HTTPException) as e2:
        server.resolve_run_params("hero-10-first-goal", "how many?", None)
    assert e2.value.status_code == 400

    # unknown canned query -> 404
    with pytest.raises(HTTPException) as e3:
        server.resolve_run_params("nope", None, None)
    assert e3.value.status_code == 404

    # unknown clip for free text -> 404
    with pytest.raises(HTTPException) as e4:
        server.resolve_run_params(None, "how many?", "no-such-clip")
    assert e4.value.status_code == 404

    # canned ok
    kind, entry = server.resolve_run_params("hero-10-first-goal", None, None)
    assert kind == "canned" and entry["id"] == "hero-10-first-goal"

    # free ok (default clip)
    kind2, payload = server.resolve_run_params(None, "how many players?", None)
    assert kind2 == "free" and payload == ("how many players?", "single-goal")


def test_free_text_rejects_overlong_query_text():
    """query_text over the cap -> 400 BEFORE codegen is called."""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        server.resolve_run_params(None, "x" * (server.QUERY_TEXT_MAX + 1), None)
    assert e.value.status_code == 400


# ---- security: numeric bounding is the OOM trust boundary -------------------------------

def test_validation_rejects_out_of_range_sample_frames():
    """A program with start/end/fps outside the clip bounds yields non-empty validation_errors,
    so it never reaches Interpreter.run (the gate must bound numeric args, not just shapes)."""
    from validate_program import validation_errors
    clip = canned.clip_by_id("single-goal")
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": -5, "end_ms": 9_000_000, "fps": 9999}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    errs = validation_errors({"program": program}, clip=clip)
    assert errs
    assert any("fps" in e for e in errs)
    assert any("duration_ms" in e for e in errs)


def test_free_text_out_of_range_program_is_ungrounded(enable_codegen, monkeypatch):
    """Codegen returns a program whose sample_frames is out of clip bounds -> invalid-program
    ungrounded, and Interpreter.run is never reached."""
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 9_000_000, "fps": 9999}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    enable_codegen(program=program)
    calls = {"run": 0}
    orig = server.Interpreter.run
    monkeypatch.setattr(server.Interpreter, "run",
                        lambda self, p, query=None: (calls.__setitem__("run", calls["run"] + 1), orig(self, p, query=query))[1])
    doc = server.build_free_text_run_doc("how many?", "single-goal")
    assert doc["findings"]["reason"] == "invalid-program"
    assert calls["run"] == 0


# ---- canned clip lookup -----------------------------------------------------------------

def test_canned_clip_by_id_resolves_cache():
    clip = canned.clip_by_id("single-goal")
    assert clip is not None
    cache_path = Path(clip["cache"])
    assert cache_path.exists(), cache_path
    assert canned.clip_by_id("does-not-exist") is None
