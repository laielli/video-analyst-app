"""
Server-level tests for the free-text branch (codegen mocked via the FakeCodegen seam in
conftest.py; execution replays over the committed hero cache). Asserts the no-fallback contract:
a novel question that can't ground resolves to an honest ungrounded run-doc — NEVER a hero-program
substitution. Mirror pattern: test_baseline_harness.py (import server-side helpers and call them).

Route param/clip checks use the plain `resolve_run_request` helper (no FastAPI TestClient), per
the plan's BLOCKER note that TestClient route tests import under neither interpreter cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
import canned
import jsonschema
from interpreter import Interpreter

API_DIR = Path(__file__).resolve().parent.parent
RUN_DOC_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


@pytest.fixture(autouse=True)
def _reset_codegen_budget():
    """Each test starts with a fresh per-process codegen-call budget."""
    server._codegen_calls = 0
    yield
    server._codegen_calls = 0


def _assert_valid_run_doc(doc):
    errs = list(jsonschema.Draft202012Validator(RUN_DOC_SCHEMA).iter_errors(doc))
    assert not errs, errs


# --------------------------------------------------------------------------------------
# grounded free-text -> live source
# --------------------------------------------------------------------------------------

def test_free_text_grounded_returns_live_source(fake_codegen, hero_program):
    """Codegen mocked to return the committed hero program + hero cache -> program_source 'live',
    a non-empty answer, grounded not False. Mirrors test_hero_cache_replays_to_grounded_answer."""
    fake_codegen(program=hero_program, enabled=True)
    doc = server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    assert doc["program_source"] == "live"
    assert doc["findings"]["answer"]
    assert doc["findings"].get("grounded") is not False
    assert doc["findings"]["verdict"] == "Yes — #10 scored the first goal"
    # the streamed query reflects the user's words, not the canned text.
    assert doc["query"] == "Does #10 score the first goal?"
    _assert_valid_run_doc(doc)


# --------------------------------------------------------------------------------------
# codegen disabled -> ungrounded (no fallback)
# --------------------------------------------------------------------------------------

def test_free_text_codegen_disabled_is_ungrounded(fake_codegen):
    fake_codegen(enabled=False)
    doc = server.build_free_text_run_doc("how many players?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False
    assert doc["findings"]["reason"] == "codegen-disabled"
    # the no-fallback contract: this must NOT equal the pinned hero run-doc.
    hero_entry = canned.by_id("hero-10-first-goal")
    hero_doc = server.build_run_doc(hero_entry)
    assert doc["program"] != hero_doc["program"]
    assert doc["findings"].get("verdict") != hero_doc["findings"].get("verdict")
    _assert_valid_run_doc(doc)


# --------------------------------------------------------------------------------------
# invalid program -> ungrounded, interpreter never reached (trust boundary)
# --------------------------------------------------------------------------------------

def test_free_text_invalid_program_is_ungrounded_not_substituted(fake_codegen, monkeypatch):
    """Codegen returns a structurally/semantically invalid program (last step not `answer`).
    validation_errors is non-empty -> ungrounded reason 'invalid-program', and Interpreter.run is
    NEVER called (no unvalidated code executes). Mirrors the validate_program gate intent."""
    invalid = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]  # no answer step -> semantic error "last step must be op 'answer'"
    fake_codegen(program=invalid, enabled=True)

    calls = {"run": 0}
    real_run = Interpreter.run

    def spy_run(self, program, query=None):
        calls["run"] += 1
        return real_run(self, program, query=query)

    monkeypatch.setattr(Interpreter, "run", spy_run)

    doc = server.build_free_text_run_doc("nonsense", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert calls["run"] == 0, "unvalidated program must never reach Interpreter.run"
    _assert_valid_run_doc(doc)


def test_validation_rejects_out_of_range_sample_frames(fake_codegen, monkeypatch):
    """A program whose sample_frames window/fps exceed the clip bounds yields non-empty
    validation_errors -> ungrounded invalid-program, never reaching Interpreter.run (the numeric
    trust boundary — a structurally valid program can still OOM via range expansion)."""
    oob = [
        # end_ms 999999 > clip.duration_ms (5067); fps 9999 > 30 -> would OOM at execution.
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 999999, "fps": 9999}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    fake_codegen(program=oob, enabled=True)

    calls = {"run": 0}
    real_run = Interpreter.run
    monkeypatch.setattr(Interpreter, "run", lambda self, p, query=None: (calls.__setitem__("run", calls["run"] + 1) or real_run(self, p, query=query)))

    doc = server.build_free_text_run_doc("how many?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert calls["run"] == 0
    _assert_valid_run_doc(doc)


# --------------------------------------------------------------------------------------
# executes but doesn't ground -> ungrounded with trace present
# --------------------------------------------------------------------------------------

def test_free_text_executes_but_does_not_ground(fake_codegen):
    """A valid program whose sampled window misses the cached frames -> partial/empty answer ->
    ungrounded reason 'no-grounded-answer', and the (validated) program's trace IS present so the
    UI can show where it ran dry. Relates to test_degraded_sibling_carries_partial_signal."""
    # Sample a window with NO cached detections (cache only has ts=4625); count -> answer.
    miss = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 500, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players early on?"}},
    ]
    fake_codegen(program=miss, enabled=True)
    doc = server.build_free_text_run_doc("how many players early on?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "no-grounded-answer"
    # the validated program's trace is present (so the UI shows where it ran dry).
    assert doc["trace"], "ungrounded-after-execution must carry the trace"
    assert any(t["op"] == "detect" for t in doc["trace"])
    _assert_valid_run_doc(doc)


# --------------------------------------------------------------------------------------
# budget guard
# --------------------------------------------------------------------------------------

def test_free_text_codegen_budget_is_bounded(fake_codegen, monkeypatch):
    """Once the per-process codegen budget is exhausted, further free-text calls ground out as
    'budget-exceeded' WITHOUT invoking codegen — bounding paid Azure calls on unauthenticated GETs."""
    fake = fake_codegen(program=[], enabled=True)
    monkeypatch.setattr(server, "MAX_CODEGEN_CALLS", 0)
    doc = server.build_free_text_run_doc("how many?", "single-goal")
    assert doc["findings"]["reason"] == "budget-exceeded"
    assert fake.calls == 0, "no codegen call once the budget is exhausted"


# --------------------------------------------------------------------------------------
# route param validation (plain helper — no TestClient)
# --------------------------------------------------------------------------------------

def test_run_route_requires_exactly_one_of_query_or_query_text():
    # neither -> 400
    kind, status, _ = server.resolve_run_request(None, None)
    assert status == 400 and kind is None
    # both -> 400
    kind, status, _ = server.resolve_run_request("hero-10-first-goal", "how many?")
    assert status == 400 and kind is None
    # canned only -> ok
    kind, status, _ = server.resolve_run_request("hero-10-first-goal", None)
    assert kind == "canned" and status is None
    # free only -> ok
    kind, status, _ = server.resolve_run_request(None, "how many?")
    assert kind == "free" and status is None


def test_free_text_rejects_overlong_query_text():
    kind, status, detail = server.resolve_run_request(None, "x" * (server.MAX_QUERY_TEXT_LEN + 1))
    assert status == 400 and kind is None
    assert "query_text" in detail


def test_unknown_clip_is_404_via_resolve_doc(fake_codegen):
    """The route must raise 404 for an unknown clip (clip_by_id returns None — the '# 404 upstream'
    comment in the plan pseudocode is false; the route raises it explicitly)."""
    from fastapi import HTTPException
    fake_codegen(enabled=False)
    with pytest.raises(HTTPException) as ei:
        server._resolve_doc(None, "how many?", "no-such-clip")
    assert ei.value.status_code == 404


# --------------------------------------------------------------------------------------
# canned clip_by_id
# --------------------------------------------------------------------------------------

def test_canned_clip_by_id_resolves_cache():
    clip = canned.clip_by_id("single-goal")
    assert clip is not None
    assert Path(clip["cache"]).exists()
    assert canned.clip_by_id("does-not-exist") is None
