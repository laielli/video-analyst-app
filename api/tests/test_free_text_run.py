"""
Server-level tests for the free-text branch (D-DR4): build_free_text_run_doc + the route
param-check helper. Codegen is mocked via the `codegen_seam` fixture (conftest.FakeCodegen);
execution replays over the committed hero cache. No Azure, no network, no TestClient — the
param-check is extracted into a plain helper (resolve_run_request) so it is unit-testable under
either interpreter (the FastAPI TestClient import seam is avoided per the plan's BLOCKER note).

Pattern mirror: test_baseline_harness.py (import server + drive helpers) and
test_precompute.py (replay over the hero cache).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import server
import canned
from interpreter import Cache

API_DIR = Path(__file__).resolve().parent.parent
HERO_PROGRAM = API_DIR / "examples" / "hero_program.json"
HERO_VERDICT = "Yes — #10 scored the first goal"


def _hero_program():
    from validate_program import strip_comments
    return strip_comments(json.loads(HERO_PROGRAM.read_text()))["program"]


@pytest.fixture
def clip():
    return canned.clip_by_id("single-goal")


# ---- grounded path --------------------------------------------------------------------------

def test_free_text_grounded_returns_live_source(codegen_seam, clip):
    """Codegen mocked to return the hero program -> grounded, program_source='live'."""
    codegen_seam.enable(program=_hero_program())
    doc = server.build_free_text_run_doc("Does #10 score the first goal?", clip)
    assert doc["program_source"] == "live"
    assert doc["findings"]["answer"]            # non-empty
    assert doc["findings"].get("grounded") is not False
    assert doc["findings"]["verdict"] == HERO_VERDICT


def test_free_text_count_question_grounds_without_hero_leak(codegen_seam, clip):
    """A count program grounds to the count and NEVER the hero verdict (free-text honesty)."""
    count_prog = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    codegen_seam.enable(program=count_prog)
    doc = server.build_free_text_run_doc("how many players?", clip)
    assert doc["program_source"] == "live"
    assert HERO_VERDICT not in (doc["findings"]["verdict"] or "")
    assert doc["findings"]["answer"] == "5"


# ---- ungrounded paths (no hero substitution) ------------------------------------------------

def test_free_text_codegen_disabled_is_ungrounded(codegen_seam, clip):
    """No creds -> ungrounded with reason='codegen-disabled', NOT the pinned hero run-doc."""
    codegen_seam.disable()
    doc = server.build_free_text_run_doc("Does #10 score the first goal?", clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False
    assert doc["findings"]["reason"] == "codegen-disabled"
    # It must NOT equal the pinned hero run-doc (no fallback substitution).
    cache = Cache.load(clip["cache"])
    from interpreter import Interpreter
    hero_doc = Interpreter(cache).run(_hero_program())
    assert doc["findings"].get("verdict") != hero_doc["findings"].get("verdict")
    assert doc["program"] == []  # no program produced


def test_free_text_codegen_error_is_ungrounded(codegen_seam, clip):
    """Codegen raising -> ungrounded reason='codegen-error', and raw exception text is NOT leaked."""
    codegen_seam.enable(raise_exc=RuntimeError("/secret/path Azure 500 boom"))
    doc = server.build_free_text_run_doc("anything", clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "codegen-error"
    assert "secret" not in json.dumps(doc)  # no exception text in the payload


def test_free_text_invalid_program_is_ungrounded_not_substituted(codegen_seam, clip, monkeypatch):
    """Codegen returns an invalid program (last step not 'answer') -> validation rejects it ->
    ungrounded reason='invalid-program', and Interpreter.run is NEVER called (trust boundary)."""
    bad = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}}]
    codegen_seam.enable(program=bad)

    import interpreter.interpreter as ii
    called = {"run": False}
    orig = ii.Interpreter.run

    def spy(self, *a, **k):
        called["run"] = True
        return orig(self, *a, **k)
    monkeypatch.setattr(ii.Interpreter, "run", spy)

    doc = server.build_free_text_run_doc("anything", clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["run"] is False, "unvalidated program must never reach the interpreter"


def test_free_text_executes_but_does_not_ground(codegen_seam, clip):
    """A valid program whose read_text finds nothing legible -> texts empty -> ungrounded
    reason='no-grounded-answer', and the (validated) program's trace IS present so the UI can
    show where it ran dry (extends D-DR5)."""
    miss = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "result", "op": "answer", "args": {"from": "numbers", "question": "what number?"}},
    ]
    codegen_seam.enable(program=miss)
    doc = server.build_free_text_run_doc("what number?", clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "no-grounded-answer"
    assert len(doc["trace"]) > 0, "the validated program's trace must stream for diagnostics"


def test_ungrounded_run_doc_validates_against_schema(codegen_seam, clip):
    """Every ungrounded run-doc shape validates against the extended run_doc.schema.json."""
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    V = jsonschema.Draft202012Validator(rd_schema)

    codegen_seam.disable()
    disabled = server.build_free_text_run_doc("q", clip)
    assert not list(V.iter_errors(disabled)), list(V.iter_errors(disabled))


# ---- route param-check helper (no TestClient) -----------------------------------------------

def test_run_route_requires_exactly_one_of_query_or_query_text():
    """resolve_run_request: neither -> 400, both -> 400, unknown query -> 404, unknown clip -> 404."""
    def status_of(query, query_text, clip):
        try:
            server.resolve_run_request(query, query_text, clip)
            return None
        except server.RequestError as e:
            return e.status

    assert status_of(None, None, None) == 400          # neither
    assert status_of("q", "qt", None) == 400           # both
    assert status_of("does-not-exist", None, None) == 404
    assert status_of(None, "how many?", "no-such-clip") == 404

    # happy paths resolve.
    assert server.resolve_run_request("hero-10-first-goal", None, None)["kind"] == "canned"
    free = server.resolve_run_request(None, "how many players?", "single-goal")
    assert free["kind"] == "free" and free["clip"]["id"] == "single-goal"


def test_free_text_rejects_overlong_query_text():
    """query_text over the cap -> 400 BEFORE any codegen call (no FakeCodegen instance made)."""
    from conftest import FakeCodegen
    FakeCodegen.instances = []
    too_long = "a" * (server.MAX_QUERY_TEXT_LEN + 1)
    try:
        server.resolve_run_request(None, too_long, "single-goal")
        assert False, "should have raised RequestError(400)"
    except server.RequestError as e:
        assert e.status == 400
    assert FakeCodegen.instances == [], "codegen must not be invoked for an over-cap query_text"


def test_free_text_budget_exhausted_is_ungrounded(codegen_seam, clip, monkeypatch):
    """When the per-process codegen budget is spent, free text is ungrounded (abuse cap)."""
    codegen_seam.enable(program=_hero_program())
    monkeypatch.setattr(server, "_codegen_calls", server.CODEGEN_CALL_BUDGET, raising=False)
    doc = server.build_free_text_run_doc("anything", clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "codegen-budget-exhausted"


# ---- validation trust boundary (numeric bounds) ---------------------------------------------

def test_validation_rejects_out_of_range_sample_frames(clip):
    """A program with start/end/fps outside the clip bounds yields non-empty validation_errors,
    so it never reaches Interpreter.run (a *validated* program can't OOM op_sample_frames)."""
    from validate_program import validation_errors
    oob = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 9_999_999, "fps": 99}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]}
    # No clip -> bounds not enforced (canned path keeps prior behavior).
    assert validation_errors(oob) == []
    # With the clip -> end > duration and fps > 30 are both flagged.
    errs = validation_errors(oob, clip=clip)
    assert errs and any("end_ms" in e for e in errs) and any("fps" in e for e in errs)


def test_free_text_oob_program_is_ungrounded_invalid(codegen_seam, clip, monkeypatch):
    """An out-of-range (but structurally valid) program from codegen -> invalid-program ungrounded,
    Interpreter.run never called (the numeric-bounds gate is part of the trust boundary)."""
    oob = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 9_999_999, "fps": 99}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    codegen_seam.enable(program=oob)
    import interpreter.interpreter as ii
    called = {"run": False}
    orig = ii.Interpreter.run
    monkeypatch.setattr(ii.Interpreter, "run", lambda self, *a, **k: called.__setitem__("run", True) or orig(self, *a, **k))
    doc = server.build_free_text_run_doc("q", clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["run"] is False


# ---- canned clip lookup ---------------------------------------------------------------------

def test_canned_clip_by_id_resolves_cache():
    """clip_by_id('single-goal') carries a loadable cache path; unknown id -> None."""
    clip = canned.clip_by_id("single-goal")
    assert clip is not None
    cache = Cache.load(clip["cache"])           # loads without error
    assert cache.clip["id"] == "single-goal"
    assert canned.clip_by_id("nope") is None
    # public_clip drops the server-internal cache path.
    assert "cache" not in canned.public_clip(clip)
