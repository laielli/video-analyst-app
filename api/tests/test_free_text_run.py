"""
Free-text path tests — the full ViperGPT thesis end to end (codegen MOCKED via the FakeCodegen
seam in conftest.py; execution replays over the committed hero cache, no network).

These exercise freetext.build_free_text_run_doc + freetext.resolve_run_request DIRECTLY (not a
FastAPI TestClient): fastapi + pytest don't co-resolve in one interpreter (plan §Tests BLOCKER),
so the trust-boundary logic was factored into freetext.py (no fastapi import) to stay testable.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import freetext
import canned
from validate_program import strip_comments

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


def _hero_program():
    return strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))["program"]


def _assert_run_doc_valid(doc):
    errs = list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))
    assert not errs, errs


# --------------------------------------------------------------------------------------
# test_free_text_grounded_returns_live_source
# --------------------------------------------------------------------------------------

def test_free_text_grounded_returns_live_source(fake_codegen):
    fake_codegen(program=_hero_program(), enabled=True)
    doc = freetext.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    assert doc["program_source"] == "live"
    assert doc["findings"]["answer"]                  # non-empty
    assert doc["findings"].get("grounded") is not False
    assert doc["findings"]["verdict"] == "Yes — #10 scored the first goal"
    _assert_run_doc_valid(doc)


# --------------------------------------------------------------------------------------
# test_free_text_codegen_disabled_is_ungrounded — the no-fallback contract
# --------------------------------------------------------------------------------------

def test_free_text_codegen_disabled_is_ungrounded(fake_codegen, monkeypatch):
    import codegen
    monkeypatch.setattr(codegen, "enabled", lambda: False)
    doc = freetext.build_free_text_run_doc("How many players are visible?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False
    assert doc["findings"]["reason"] == "codegen-disabled"
    # Crucially NOT the pinned hero run-doc — no silent substitution for free text.
    assert doc["program"] == []
    assert "#10 scored the first goal" not in json.dumps(doc)
    _assert_run_doc_valid(doc)


# --------------------------------------------------------------------------------------
# test_free_text_invalid_program_is_ungrounded_not_substituted — TRUST BOUNDARY
# --------------------------------------------------------------------------------------

def test_free_text_invalid_program_is_ungrounded_not_substituted(fake_codegen, monkeypatch):
    # Codegen returns a structurally/semantically invalid program (last step is NOT answer).
    bad = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]
    fake_codegen(program=bad, enabled=True)

    # Spy: Interpreter.run must NEVER be called for an invalid program (unvalidated code can't run).
    import interpreter as interp_mod
    called = {"run": 0}
    orig_run = interp_mod.Interpreter.run

    def spy_run(self, *a, **kw):
        called["run"] += 1
        return orig_run(self, *a, **kw)

    monkeypatch.setattr(interp_mod.Interpreter, "run", spy_run)

    doc = freetext.build_free_text_run_doc("anything", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert called["run"] == 0, "Interpreter.run ran on an UNVALIDATED program (trust boundary breach)"
    _assert_run_doc_valid(doc)


# --------------------------------------------------------------------------------------
# test_free_text_executes_but_does_not_ground — validated program, no grounding -> trace present
# --------------------------------------------------------------------------------------

def test_free_text_executes_but_does_not_ground(fake_codegen):
    # A valid program whose sampled window MISSES the cached frames (cache only has detect @4625).
    miss = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score the first goal?"}},
    ]
    fake_codegen(program=miss, enabled=True)
    doc = freetext.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "no-grounded-answer"
    # The validated program's trace IS present so the UI can show WHERE it ran dry (D-DR5).
    assert doc["trace"], "validated-but-ungrounded run must stream its trace"
    assert len(doc["program"]) == len(miss)
    _assert_run_doc_valid(doc)


# --------------------------------------------------------------------------------------
# test_free_text_codegen_error_is_ungrounded — generation exception -> ungrounded, no raw text
# --------------------------------------------------------------------------------------

def test_free_text_codegen_error_is_ungrounded(fake_codegen):
    fake_codegen(raise_exc=RuntimeError("azure 500: /secret/path leaked"), enabled=True)
    doc = freetext.build_free_text_run_doc("anything", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "codegen-error"
    # Info-leakage guard: the raw exception text NEVER appears in the run-doc.
    assert "secret" not in json.dumps(doc)
    _assert_run_doc_valid(doc)


# --------------------------------------------------------------------------------------
# test_run_route_requires_exactly_one_of_query_or_query_text (plain helper, no TestClient)
# --------------------------------------------------------------------------------------

def test_run_route_requires_exactly_one_of_query_or_query_text():
    # neither -> 400
    with pytest.raises(freetext.RequestError) as e1:
        freetext.resolve_run_request(None, None)
    assert e1.value.status_code == 400
    # both -> 400
    with pytest.raises(freetext.RequestError) as e2:
        freetext.resolve_run_request("hero-10-first-goal", "how many?")
    assert e2.value.status_code == 400
    # canned only -> ok
    kind, entry = freetext.resolve_run_request("hero-10-first-goal", None)
    assert kind == "canned" and entry["id"] == "hero-10-first-goal"
    # free only -> ok
    kind2, text = freetext.resolve_run_request(None, "how many players?")
    assert kind2 == "free" and text == "how many players?"
    # unknown canned query -> 404
    with pytest.raises(freetext.RequestError) as e3:
        freetext.resolve_run_request("nope", None)
    assert e3.value.status_code == 404


def test_free_text_unknown_clip_is_404(fake_codegen):
    fake_codegen(program=_hero_program(), enabled=True)
    with pytest.raises(freetext.RequestError) as e:
        freetext.build_free_text_run_doc("anything", "does-not-exist")
    assert e.value.status_code == 404


def test_canned_clip_by_id_resolves_cache():
    clip = canned.clip_by_id("single-goal")
    assert clip is not None
    # carries a loadable cache path + the codegen hint (Q2 -> Alt C).
    assert Path(clip["cache"]).exists()
    assert clip.get("hint")
    from interpreter import Cache
    cache = Cache.load(clip["cache"])  # loads without error
    assert cache.clip["id"] == "single-goal"
    # unknown id -> None (mirrors by_id over QUERIES).
    assert canned.clip_by_id("nope") is None


# --------------------------------------------------------------------------------------
# Security / trust-boundary
# --------------------------------------------------------------------------------------

def test_free_text_rejects_overlong_query_text():
    # query_text over the cap -> 400 in resolve_run_request, BEFORE any codegen call.
    long = "x" * (freetext.MAX_QUERY_TEXT + 1)
    with pytest.raises(freetext.RequestError) as e:
        freetext.resolve_run_request(None, long)
    assert e.value.status_code == 400
    # No FakeCodegen was ever constructed (cap is enforced before build_free_text_run_doc).
    from conftest import FakeCodegen
    assert FakeCodegen.instances == []


def test_validation_rejects_out_of_range_sample_frames():
    from validate_program import validation_errors
    clip = canned.CLIPS["single-goal"]
    # A schema-valid program whose sample window blows past the clip duration (OOM vector).
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 10_000_000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    # Without clip bounds it passes the structural+semantic gate...
    assert validation_errors({"program": program}) == []
    # ...but WITH the clip it's rejected, so it never reaches Interpreter.run (trust boundary).
    errs = validation_errors({"program": program}, clip=clip)
    assert errs and any("duration" in e for e in errs)

    # fps out of range too.
    bad_fps = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 9999}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    assert any("fps" in e for e in validation_errors({"program": bad_fps}, clip=clip))


def test_free_text_invalid_program_via_out_of_range_is_ungrounded(fake_codegen):
    # End-to-end: codegen returns a program that validates structurally but exceeds the clip
    # bounds -> ungrounded with reason invalid-program (the numeric-bounds gate caught it).
    oversized = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 10_000_000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    fake_codegen(program=oversized, enabled=True)
    doc = freetext.build_free_text_run_doc("how many?", "single-goal")
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"


def test_free_text_codegen_budget_caps_calls(fake_codegen, monkeypatch):
    # The per-process budget stops unbounded paid codegen calls on unauthenticated GETs.
    monkeypatch.setattr(freetext, "CODEGEN_CALL_BUDGET", 2)
    inst = fake_codegen(program=_hero_program(), enabled=True)
    freetext.build_free_text_run_doc("q1", "single-goal")
    freetext.build_free_text_run_doc("q2", "single-goal")
    # Third call is over budget -> ungrounded codegen-error, generate() not invoked a 3rd time.
    doc3 = freetext.build_free_text_run_doc("q3", "single-goal")
    assert doc3["program_source"] == "ungrounded"
    assert doc3["findings"]["reason"] == "codegen-error"
    assert inst.calls == 2
