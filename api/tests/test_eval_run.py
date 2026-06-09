"""Runner / CLI tests — the coverage-discipline layer for run_eval.py + seed_fixtures.py.

Covers: replay over the committed seeds, seed regeneration, the threshold gate (pass/fail), the
uniform-vs-partial stale rule (D6 anti-gaming), creds-gating, capture-never-touches-server, and
replay-never-imports-codegen. Drives the capture argparse branches through the mock_codegen seam.
"""
from __future__ import annotations

import json

import pytest

import canned
from eval import bank, run_eval, scoring, seed_fixtures
from eval.fixtures import Fixture


# ---- replay over committed seeds ---------------------------------------------------------------

def test_replay_over_committed_seeds_scores(tmp_path, capsys):
    """Run replay in-process over the COMMITTED seed fixtures; every in-scope seed reaches its
    target tier, out-of-scope all pass, exit 0. (A bank case without a seed would surface MISSING.)"""
    args = _replay_args(report=tmp_path / "r.md", json_path=tmp_path / "r.json")
    code = run_eval.run_replay(args)
    assert code == 0
    obj = json.loads((tmp_path / "r.json").read_text())
    funnel = obj["summary"]["funnel"]
    assert funnel["total"] > 0
    # Every in-scope groundable seed reaches T4 (the pinned templates ground correctly).
    assert funnel["counts"][scoring.T4_CORRECT] == funnel["total"]
    assert obj["summary"]["out_of_scope_honest"]["rate"] == 1.0
    assert obj["summary"]["adversarial_safe"]["rate"] == 1.0
    assert obj["summary"]["missing"] == 0 and obj["summary"]["stale"] == 0


def test_replay_filters_by_shape(tmp_path):
    # no_gate: a filtered replay can't satisfy the full min_fixtures/out-of-scope floors — this test
    # exercises the filter, not the gate (the gate is covered by test_threshold_gate_*).
    args = _replay_args(report=tmp_path / "r.md", json_path=tmp_path / "r.json", shape="count",
                        no_gate=True)
    code = run_eval.run_replay(args)
    assert code == 0
    obj = json.loads((tmp_path / "r.json").read_text())
    assert all(c["shape"] == "count" for c in obj["cases"])


def test_replay_no_gate(tmp_path):
    args = _replay_args(report=tmp_path / "r.md", json_path=tmp_path / "r.json", no_gate=True)
    assert run_eval.run_replay(args) == 0


def test_seed_fixtures_regenerate_match():
    """Regenerate every seed into a tmp dir and compare to the committed fixtures, modulo the
    prompt_version + captured_at stamps. A bank edit without re-seeding fails here."""
    import tempfile
    from pathlib import Path

    cases = bank.load_cases()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        seed_fixtures.generate_seeds(cases, fixtures_dir=tmp)
        for case in cases:
            cid = case["case_id"]
            committed = json.loads((seed_fixtures.FIXTURES_DIR / f"{cid}.json").read_text())
            regen = json.loads((tmp / f"{cid}.json").read_text())
            # Compare modulo the volatile stamps.
            for d_ in (committed, regen):
                d_.pop("prompt_version", None)
                d_.pop("captured_at", None)
            assert committed == regen, f"seed {cid} drifted; re-run seed_fixtures.py"


def test_committed_seeds_carry_current_prompt_version():
    """The committed seeds carry the CURRENT prompt_version (so they are not stale and the gate
    exercises the real ladder)."""
    pv_by_clip = {cid: bank.prompt_version(clip) for cid, clip in canned.CLIPS.items()}
    for case in bank.load_cases():
        committed = json.loads((seed_fixtures.FIXTURES_DIR / f"{case['case_id']}.json").read_text())
        assert committed["prompt_version"] == pv_by_clip[case["clip_id"]], case["case_id"]


# ---- threshold gate (D6) -----------------------------------------------------------------------

def _result(case_id, reached, target, passed, stale=False, missing=False,
            phrasing_class="paraphrase", distance="near"):
    return scoring.CaseResult(
        case_id=case_id, clip_id="bernabeu-counter", shape="count",
        phrasing_class=phrasing_class, distance=distance, target_tier=target,
        reached_tier=reached, passed=passed, stale=stale, missing=missing, detail="d")


def _json_for(results):
    from eval import report
    _, obj = report.build_report(results, {"bank_version": 1})
    return obj


def test_threshold_gate_fails_on_regression():
    # Two in-scope cases, one only reaches T2 -> T4 rate 0.5 < floor 0.85 -> FAIL.
    results = [
        _result("a", scoring.T4_CORRECT, scoring.T4_CORRECT, True),
        _result("b", scoring.T2_SEMANTIC, scoring.T4_CORRECT, False),
    ]
    obj = _json_for(results)
    code, msg = run_eval.apply_gate(results, obj, {"T4_answer_correct": 0.85, "min_fixtures": 0,
                                                   "max_stale": 0}, {})
    assert code == 1 and "T4_answer_correct" in msg

    # Both at T4 -> rate 1.0 >= floor -> PASS.
    results2 = [
        _result("a", scoring.T4_CORRECT, scoring.T4_CORRECT, True),
        _result("b", scoring.T4_CORRECT, scoring.T4_CORRECT, True),
    ]
    code2, _ = run_eval.apply_gate(results2, _json_for(results2),
                                   {"T4_answer_correct": 0.85, "min_fixtures": 0, "max_stale": 0}, {})
    assert code2 == 0


def test_threshold_gate_min_fixtures_floor():
    results = [_result("a", scoring.T4_CORRECT, scoring.T4_CORRECT, True)]
    code, msg = run_eval.apply_gate(results, _json_for(results),
                                    {"min_fixtures": 57, "max_stale": 0}, {})
    assert code == 1 and "min_fixtures" in msg


def test_stale_handling_uniform_vs_partial():
    # ALL stale -> advisory downgrade (exit 0).
    all_stale = [
        _result("a", scoring.T4_CORRECT, scoring.T4_CORRECT, True, stale=True),
        _result("b", scoring.T4_CORRECT, scoring.T4_CORRECT, True, stale=True),
    ]
    code, msg = run_eval.apply_gate(all_stale, _json_for(all_stale),
                                    {"T4_answer_correct": 0.85, "min_fixtures": 0, "max_stale": 0}, {})
    assert code == 0 and "ADVISORY" in msg and "STALE" in msg

    # ONE stale among fresh -> partial staleness > max_stale -> FAIL (anti-gaming).
    partial = [
        _result("a", scoring.T4_CORRECT, scoring.T4_CORRECT, True, stale=False),
        _result("b", scoring.T4_CORRECT, scoring.T4_CORRECT, True, stale=True),
    ]
    code2, msg2 = run_eval.apply_gate(partial, _json_for(partial),
                                      {"T4_answer_correct": 0.85, "min_fixtures": 0, "max_stale": 0}, {})
    assert code2 == 1 and "max_stale" in msg2


def test_gate_fails_on_missing_fixture():
    results = [_result("a", scoring.MISSING, scoring.T4_CORRECT, False, missing=True)]
    code, msg = run_eval.apply_gate(results, _json_for(results), {"min_fixtures": 0, "max_stale": 0}, {})
    assert code == 1 and "MISSING" in msg


def test_gate_out_of_scope_honest_floor():
    # An out-of-scope case that FAILS (overclaim) -> oos rate 0.0 < floor -> FAIL.
    results = [_result("oos", scoring.T3_GROUNDED, scoring.REJECTED, False,
                       phrasing_class="out_of_scope", distance="n/a")]
    code, msg = run_eval.apply_gate(results, _json_for(results),
                                    {"out_of_scope_honest": 0.9, "min_fixtures": 0, "max_stale": 0}, {})
    assert code == 1 and "out_of_scope_honest" in msg


def test_fail_under_override_parsing():
    assert run_eval._parse_fail_under(["T2_semantic_valid=1.0", "T4_answer_correct=0.5"]) == {
        "T2_semantic_valid": 1.0, "T4_answer_correct": 0.5}
    with pytest.raises(SystemExit):
        run_eval._parse_fail_under(["bogus"])


# ---- capture (mocked seam) ---------------------------------------------------------------------

def test_capture_is_creds_gated(monkeypatch, tmp_path):
    """capture with codegen.enabled() False -> exit 2, writes no fixture, never constructs client."""
    import codegen

    monkeypatch.setattr(codegen, "enabled", lambda: False)

    def boom(cls):
        raise AssertionError("client must not be constructed when creds are missing")

    monkeypatch.setattr(codegen.AzureCodegen, "from_env", classmethod(boom))
    monkeypatch.setattr(run_eval, "FIXTURES_DIR_unused", tmp_path, raising=False)
    code = run_eval.run_capture(_capture_args(only="bernabeu-count-canonical", limit=1))
    assert code == 2


def test_capture_dry_run_makes_no_calls(monkeypatch, mock_codegen, capsys):
    """--dry-run prints a cost estimate and makes no calls (the informed-consent step)."""
    fake = mock_codegen(program=[{"id": "r", "op": "answer", "args": {}}], enabled=True)
    code = run_eval.run_capture(_capture_args(dry_run=True, limit=5))
    assert code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "live call" in out
    from tests.conftest import FakeCodegen
    assert FakeCodegen.total_calls == 0


def test_capture_writes_fixture_and_never_touches_server(monkeypatch, mock_codegen, tmp_path):
    """capture one case via the mock seam: writes a fixture with raw_program + usage:null, and
    server.build_free_text_run_doc is never called. Also assert no eval module imports server."""
    import server

    program = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "p"}},
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    mock_codegen(program=program, enabled=True)

    def boom(*a, **k):
        raise AssertionError("capture must NEVER call server.build_free_text_run_doc")

    monkeypatch.setattr(server, "build_free_text_run_doc", boom)
    # Redirect fixture writes to tmp_path.
    monkeypatch.setattr("eval.run_eval.write_fixture",
                        lambda fx, fixtures_dir=tmp_path: __import__("eval.fixtures",
                        fromlist=["write_fixture"]).write_fixture(fx, fixtures_dir=tmp_path))

    code = run_eval.run_capture(_capture_args(only="bernabeu-count-canonical", limit=1, force=True))
    assert code == 0
    from eval import fixtures as fx_mod
    written = fx_mod.load_fixture("bernabeu-count-canonical", fixtures_dir=tmp_path)
    assert written is not None
    assert written.raw_program == program
    assert written.usage is None

    # No eval module imports server (static source check).
    for mod in ("bank", "fixtures", "scoring", "report", "seed_fixtures", "run_eval"):
        src = (canned.API_DIR / "eval" / f"{mod}.py").read_text()
        assert "import server" not in src, f"eval/{mod}.py must not import server"


def test_capture_resume_skips_fresh(monkeypatch, mock_codegen, tmp_path):
    """A case whose fixture is already fresh is resume-skipped (no call) unless --force."""
    case_id = "bernabeu-count-canonical"
    case = next(c for c in bank.load_cases() if c["case_id"] == case_id)
    clip = canned.clip_by_id(case["clip_id"])
    pv = bank.prompt_version(clip)
    fx = Fixture(case_id=case_id, clip_id=case["clip_id"], question=case["question"],
                 prompt_version=pv, raw_program=[{"id": "r", "op": "answer", "args": {}}],
                 usage=None)
    from eval import fixtures as fx_mod
    fx_mod.write_fixture(fx, fixtures_dir=tmp_path)
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    monkeypatch.setattr("eval.run_eval.load_fixture",
                        lambda cid: fx_mod.load_fixture(cid, fixtures_dir=tmp_path))

    mock_codegen(program=[{"id": "r", "op": "answer", "args": {}}], enabled=True)
    code = run_eval.run_capture(_capture_args(only=case_id, dry_run=True))
    assert code == 0  # resume-skip: dry-run reports 0 pending


def test_replay_never_imports_codegen(monkeypatch, tmp_path):
    """Monkeypatch codegen.AzureCodegen.from_env to raise; replay must complete (no client built)
    and a FakeCodegen-style counter stays 0."""
    import codegen

    monkeypatch.setattr(codegen.AzureCodegen, "from_env",
                        classmethod(lambda cls: (_ for _ in ()).throw(
                            AssertionError("replay must not construct the Azure client"))))
    from tests.conftest import FakeCodegen
    FakeCodegen.total_calls = 0

    args = _replay_args(report=tmp_path / "r.md", json_path=tmp_path / "r.json")
    code = run_eval.run_replay(args)
    assert code == 0
    assert FakeCodegen.total_calls == 0


def test_main_dispatches(monkeypatch, tmp_path):
    code = run_eval.main(["replay", "--report", str(tmp_path / "r.md"),
                          "--json", str(tmp_path / "r.json"), "--no-gate"])
    assert code == 0


def test_main_capture_dispatch_creds_gated(monkeypatch):
    """main() routes 'capture' to run_capture; with no creds it exits 2 (covers the dispatch arm)."""
    import codegen
    monkeypatch.setattr(codegen, "enabled", lambda: False)
    code = run_eval.main(["capture", "--limit", "1"])
    assert code == 2


def test_replay_reports_missing_fixture(monkeypatch, tmp_path):
    """A bank case whose fixture is absent surfaces as MISSING (run_replay missing branch) and the
    gate fails on it."""
    real_load = run_eval.load_fixture

    def patched(cid):
        if cid == "bernabeu-count-canonical":
            return None
        return real_load(cid)

    monkeypatch.setattr(run_eval, "load_fixture", patched)
    args = _replay_args(report=tmp_path / "r.md", json_path=tmp_path / "r.json")
    code = run_eval.run_replay(args)
    obj = json.loads((tmp_path / "r.json").read_text())
    assert obj["summary"]["missing"] >= 1
    assert code == 1  # MISSING fails the gate


def test_gate_T2_floor_fail():
    results = [
        _result("a", scoring.T1_SCHEMA, scoring.T4_CORRECT, False),
        _result("b", scoring.T4_CORRECT, scoring.T4_CORRECT, True),
    ]
    code, msg = run_eval.apply_gate(results, _json_for(results),
                                    {"T2_semantic_valid": 1.0, "min_fixtures": 0, "max_stale": 0}, {})
    assert code == 1 and "T2_semantic_valid" in msg


def test_estimate_cost_and_prompt_chars():
    assert run_eval._estimate_cost(60) > 0
    chars = run_eval._prompt_chars()
    assert set(chars) == {"single-goal", "bernabeu-counter"}
    assert all(v > 0 for v in chars.values())


# ---- helpers -----------------------------------------------------------------------------------

def _replay_args(report=None, json_path=None, shape=None, clip=None, no_gate=False, fail_under=None):
    import argparse
    return argparse.Namespace(mode="replay", report=str(report) if report else None,
                              json=str(json_path) if json_path else None, shape=shape, clip=clip,
                              fail_under=fail_under or [], no_gate=no_gate)


def _capture_args(only=None, limit=None, force=False, dry_run=False):
    import argparse
    return argparse.Namespace(mode="capture", only=only, limit=limit, force=force, dry_run=dry_run)
