"""Runner / CLI tests — the coverage-critical layer. Drives replay over the committed seeds, the
seed regeneration, the threshold gate, stale handling, capture creds-gating, and the
no-server-touch / no-codegen-touch invariants. Creds-free; uses the mock_codegen seam from
conftest for the capture path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = API_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import bank  # noqa: E402
import canned  # noqa: E402
import fixtures as fx  # noqa: E402
import run_eval  # noqa: E402
import scoring  # noqa: E402
import seed_fixtures  # noqa: E402


# ---- replay over committed seeds ---------------------------------------------------------

def test_replay_over_committed_seeds_scores():
    cases = bank.load_bank()
    results = run_eval.replay(cases)
    by_id = {r.case_id: r for r in results}
    assert len(results) == len(cases)
    # No MISSING / STALE seeds (every bank case has a fresh committed seed).
    assert not any(r.missing for r in results), "a bank case lacks a committed seed"
    assert not any(r.stale for r in results), "a committed seed is stale (re-seed)"
    # Every in-scope groundable seed reaches its target tier.
    for r in results:
        if r.out_of_scope or r.safety_axis:
            continue
        assert scoring.TIER_RANK[r.reached_tier] >= scoring.TIER_RANK[r.target_tier], \
            f"{r.case_id} reached {r.reached_tier}, target {r.target_tier}"
    # Spot-check the four pinned answers.
    assert by_id["bernabeu-count-canonical"].answer == "5"
    assert by_id["single-goal-first-goal-canonical"].answer == "Yes"
    assert by_id["bernabeu-scorer-number-canonical"].answer == "7"


def test_seed_fixtures_regenerate_match(tmp_path):
    """Regenerate all seeds into a tmpdir; assert byte-for-byte match with the committed seeds
    modulo the captured_at/model sentinels (which are fixed, so they match too). A bank edit
    without re-seeding fails here."""
    seed_fixtures.generate_all(fixtures_dir=tmp_path)
    committed_dir = EVAL_DIR / "fixtures"
    regenerated = sorted(tmp_path.glob("*.json"))
    committed = sorted(committed_dir.glob("*.json"))
    assert {p.name for p in regenerated} == {p.name for p in committed}, \
        "regenerated seed set differs from committed (re-run seed_fixtures.generate_all and commit)"
    for p in regenerated:
        a = json.loads(p.read_text())
        b = json.loads((committed_dir / p.name).read_text())
        assert a == b, f"committed seed {p.name} differs from regeneration"


# ---- threshold gate ----------------------------------------------------------------------

def _result(case_id, reached, passed, out_of_scope=False, stale=False, missing=False):
    return scoring.CaseResult(
        case_id=case_id, target_tier=scoring.T4, reached_tier=reached, passed=passed,
        detail="d", out_of_scope=out_of_scope, stale=stale, missing=missing,
        shape="count", clip_id="bernabeu-counter", phrasing_class="canonical", distance="canonical",
    )


def test_threshold_gate_fails_on_regression():
    thresholds = {"T2_semantic_valid": 1.0, "T4_answer_correct": 0.85, "min_fixtures": 2, "max_stale": 0}
    # Below the T4 floor: 1/2 correct = 0.50 < 0.85.
    below = [_result("a", scoring.T4, True), _result("b", scoring.T2, False)]
    code, msgs = run_eval.apply_gate(below, thresholds)
    assert code == 1 and any("T4_answer_correct" in m for m in msgs)
    # Above the floor: both correct.
    above = [_result("a", scoring.T4, True), _result("b", scoring.T4, True)]
    code, msgs = run_eval.apply_gate(above, thresholds)
    assert code == 0 and any("GATE PASS" in m for m in msgs)


def test_gate_min_fixtures_catches_deletion():
    thresholds = {"min_fixtures": 3, "max_stale": 0}
    results = [_result("a", scoring.T4, True), _result("b", scoring.T4, True),
               _result("c", scoring.MISSING, False, missing=True)]
    code, msgs = run_eval.apply_gate(results, thresholds)
    assert code == 1 and any("min_fixtures" in m for m in msgs)


def test_stale_handling_uniform_vs_partial():
    thresholds = {"T2_semantic_valid": 1.0, "min_fixtures": 1, "max_stale": 0}
    # ALL stale -> advisory downgrade (exit 0).
    all_stale = [_result("a", scoring.STALE, False, stale=True),
                 _result("b", scoring.STALE, False, stale=True)]
    code, msgs = run_eval.apply_gate(all_stale, thresholds)
    assert code == 0 and any("ADVISORY" in m and "STALE" in m for m in msgs)
    # ONE stale among fresh -> partial staleness > max_stale -> FAIL.
    partial = [_result("a", scoring.T4, True), _result("b", scoring.STALE, False, stale=True)]
    code, msgs = run_eval.apply_gate(partial, thresholds)
    assert code == 1 and any("STALE" in m and "max_stale" in m for m in msgs)


def test_out_of_scope_honest_floor():
    thresholds = {"out_of_scope_honest": 0.9, "min_fixtures": 1, "max_stale": 0}
    # 1/2 honest = 0.50 < 0.90 -> FAIL.
    oos = [_result("a", scoring.T2, True, out_of_scope=True),
           _result("b", scoring.T3, False, out_of_scope=True)]
    code, msgs = run_eval.apply_gate(oos, thresholds)
    assert code == 1 and any("out_of_scope_honest" in m for m in msgs)


# ---- CLI argparse branches ---------------------------------------------------------------

def test_cli_replay_writes_report_and_gates(tmp_path):
    md = tmp_path / "r.md"
    js = tmp_path / "r.json"
    code = run_eval.main(["replay", "--report", str(md), "--json", str(js)])
    assert code == 0  # committed seeds pass the committed thresholds
    assert md.exists() and js.exists()
    j = json.loads(js.read_text())
    assert j["summary"]["total_cases"] == len(bank.load_bank())


def test_cli_replay_no_gate_and_filters(tmp_path, capsys):
    code = run_eval.main(["replay", "--report", str(tmp_path / "r.md"),
                          "--json", str(tmp_path / "r.json"), "--shape", "count", "--no-gate"])
    assert code == 0
    j = json.loads((tmp_path / "r.json").read_text())
    # only count cases scored
    assert all(c["shape"] == "count" for c in j["cases"])


def test_cli_replay_fail_under_override(tmp_path):
    # An impossible T4 floor forces a gate failure even on passing seeds.
    code = run_eval.main(["replay", "--report", str(tmp_path / "r.md"),
                          "--json", str(tmp_path / "r.json"),
                          "--fail-under", "T4_answer_correct=2.0"])
    assert code == 1


def test_fail_under_parse_rejects_bad_format():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        run_eval._parse_fail_under(["T4nope"])


# ---- capture: creds-gated, never touches server, never bills under dry-run ---------------

def test_capture_is_creds_gated(monkeypatch, tmp_path):
    """capture with codegen.enabled() False -> exit 2, writes no fixture, never constructs client."""
    import codegen

    monkeypatch.setattr(codegen, "enabled", lambda: False)

    def boom(cls):
        raise AssertionError("client constructed despite no creds")

    monkeypatch.setattr(codegen.AzureCodegen, "from_env", classmethod(boom))
    monkeypatch.setattr(fx, "FIXTURES_DIR", tmp_path)
    args = run_eval.build_parser().parse_args(["capture", "--limit", "1"])
    code = run_eval.cmd_capture(args)
    assert code == 2
    assert not list(tmp_path.glob("*.json"))


def test_capture_dry_run_makes_no_calls(monkeypatch, tmp_path, mock_codegen, capsys):
    mock_codegen(program=[], enabled=True)
    import codegen

    monkeypatch.setattr(fx, "FIXTURES_DIR", tmp_path)
    args = run_eval.build_parser().parse_args(["capture", "--only", "bernabeu-count-canonical", "--dry-run"])
    code = run_eval.cmd_capture(args)
    assert code == 0
    assert not list(tmp_path.glob("*.json"))  # dry-run writes nothing
    from tests.conftest import FakeCodegen
    assert FakeCodegen.total_calls == 0  # no live calls under dry-run
    err = capsys.readouterr().err
    assert "will make" in err and "live calls" in err


def test_capture_writes_fixture_via_mock_seam(monkeypatch, tmp_path, mock_codegen):
    """capture one case via the mock_codegen seam writes a fixture with raw_program + usage:null,
    and resume-skips it on a second run."""
    program = [{"id": "result", "op": "answer", "args": {"from": "x", "question": "q"}}]
    mock_codegen(program=program, enabled=True)
    monkeypatch.setattr(fx, "FIXTURES_DIR", tmp_path)
    # Patch the live block's _capture_one to use the mocked client path (it is # pragma: no cover).
    import codegen

    def fake_capture_one(client, case, clip, prompt_version):
        raw = client.generate(case["question"], clip=clip)
        return fx.Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                          prompt_version=prompt_version, captured_at="t", model="mock",
                          raw_program=raw, usage=None, error=None)

    monkeypatch.setattr(run_eval, "_capture_one", fake_capture_one)
    args = run_eval.build_parser().parse_args(["capture", "--only", "bernabeu-count-canonical"])
    code = run_eval.cmd_capture(args)
    assert code == 0
    loaded = fx.load_fixture("bernabeu-count-canonical", fixtures_dir=tmp_path)
    assert loaded is not None and loaded.raw_program == program and loaded.usage is None


def test_capture_never_touches_server(monkeypatch, tmp_path, mock_codegen):
    """capture must not touch server: monkeypatch server.build_free_text_run_doc to raise; assert
    the eval modules contain no `server` import. (NOT a _codegen_calls counter assertion — the
    autouse budget reset makes that vacuous.)"""
    import server

    def boom(*a, **k):
        raise AssertionError("eval capture must never call server.build_free_text_run_doc")

    monkeypatch.setattr(server, "build_free_text_run_doc", boom)
    mock_codegen(program=[{"id": "result", "op": "answer", "args": {"from": "x", "question": "q"}}], enabled=True)
    monkeypatch.setattr(fx, "FIXTURES_DIR", tmp_path)

    def fake_capture_one(client, case, clip, prompt_version):
        return fx.Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                          prompt_version=prompt_version, captured_at="t", model="mock",
                          raw_program=[], usage=None, error=None)

    monkeypatch.setattr(run_eval, "_capture_one", fake_capture_one)
    args = run_eval.build_parser().parse_args(["capture", "--only", "bernabeu-count-canonical"])
    run_eval.cmd_capture(args)  # must not raise

    # No `server` import in the eval source modules.
    for mod in ("bank.py", "fixtures.py", "scoring.py", "report.py", "run_eval.py", "seed_fixtures.py"):
        src = (EVAL_DIR / mod).read_text()
        assert "import server" not in src, f"{mod} imports server"


def test_replay_never_imports_codegen(monkeypatch, tmp_path):
    """Replay must complete without constructing the Azure client. Make from_env raise; replay
    end-to-end must still succeed (proves no client construction)."""
    import codegen

    monkeypatch.setattr(codegen.AzureCodegen, "from_env",
                        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("client constructed"))))
    code = run_eval.main(["replay", "--report", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")])
    assert code == 0
    from tests.conftest import FakeCodegen
    # If a mock seam was installed elsewhere it would count; here replay constructs nothing.
    assert FakeCodegen.total_calls == 0


def test_replay_marks_missing_and_stale(monkeypatch, tmp_path):
    """A bank with one case whose fixture is absent -> MISSING; a stale-stamped fixture -> STALE."""
    cases = bank.load_bank()[:2]
    # Point fixture loading at an empty tmpdir for the first case (MISSING), write a stale one for
    # the second.
    monkeypatch.setattr(fx, "FIXTURES_DIR", tmp_path)
    stale = fx.Fixture(case_id=cases[1]["case_id"], clip_id=cases[1]["clip_id"],
                       question=cases[1]["question"], prompt_version="STALEXXXXXXXXXXX",
                       captured_at="t", model="m", raw_program=[], usage=None, error=None)
    fx.write_fixture(stale, fixtures_dir=tmp_path)
    results = run_eval.replay(cases)
    by_id = {r.case_id: r for r in results}
    assert by_id[cases[0]["case_id"]].missing is True
    assert by_id[cases[1]["case_id"]].stale is True
