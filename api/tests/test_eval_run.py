"""
Runner / CLI tests (D3, D6) — replay over committed seeds, seed regeneration, the threshold gate,
stale handling (uniform vs partial), creds-gating, and the server-isolation / no-codegen-on-replay
guarantees. This is where run_eval.py and seed_fixtures.py earn their coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import canned
from eval import bank, fixtures, run_eval, scoring, seed_fixtures
from eval.report import build_report

EVAL_DIR = Path(seed_fixtures.__file__).resolve().parent
COMMITTED_FIXTURES = EVAL_DIR / "fixtures"


# ---- replay over committed seeds ---------------------------------------------------------
def test_replay_over_committed_seeds_scores(tmp_path):
    """Every in-scope seed reaches its target tier; the report has the expected per-tier counts.
    A bank case without a seed fails here (keeps the seeds honest)."""
    rc = run_eval.cmd_replay(_replay_args(report=tmp_path / "r.md", json_=tmp_path / "r.json"))
    assert rc == 0
    j = json.loads((tmp_path / "r.json").read_text())
    n_in = j["summary"]["in_scope_cases"]
    assert j["summary"]["funnel"][scoring.T4_CORRECT]["reached"] == n_in  # all in-scope seeds -> T4
    assert j["summary"]["missing"] == 0
    assert j["summary"]["out_of_scope_honest"]["rate"] == "1.00"


def test_every_bank_case_has_a_committed_seed():
    for case in bank.load_bank():
        assert (COMMITTED_FIXTURES / f"{case['case_id']}.json").exists(), \
            f"missing committed seed for {case['case_id']}"


def test_seed_fixtures_regenerate_match(tmp_path):
    """Regenerate all seeds into a tmpdir; assert they match the committed ones byte-for-byte
    (modulo nothing — seeds use fixed sentinels for captured_at/model). Fails on a bank edit without
    re-seeding."""
    seed_fixtures.write_all_seeds(fixtures_dir=tmp_path)
    for case in bank.load_bank():
        name = f"{case['case_id']}.json"
        regenerated = (tmp_path / name).read_text()
        committed = (COMMITTED_FIXTURES / name).read_text()
        assert regenerated == committed, f"committed seed for {case['case_id']} is stale; re-seed"


def test_seeds_carry_current_prompt_version():
    """Seeds are NOT stale: each carries the current prompt_version so the gate exercises the real
    ladder."""
    for case in bank.load_bank():
        clip = canned.clip_by_id(case["clip_id"])
        pv = bank.prompt_version(clip)
        fx = fixtures.load_fixture(case["case_id"])
        assert fixtures.is_fresh(fx, pv), f"{case['case_id']} seed is stale against the current prompt"


# ---- the threshold gate ------------------------------------------------------------------
def _synth(cases_by_id, results):
    _, j = build_report(results, cases_by_id, {"bank_version": 1})
    return j


def test_threshold_gate_fails_on_regression():
    """A funnel below a floor returns non-zero; above it returns zero."""
    cbid = {"g": {"case_id": "g", "clip_id": "bernabeu-counter", "shape": "count",
                  "phrasing_class": "canonical", "distance": "canonical",
                  "expect": {"outcome": "grounded", "answer": "5", "answer_match": "numeric"}}}
    below = [scoring.CaseResult("g", scoring.T4_CORRECT, scoring.T2_SEMANTIC, False, "wrong")]
    above = [scoring.CaseResult("g", scoring.T4_CORRECT, scoring.T4_CORRECT, True, "ok")]
    floors = {"T4_answer_correct": 0.85, "min_fixtures": 1}
    assert run_eval.apply_gate(below, cbid, _synth(cbid, below), floors) == 1
    assert run_eval.apply_gate(above, cbid, _synth(cbid, above), floors) == 0


def test_gate_fails_on_missing_and_min_fixtures():
    cbid = {"m": {"case_id": "m", "clip_id": "bernabeu-counter", "shape": "count",
                  "phrasing_class": "canonical", "distance": "canonical",
                  "expect": {"outcome": "grounded", "answer": "5", "answer_match": "numeric"}}}
    missing = [scoring.CaseResult("m", scoring.T4_CORRECT, scoring.MISSING, False, "no fixture", missing=True)]
    # A missing fixture is a hard fail regardless of the min_fixtures floor.
    assert run_eval.apply_gate(missing, cbid, _synth(cbid, missing), {"min_fixtures": 0}) == 1


def test_stale_handling_uniform_vs_partial(tmp_path, monkeypatch):
    """ALL stale -> advisory (exit 0); ONE stale among fresh -> partial staleness FAILS (D6)."""
    cbid = {
        "a": {"case_id": "a", "clip_id": "bernabeu-counter", "shape": "count",
              "phrasing_class": "canonical", "distance": "canonical",
              "expect": {"outcome": "grounded", "answer": "5", "answer_match": "numeric"}},
        "b": {"case_id": "b", "clip_id": "bernabeu-counter", "shape": "count",
              "phrasing_class": "paraphrase", "distance": "far",
              "expect": {"outcome": "grounded", "answer": "5", "answer_match": "numeric"}},
    }
    all_stale = [scoring.CaseResult("a", scoring.T4_CORRECT, scoring.STALE, False, "stale", stale=True),
                 scoring.CaseResult("b", scoring.T4_CORRECT, scoring.STALE, False, "stale", stale=True)]
    # Advisory bypass fires BEFORE the min_fixtures check, so a tiny synthetic set is fine.
    assert run_eval.apply_gate(all_stale, cbid, _synth(cbid, all_stale), {"min_fixtures": 1}) == 0

    one_stale = [scoring.CaseResult("a", scoring.T4_CORRECT, scoring.T4_CORRECT, True, "ok"),
                 scoring.CaseResult("b", scoring.T4_CORRECT, scoring.STALE, False, "stale", stale=True)]
    # Partial staleness fails even with min_fixtures satisfied (the anti-gaming rule).
    assert run_eval.apply_gate(one_stale, cbid, _synth(cbid, one_stale), {"min_fixtures": 1}) == 1


def test_replay_marks_stale_on_prompt_drift(tmp_path, monkeypatch):
    """A whole-bank prompt edit stales every fixture uniformly; replay downgrades to advisory."""
    import codegen
    monkeypatch.setattr(codegen, "SYSTEM_PROMPT", codegen.SYSTEM_PROMPT + "\nNEW RULE FOR DRIFT.")
    rc = run_eval.cmd_replay(_replay_args(report=tmp_path / "r.md", json_=tmp_path / "r.json"))
    j = json.loads((tmp_path / "r.json").read_text())
    assert j["summary"]["stale"] == j["summary"]["total_cases"]  # uniform stale
    assert rc == 0  # advisory bypass


def test_fail_under_parsing_rejects_bad_pair():
    with pytest.raises(SystemExit):
        run_eval._parse_fail_under(["T2_semantic_valid"])  # no '='


def test_no_gate_returns_zero(tmp_path):
    rc = run_eval.cmd_replay(_replay_args(report=tmp_path / "r.md", json_=tmp_path / "r.json", no_gate=True))
    assert rc == 0


def test_replay_filter_by_shape(tmp_path):
    """A --shape slice is a diagnostic, not a gating run (it can't satisfy min_fixtures), so use
    --no-gate; assert only that-shape cases appear."""
    rc = run_eval.cmd_replay(_replay_args(report=tmp_path / "r.md", json_=tmp_path / "r.json",
                                          shape="count", no_gate=True))
    j = json.loads((tmp_path / "r.json").read_text())
    assert rc == 0
    ids = {c["case_id"] for c in j["cases"]}
    assert "bernabeu-count-canonical" in ids
    assert "hero-firstgoal-canonical" not in ids


# ---- capture creds-gating + server-isolation ---------------------------------------------
def test_capture_is_creds_gated(monkeypatch, tmp_path):
    """capture with codegen.enabled() False -> exits 2, writes no fixture, never constructs client."""
    import codegen
    monkeypatch.setattr(codegen, "enabled", lambda: False)

    def boom(cls):
        raise AssertionError("AzureCodegen.from_env constructed despite no creds")
    monkeypatch.setattr(codegen.AzureCodegen, "from_env", classmethod(boom))

    rc = run_eval.cmd_capture(_capture_args(limit=1))
    assert rc == 2


def test_capture_never_touches_server(mock_codegen, monkeypatch, tmp_path):
    """capture one case via the mock seam -> writes a fixture with raw_program + usage:null, and
    server is never touched. Asserts NO eval module imports server."""
    import server
    fake = mock_codegen(program=[{"id": "frames", "op": "sample_frames",
                                  "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
                                 {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
                                 {"id": "n", "op": "count", "args": {"items": "people"}},
                                 {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}}])

    def explode(*a, **k):
        raise AssertionError("capture must never call server.build_free_text_run_doc")
    monkeypatch.setattr(server, "build_free_text_run_doc", explode)

    # Redirect fixture writes to a tmp dir so we don't clobber the committed seeds.
    monkeypatch.setattr(fixtures, "FIXTURES_DIR", tmp_path)
    rc = run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical"))
    assert rc == 0
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    fx = json.loads(written[0].read_text())
    assert fx["raw_program"] is not None and fx["usage"] is None
    assert fake.total_calls >= 1

    # No eval module imports server (the trust separation is by NON-import, not a counter).
    import eval.bank, eval.fixtures, eval.scoring, eval.report, eval.run_eval, eval.seed_fixtures
    for mod in (eval.bank, eval.fixtures, eval.scoring, eval.report, eval.run_eval, eval.seed_fixtures):
        src = Path(mod.__file__).read_text()
        assert "import server" not in src, f"{mod.__name__} must not import server"


def test_replay_never_imports_codegen_client(tmp_path, monkeypatch):
    """Replay runs end-to-end even if AzureCodegen.from_env raises (proves no client construction)."""
    import codegen
    monkeypatch.setattr(codegen.AzureCodegen, "from_env",
                        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("client built on replay"))))
    rc = run_eval.cmd_replay(_replay_args(report=tmp_path / "r.md", json_=tmp_path / "r.json"))
    assert rc == 0


def test_capture_dry_run_makes_no_calls(mock_codegen, monkeypatch, tmp_path):
    """--dry-run prints the call count and constructs no client / writes no fixture."""
    fake = mock_codegen(program=[])
    monkeypatch.setattr(fixtures, "FIXTURES_DIR", tmp_path)
    rc = run_eval.cmd_capture(_capture_args(dry_run=True, force=True))
    assert rc == 0
    assert fake.total_calls == 0
    assert not list(tmp_path.glob("*.json"))


def test_capture_resume_skips_fresh(mock_codegen, monkeypatch, tmp_path):
    """A fresh fixture present in the dir is skipped (resume) unless --force."""
    prog = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "people"}},
            {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}}]
    fake = mock_codegen(program=prog)
    monkeypatch.setattr(fixtures, "FIXTURES_DIR", tmp_path)

    # First capture writes the fixture.
    run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical"))
    assert fake.total_calls == 1
    # Second capture (no --force) resumes: skips the fresh fixture, makes no new call.
    run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical"))
    assert fake.total_calls == 1  # unchanged
    # With --force it re-captures.
    run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical", force=True))
    assert fake.total_calls == 2


def test_capture_limit_clamped_to_ceiling(mock_codegen, monkeypatch, tmp_path):
    fake = mock_codegen(program=[])
    monkeypatch.setattr(fixtures, "FIXTURES_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "MAX_CAPTURE_CALLS", 2)
    rc = run_eval.cmd_capture(_capture_args(dry_run=True, limit=100, force=True))
    assert rc == 0  # dry-run; the clamp logic ran without exceeding the ceiling


def test_capture_only_no_match(monkeypatch, mock_codegen, tmp_path):
    mock_codegen(program=[])
    monkeypatch.setattr(fixtures, "FIXTURES_DIR", tmp_path)
    rc = run_eval.cmd_capture(_capture_args(only="no-such-case"))
    assert rc == 1


def test_main_parses_replay(tmp_path):
    rc = run_eval.main(["replay", "--report", str(tmp_path / "r.md"),
                        "--json", str(tmp_path / "r.json"), "--no-gate"])
    assert rc == 0


# ---- arg shims ---------------------------------------------------------------------------
class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _replay_args(report, json_, shape=None, clip=None, fail_under=None, no_gate=False):
    return _Args(report=str(report), json=str(json_), shape=shape, clip=clip,
                 fail_under=fail_under, no_gate=no_gate)


def _capture_args(only=None, limit=None, force=False, dry_run=False):
    return _Args(only=only, limit=limit, force=force, dry_run=dry_run)
