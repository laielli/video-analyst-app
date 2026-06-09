"""Runner / CLI tests — replay over committed seeds, seed regeneration match, the threshold gate,
stale handling (uniform vs partial), creds gating, never-touches-server, never-imports-codegen,
and the capture resume/limit/force paths (driven through the mock_codegen seam)."""
from __future__ import annotations

import json
from pathlib import Path

import bank
import canned
import fixtures as fx_mod
import run_eval
import scoring
import seed_fixtures

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
COMMITTED_FIXTURES = EVAL_DIR / "fixtures"


def test_replay_over_committed_seeds_scores():
    results, meta = run_eval.replay()
    # Every in-scope seed reaches its target tier; every case passes.
    by_id = {r.case_id: r for r in results}
    cases = bank.load_bank()["cases"]
    assert len(results) == len(cases)
    for r in results:
        assert not r.missing, f"{r.case_id} has no committed seed fixture"
        assert r.passed, f"{r.case_id} failed: {r.reached_tier} / {r.detail}"
    # Headline funnel: all in-scope reach T4.
    in_scope = [r for r in results if r.expected_outcome == "grounded"]
    assert all(scoring.TIER_RANK[r.reached_tier] >= scoring.TIER_RANK[r.target_tier] for r in in_scope)
    # A bank case present implies a scored fixture (keeps seeds honest).
    for c in cases:
        assert c["case_id"] in by_id


def test_seed_fixtures_regenerate_match(tmp_path):
    """Regenerate all seeds into a tmpdir and assert they match the committed fixtures byte-for-byte
    (covers seed_fixtures.py; a bank edit without re-seeding fails here)."""
    seed_fixtures.generate_all(tmp_path)
    for committed in sorted(COMMITTED_FIXTURES.glob("*.json")):
        regenerated = tmp_path / committed.name
        assert regenerated.exists(), f"missing regenerated seed for {committed.name}"
        assert regenerated.read_bytes() == committed.read_bytes(), committed.name
    # Same set of files (no orphan / no extra).
    assert {p.name for p in COMMITTED_FIXTURES.glob("*.json")} == {p.name for p in tmp_path.glob("*.json")}


def test_threshold_gate_fails_on_regression():
    results, meta = run_eval.replay()
    # Above floor (the seeds all pass) -> gate passes.
    passed, _ = run_eval.evaluate_gate(results, meta, {"T4_answer_correct": 0.85, "min_fixtures": 1})
    assert passed
    # A floor the rate can't meet -> gate fails.
    # Force one in-scope case below T4.
    for r in results:
        if r.expected_outcome == "grounded":
            r.reached_tier = scoring.T3_GROUNDED
            r.passed = False
            break
    passed, msgs = run_eval.evaluate_gate(results, meta, {"T4_answer_correct": 1.0, "min_fixtures": 1})
    assert not passed
    assert any("FAIL" in m for m in msgs)


def test_min_fixtures_gate_catches_deletion():
    results, meta = run_eval.replay()
    passed, msgs = run_eval.evaluate_gate(results, meta, {"min_fixtures": 9999})
    assert not passed
    assert any("min_fixtures" in m for m in msgs)


def test_stale_handling_uniform_vs_partial(tmp_path):
    cases = bank.load_bank()["cases"]
    # Build a fresh-seed set, then re-stamp them. Use the REAL run_eval.replay(fixtures_dir=...) so
    # the production STALE-marking path is exercised.
    seed_fixtures.generate_all(tmp_path)

    # UNIFORM stale: stamp every fixture with a wrong prompt_version -> all STALE -> advisory (exit 0).
    for case in cases:
        fx = fx_mod.load_fixture(case["case_id"], tmp_path)
        fx.prompt_version = "deadbeefdeadbeef"
        fx_mod.write_fixture(fx, tmp_path)
    results, meta = run_eval.replay(fixtures_dir=tmp_path)
    assert all(r.stale for r in results if not r.missing)
    assert all(r.reached_tier == scoring.STALE for r in results if not r.missing)
    passed, msgs = run_eval.evaluate_gate(results, meta, {"T4_answer_correct": 1.0, "min_fixtures": 1, "max_stale": 0})
    assert passed  # uniform stale -> advisory downgrade
    assert any("ADVISORY" in m for m in msgs)

    # PARTIAL stale: regenerate fresh, then doctor ONE fixture's prompt_version -> gate FAILS.
    seed_fixtures.generate_all(tmp_path)
    one = cases[0]
    fx = fx_mod.load_fixture(one["case_id"], tmp_path)
    fx.prompt_version = "deadbeefdeadbeef"
    fx_mod.write_fixture(fx, tmp_path)
    results, meta = run_eval.replay(fixtures_dir=tmp_path)
    stale = [r for r in results if r.stale]
    assert len(stale) == 1
    passed, msgs = run_eval.evaluate_gate(results, meta, {"min_fixtures": 1, "max_stale": 0})
    assert not passed
    assert any("max_stale" in m for m in msgs)


def test_capture_is_creds_gated(monkeypatch, tmp_path):
    """capture with codegen.enabled() False -> exits 2, writes no fixture, never constructs client."""
    import codegen

    monkeypatch.setattr(codegen, "enabled", lambda: False)

    def boom(cls):
        raise AssertionError("client constructed despite no creds")

    monkeypatch.setattr(codegen.AzureCodegen, "from_env", classmethod(boom))
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)

    rc = run_eval.main(["capture", "--only", "bernabeu-count-canonical"])
    assert rc == 2
    assert not list(tmp_path.glob("*.json"))


def test_capture_dry_run_makes_no_calls(monkeypatch, mock_codegen, tmp_path):
    fake = mock_codegen(program=[], enabled=True)
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    rc = run_eval.main(["capture", "--dry-run", "--only", "bernabeu-count-canonical"])
    assert rc == 0
    assert fake.calls == []  # no live calls in dry-run


def test_capture_writes_fixture_and_never_touches_server(monkeypatch, mock_codegen, tmp_path):
    """capture one case via the mock seam: writes a fixture with raw_program + usage:null, and
    never touches server (a raising monkeypatch on build_free_text_run_doc) — and the eval modules
    import no server. (NOT a _codegen_calls counter assertion — conftest zeroes it per test.)"""
    import server

    monkeypatch.setattr(server, "build_free_text_run_doc",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("server touched")))
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)

    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "How many players are visible?"}},
    ]
    fake = mock_codegen(program=program, enabled=True)
    rc = run_eval.main(["capture", "--only", "bernabeu-count-canonical", "--force"])
    assert rc == 0
    fx = fx_mod.load_fixture("bernabeu-count-canonical", tmp_path)
    assert fx is not None
    assert fx.raw_program == program
    assert fx.usage is None
    assert len(fake.calls) == 1

    # The eval modules never import server.
    for mod in ("bank", "fixtures", "scoring", "report", "run_eval", "seed_fixtures"):
        src = (EVAL_DIR / f"{mod}.py").read_text()
        assert "import server" not in src
        assert "from server" not in src


def test_capture_resume_skips_fresh(monkeypatch, mock_codegen, tmp_path):
    """A case whose fresh fixture already exists is skipped unless --force."""
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    clip = canned.clip_by_id("bernabeu-counter")
    pv = bank.prompt_version(clip)
    # Pre-write a fresh fixture.
    fx_mod.write_fixture(fx_mod.Fixture(
        case_id="bernabeu-count-canonical", clip_id="bernabeu-counter", question="q",
        prompt_version=pv, raw_program=[{"id": "x", "op": "answer", "args": {}}],
        captured_at="t", model="seed"), tmp_path)

    fake = mock_codegen(program=[], enabled=True)
    rc = run_eval.main(["capture", "--only", "bernabeu-count-canonical"])
    assert rc == 0
    assert fake.calls == []  # resume-skip: fresh fixture not re-captured


def test_capture_limit_clamps(monkeypatch, mock_codegen, tmp_path):
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    fake = mock_codegen(program=[{"id": "x", "op": "answer", "args": {}}], enabled=True)
    rc = run_eval.main(["capture", "--limit", "2", "--force"])
    assert rc == 0
    assert len(fake.calls) == 2  # --limit caps live calls this invocation


def test_capture_records_error_tag_on_generate_failure(monkeypatch, mock_codegen, tmp_path):
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    fake = mock_codegen(raise_exc=ValueError("codegen output exceeds 65536 bytes"), enabled=True)
    rc = run_eval.main(["capture", "--only", "bernabeu-count-canonical", "--force"])
    assert rc == 0
    fx = fx_mod.load_fixture("bernabeu-count-canonical", tmp_path)
    assert fx.raw_program is None
    assert fx.error == fx_mod.ERROR_OVERSIZED


def test_capture_records_unparseable_and_generic_error_tags(monkeypatch, mock_codegen, tmp_path):
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    # unparseable: a "missing 'program'" ValueError.
    mock_codegen(raise_exc=ValueError("codegen output missing 'program' array"), enabled=True)
    run_eval.main(["capture", "--only", "bernabeu-count-canonical", "--force"])
    assert fx_mod.load_fixture("bernabeu-count-canonical", tmp_path).error == fx_mod.ERROR_UNPARSEABLE
    # generic: a non-ValueError exception.
    mock_codegen(raise_exc=RuntimeError("network down"), enabled=True)
    run_eval.main(["capture", "--only", "bernabeu-scorer-number-canonical", "--force"])
    assert fx_mod.load_fixture("bernabeu-scorer-number-canonical", tmp_path).error == fx_mod.ERROR_GENERATION_FAILED


def test_generate_one_maps_value_error_without_keyword():
    """A ValueError with no oversized/json keyword maps to the generic generation-failed tag."""
    class _Raiser:
        def generate(self, q, c):
            raise ValueError("some other problem")

    raw, err = run_eval._generate_one(_Raiser(), "q", canned.clip_by_id("single-goal"))
    assert raw is None and err == fx_mod.ERROR_GENERATION_FAILED


def test_capture_only_no_match(monkeypatch, mock_codegen, tmp_path):
    monkeypatch.setattr(fx_mod, "FIXTURES_DIR", tmp_path)
    mock_codegen(program=[], enabled=True)
    rc = run_eval.main(["capture", "--only", "nonexistent-case"])
    assert rc == 1


def test_replay_never_imports_codegen(monkeypatch, tmp_path):
    """Monkeypatch codegen.AzureCodegen.from_env to raise; replay must complete (no client
    construction) and a FakeCodegen-style counter stays 0."""
    import codegen
    from tests.conftest import FakeCodegen

    FakeCodegen.total_calls = 0
    monkeypatch.setattr(codegen.AzureCodegen, "from_env",
                        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("client built"))))

    rc = run_eval.main(["replay", "--report", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json")])
    assert rc == 0
    assert FakeCodegen.total_calls == 0


def test_replay_cli_writes_artifacts_and_gates(tmp_path):
    mdp, jp = tmp_path / "r.md", tmp_path / "r.json"
    rc = run_eval.main(["replay", "--report", str(mdp), "--json", str(jp),
                        "--fail-under", "T4_answer_correct=1.0"])
    assert rc == 0
    assert mdp.exists() and jp.exists()
    j = json.loads(jp.read_text())
    assert j["summary"]["passed"] == j["summary"]["total_cases"]


def test_replay_no_gate(tmp_path):
    rc = run_eval.main(["replay", "--report", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json"),
                        "--no-gate", "--fail-under", "T4_answer_correct=1.0"])
    assert rc == 0


def test_replay_shape_filter(tmp_path):
    results, _ = run_eval.replay(shape="count")
    assert results and all(r.shape == "count" for r in results)


def test_load_thresholds_overrides():
    t = run_eval._load_thresholds(["T2_semantic_valid=0.9", "T4_answer_correct=0.5"])
    assert t["T2_semantic_valid"] == 0.9
    assert t["T4_answer_correct"] == 0.5
    assert "_comment" not in t


def test_est_cost_offline():
    assert run_eval._est_cost(60).startswith("~$")
