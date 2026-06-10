"""
Runner / CLI tests — replay over committed seeds, seed regeneration, threshold gate, stale
handling, creds gating, and the trust/server-isolation invariants. Creds-free; capture is driven
through the mock_codegen seam (no Azure).
"""
from __future__ import annotations

import json

import codegen
import canned
from eval import run_eval, seed_fixtures, bank
from eval import scoring


# ---- replay over the committed seeds ----------------------------------------------------

def test_replay_over_committed_seeds_scores(tmp_path, capsys):
    """Run replay in-process over the COMMITTED seed fixtures; every in-scope seed reaches its
    target tier and the report has the expected per-tier counts. A bank case without a seed would
    fail here (MISSING)."""
    rc = run_eval.cmd_replay(_replay_args(tmp_path))
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    s = report["summary"]
    assert s["missing"] == 0, "every bank case must have a committed seed"
    assert s["stale"] == 0, "seeds carry the current prompt_version"
    # In-scope seeds reproduce the pinned grounded programs -> 100% through T4.
    assert s["T2_semantic_valid"] == 1.0
    assert s["T4_answer_correct"] == 1.0
    assert s["out_of_scope_honest_rate"] == 1.0
    assert s["adversarial_safe_rate"] == 1.0


def test_seed_fixtures_regenerate_match(tmp_path):
    """Regenerate all seeds into a tmpdir; they match the committed fixtures byte-for-byte modulo
    the captured_at/prompt_version stamps (covers seed_fixtures.py; a bank edit without re-seeding
    fails here)."""
    seed_fixtures.generate_all(fixtures_dir=tmp_path)
    committed_dir = seed_fixtures._HERE / "fixtures"
    regenerated = sorted(tmp_path.glob("*.json"))
    committed = sorted(committed_dir.glob("*.json"))
    assert [p.name for p in regenerated] == [p.name for p in committed], "case set drifted from seeds"
    for p in regenerated:
        a = json.loads(p.read_text())
        b = json.loads((committed_dir / p.name).read_text())
        for vol in ("captured_at",):  # stamps that may legitimately differ
            a.pop(vol, None)
            b.pop(vol, None)
        assert a == b, f"seed {p.name} drifted; re-run python eval/seed_fixtures.py"


# ---- threshold gate ---------------------------------------------------------------------

def _summary(t2=1.0, t4=1.0, oos=1.0, adv=1.0, total=60, stale=0, missing=0):
    return {
        "summary": {
            "total_cases": total, "stale": stale, "missing": missing,
            "T2_semantic_valid": t2, "T4_answer_correct": t4,
            "out_of_scope_honest_rate": oos, "adversarial_safe_rate": adv,
        }
    }


class _Args:
    def __init__(self, **kw):
        self.fail_under = kw.get("fail_under")
        self.no_gate = kw.get("no_gate", False)


def test_threshold_gate_fails_on_regression():
    # Below the T4 floor (0.85) -> non-zero.
    rc = run_eval._apply_gate([], _summary(t4=0.50), _Args())
    assert rc == 1
    # Above the floor -> zero.
    rc = run_eval._apply_gate([], _summary(t4=1.0), _Args())
    assert rc == 0


def test_threshold_gate_min_fixtures_catches_deletion():
    # Most fixtures missing -> present < min_fixtures floor -> fail.
    rc = run_eval._apply_gate([], _summary(total=53, missing=53), _Args())
    assert rc == 1


def test_stale_handling_uniform_vs_partial():
    # ALL stale (uniform) -> advisory downgrade, exit 0 (the honest post-prompt-edit state).
    rc = run_eval._apply_gate([], _summary(total=60, stale=60), _Args())
    assert rc == 0
    # ONE stale among fresh (partial, > max_stale=0) -> FAIL (the anti-gaming rule, D6).
    rc = run_eval._apply_gate([], _summary(total=60, stale=1), _Args())
    assert rc == 1


def test_fail_under_override_parses():
    # An impossible floor via --fail-under overrides the committed threshold and fails.
    rc = run_eval._apply_gate([], _summary(t2=0.9), _Args(fail_under=["T2_semantic_valid=1.0"]))
    assert rc == 1


# ---- capture (mocked seam — never Azure) ------------------------------------------------

def test_capture_is_creds_gated(monkeypatch, tmp_path):
    """capture with codegen.enabled() False -> exit 2, writes no fixture, never constructs client."""
    monkeypatch.setattr(codegen, "enabled", lambda: False)

    def boom(cls):
        raise AssertionError("client must not be constructed when creds are absent")

    monkeypatch.setattr(codegen.AzureCodegen, "from_env", classmethod(boom))
    args = _capture_args(only="bernabeu-count-canonical", limit=1)
    rc = run_eval.cmd_capture(args)
    assert rc == 2


def test_capture_never_touches_server(monkeypatch, mock_codegen, tmp_path):
    """Capture one case via the mock seam: it writes a fixture with raw_program + usage:null, and
    NEVER touches server (a raising monkeypatch on server.build_free_text_run_doc + no server import
    in the eval modules). NOT a budget-counter assertion (conftest's autouse reset makes it vacuous)."""
    import server
    monkeypatch.setattr(server, "build_free_text_run_doc",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("capture must not touch server")))

    prog = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "people"}},
            {"id": "result", "op": "answer", "args": {"from": "n", "question": "How many players are visible?"}}]
    mock_codegen(program=prog, enabled=True)
    # Redirect fixture writes to tmp by monkeypatching the fixtures module the runner uses
    # (run_eval.fx is the single eval.fixtures identity thanks to the package-relative imports).
    monkeypatch.setattr(run_eval.fx, "FIXTURES_DIR", tmp_path)

    rc = run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical"))
    assert rc == 0
    written = list(tmp_path.glob("bernabeu-count-canonical.json"))
    assert written, "capture must write a fixture"
    data = json.loads(written[0].read_text())
    assert data["raw_program"] == prog
    assert data["usage"] is None

    # The eval modules contain no actual `import server` / `from server import` STATEMENT (the
    # docstrings legitimately mention server; match real import lines, not prose).
    import re
    server_import = re.compile(r"^\s*(import\s+server\b|from\s+server\s+import)", re.MULTILINE)
    for mod in ("bank", "fixtures", "scoring", "report", "run_eval", "seed_fixtures"):
        src = (run_eval.HERE / f"{mod}.py").read_text()
        assert not server_import.search(src), f"eval/{mod}.py must not import server"


def test_capture_dry_run_makes_no_calls(monkeypatch, mock_codegen, capsys, tmp_path):
    # Empty fixtures dir so resume doesn't skip the case (the dry-run plan should show 1 call).
    monkeypatch.setattr(run_eval.fx, "FIXTURES_DIR", tmp_path)
    fake = mock_codegen(program=[], enabled=True)
    rc = run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical", dry_run=True))
    assert rc == 0
    # generate() is never called in dry-run; the dry-run line names the call count + cost estimate.
    assert len(fake.calls) == 0
    out = capsys.readouterr().out
    assert "will make 1 live calls" in out and "~$" in out


def test_capture_resume_skips_fresh_fixture(monkeypatch, mock_codegen, tmp_path):
    """A case whose fixture is fresh for the current prompt is skipped (resume) unless --force."""
    fxmod = run_eval.fx
    monkeypatch.setattr(fxmod, "FIXTURES_DIR", tmp_path)
    # Write a fresh fixture for the case.
    clip = canned.clip_by_id("bernabeu-counter")
    pv = bank.prompt_version(clip)
    fxmod.write_fixture(fxmod.Fixture(case_id="bernabeu-count-canonical", clip_id="bernabeu-counter",
                                      question="How many players are visible?", prompt_version=pv,
                                      raw_program=[{"id": "r", "op": "answer", "args": {}}]),
                        fixtures_dir=tmp_path)
    fake = mock_codegen(program=[{"id": "r", "op": "answer", "args": {}}], enabled=True)
    rc = run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical"))
    assert rc == 0
    assert len(fake.calls) == 0, "a fresh fixture must be skipped on resume"
    # --force re-captures (one call).
    rc = run_eval.cmd_capture(_capture_args(only="bernabeu-count-canonical", force=True))
    assert len(fake.calls) == 1


def test_capture_only_no_match_returns_1(monkeypatch, mock_codegen):
    mock_codegen(program=[], enabled=True)
    rc = run_eval.cmd_capture(_capture_args(only="no-such-case-shape-or-clip"))
    assert rc == 1


def test_capture_limit_clamps_calls(monkeypatch, mock_codegen, tmp_path):
    """--limit caps live calls this invocation (with an empty fixtures dir, all cases are pending)."""
    monkeypatch.setattr(run_eval.fx, "FIXTURES_DIR", tmp_path)
    prog = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "people"}},
            {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}}]
    fake = mock_codegen(program=prog, enabled=True)
    rc = run_eval.cmd_capture(_capture_args(only="bernabeu-counter", limit=2))
    assert rc == 0
    assert len(fake.calls) == 2, "--limit must clamp the number of live calls"


def test_replay_filters_by_shape_and_clip(tmp_path):
    """--shape / --clip narrow the scored set (reusing filter_cases)."""
    ns = run_eval.build_parser().parse_args([
        "replay", "--report", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json"),
        "--shape", "count", "--clip", "bernabeu-counter", "--no-gate",
    ])
    rc = run_eval.cmd_replay(ns)
    assert rc == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert {c["shape"] for c in report["cases"]} == {"count"}
    assert {c["clip_id"] for c in report["cases"]} == {"bernabeu-counter"}


def test_replay_marks_stale_on_prompt_drift(monkeypatch, tmp_path):
    """If the current prompt_version drifts from the (uniform) committed fixtures, replay marks them
    STALE and the gate downgrades to advisory (uniform-stale bypass)."""
    import codegen as cg
    monkeypatch.setattr(cg, "SYSTEM_PROMPT", cg.SYSTEM_PROMPT + "\nDRIFT")
    ns = run_eval.build_parser().parse_args([
        "replay", "--report", str(tmp_path / "r.md"), "--json", str(tmp_path / "r.json"),
    ])
    rc = run_eval.cmd_replay(ns)
    assert rc == 0  # uniform-stale advisory downgrade
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["summary"]["stale"] == report["summary"]["total_cases"]


def test_replay_never_imports_codegen(monkeypatch, tmp_path):
    """Monkeypatch codegen.AzureCodegen.from_env to RAISE; replay must complete end-to-end. If
    replay constructed the Azure client the raise would surface, so a clean exit 0 proves replay
    never touches the client (it only uses build_system_message for the offline prompt-size stamp)."""
    monkeypatch.setattr(codegen.AzureCodegen, "from_env",
                        classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("no client in replay"))))
    rc = run_eval.cmd_replay(_replay_args(tmp_path))
    assert rc == 0


# ---- argparse ---------------------------------------------------------------------------

def test_parser_requires_subcommand():
    import pytest
    with pytest.raises(SystemExit):
        run_eval.build_parser().parse_args([])


def test_main_dispatches_replay(tmp_path):
    rc = run_eval.main(["replay", "--report", str(tmp_path / "r.md"),
                        "--json", str(tmp_path / "r.json"), "--no-gate"])
    assert rc == 0
    assert (tmp_path / "r.md").exists()


# ---- helpers ----------------------------------------------------------------------------

def _replay_args(tmp_path, **kw):
    ns = run_eval.build_parser().parse_args([
        "replay", "--report", str(tmp_path / "report.md"), "--json", str(tmp_path / "report.json"),
    ])
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _capture_args(only=None, limit=None, force=False, dry_run=False):
    args = ["capture"]
    if only:
        args += ["--only", only]
    if limit is not None:
        args += ["--limit", str(limit)]
    if force:
        args += ["--force"]
    if dry_run:
        args += ["--dry-run"]
    return run_eval.build_parser().parse_args(args)
