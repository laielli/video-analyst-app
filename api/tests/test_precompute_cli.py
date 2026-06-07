"""
CLI-level tests for precompute.py + cross-checks against the existing harness (run_program.py,
validate_run_doc.py). Exit codes are the contract: 0 ok, 1 produced-but-failed, 2 setup problem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))

import precompute as pc  # noqa: E402


def test_manifest_absent_errors_exit_2(monkeypatch):
    # --clip with no manifest must error clearly (exit 2), never synthesize defaults.
    rc = pc.main(["--clip", "does-not-exist", "--dry-run"])
    assert rc == 2


def test_no_creds_live_run_exits_2(monkeypatch):
    # Live (non-dry-run) without AZURE_VISION_* must exit 2 with a clear message.
    monkeypatch.delenv("AZURE_VISION_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_VISION_KEY", raising=False)
    # Prevent .env from re-supplying creds during this test.
    monkeypatch.setattr(pc, "load_dotenv", lambda *a, **k: None)
    rc = pc.main(["--clip", "single-goal"])
    assert rc == 2


def test_dry_run_cli_exit_0(monkeypatch):
    monkeypatch.setattr(pc, "load_dotenv", lambda *a, **k: None)
    rc = pc.main(["--clip", "single-goal", "--query", "hero-10-first-goal", "--dry-run"])
    assert rc == 0


def test_run_program_main_validates_generated_cache(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract):
    # End-to-end through the EXISTING harness: generate a cache, then feed it to run_program.main()
    # and assert exit 0 (program valid + replays + run-doc valid).
    import run_program

    out = tmp_path / "gen_cache.json"
    pc.run_precompute(
        manifest, manifest_path, out=out,
        program_path=API_DIR / "examples" / "hero_program.json",
        vision_factory=lambda: fake_vision,
        vlm_factory=lambda: fake_vlm_none,
        extract=fake_extract,
        emit_frames=False,
    )
    rc = run_program.main(["--cache", str(out)]) if _accepts_argv(run_program.main) else _run_program_via_argv(run_program, out)
    assert rc == 0


def _accepts_argv(fn) -> bool:
    import inspect
    return len(inspect.signature(fn).parameters) >= 1


def _run_program_via_argv(run_program, out: Path) -> int:
    # run_program.main() reads sys.argv; drive it that way.
    argv = sys.argv
    try:
        sys.argv = ["run_program.py", "--cache", str(out)]
        return run_program.main()
    finally:
        sys.argv = argv


def test_generated_cache_passes_validate_run_doc(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract):
    # Generate a cache, replay to a run-doc on disk, then run the standalone validate_run_doc.main.
    import validate_run_doc
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    out = tmp_path / "gen_cache.json"
    pc.run_precompute(
        manifest, manifest_path, out=out,
        program_path=API_DIR / "examples" / "hero_program.json",
        vision_factory=lambda: fake_vision,
        vlm_factory=lambda: fake_vlm_none,
        extract=fake_extract,
        emit_frames=False,
    )
    program = strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))["program"]
    run_doc = Interpreter(Cache.load(out)).run(program)
    rd_path = tmp_path / "gen_run_doc.json"
    rd_path.write_text(json.dumps(run_doc, indent=2))

    argv = sys.argv
    try:
        sys.argv = ["validate_run_doc.py", str(rd_path)]
        rc = validate_run_doc.main()
    finally:
        sys.argv = argv
    assert rc == 0
