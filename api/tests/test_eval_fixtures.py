"""Fixture IO tests — atomic round-trip, missing -> None, prompt-drift detection. Creds-free."""
from __future__ import annotations

import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = API_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import fixtures as fx  # noqa: E402


def _fixture(case_id="c1", pv="pv1"):
    return fx.Fixture(case_id=case_id, clip_id="single-goal", question="q?",
                      prompt_version=pv, captured_at="2026-01-01T00:00:00Z", model="gpt-4o",
                      raw_program=[{"id": "result", "op": "answer", "args": {"from": "x", "question": "q?"}}],
                      usage=None, error=None)


def test_write_load_roundtrip_atomic(tmp_path):
    fxt = _fixture()
    path = fx.write_fixture(fxt, fixtures_dir=tmp_path)
    assert path.exists()
    loaded = fx.load_fixture("c1", fixtures_dir=tmp_path)
    assert loaded is not None
    assert loaded.case_id == "c1"
    assert loaded.raw_program == fxt.raw_program
    assert loaded.usage is None
    # no temp files left behind
    assert not list(tmp_path.glob(".*.tmp"))


def test_missing_fixture_loads_none(tmp_path):
    assert fx.load_fixture("nope", fixtures_dir=tmp_path) is None


def test_is_fresh_detects_prompt_drift():
    fxt = _fixture(pv="abc123")
    assert fx.is_fresh(fxt, "abc123") is True
    assert fx.is_fresh(fxt, "different") is False


def test_field_order_raw_program_first(tmp_path):
    """raw_program reads before the captured_at/model provenance stamps so a re-capture's program
    change shows first in a diff (Deferred note)."""
    fx.write_fixture(_fixture(), fixtures_dir=tmp_path)
    body = (tmp_path / "c1.json").read_text()
    assert body.index('"raw_program"') < body.index('"captured_at"')
    assert body.index('"raw_program"') < body.index('"model"')


def test_error_fixture_roundtrip(tmp_path):
    fxt = fx.Fixture(case_id="e1", clip_id="single-goal", question="q?",
                     prompt_version="pv", captured_at="t", model="m",
                     raw_program=None, usage=None, error=fx.ERR_OVERSIZED)
    fx.write_fixture(fxt, fixtures_dir=tmp_path)
    loaded = fx.load_fixture("e1", fixtures_dir=tmp_path)
    assert loaded.raw_program is None
    assert loaded.error == fx.ERR_OVERSIZED
