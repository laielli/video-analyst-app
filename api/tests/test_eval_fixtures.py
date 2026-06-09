"""Fixture IO tests (D2): atomic round-trip, missing -> None, prompt-version freshness."""
from __future__ import annotations

import json

from eval.fixtures import Fixture, is_fresh, load_fixture, write_fixture


def _fx(case_id="c1", pv="abc123"):
    return Fixture(case_id=case_id, clip_id="bernabeu-counter", question="How many?",
                   prompt_version=pv, raw_program=[{"id": "a", "op": "answer", "args": {}}],
                   model="seed", captured_at="seed", usage=None)


def test_write_load_roundtrip_atomic(tmp_path):
    fx = _fx()
    path = write_fixture(fx, fixtures_dir=tmp_path)
    assert path.exists()
    loaded = load_fixture("c1", fixtures_dir=tmp_path)
    assert loaded is not None
    assert loaded.case_id == fx.case_id
    assert loaded.raw_program == fx.raw_program
    assert loaded.usage is None
    # No temp file left behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_fixture_field_order_raw_program_before_provenance(tmp_path):
    """raw_program diffs read before the captured_at restamp (Deferred-notes cosmetic)."""
    write_fixture(_fx(), fixtures_dir=tmp_path)
    keys = list(json.loads((tmp_path / "c1.json").read_text()).keys())
    assert keys.index("raw_program") < keys.index("captured_at")


def test_missing_fixture_loads_none(tmp_path):
    assert load_fixture("nope", fixtures_dir=tmp_path) is None


def test_is_fresh_detects_prompt_drift():
    fx = _fx(pv="current")
    assert is_fresh(fx, "current") is True
    assert is_fresh(fx, "different") is False


def test_error_fixture_roundtrip(tmp_path):
    fx = Fixture(case_id="e1", clip_id="bernabeu-counter", question="?",
                 prompt_version="pv", raw_program=None, error="generation-failed",
                 model="seed", captured_at="seed")
    write_fixture(fx, fixtures_dir=tmp_path)
    loaded = load_fixture("e1", fixtures_dir=tmp_path)
    assert loaded.raw_program is None and loaded.error == "generation-failed"
