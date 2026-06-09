"""Fixture IO tests — round-trip, missing, prompt-drift detection, atomic write field order."""
from __future__ import annotations

import json

from eval import fixtures
from eval.fixtures import Fixture


def _fx(case_id="c1", pv="abc123"):
    return Fixture(case_id=case_id, clip_id="bernabeu-counter", question="How many?",
                   prompt_version=pv, raw_program=[{"id": "r", "op": "answer", "args": {}}],
                   error=None, usage=None, captured_at="2026-06-09T00:00:00Z", model="gpt-4o")


def test_write_load_roundtrip_atomic(tmp_path):
    fx = _fx()
    p = fixtures.write_fixture(fx, fixtures_dir=tmp_path)
    assert p.exists()
    loaded = fixtures.load_fixture("c1", fixtures_dir=tmp_path)
    assert loaded is not None
    assert loaded.case_id == "c1"
    assert loaded.raw_program == fx.raw_program
    assert loaded.usage is None
    assert loaded.prompt_version == "abc123"
    # No leftover temp files.
    assert not any(f.name.startswith(".tmp-") for f in tmp_path.iterdir())


def test_fixture_field_order_raw_program_before_provenance(tmp_path):
    """raw_program is serialized before captured_at/model so a re-capture diff reads model output
    first (Deferred-notes cosmetic)."""
    fixtures.write_fixture(_fx(), fixtures_dir=tmp_path)
    text = (tmp_path / "c1.json").read_text()
    assert text.index('"raw_program"') < text.index('"captured_at"')
    assert text.index('"raw_program"') < text.index('"model"')


def test_missing_fixture_loads_none(tmp_path):
    assert fixtures.load_fixture("nope", fixtures_dir=tmp_path) is None


def test_is_fresh_detects_prompt_drift():
    fx = _fx(pv="currenthash")
    assert fixtures.is_fresh(fx, "currenthash") is True
    assert fixtures.is_fresh(fx, "differenthash") is False


def test_error_fixture_roundtrip(tmp_path):
    fx = Fixture(case_id="err", clip_id="single-goal", question="?", prompt_version="pv",
                 raw_program=None, error=fixtures.ERROR_GENERATION_FAILED, usage=None)
    fixtures.write_fixture(fx, fixtures_dir=tmp_path)
    loaded = fixtures.load_fixture("err", fixtures_dir=tmp_path)
    assert loaded.raw_program is None and loaded.error == "generation-failed"


def test_fixture_format_constant_recorded(tmp_path):
    fixtures.write_fixture(_fx(), fixtures_dir=tmp_path)
    d = json.loads((tmp_path / "c1.json").read_text())
    assert d["fixture_format"] == fixtures.FIXTURE_FORMAT
