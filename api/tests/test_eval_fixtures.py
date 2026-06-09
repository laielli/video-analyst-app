"""Fixture IO tests — atomic round-trip, missing -> None, prompt-drift detection, field order."""
from __future__ import annotations

import json

import fixtures as fx_mod


def _make(case_id="c1", pv="abc123", raw=None):
    return fx_mod.Fixture(
        case_id=case_id, clip_id="single-goal", question="q?", prompt_version=pv,
        raw_program=raw if raw is not None else [{"id": "x", "op": "answer", "args": {}}],
        captured_at="2026-06-09T00:00:00Z", model="seed",
    )


def test_write_load_roundtrip_atomic(tmp_path):
    fx = _make()
    path = fx_mod.write_fixture(fx, tmp_path)
    assert path.exists()
    # No temp file left behind.
    assert not list(tmp_path.glob(".*.tmp"))
    loaded = fx_mod.load_fixture("c1", tmp_path)
    assert loaded is not None
    assert loaded.case_id == fx.case_id
    assert loaded.raw_program == fx.raw_program
    assert loaded.usage is None
    assert loaded.fixture_format == fx_mod.FIXTURE_FORMAT


def test_missing_fixture_loads_none(tmp_path):
    assert fx_mod.load_fixture("nope", tmp_path) is None


def test_is_fresh_detects_prompt_drift():
    fx = _make(pv="current")
    assert fx_mod.is_fresh(fx, "current")
    assert not fx_mod.is_fresh(fx, "different")


def test_fixture_field_order_raw_program_before_provenance(tmp_path):
    """raw_program (the load-bearing diff) reads before captured_at (the noisy stamp)."""
    path = fx_mod.write_fixture(_make(), tmp_path)
    text = path.read_text()
    assert text.index('"raw_program"') < text.index('"captured_at"')


def test_error_fixture_roundtrip(tmp_path):
    fx = fx_mod.Fixture(case_id="e", clip_id="single-goal", question="q", prompt_version="p",
                        raw_program=None, captured_at="t", model="seed",
                        error=fx_mod.ERROR_GENERATION_FAILED)
    fx_mod.write_fixture(fx, tmp_path)
    loaded = fx_mod.load_fixture("e", tmp_path)
    assert loaded.raw_program is None
    assert loaded.error == fx_mod.ERROR_GENERATION_FAILED


def test_write_is_byte_stable(tmp_path):
    """Writing the same fixture twice yields identical bytes (diff-stable fixtures)."""
    fx = _make()
    p1 = fx_mod.write_fixture(fx, tmp_path)
    b1 = p1.read_bytes()
    p2 = fx_mod.write_fixture(fx, tmp_path)
    assert p2.read_bytes() == b1
