"""Fixture IO tests — atomic round-trip, MISSING -> None, prompt-drift detection. Creds-free."""
from __future__ import annotations

from eval.fixtures import Fixture, fixture_path, is_fresh, load_fixture, write_fixture


def _fx(case_id="c1", pv="pv1", raw=None):
    return Fixture(case_id=case_id, clip_id="single-goal", question="q?",
                   prompt_version=pv, raw_program=raw if raw is not None else [{"id": "x", "op": "answer", "args": {}}])


def test_write_load_roundtrip_atomic(tmp_path):
    fx = _fx()
    path = write_fixture(fx, fixtures_dir=tmp_path)
    assert path.exists()
    loaded = load_fixture("c1", fixtures_dir=tmp_path)
    assert loaded is not None
    assert loaded.case_id == "c1"
    assert loaded.raw_program == fx.raw_program
    assert loaded.usage is None
    # No stray temp files left behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_fixture_loads_none(tmp_path):
    assert load_fixture("absent-case", fixtures_dir=tmp_path) is None


def test_is_fresh_detects_prompt_drift():
    fx = _fx(pv="aaaa")
    assert is_fresh(fx, "aaaa") is True
    assert is_fresh(fx, "bbbb") is False


def test_error_fixture_roundtrip(tmp_path):
    fx = Fixture(case_id="e1", clip_id="single-goal", question="?", prompt_version="pv",
                 raw_program=None, error="generation-failed")
    write_fixture(fx, fixtures_dir=tmp_path)
    loaded = load_fixture("e1", fixtures_dir=tmp_path)
    assert loaded.raw_program is None and loaded.error == "generation-failed"


def test_fixture_path_uses_case_id(tmp_path):
    assert fixture_path("my-case", fixtures_dir=tmp_path).name == "my-case.json"


def test_to_dict_field_order_puts_raw_program_before_provenance():
    """Field order keeps re-capture restamp noise (captured_at) out of the lines a reviewer scans
    first (Deferred note)."""
    keys = list(_fx().to_dict().keys())
    assert keys.index("raw_program") < keys.index("captured_at")
    assert keys.index("case_id") < keys.index("raw_program")
