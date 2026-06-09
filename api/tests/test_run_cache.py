"""
Unit tests for api/run_cache.py: the content-addressed key normalization + derivation, the
FileRunStore round-trip / atomic-write discipline, eviction, and the self-healing miss rules
(corrupt entry, stale CACHE_FORMAT). These pin the cache layer in isolation from the server.

The normalizer here encodes the CONSERVATIVE default (casefold + strip + collapse internal
whitespace, nothing else) — the one DESIGN-ROOM knob. If a more aggressive normalizer is
chosen, `test_normalize_query_casefold_and_whitespace` must change with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_cache
from run_cache import CACHE_FORMAT, FileRunStore, cache_key, normalize_query

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


def _grounded_doc(query="Does #10 score the first goal?", clip_id="single-goal"):
    """A minimal schema-valid grounded run-doc (enough to pass run_doc.schema.json on store.get)."""
    return {
        "schema_version": "1",
        "query": query,
        "clip": {"id": clip_id, "width": 1872, "height": 1042, "duration_ms": 5067},
        "program": [{"id": "r", "op": "answer", "args": {"from": "x", "question": query}}],
        "trace": [{"id": "r", "op": "answer", "status": "done", "source": "cached"}],
        "findings": {"answer": "Yes", "verdict": "Yes — #10 scored", "grounded": True,
                     "reason": None, "partial": False},
        "program_source": "live",
    }


# ---- normalization + key derivation ------------------------------------------------------

def test_normalize_query_casefold_and_whitespace():
    # casefold + strip + collapse internal whitespace; punctuation is PRESERVED (conservative).
    assert normalize_query("  Does #10 SCORE? ") == "does #10 score?"
    assert normalize_query("how   many\tplayers\n?") == "how many players ?"
    # idempotent: normalizing an already-normalized string is a no-op.
    n = normalize_query("does #10 score?")
    assert normalize_query(n) == n


def test_cache_key_stable_and_clip_scoped():
    # identical inputs -> identical key across calls (deterministic content address).
    k1 = cache_key("Does #10 score?", "single-goal")
    k2 = cache_key("Does #10 score?", "single-goal")
    assert k1 == k2
    # same text, different clip -> different key (clip id is part of the address).
    assert cache_key("Does #10 score?", "single-goal") != cache_key("Does #10 score?", "bernabeu-counter")
    # the key is a 64-char sha256 hex digest (the route's ^[0-9a-f]{64}$ guard depends on this).
    assert run_cache.is_valid_key(k1)
    assert len(k1) == 64

    # normalized variants collide (whitespace/case), meaningfully-different questions do not.
    assert cache_key("Does #10 score?", "single-goal") == cache_key("  does #10 SCORE?  ", "single-goal")
    assert cache_key("does #10 score?", "single-goal") != cache_key("does #11 score?", "single-goal")


def test_is_valid_key_rejects_path_chars():
    assert not run_cache.is_valid_key("../../etc/passwd")
    assert not run_cache.is_valid_key("deadbeef")            # too short
    assert not run_cache.is_valid_key("g" * 64)              # non-hex char
    assert not run_cache.is_valid_key("a" * 63)              # 63 chars
    assert run_cache.is_valid_key("a" * 64)


# ---- file store round-trip + atomicity ---------------------------------------------------

def test_file_store_round_trip_and_atomic(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("Does #10 score?", "single-goal")
    doc = _grounded_doc()
    store.put(key, doc)

    # put then get returns an equal dict.
    got = store.get(key)
    assert got == doc

    # the on-disk file is valid JSON wrapped in the versioned envelope.
    path = tmp_path / f"{key}.json"
    assert path.exists()
    envelope = json.loads(path.read_text())
    assert envelope["cache_format"] == CACHE_FORMAT
    assert envelope["doc"] == doc

    # no temp .tmp file left behind after an atomic write.
    assert list(tmp_path.glob("*.tmp")) == []


def test_get_unknown_key_is_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    assert store.get(cache_key("never stored", "single-goal")) is None


def test_get_malformed_key_is_miss_not_path_read(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    # a path-traversal-ish key never composes a path; it is a plain miss.
    assert store.get("../../etc/passwd") is None
    # put on a malformed key raises (never writes outside root).
    with pytest.raises(ValueError):
        store.put("../../evil", _grounded_doc())


# ---- eviction ----------------------------------------------------------------------------

def test_eviction_caps_entry_count(tmp_path):
    store = FileRunStore(tmp_path, max_entries=2)
    keys = [cache_key(f"q{i}", "single-goal") for i in range(3)]
    import os
    import time
    for i, k in enumerate(keys):
        store.put(k, _grounded_doc(query=f"q{i}"))
        # force distinct mtimes so evict-oldest is deterministic (mtime resolution can be coarse).
        os.utime(tmp_path / f"{k}.json", (1000 + i, 1000 + i))
        time.sleep(0.001)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2, "soft cap of 2 enforced after the third put"
    # the oldest (keys[0]) was evicted; the two newest survive.
    assert store.get(keys[0]) is None
    assert store.get(keys[1]) is not None
    assert store.get(keys[2]) is not None


def test_no_eviction_when_max_entries_none(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    for i in range(5):
        store.put(cache_key(f"q{i}", "single-goal"), _grounded_doc(query=f"q{i}"))
    assert len(list(tmp_path.glob("*.json"))) == 5


# ---- self-healing miss rules -------------------------------------------------------------

def test_corrupt_cache_entry_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("Does #10 score?", "single-goal")
    path = tmp_path / f"{key}.json"

    # (1) non-JSON garbage -> miss + file deleted (self-healing).
    path.write_text("{not json at all")
    assert store.get(key) is None
    assert not path.exists()

    # (2) schema-invalid JSON (valid envelope, but the doc violates run_doc.schema.json) -> miss
    #     + file deleted.
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT, "doc": {"not": "a run doc"}}))
    assert store.get(key) is None
    assert not path.exists()


def test_stale_cache_format_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("Does #10 score?", "single-goal")
    path = tmp_path / f"{key}.json"
    # a valid doc but an OLD cache_format -> the kill-switch invalidates it (miss + delete).
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT - 1, "doc": _grounded_doc()}))
    assert store.get(key) is None
    assert not path.exists()


def test_round_trip_doc_conforms_to_schema():
    # the helper doc this suite stores must itself be schema-valid (guards the test fixture).
    import jsonschema
    errs = list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(_grounded_doc()))
    assert errs == [], errs
