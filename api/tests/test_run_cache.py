"""
Unit tests for api/run_cache.py — the content-addressed cache + file-backed persistence seam.

Covers: the normalization contract (the one design knob), key stability + clip-scoping, the
file store round-trip + atomic write, evict-oldest-by-mtime, the CACHE_FORMAT kill-switch, and
the self-healing "corrupt/stale entry -> miss + delete" rules. No server / TestClient here; this
file pins run_cache in isolation (the integration wiring lives in test_free_text_cache.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from run_cache import (
    CACHE_FORMAT,
    FileRunStore,
    cache_key,
    normalize_query,
)

API_DIR = Path(__file__).resolve().parent.parent


def _grounded_doc(query="Does #10 score the first goal?", clip_id="single-goal") -> dict:
    """A minimal schema-valid run-doc (so store.get's schema validation passes). Mirrors the
    run_doc.schema.json required keys; the cache never inspects the doc beyond schema validity."""
    return {
        "schema_version": "1",
        "query": query,
        "clip": {"id": clip_id, "width": 1872, "height": 1042, "duration_ms": 5067},
        "program": [{"id": "r", "op": "answer", "args": {}}],
        "trace": [{"id": "r", "op": "answer", "status": "done", "source": "cached"}],
        "findings": {"answer": "Yes", "verdict": "Yes — #10 scored", "grounded": True, "reason": None},
        "program_source": "live",
    }


# --------------------------------------------------------------------------------------
# normalize_query — the contract (casefold + strip + collapse internal whitespace, nothing else)
# --------------------------------------------------------------------------------------

def test_normalize_query_casefold_and_whitespace():
    # exact contract for the chosen (conservative) normalizer.
    assert normalize_query("  Does #10 SCORE? ") == "does #10 score?"
    # internal whitespace collapses (tabs/newlines/multi-space -> single space).
    assert normalize_query("does   #10\tscore\nthe  goal?") == "does #10 score the goal?"
    # casefold is applied; punctuation is DELIBERATELY preserved (no aggressive stripping).
    assert normalize_query("DOES IT?") == "does it?"
    assert normalize_query("a. b.") == "a. b."  # trailing punctuation NOT stripped


def test_normalize_query_does_not_merge_meaningfully_different():
    # the conservative normalizer must NOT collapse two distinct questions.
    assert normalize_query("does #10 score?") != normalize_query("does #11 score?")
    assert normalize_query("how many players?") != normalize_query("how many goals?")


# --------------------------------------------------------------------------------------
# cache_key — stable, clip-scoped, full sha256 hex
# --------------------------------------------------------------------------------------

def test_cache_key_stable_and_clip_scoped():
    text = "Does #10 score the first goal?"
    k1 = cache_key(text, "single-goal")
    k2 = cache_key(text, "single-goal")
    assert k1 == k2  # deterministic across calls
    # same text, different clip -> different key (clip is part of the address, raw).
    assert cache_key(text, "single-goal") != cache_key(text, "bernabeu-counter")
    # normalized variants share a key.
    assert cache_key("  DOES #10 score the first goal? ", "single-goal") == k1
    # full sha256 hex (the route guard depends on the fixed 64-char length).
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


# --------------------------------------------------------------------------------------
# FileRunStore — round trip + atomic write
# --------------------------------------------------------------------------------------

def test_file_store_round_trip_and_atomic(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=10)
    key = cache_key("q", "single-goal")
    doc = _grounded_doc()
    assert store.get(key) is None  # cold miss
    store.put(key, doc)
    got = store.get(key)
    assert got == doc  # equal dict round-trips

    # the on-disk file is valid JSON wrapping the doc under the version envelope.
    path = tmp_path / "rc" / f"{key}.json"
    assert path.exists()
    envelope = json.loads(path.read_text())
    assert envelope["cache_format"] == CACHE_FORMAT
    assert envelope["doc"] == doc
    # no leftover temp file from the atomic write.
    assert not list((tmp_path / "rc").glob("*.tmp"))


def test_put_is_idempotent(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=10)
    key = cache_key("q", "single-goal")
    doc = _grounded_doc()
    store.put(key, doc)
    store.put(key, doc)  # re-put same key -> still exactly one file
    assert len(list((tmp_path / "rc").glob("*.json"))) == 1
    assert store.get(key) == doc


# --------------------------------------------------------------------------------------
# path-safety: a traversal key is refused (defense in depth)
# --------------------------------------------------------------------------------------

def test_store_refuses_unsafe_keys(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=10)
    for bad in ("../../etc/passwd", "deadbeef", "a/b", "x" * 63, "G" * 64, "x" * 65):
        assert store.get(bad) is None
        store.put(bad, _grounded_doc())  # must be a no-op, never compose a path
    # nothing was written outside the (still-empty) store root.
    assert not list((tmp_path / "rc").glob("*.json"))


# --------------------------------------------------------------------------------------
# eviction — evict-oldest-by-mtime, soft cap
# --------------------------------------------------------------------------------------

def test_eviction_caps_entry_count(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=2)
    keys = [cache_key(f"q{i}", "single-goal") for i in range(3)]
    for k in keys:
        store.put(k, _grounded_doc(query=k))
        time.sleep(0.01)  # ensure distinct mtimes so oldest is unambiguous
    files = list((tmp_path / "rc").glob("*.json"))
    assert len(files) == 2  # soft cap holds
    # the OLDEST (first written) was evicted; the two newest survive.
    assert store.get(keys[0]) is None
    assert store.get(keys[1]) is not None
    assert store.get(keys[2]) is not None


def test_eviction_disabled_when_max_entries_none(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=None)
    for i in range(5):
        store.put(cache_key(f"q{i}", "single-goal"), _grounded_doc())
    assert len(list((tmp_path / "rc").glob("*.json"))) == 5


# --------------------------------------------------------------------------------------
# self-healing: corrupt entry -> miss + deleted
# --------------------------------------------------------------------------------------

def test_corrupt_cache_entry_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=10)
    key = cache_key("q", "single-goal")
    path = tmp_path / "rc" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # (a) non-JSON garbage.
    path.write_text("{not json at all")
    assert store.get(key) is None
    assert not path.exists()  # self-healed (bad file deleted)

    # (b) schema-invalid JSON (valid envelope, but the doc is missing required keys).
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT, "doc": {"nope": 1}}))
    assert store.get(key) is None
    assert not path.exists()


def test_stale_cache_format_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path / "rc", max_entries=10)
    key = cache_key("q", "single-goal")
    path = tmp_path / "rc" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # a perfectly valid doc, but stamped with an OLD cache_format -> the kill-switch fires.
    path.write_text(json.dumps({"cache_format": 0, "doc": _grounded_doc()}))
    assert store.get(key) is None  # version mismatch -> miss
    assert not path.exists()  # and deleted


def test_stored_doc_is_schema_pure(tmp_path):
    """The stored doc must NOT carry run_id/cached (those live in the meta payload only). The
    cache stores exactly what it was given under the envelope; the server is responsible for
    never handing it a polluted doc — assert the wrapper keeps the doc verbatim."""
    store = FileRunStore(tmp_path / "rc", max_entries=10)
    key = cache_key("q", "single-goal")
    doc = _grounded_doc()
    store.put(key, doc)
    envelope = json.loads((tmp_path / "rc" / f"{key}.json").read_text())
    assert "run_id" not in envelope["doc"]
    assert "cached" not in envelope["doc"]
