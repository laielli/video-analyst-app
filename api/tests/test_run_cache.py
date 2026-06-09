"""
Unit tests for the content-addressed run cache (api/run_cache.py): key normalization, the
file-backed RunStore round-trip + atomic writes, eviction (FIFO by mtime), CACHE_FORMAT versioning,
schema-validate-on-read, and the self-healing "corrupt/stale entry -> miss + delete" rule.

These are pure-layer tests (no server, no codegen): they construct a FileRunStore over tmp_path and
exercise its contract directly. The integration with build_free_text_run_doc + the permalink route
lives in test_free_text_cache.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_cache
from run_cache import (
    CACHE_FORMAT,
    FileRunStore,
    cache_key,
    is_valid_key,
    normalize_query,
)

API_DIR = Path(__file__).resolve().parent.parent


def _grounded_doc(query="Does #10 score the first goal?", clip_id="single-goal") -> dict:
    """A minimal run-doc that conforms to run_doc.schema.json (so store.put accepts it and
    store.get returns it). Mirrors the schema's required keys + a grounded findings block."""
    return {
        "schema_version": "1",
        "query": query,
        "clip": {"id": clip_id, "width": 1872, "height": 1042, "duration_ms": 5067},
        "program": [{"id": "result", "op": "answer", "args": {}}],
        "trace": [{"id": "result", "op": "answer", "status": "done", "source": "cached"}],
        "findings": {"answer": "Yes", "verdict": "Yes — #10 scored", "grounded": True, "reason": None},
        "program_source": "live",
    }


# --------------------------------------------------------------------------------------
# normalization contract
# --------------------------------------------------------------------------------------

def test_normalize_query_casefold_and_whitespace():
    # The conservative default: casefold + strip + collapse internal whitespace, nothing else.
    assert normalize_query("  Does #10 SCORE? ") == "does #10 score?"
    # internal whitespace collapses (tabs/newlines too).
    assert normalize_query("does   #10\tscore?") == "does #10 score?"
    assert normalize_query("does\n#10 score?") == "does #10 score?"
    # punctuation is NOT stripped (lowest false-merge risk) — '?' survives.
    assert normalize_query("does 10 score") != normalize_query("does 10 score?")


def test_cache_key_stable_and_clip_scoped():
    text = "Does #10 score the first goal?"
    # identical inputs -> identical key across calls.
    assert cache_key(text, "single-goal") == cache_key(text, "single-goal")
    # same text, different clip -> different key (clip id is part of the key, raw).
    assert cache_key(text, "single-goal") != cache_key(text, "bernabeu-counter")
    # the key is a 64-char sha256 hex (the run_id shape the route guard depends on).
    assert is_valid_key(cache_key(text, "single-goal"))
    # whitespace/case variants collapse to the same key (the normalizer feeds the key).
    assert cache_key("  does #10 score the first goal?  ", "single-goal") == cache_key(text, "single-goal")


def test_is_valid_key_rejects_traversal_and_malformed():
    assert is_valid_key("a" * 64)
    assert not is_valid_key("a" * 63)       # too short
    assert not is_valid_key("a" * 65)       # too long
    assert not is_valid_key("A" * 64)       # uppercase (sha256 hex is lowercase)
    assert not is_valid_key("../../etc/passwd")
    assert not is_valid_key("deadbeef")
    assert not is_valid_key("g" * 64)       # non-hex char


# --------------------------------------------------------------------------------------
# store round-trip + atomic write
# --------------------------------------------------------------------------------------

def test_file_store_round_trip_and_atomic(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("Does #10 score the first goal?", "single-goal")
    doc = _grounded_doc()
    store.put(key, doc)

    got = store.get(key)
    assert got == doc

    # the on-disk file is valid JSON and is the versioned envelope, not the bare doc.
    on_disk = json.loads((tmp_path / f"{key}.json").read_text())
    assert on_disk["cache_format"] == CACHE_FORMAT
    assert on_disk["doc"] == doc

    # no temp .tmp file left behind (atomic write cleans up).
    assert not list(tmp_path.glob("*.tmp"))


def test_put_idempotent_overwrite(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    store.put(key, _grounded_doc())
    store.put(key, _grounded_doc(query="q2"))  # same key, second write overwrites
    assert store.get(key)["query"] == "q2"
    assert len(list(tmp_path.glob("*.json"))) == 1  # still one entry


def test_get_unknown_key_is_none(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    assert store.get(cache_key("never stored", "single-goal")) is None


def test_get_rejects_malformed_key_without_touching_fs(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    # a traversal-shaped key must never compose a path read.
    assert store.get("../../etc/passwd") is None
    assert store.get("a" * 63) is None


def test_put_rejects_non_conforming_doc(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    with pytest.raises(ValueError):
        store.put(key, {"not": "a run-doc"})  # fails run_doc.schema.json
    assert store.get(key) is None  # nothing persisted


# --------------------------------------------------------------------------------------
# eviction (FIFO by mtime)
# --------------------------------------------------------------------------------------

def test_eviction_caps_entry_count(tmp_path):
    import os
    store = FileRunStore(tmp_path, max_entries=2)
    keys = [cache_key(f"q{i}", "single-goal") for i in range(3)]
    for i, k in enumerate(keys):
        store.put(k, _grounded_doc(query=f"q{i}"))
        # force a strictly-increasing mtime so "oldest" is unambiguous on fast filesystems.
        os.utime(tmp_path / f"{k}.json", (1000 + i, 1000 + i))
        # re-run the cap sweep now that mtime is deterministic.
        store._evict()

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2  # soft cap holds
    # the oldest (q0) was evicted; the two newest survive.
    assert store.get(keys[0]) is None
    assert store.get(keys[1]) is not None
    assert store.get(keys[2]) is not None


def test_no_eviction_when_max_entries_none(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    for i in range(5):
        store.put(cache_key(f"q{i}", "single-goal"), _grounded_doc(query=f"q{i}"))
    assert len(list(tmp_path.glob("*.json"))) == 5


# --------------------------------------------------------------------------------------
# self-healing: corrupt / stale entries treated as a miss + deleted
# --------------------------------------------------------------------------------------

def test_corrupt_cache_entry_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    path = tmp_path / f"{key}.json"

    # (a) non-JSON garbage -> miss + self-heal (file deleted).
    path.write_text("}{ not json")
    assert store.get(key) is None
    assert not path.exists()

    # (b) schema-invalid JSON in the envelope -> miss + self-heal.
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT, "doc": {"bogus": True}}))
    assert store.get(key) is None
    assert not path.exists()


def test_stale_cache_format_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    path = tmp_path / f"{key}.json"
    # a valid doc but an OLD cache_format -> the CACHE_FORMAT kill-switch -> miss + delete.
    path.write_text(json.dumps({"cache_format": 0, "doc": _grounded_doc()}))
    assert store.get(key) is None
    assert not path.exists()


def test_runtime_protocol_check():
    # FileRunStore satisfies the RunStore Protocol (the Blob seam) at runtime.
    assert isinstance(FileRunStore("/tmp/x", max_entries=None), run_cache.RunStore)
