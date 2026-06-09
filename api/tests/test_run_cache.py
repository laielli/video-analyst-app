"""
Unit tests for api/run_cache.py: key normalization, the content-address (sha256) key, the
FileRunStore round-trip + atomic-write discipline, eviction (evict-oldest-by-mtime), the
CACHE_FORMAT version kill-switch, and the "corrupt/stale/schema-invalid entry is a MISS and
self-heals (deletes the bad file)" rule.

These pin the seam the server depends on (api/server.py wires ONE FileRunStore at startup).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jsonschema
import pytest

import run_cache
from run_cache import CACHE_FORMAT, FileRunStore, cache_key, is_valid_key, normalize_query

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


# A minimal run-doc that conforms to run_doc.schema.json (so store.get's validation-on-read
# passes). Mirrors the schema's required keys; not a real interpreter output, just schema-valid.
def _valid_doc(query="q", answer="Yes") -> dict:
    return {
        "schema_version": "1",
        "query": query,
        "clip": {"id": "single-goal", "width": 1872, "height": 1042, "duration_ms": 5067},
        "program": [{"id": "r", "op": "answer", "args": {}}],
        "trace": [{"id": "r", "op": "answer", "status": "done", "source": "cached"}],
        "findings": {"answer": answer, "verdict": f"{answer} — grounded", "grounded": True,
                     "reason": None, "partial": False},
        "program_source": "live",
    }


def _assert_schema_valid(doc):
    assert not list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


# --------------------------------------------------------------------------------------
# normalization contract (the one knob: casefold + strip + collapse internal whitespace)
# --------------------------------------------------------------------------------------

def test_normalize_query_casefold_and_whitespace():
    # default contract: strip + casefold + collapse internal runs of whitespace to one space.
    assert normalize_query("  Does #10 SCORE? ") == "does #10 score?"
    assert normalize_query("how   many\tplayers\nare visible") == "how many players are visible"
    # casefold (not just lower) + idempotent on an already-normal string.
    assert normalize_query("does #10 score?") == "does #10 score?"
    # punctuation is NOT stripped (conservative default — distinct questions never collide).
    assert normalize_query("does #10 score") != normalize_query("does #10 score?")


# --------------------------------------------------------------------------------------
# the content-address key: stable, clip-scoped, full sha256 hex
# --------------------------------------------------------------------------------------

def test_cache_key_stable_and_clip_scoped():
    # identical inputs -> identical key across calls (deterministic permalink).
    k1 = cache_key("Does #10 score?", "single-goal")
    k2 = cache_key("Does #10 score?", "single-goal")
    assert k1 == k2
    # same text, DIFFERENT clip -> different key (clip id is part of the key, raw).
    assert cache_key("Does #10 score?", "single-goal") != cache_key("Does #10 score?", "bernabeu-counter")
    # normalization folds whitespace/case variants to the SAME key.
    assert cache_key("  does #10 SCORE? ", "single-goal") == cache_key("Does #10 score?", "single-goal")
    # the key is a full 64-char sha256 hex (the route's ^[0-9a-f]{64}$ guard depends on this).
    assert is_valid_key(k1)
    assert len(k1) == 64


def test_is_valid_key_rejects_traversal_and_short_ids():
    assert is_valid_key("a" * 64)
    assert not is_valid_key("deadbeef")                 # too short
    assert not is_valid_key("../../etc/passwd")         # traversal
    assert not is_valid_key("g" * 64)                   # non-hex char
    assert not is_valid_key("A" * 64)                   # uppercase not allowed
    assert not is_valid_key("a" * 63)                   # off-by-one length


# --------------------------------------------------------------------------------------
# FileRunStore round-trip + atomic write (no .tmp left behind)
# --------------------------------------------------------------------------------------

def test_file_store_round_trip_and_atomic(tmp_path):
    store = FileRunStore(tmp_path, max_entries=10)
    key = cache_key("q", "single-goal")
    doc = _valid_doc()
    assert store.get(key) is None  # cold miss
    store.put(key, doc)
    got = store.get(key)
    assert got == doc  # equal dict round-trips
    # the on-disk file is a valid JSON ENVELOPE (cache_format + doc), not a bare doc.
    on_disk = json.loads((tmp_path / f"{key}.json").read_text())
    assert on_disk["cache_format"] == CACHE_FORMAT
    assert on_disk["doc"] == doc
    # no temp .tmp file left behind by the atomic write.
    assert list(tmp_path.glob("*.tmp")) == []


def test_put_is_idempotent_overwrite(tmp_path):
    store = FileRunStore(tmp_path, max_entries=10)
    key = cache_key("q", "single-goal")
    store.put(key, _valid_doc(answer="Yes"))
    store.put(key, _valid_doc(answer="No"))  # overwrite same key
    assert store.get(key)["findings"]["answer"] == "No"
    assert len(list(tmp_path.glob("*.json"))) == 1  # one entry, not two


def test_store_refuses_malformed_key_on_put(tmp_path):
    store = FileRunStore(tmp_path, max_entries=10)
    # defense in depth: put with a non-key never composes a path.
    with pytest.raises(ValueError):
        store.put("../../etc/passwd", _valid_doc())
    # get with a malformed key is a hard miss (never a file read).
    assert store.get("../../etc/passwd") is None
    assert store.get("deadbeef") is None


# --------------------------------------------------------------------------------------
# eviction: soft cap, evict-oldest-by-mtime (FIFO, not LRU)
# --------------------------------------------------------------------------------------

def test_eviction_caps_entry_count(tmp_path):
    store = FileRunStore(tmp_path, max_entries=2)
    keys = [cache_key(f"q{i}", "single-goal") for i in range(3)]
    for k in keys:
        store.put(k, _valid_doc(query=k))
        time.sleep(0.01)  # ensure distinct mtimes so "oldest" is well-defined
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2, "soft cap not enforced"
    # the OLDEST (first written) was evicted; the two newest remain.
    assert store.get(keys[0]) is None
    assert store.get(keys[1]) is not None
    assert store.get(keys[2]) is not None


def test_eviction_disabled_when_max_entries_none(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    for i in range(5):
        store.put(cache_key(f"q{i}", "single-goal"), _valid_doc())
    assert len(list(tmp_path.glob("*.json"))) == 5


# --------------------------------------------------------------------------------------
# corrupt / schema-invalid entry -> MISS + self-heal (delete the bad file)
# --------------------------------------------------------------------------------------

def test_corrupt_cache_entry_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=10)
    key = cache_key("q", "single-goal")
    path = tmp_path / f"{key}.json"

    # 1) non-JSON garbage.
    path.write_text("this is not json {{{")
    assert store.get(key) is None
    assert not path.exists(), "garbage entry must self-heal (be deleted)"

    # 2) valid JSON envelope but the doc is schema-INVALID (missing required keys).
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT, "doc": {"nope": True}}))
    assert store.get(key) is None
    assert not path.exists(), "schema-invalid entry must self-heal"


def test_stale_cache_format_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=10)
    key = cache_key("q", "single-goal")
    path = tmp_path / f"{key}.json"
    # a valid doc wrapped with a STALE cache_format -> the kill-switch fires.
    doc = _valid_doc()
    _assert_schema_valid(doc)
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT - 1, "doc": doc}))
    assert store.get(key) is None
    assert not path.exists(), "stale-format entry must self-heal (CACHE_FORMAT kill-switch)"


def test_get_missing_file_is_plain_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=10)
    # a well-formed but absent key -> None, no exception, no file created.
    assert store.get("a" * 64) is None
    assert list(tmp_path.glob("*.json")) == []


def test_runstore_protocol_is_satisfied(tmp_path):
    # FileRunStore structurally satisfies the RunStore Protocol (the Blob seam).
    store = FileRunStore(tmp_path, max_entries=10)
    assert isinstance(store, run_cache.RunStore)
