"""
Unit tests for the content-addressed run-doc cache (api/run_cache.py): key normalization, the
file-backed store round-trip + atomicity, eviction, CACHE_FORMAT versioning, and the
"don't serve corrupt/stale entries" self-healing-miss rule.

These are pure-store tests (no server, no codegen). The free-text integration — codegen-call
counts, replay-from-id, poisoning — lives in test_free_text_cache.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_cache
from run_cache import FileRunStore, CACHE_FORMAT, normalize_query, cache_key, is_valid_key

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


def _grounded_doc(clip_id: str = "single-goal", answer: str = "Yes") -> dict:
    """A minimal run-doc that conforms to run_doc.schema.json (so it survives validate-on-read)."""
    return {
        "schema_version": "1",
        "query": "Does #10 score the first goal?",
        "clip": {"id": clip_id, "width": 1872, "height": 1042, "duration_ms": 5067},
        "program": [{"id": "r", "op": "answer", "args": {"from": "x", "question": "q"}}],
        "trace": [{"id": "r", "op": "answer", "status": "done", "source": "live"}],
        "findings": {"answer": answer, "verdict": f"{answer} — proven", "grounded": True},
        "program_source": "live",
    }


# --------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------

def test_normalize_query_casefold_and_whitespace():
    # The chosen (conservative) normalizer: casefold + strip + collapse internal whitespace.
    assert normalize_query("  Does #10 SCORE? ") == "does #10 score?"
    assert normalize_query("how   many\tplayers\n?") == "how many players ?"
    # casefold is Unicode-aware lowercase; punctuation is preserved (no stripping).
    assert normalize_query("Does #10 score?") == "does #10 score?"
    # a whitespace/case variant collapses to the same canonical form.
    assert normalize_query("  does #10 score the first goal?  ") == \
        normalize_query("Does #10 score the first goal?")


def test_cache_key_stable_and_clip_scoped():
    # identical inputs -> identical key across calls (deterministic content-address).
    k1 = cache_key("How many players?", "single-goal")
    k2 = cache_key("How many players?", "single-goal")
    assert k1 == k2
    # same text + DIFFERENT clip -> different key (clip is part of the address).
    assert cache_key("How many players?", "single-goal") != cache_key("How many players?", "bernabeu-counter")
    # the key is a 64-char lowercase hex sha256 digest (the permalink-guard shape).
    assert is_valid_key(k1)
    assert len(k1) == 64

    # a normalized-variant text shares the key (encodes the conservative normalizer).
    assert cache_key("  Does #10 SCORE the first goal?  ", "single-goal") == \
        cache_key("Does #10 score the first goal?", "single-goal")
    # a meaningfully different question does NOT collide.
    assert cache_key("Does #10 score?", "single-goal") != cache_key("Does #7 score?", "single-goal")


def test_is_valid_key_rejects_path_traversal_and_bad_shapes():
    assert is_valid_key("a" * 64)
    assert not is_valid_key("deadbeef")              # too short
    assert not is_valid_key("g" * 64)                # non-hex char
    assert not is_valid_key("../../etc/passwd")      # path traversal
    assert not is_valid_key("A" * 64)                # uppercase hex not allowed (key is lowercased)
    assert not is_valid_key("a" * 63 + "/")          # contains a separator


# --------------------------------------------------------------------------------------
# store round-trip + atomicity
# --------------------------------------------------------------------------------------

def test_file_store_round_trip_and_atomic(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    doc = _grounded_doc()
    store.put(key, doc)
    got = store.get(key)
    assert got == doc  # round-trips to an equal dict

    # the on-disk file is valid JSON wrapped in the versioned envelope.
    on_disk = json.loads((tmp_path / f"{key}.json").read_text())
    assert on_disk["cache_format"] == CACHE_FORMAT
    assert on_disk["doc"] == doc
    # no stray temp file left behind after a successful write.
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_key_is_a_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    assert store.get(cache_key("never-stored", "single-goal")) is None


def test_store_refuses_path_traversal_keys(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    # get/put on a malformed key never composes a path — get is a miss, put a no-op.
    assert store.get("../../etc/passwd") is None
    store.put("../../etc/passwd", _grounded_doc())  # no-op, no file escapes the root
    assert not any(p.name != "" for p in tmp_path.parent.glob("passwd*"))


# --------------------------------------------------------------------------------------
# eviction
# --------------------------------------------------------------------------------------

def test_eviction_caps_entry_count(tmp_path):
    import os
    import time

    store = FileRunStore(tmp_path, max_entries=2)
    keys = [cache_key(f"q{i}", "single-goal") for i in range(3)]
    for i, k in enumerate(keys):
        store.put(k, _grounded_doc(answer=f"A{i}"))
        # force a strictly increasing mtime so "oldest" is unambiguous (avoids same-tick ties).
        os.utime(tmp_path / f"{k}.json", (time.time() + i, time.time() + i))
        store._evict_over_cap()  # re-run after fixing mtime so the cap holds on the true ordering

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2, "soft cap of 2 must leave exactly 2 entries"
    # the oldest (keys[0]) was evicted; the two newest survive.
    assert store.get(keys[0]) is None
    assert store.get(keys[1]) is not None
    assert store.get(keys[2]) is not None


# --------------------------------------------------------------------------------------
# corruption + version self-healing
# --------------------------------------------------------------------------------------

def test_corrupt_cache_entry_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    path = tmp_path / f"{key}.json"

    # (1) non-JSON garbage -> miss + self-heal (file deleted).
    path.write_text("{ not json at all")
    assert store.get(key) is None
    assert not path.exists(), "corrupt file must be self-healed (deleted) on read"

    # (2) valid JSON but schema-invalid doc -> miss + self-heal.
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT, "doc": {"nope": True}}))
    assert store.get(key) is None
    assert not path.exists()

    # (3) a valid envelope round-trips again after the self-heal (store is usable).
    store.put(key, _grounded_doc())
    assert store.get(key) is not None


def test_stale_cache_format_treated_as_miss(tmp_path):
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    path = tmp_path / f"{key}.json"
    # a structurally-valid doc but an OLD cache_format -> the kill-switch invalidates it.
    path.write_text(json.dumps({"cache_format": CACHE_FORMAT - 1, "doc": _grounded_doc()}))
    assert store.get(key) is None
    assert not path.exists(), "stale-format file must be self-healed (the CACHE_FORMAT kill-switch)"


def test_put_then_corrupt_then_repopulate_via_codegen_is_clean(tmp_path):
    # round-trip resilience: a corrupt entry never wedges the key permanently.
    store = FileRunStore(tmp_path, max_entries=None)
    key = cache_key("q", "single-goal")
    store.put(key, _grounded_doc(answer="Yes"))
    (tmp_path / f"{key}.json").write_text("garbage")
    assert store.get(key) is None
    store.put(key, _grounded_doc(answer="No"))
    assert store.get(key)["findings"]["answer"] == "No"
