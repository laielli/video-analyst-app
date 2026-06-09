"""
Integration tests for the free-text run cache + the permalink replay route.

Headline contracts (the cost lever + the share feature):
  - Two identical free-text questions fire EXACTLY ONE codegen call (the second is a pure hit).
  - A cache hit does NOT consume the MAX_CODEGEN_CALLS budget nor increment _codegen_calls.
  - Only validated + grounded runs are cached (no poisoning from ungrounded questions).
  - A grounded run yields a stable run_id that replays the SAME doc cold (codegen disabled).
  - run_id is attached on EXACTLY the grounded path; the four non-grounded exits carry none.

Uses the conftest `mock_codegen` fixture + the class-level FakeCodegen.total_calls counter, the
autouse _reset_codegen_budget + _isolated_run_store fixtures, and a TestClient for the route.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
import canned
from conftest import FakeCodegen
from run_cache import cache_key

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())

client = TestClient(server.app)


def _schema_errors(doc):
    import jsonschema
    return list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


def _store():
    return server.get_store()


# A 64-char well-formed sha256 hex id that is NOT in the store (for the miss-path 404 test).
ABSENT_ID = "0" * 64


def _events(resp_text: str):
    out = []
    for block in resp_text.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((name, data))
    return out


# --------------------------------------------------------------------------------------
# the cost lever: identical question -> exactly one codegen call
# --------------------------------------------------------------------------------------

def test_identical_free_text_question_codegen_called_once(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    q, clip = "Does #10 score the first goal?", "single-goal"
    doc1 = server.build_free_text_run_doc(q, clip)
    doc2 = server.build_free_text_run_doc(q, clip)  # pure cache hit
    assert FakeCodegen.total_calls == 1
    assert doc1["findings"] == doc2["findings"]
    assert doc1["program"] == doc2["program"]
    assert doc2["program_source"] == "live"  # hit keeps program_source "live" (not a doc field flip)


def test_normalized_variants_share_cache_entry(mock_codegen, hero_program):
    # whitespace/case variant per the chosen (conservative) normalizer: casefold + strip + collapse
    # internal whitespace. If the normalizer were made more aggressive, these inputs would change.
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    server.build_free_text_run_doc("  does #10 score the first goal?  ", "single-goal")
    assert FakeCodegen.total_calls == 1  # the variant hit the same entry


def test_meaningfully_different_questions_do_not_collide(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    server.build_free_text_run_doc("Does #11 score the first goal?", "single-goal")
    assert FakeCodegen.total_calls == 2  # over-aggressive normalization would merge these -> 1
    # two distinct run_ids (the content-address of distinct questions differs).
    k1 = cache_key("Does #10 score the first goal?", "single-goal")
    k2 = cache_key("Does #11 score the first goal?", "single-goal")
    assert k1 != k2


# --------------------------------------------------------------------------------------
# no poisoning: ungrounded runs are NOT cached
# --------------------------------------------------------------------------------------

def test_ungrounded_run_is_not_cached(mock_codegen, hero_program):
    q, clip = "Does #10 score the first goal?", "single-goal"
    key = cache_key(q, clip)

    # codegen disabled -> ungrounded twice; nothing cached.
    mock_codegen(enabled=False)
    d1 = server.build_free_text_run_doc(q, clip)
    d2 = server.build_free_text_run_doc(q, clip)
    assert d1["program_source"] == "ungrounded" and d2["program_source"] == "ungrounded"
    assert _store().get(key) is None, "an ungrounded run must NOT populate the cache (no poisoning)"

    # a subsequent GROUNDED call for the same key is NOT short-circuited by a poisoned entry.
    mock_codegen(program=hero_program)
    d3 = server.build_free_text_run_doc(q, clip)
    assert d3["program_source"] == "live" and d3["findings"]["grounded"] is True
    assert FakeCodegen.total_calls == 1, "the grounded call must actually run codegen (no poison hit)"
    assert _store().get(key) is not None  # now it IS cached


def test_executed_but_ungrounded_run_is_not_cached(mock_codegen):
    # a VALID program that runs but doesn't ground (count over an empty window) is NOT cached.
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    mock_codegen(program=program)
    q, clip = "how many players?", "single-goal"
    doc = server.build_free_text_run_doc(q, clip)
    assert doc["program_source"] == "ungrounded"
    assert _store().get(cache_key(q, clip)) is None


# --------------------------------------------------------------------------------------
# the whole point: a cache hit does NOT consume the codegen budget
# --------------------------------------------------------------------------------------

def test_cache_hit_does_not_consume_codegen_budget(mock_codegen, hero_program, monkeypatch):
    mock_codegen(program=hero_program)
    q, clip = "Does #10 score the first goal?", "single-goal"
    # 1) warm the cache FIRST, under budget (warming increments _codegen_calls).
    server.build_free_text_run_doc(q, clip)
    assert FakeCodegen.total_calls == 1
    calls_after_warm = server._codegen_calls

    # 2) now PIN the budget at the cap — a fresh codegen would be refused (ungrounded).
    monkeypatch.setattr(server, "_codegen_calls", server.MAX_CODEGEN_CALLS)

    # 3) a repeat of the SAME question is served from the cache: grounded, no codegen, no increment.
    doc = server.build_free_text_run_doc(q, clip)
    assert doc["program_source"] == "live"
    assert doc["findings"]["grounded"] is True
    assert FakeCodegen.total_calls == 1, "the hit must not bill another codegen call"
    assert server._codegen_calls == server.MAX_CODEGEN_CALLS, "the hit must not touch the counter"
    # sanity: warming itself only billed one call.
    assert calls_after_warm == 1


# --------------------------------------------------------------------------------------
# permalink route: replay a stored run by id, cold (codegen disabled)
# --------------------------------------------------------------------------------------

def test_permalink_route_replays_stored_run(mock_codegen, hero_program):
    # warm a grounded free-text run; capture its run_id from the run_doc envelope.
    mock_codegen(program=hero_program)
    q, clip = "Does #10 score the first goal?", "single-goal"
    warm = client.get("/api/run_doc", params={"query_text": q, "clip": clip})
    assert warm.status_code == 200
    warm_doc = warm.json()
    run_id = warm_doc["run_id"]
    assert len(run_id) == 64
    assert warm_doc["cached"] is False  # first time: served live, then stored

    # now fetch the permalink COLD with codegen DISABLED — pins Phase 5 item 3 (the run branch
    # must bypass build_free_text_run_doc, so it works on a credential-less deploy).
    import codegen
    import server as srv
    # disable codegen at the seam for the replay (the permalink path must not touch it).
    saved = codegen.enabled
    codegen.enabled = lambda: False
    try:
        replay = client.get("/api/run_doc", params={"run": run_id})
    finally:
        codegen.enabled = saved
    assert replay.status_code == 200
    rdoc = replay.json()
    assert rdoc["cached"] is True
    assert rdoc["run_id"] == run_id
    # program + findings replay byte-for-byte (modulo the envelope-only run_id/cached fields).
    assert rdoc["program"] == warm_doc["program"]
    assert rdoc["findings"] == warm_doc["findings"]
    # the served doc (minus envelope-only fields) conforms to run_doc.schema.json.
    pure = {k: v for k, v in rdoc.items() if k not in ("run_id", "cached")}
    assert not _schema_errors(pure)


def test_sse_meta_carries_run_id_and_cached(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    q = "Does #10 score the first goal?"
    # first request: grounded free-text -> run_id present, cached False.
    r1 = client.get("/api/run", params={"query_text": q, "pace_ms": 0})
    assert r1.status_code == 200
    meta1 = _events(r1.text)[0][1]
    assert meta1["run_id"] is not None and len(meta1["run_id"]) == 64
    assert meta1["cached"] is False

    # repeat: served from cache -> same run_id, cached True.
    r2 = client.get("/api/run", params={"query_text": q, "pace_ms": 0})
    meta2 = _events(r2.text)[0][1]
    assert meta2["cached"] is True
    assert meta2["run_id"] == meta1["run_id"]


def test_ungrounded_sse_meta_has_no_run_id(mock_codegen):
    # codegen disabled -> ungrounded -> meta carries run_id None, cached False (no share link).
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    meta = _events(resp.text)[0][1]
    assert meta["program_source"] == "ungrounded"
    assert meta["run_id"] is None
    assert meta["cached"] is False


def test_canned_run_has_no_run_id(mock_codegen):
    # a canned run is never minted a run_id (permalinks are free-text cached runs only).
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    meta = _events(resp.text)[0][1]
    assert meta["run_id"] is None
    assert meta["cached"] is False


# --------------------------------------------------------------------------------------
# permalink route: 404 / 400 input validation
# --------------------------------------------------------------------------------------

def test_permalink_unknown_id_returns_404():
    # a well-formed but absent id -> 404 (the MISS path, not the format guard).
    for path in ("/api/run_doc", "/api/run"):
        resp = client.get(path, params={"run": ABSENT_ID})
        assert resp.status_code == 404, path


def test_permalink_malformed_id_rejected(monkeypatch):
    # spy on the store so we can assert it is NEVER touched for a malformed id.
    store = server.get_store()
    touched = {"n": 0}
    real_get = store.get

    def spy_get(key):
        touched["n"] += 1
        return real_get(key)
    monkeypatch.setattr(store, "get", spy_get)

    for bad in ("deadbeef", "../../etc/passwd", "G" * 64, "x" * 63, "x" * 65):
        for path in ("/api/run_doc", "/api/run"):
            resp = client.get(path, params={"run": bad})
            assert resp.status_code == 400, f"{path} {bad!r}"
    assert touched["n"] == 0, "a malformed run id must be rejected BEFORE the store is touched"


# --------------------------------------------------------------------------------------
# exactly-one-of-three (run | query | query_text) — the rewritten mutual exclusion
# --------------------------------------------------------------------------------------

def _status(fn):
    from fastapi import HTTPException
    try:
        fn()
        return 200
    except HTTPException as e:
        return e.status_code


def test_resolve_run_request_exactly_one_of_three():
    # bare ?run=<id> is ACCEPTED (the old two-way XOR would 400 it).
    kind, payload = server._resolve_run_request(None, None, None, ABSENT_ID)
    assert kind == "permalink" and payload == ABSENT_ID
    # run is mutually exclusive with query / query_text.
    assert _status(lambda: server._resolve_run_request("hero-10-first-goal", None, None, ABSENT_ID)) == 400
    assert _status(lambda: server._resolve_run_request(None, "how many?", None, ABSENT_ID)) == 400
    # all three -> 400; none -> 400 (unchanged).
    assert _status(lambda: server._resolve_run_request("q", "t", None, ABSENT_ID)) == 400
    assert _status(lambda: server._resolve_run_request(None, None, None, None)) == 400
    # a malformed run id -> 400 (format guard).
    assert _status(lambda: server._resolve_run_request(None, None, None, "not-hex")) == 400
    # the original two happy paths still resolve.
    assert server._resolve_run_request("hero-10-first-goal", None, None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None, None)[0] == "free"


def test_permalink_bypasses_codegen_disabled(mock_codegen, hero_program):
    """A stored run replays even with codegen disabled — the run branch must NOT route through
    build_free_text_run_doc (which would return _ungrounded('codegen-disabled')). Direct-builder
    version of the route test, pinning Phase 5 item 3 at the unit level."""
    mock_codegen(program=hero_program)
    q, clip = "Does #10 score the first goal?", "single-goal"
    server.build_free_text_run_doc(q, clip)  # warm
    run_id = cache_key(q, clip)
    # now disable codegen and replay via the permalink builder.
    mock_codegen(enabled=False)
    doc = server.build_permalink_run_doc(run_id)
    assert doc["program_source"] == "live"
    assert doc["findings"]["grounded"] is True
