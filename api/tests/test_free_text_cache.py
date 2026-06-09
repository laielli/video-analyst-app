"""
Integration tests for the free-text program cache + shareable permalink replays, over
server.build_free_text_run_doc + the /api/run, /api/run_doc routes.

Headline contracts (the cost win + the trust boundary + the share story):
  1. two identical free-text questions => EXACTLY ONE codegen call (FakeCodegen.total_calls == 1).
  2. a grounded free-text run yields a run_id that, fetched cold (codegen disabled), replays the
     same run-doc (program/findings) byte-for-byte and conforms to run_doc.schema.json.
  3. ungrounded/failed runs are NOT cached as if successful (no poisoning), and never carry a run_id.
  4. a cache hit does NOT consume the MAX_CODEGEN_CALLS budget nor increment _codegen_calls.

Test isolation: conftest autouse fixtures reset server._codegen_calls to 0 and point
server._run_store at a tmp_path-rooted FileRunStore per test (so these assertions are not
order-dependent and never touch the real api/.run_cache/).

Pattern mirror: test_free_text_run.py (build_free_text_run_doc + _schema_errors), test_sse_stream.py
(TestClient + the SSE `events` parser).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
import run_cache
from conftest import FakeCodegen

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())

client = TestClient(server.app)

HERO_TEXT = "Does #10 score the first goal?"
HERO_CLIP = "single-goal"


def _schema_errors(doc):
    import jsonschema
    doc = {k: v for k, v in doc.items() if k != "_serve"}
    return list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


def _events(resp_text: str):
    """Parse an SSE body into [(name, dict), ...] (copied from test_sse_stream.py)."""
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
# (1) identical question -> exactly one codegen call
# --------------------------------------------------------------------------------------

def test_identical_free_text_question_codegen_called_once(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    d1 = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    d2 = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    assert FakeCodegen.total_calls == 1, "the second identical question must be a pure cache hit"
    # both docs ground to the same findings (the second is served from the store).
    assert d1["findings"] == d2["findings"]
    assert d1["program"] == d2["program"]
    # the first run minted + stored a run_id; the second is a cache hit (cached=True).
    assert d1["_serve"]["run_id"] == d2["_serve"]["run_id"]
    assert d1["_serve"]["cached"] is False
    assert d2["_serve"]["cached"] is True


def test_normalized_variants_share_cache_entry(mock_codegen, hero_program):
    # whitespace/case variant per the Phase-1 default normalizer (casefold + strip + collapse ws).
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc("Does #10 score the first goal?", HERO_CLIP)
    server.build_free_text_run_doc("  does #10 score the first goal?  ", HERO_CLIP)
    assert FakeCodegen.total_calls == 1, "case/whitespace variants must share one cache entry"


def test_meaningfully_different_questions_do_not_collide(mock_codegen, hero_program):
    # Two DIFFERENT questions -> two codegen calls + two distinct run_ids (no over-merge).
    mock_codegen(program=hero_program)
    d1 = server.build_free_text_run_doc("Does #10 score the first goal?", HERO_CLIP)
    d2 = server.build_free_text_run_doc("Does #11 score the first goal?", HERO_CLIP)
    assert FakeCodegen.total_calls == 2
    assert d1["_serve"]["run_id"] != d2["_serve"]["run_id"]


# --------------------------------------------------------------------------------------
# (3) trust boundary: ungrounded runs are NOT cached, carry NO run_id
# --------------------------------------------------------------------------------------

def test_ungrounded_run_is_not_cached(mock_codegen, hero_program, run_store):
    # codegen disabled -> ungrounded. Calling twice must stay ungrounded; the store stays empty so
    # a later grounded call for the SAME key is NOT short-circuited by a poisoned entry.
    key = run_cache.cache_key(HERO_TEXT, HERO_CLIP)
    mock_codegen(enabled=False)
    d1 = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    d2 = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    assert d1["program_source"] == "ungrounded"
    assert d2["program_source"] == "ungrounded"
    # NOT cached (no poisoning) and no run_id on an ungrounded exit.
    assert run_store.get(key) is None
    assert "_serve" not in d1 and "_serve" not in d2

    # now a real grounded run for the same key is NOT short-circuited by a poisoned entry.
    mock_codegen(program=hero_program)
    d3 = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    assert d3["program_source"] == "live"
    assert d3["findings"]["grounded"] is True
    assert FakeCodegen.total_calls == 1, "the grounded run must reach codegen (cache wasn't poisoned)"
    assert run_store.get(key) is not None  # now it IS cached


def test_ungrounded_after_execution_is_not_cached(mock_codegen, run_store):
    # A VALID program that runs but doesn't ground (count over an empty window) -> ungrounded,
    # NOT cached, no run_id.
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    mock_codegen(program=program)
    doc = server.build_free_text_run_doc("how many players?", HERO_CLIP)
    assert doc["program_source"] == "ungrounded"
    assert "_serve" not in doc
    assert run_store.get(run_cache.cache_key("how many players?", HERO_CLIP)) is None


# --------------------------------------------------------------------------------------
# (4) cache hit does NOT consume the codegen budget
# --------------------------------------------------------------------------------------

def test_cache_hit_does_not_consume_codegen_budget(mock_codegen, hero_program, monkeypatch):
    # ORDER MATTERS: warm the cache FIRST while under budget (warming increments _codegen_calls;
    # warming at the cap would return _ungrounded and never populate the cache).
    mock_codegen(program=hero_program)
    warm = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    assert warm["program_source"] == "live"
    assert server._codegen_calls == 1

    # now slam the budget to the cap. A NEW question would be refused; the cached one must not.
    monkeypatch.setattr(server, "_codegen_calls", server.MAX_CODEGEN_CALLS)
    hit = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    assert hit["program_source"] == "live"
    assert hit["findings"]["grounded"] is True
    # the counter did NOT move (the hit short-circuited before the budget check AND the increment).
    assert server._codegen_calls == server.MAX_CODEGEN_CALLS
    assert FakeCodegen.total_calls == 1  # still just the warming call


# --------------------------------------------------------------------------------------
# (5) permalink route: replay a stored run cold (codegen disabled)
# --------------------------------------------------------------------------------------

def test_permalink_route_replays_stored_run(mock_codegen, hero_program):
    # warm a grounded free-text run via the route, capture its run_id from the SSE meta.
    mock_codegen(program=hero_program)
    warm = client.get("/api/run", params={"query_text": HERO_TEXT, "pace_ms": 0})
    assert warm.status_code == 200
    meta = _events(warm.text)[0][1]
    run_id = meta["run_id"]
    assert run_cache.is_valid_key(run_id)
    warm_doc = client.get("/api/run_doc", params={"query_text": HERO_TEXT}).json()

    # cold replay: codegen disabled (the credential-less deploy case the permalink exists for).
    mock_codegen(enabled=False)
    resp = client.get("/api/run_doc", params={"run": run_id})
    assert resp.status_code == 200
    doc = resp.json()
    # the run branch BYPASSES build_free_text_run_doc (no codegen-disabled degradation).
    assert doc["program_source"] == "live"
    assert doc["findings"]["grounded"] is True
    assert doc["program"] == warm_doc["program"]
    assert doc["findings"] == warm_doc["findings"]
    assert doc["cached"] is True
    # the run-doc body (sans envelope-only run_id/cached) conforms to run_doc.schema.json.
    body = {k: v for k, v in doc.items() if k not in ("run_id", "cached")}
    assert not _schema_errors(body)


def test_permalink_sse_replay_does_not_call_codegen(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    warm = client.get("/api/run", params={"query_text": HERO_TEXT, "pace_ms": 0})
    run_id = _events(warm.text)[0][1]["run_id"]
    calls_after_warm = FakeCodegen.total_calls

    mock_codegen(enabled=False)  # cold
    resp = client.get("/api/run", params={"run": run_id, "pace_ms": 0})
    assert resp.status_code == 200
    evs = _events(resp.text)
    meta = evs[0][1]
    assert meta["program_source"] == "live"
    assert meta["cached"] is True
    assert meta["run_id"] == run_id
    # the permalink replay made NO NEW codegen call (it bypasses build_free_text_run_doc). The
    # mock_codegen installer does not reset total_calls, so compare against the post-warm count.
    assert FakeCodegen.total_calls == calls_after_warm


# --------------------------------------------------------------------------------------
# (5/6) SSE meta carries run_id + cached, derived at serve time
# --------------------------------------------------------------------------------------

def test_sse_meta_carries_run_id_and_cached(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    r1 = client.get("/api/run", params={"query_text": HERO_TEXT, "pace_ms": 0})
    m1 = _events(r1.text)[0][1]
    assert "run_id" in m1 and run_cache.is_valid_key(m1["run_id"])
    assert m1["cached"] is False  # fresh run, not from the store

    r2 = client.get("/api/run", params={"query_text": HERO_TEXT, "pace_ms": 0})
    m2 = _events(r2.text)[0][1]
    assert m2["cached"] is True            # second identical request -> served from the store
    assert m2["run_id"] == m1["run_id"]    # same stable run_id
    assert FakeCodegen.total_calls == 1    # codegen called exactly once across both requests


def test_ungrounded_run_has_no_run_id_in_meta(mock_codegen):
    # an ungrounded free-text run must NOT carry a run_id (no run was stored to share).
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    meta = _events(resp.text)[0][1]
    assert "run_id" not in meta
    assert meta["cached"] is False
    assert meta["program_source"] == "ungrounded"


def test_canned_run_has_no_run_id_in_meta():
    # the canned path mints no run_id (permalinks are free-text-cached runs only).
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    meta = _events(resp.text)[0][1]
    assert "run_id" not in meta
    assert meta["cached"] is False


# --------------------------------------------------------------------------------------
# (5) permalink route: 404 on unknown id, 400 on malformed id (store untouched)
# --------------------------------------------------------------------------------------

def test_permalink_unknown_id_returns_404():
    # a well-formed but absent id -> 404 (the miss path, not the format guard).
    unknown = "a" * 64
    resp = client.get("/api/run_doc", params={"run": unknown})
    assert resp.status_code == 404
    resp2 = client.get("/api/run", params={"run": unknown, "pace_ms": 0})
    assert resp2.status_code == 404


def test_permalink_malformed_id_rejected(monkeypatch):
    # malformed ids -> 400, and the store is NEVER touched (rejected before the lookup).
    touched = {"get": 0}
    real_get = server._run_store.get

    def spy_get(key):
        touched["get"] += 1
        return real_get(key)
    monkeypatch.setattr(server._run_store, "get", spy_get)

    for bad in ("deadbeef", "../../etc/passwd", "g" * 64, "A" * 64):
        resp = client.get("/api/run_doc", params={"run": bad})
        assert resp.status_code == 400, f"{bad!r} should be a 400"
    assert touched["get"] == 0, "the store must not be touched for a malformed id"


# --------------------------------------------------------------------------------------
# (2) _resolve_run_request is now exactly-one-of-THREE (run | query | query_text)
# --------------------------------------------------------------------------------------

def _status(fn):
    from fastapi import HTTPException
    try:
        fn()
        return 200
    except HTTPException as e:
        return e.status_code


def test_resolve_run_request_exactly_one_of_three():
    rid = "a" * 64
    # a BARE ?run=<id> is ACCEPTED (the old two-way XOR would have 400'd this).
    kind, payload = server._resolve_run_request(None, None, None, rid)
    assert kind == "permalink" and payload == rid
    # run is mutually exclusive with query / query_text (two given -> 400).
    assert _status(lambda: server._resolve_run_request("hero-10-first-goal", None, None, rid)) == 400
    assert _status(lambda: server._resolve_run_request(None, "how many?", None, rid)) == 400
    # none of the three -> 400.
    assert _status(lambda: server._resolve_run_request(None, None, None, None)) == 400
    # malformed run id -> 400.
    assert _status(lambda: server._resolve_run_request(None, None, None, "deadbeef")) == 400
    # the canned + free paths still resolve.
    assert server._resolve_run_request("hero-10-first-goal", None, None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None, None)[0] == "free"


# --------------------------------------------------------------------------------------
# end-to-end: warm -> share -> cold replay produces identical findings/program
# --------------------------------------------------------------------------------------

def test_permalink_replay_matches_byte_for_byte(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    warm = server.build_free_text_run_doc(HERO_TEXT, HERO_CLIP)
    run_id = warm["_serve"]["run_id"]

    mock_codegen(enabled=False)
    replay = server.replay_permalink_doc(run_id)
    # the doc body (excluding the transient _serve carrier) replays identically.
    warm_body = {k: v for k, v in warm.items() if k != "_serve"}
    replay_body = {k: v for k, v in replay.items() if k != "_serve"}
    assert replay_body == warm_body
    assert replay["_serve"]["cached"] is True
    assert replay["_serve"]["run_id"] == run_id
