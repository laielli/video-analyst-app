"""
Integration tests for the free-text program cache + shareable permalink replays, over
build_free_text_run_doc + the permalink route. Headline contracts:
  (1) two identical free-text questions produce EXACTLY ONE codegen call (the cost win),
  (2) a grounded free-text run yields a run_id that, fetched cold (codegen disabled), replays the
      stored run-doc (program/findings) byte-for-byte and conforms to run_doc.schema.json,
  (3) ungrounded/failed runs are NOT cached (no poisoning),
  (4) a cache hit does NOT consume the MAX_CODEGEN_CALLS budget nor increment _codegen_calls.

Mirrors test_free_text_run.py / test_sse_stream.py: the codegen seam is the conftest mock_codegen
fixture (FakeCodegen.total_calls counts billed calls), and the run cache is redirected to a tmp dir
by the autouse _isolate_run_store fixture, with the budget counter reset by _reset_codegen_budget.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
import canned
from run_cache import cache_key
from conftest import FakeCodegen

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())

client = TestClient(server.app)

HERO_Q = "Does #10 score the first goal?"
HERO_CLIP = "single-goal"


def _schema_errors(doc):
    import jsonschema
    return list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


def _sse_events(resp_text: str):
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
# (1) the cost win: identical question -> exactly one codegen call
# --------------------------------------------------------------------------------------

def test_identical_free_text_question_codegen_called_once(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    doc1 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    doc2 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)  # pure cache hit
    assert FakeCodegen.total_calls == 1, "the second identical question must be served from cache"
    assert doc1["findings"] == doc2["findings"]
    assert doc1["program"] == doc2["program"]
    assert doc1["program_source"] == "live" and doc2["program_source"] == "live"


def test_normalized_variants_share_cache_entry(mock_codegen, hero_program):
    # whitespace/case variant per the conservative normalizer (Phase-1 default).
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    server.build_free_text_run_doc("  does #10 score the first goal?  ", HERO_CLIP)
    assert FakeCodegen.total_calls == 1, "case/whitespace variants must hit the same cache entry"


def test_meaningfully_different_questions_do_not_collide(mock_codegen, hero_program):
    # two DIFFERENT questions must each compile (guards against over-aggressive normalization) and
    # mint two distinct run_ids.
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc("Does #10 score the first goal?", HERO_CLIP)
    server.build_free_text_run_doc("Does #11 score the first goal?", HERO_CLIP)
    assert FakeCodegen.total_calls == 2
    id_a = cache_key("Does #10 score the first goal?", HERO_CLIP)
    id_b = cache_key("Does #11 score the first goal?", HERO_CLIP)
    assert id_a != id_b


# --------------------------------------------------------------------------------------
# (3) trust boundary: ungrounded runs are NOT cached (no poisoning)
# --------------------------------------------------------------------------------------

def test_ungrounded_run_is_not_cached(mock_codegen, hero_program):
    # an ungrounded run (codegen disabled) must NOT populate the cache.
    mock_codegen(enabled=False)
    doc1 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    doc2 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert doc1["program_source"] == "ungrounded" and doc2["program_source"] == "ungrounded"
    # nothing stored under the key (no poisoning) — the store has no entry for it.
    key = cache_key(HERO_Q, HERO_CLIP)
    assert server.get_store().get(key) is None

    # a subsequent GROUNDED call for the same key is NOT short-circuited by a poisoned entry:
    # it actually compiles (codegen invoked) and now grounds + caches.
    mock_codegen(program=hero_program)
    doc3 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert doc3["findings"]["grounded"] is True
    assert FakeCodegen.total_calls == 1, "the grounded run compiled (the ungrounded run cached nothing)"
    assert server.get_store().get(key) is not None  # now cached


# --------------------------------------------------------------------------------------
# (4) a cache hit does not consume the codegen budget
# --------------------------------------------------------------------------------------

def test_cache_hit_does_not_consume_codegen_budget(mock_codegen, hero_program, monkeypatch):
    # ORDER MATTERS: warm the cache FIRST while under budget (warming increments _codegen_calls),
    # THEN exhaust the budget, then assert the repeat returns grounded WITHOUT touching the counter.
    mock_codegen(program=hero_program)
    warm = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert warm["findings"]["grounded"] is True

    monkeypatch.setattr(server, "_codegen_calls", server.MAX_CODEGEN_CALLS)  # budget exhausted
    FakeCodegen.total_calls = 0  # reset to prove the hit path bills zero
    hit = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert hit["findings"]["grounded"] is True, "a cache hit must short-circuit the budget guard"
    assert FakeCodegen.total_calls == 0, "a cache hit must not bill a codegen call"
    assert server._codegen_calls == server.MAX_CODEGEN_CALLS, "the counter must be untouched by a hit"


# --------------------------------------------------------------------------------------
# (2) permalink replay: cold fetch (codegen disabled) replays the stored run
# --------------------------------------------------------------------------------------

def test_permalink_route_replays_stored_run(mock_codegen, hero_program):
    # warm a grounded free-text run via the SSE route and capture its run_id from meta.
    mock_codegen(program=hero_program)
    resp = client.get("/api/run", params={"query_text": HERO_Q, "pace_ms": 0})
    assert resp.status_code == 200
    meta = _sse_events(resp.text)[0][1]
    run_id = meta["run_id"]
    assert run_id == cache_key(HERO_Q, HERO_CLIP)

    # COLD permalink fetch with codegen DISABLED: the run branch must BYPASS the free-text builder
    # entirely (Phase 5 item 3), so it replays even credential-less.
    mock_codegen(enabled=False)
    cold = client.get("/api/run_doc", params={"run": run_id})
    assert cold.status_code == 200
    doc = cold.json()
    assert doc["program_source"] == "live"  # the stored grounded run, not an ungrounded fallback
    assert doc["findings"]["grounded"] is True
    # program + findings match the warm run, and the stored doc (envelope keys stripped) conforms.
    warm = client.get("/api/run_doc", params={"query_text": HERO_Q})  # served from cache now
    warm_doc = warm.json()
    assert doc["program"] == warm_doc["program"]
    assert doc["findings"] == warm_doc["findings"]
    core = {k: v for k, v in doc.items() if k not in ("run_id", "cached")}
    assert not _schema_errors(core)


def test_sse_meta_carries_run_id_and_cached(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    r1 = client.get("/api/run", params={"query_text": HERO_Q, "pace_ms": 0})
    meta1 = _sse_events(r1.text)[0][1]
    assert "run_id" in meta1 and meta1["run_id"] == cache_key(HERO_Q, HERO_CLIP)
    assert meta1["cached"] is False  # first run: freshly compiled

    r2 = client.get("/api/run", params={"query_text": HERO_Q, "pace_ms": 0})  # repeat -> hit
    meta2 = _sse_events(r2.text)[0][1]
    assert meta2["cached"] is True
    assert meta2["run_id"] == meta1["run_id"]


def test_ungrounded_sse_meta_carries_no_run_id(mock_codegen):
    # the four non-grounded exits must NOT carry a run_id (the UI must not offer a share link to a
    # run that was never stored).
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"query_text": HERO_Q, "pace_ms": 0})
    meta = _sse_events(resp.text)[0][1]
    assert "run_id" not in meta
    assert meta["cached"] is False


# --------------------------------------------------------------------------------------
# permalink security + 404
# --------------------------------------------------------------------------------------

def test_permalink_unknown_id_returns_404(mock_codegen):
    mock_codegen(enabled=False)
    # a well-formed (64 hex) id that is NOT in the store -> 404 (the miss path, not the format guard).
    unknown = "0" * 64
    resp = client.get("/api/run_doc", params={"run": unknown})
    assert resp.status_code == 404


def test_permalink_malformed_id_rejected(monkeypatch):
    # malformed ids must 400 BEFORE the store is touched (defense in depth).
    touched = {"n": 0}
    real_get = server.get_store().get

    def spy(key):
        touched["n"] += 1
        return real_get(key)
    monkeypatch.setattr(server.get_store(), "get", spy)

    for bad in ("deadbeef", "../../etc/passwd", "g" * 64, "A" * 64):
        resp = client.get("/api/run_doc", params={"run": bad})
        assert resp.status_code == 400, f"{bad!r} should be 400"
    assert touched["n"] == 0, "the store must never be touched for a malformed run id"


def test_permalink_404_raised_before_stream_on_run_route(mock_codegen):
    # On the SSE /api/run route, a missing id must be a real 404 (raised in the sync body before
    # StreamingResponse), NOT a 200 with a broken stream.
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"run": "0" * 64, "pace_ms": 0})
    assert resp.status_code == 404


# --------------------------------------------------------------------------------------
# exactly-one-of-three resolution (run | query | query_text)
# --------------------------------------------------------------------------------------

def test_resolve_run_request_exactly_one_of_three():
    from fastapi import HTTPException

    def status(fn):
        try:
            fn()
            return 200
        except HTTPException as e:
            return e.status_code

    valid_id = "a" * 64
    # a bare ?run=<id> is ACCEPTED (the old two-way XOR would have 400'd it).
    assert server._resolve_run_request(None, None, None, valid_id)[0] == "permalink"
    # run is mutually exclusive with query / query_text.
    assert status(lambda: server._resolve_run_request("hero-10-first-goal", None, None, valid_id)) == 400
    assert status(lambda: server._resolve_run_request(None, "how many?", None, valid_id)) == 400
    # none of the three -> 400.
    assert status(lambda: server._resolve_run_request(None, None, None, None)) == 400
    # a malformed run id -> 400.
    assert status(lambda: server._resolve_run_request(None, None, None, "deadbeef")) == 400
    # canned + free still resolve (back-compat).
    assert server._resolve_run_request("hero-10-first-goal", None, None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None, None)[0] == "free"


# --------------------------------------------------------------------------------------
# the four non-grounded exits attach no run_id (builder-level)
# --------------------------------------------------------------------------------------

def test_non_grounded_doc_carries_no_run_id(mock_codegen):
    # the builder returns schema-pure docs; the route attaches run_id only on the grounded path.
    # Drive each non-grounded exit through _doc_for_request and assert no run_id leaks.
    mock_codegen(enabled=False)  # codegen-disabled exit
    doc = server._doc_for_request(None, HERO_Q, None, None)
    assert doc["findings"]["grounded"] is False
    assert "run_id" not in doc


def test_corrupt_stored_entry_falls_back_to_codegen(mock_codegen, hero_program):
    # a corrupt entry at a key path is a MISS on the ask path: the question proceeds to codegen.
    mock_codegen(program=hero_program)
    key = cache_key(HERO_Q, HERO_CLIP)
    # write garbage at the entry path, then ask: the store self-heals (miss) and codegen runs.
    store = server.get_store()
    store._path(key).parent.mkdir(parents=True, exist_ok=True)
    store._path(key).write_text("}{ corrupt")
    doc = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert FakeCodegen.total_calls == 1, "a corrupt entry must be a miss -> codegen runs"
    assert doc["findings"]["grounded"] is True
