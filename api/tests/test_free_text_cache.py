"""
Integration tests over build_free_text_run_doc + the ?run=<id> permalink route. These pin the
cost win (an identical repeat question fires EXACTLY ONE billed codegen call), the trust boundary
(ungrounded runs are never cached → no poisoning, no run_id), and the share story (a grounded
free-text run mints a stable run_id that, fetched cold with codegen disabled, replays the same
doc byte-for-byte in program/findings).

Mirrors test_free_text_run.py + test_sse_stream.py; uses the conftest mock_codegen fixture and
the FakeCodegen.total_calls counter. The autouse _isolated_run_store + _reset_codegen_budget
fixtures (conftest) keep these order-independent.
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

HERO_Q = "Does #10 score the first goal?"
HERO_CLIP = "single-goal"


def _schema_errors(doc):
    import jsonschema
    return list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


def events(resp_text: str):
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
# the cost win: an identical repeat fires exactly one codegen call
# --------------------------------------------------------------------------------------

def test_identical_free_text_question_codegen_called_once(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    doc1 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    doc2 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    # exactly one billed codegen call across the two identical questions.
    assert FakeCodegen.total_calls == 1
    # both are grounded and carry identical findings (the second is a pure cache hit).
    assert doc1["findings"]["grounded"] is True
    assert doc1["findings"] == doc2["findings"]
    assert doc1["program"] == doc2["program"]


def test_normalized_variants_share_cache_entry(mock_codegen, hero_program):
    # whitespace/case variant per the Phase-1 conservative normalizer (casefold + strip +
    # collapse whitespace) -> same cache entry -> still exactly one codegen call.
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    server.build_free_text_run_doc(f"   {HERO_Q.upper()}   ", HERO_CLIP)
    assert FakeCodegen.total_calls == 1


def test_meaningfully_different_questions_do_not_collide(mock_codegen, hero_program):
    # two genuinely different questions -> two codegen calls, two distinct run_ids. Guards against
    # an over-aggressive normalizer merging questions whose answers could differ.
    mock_codegen(program=hero_program)
    m1: dict = {}
    m2: dict = {}
    server.build_free_text_run_doc(HERO_Q, HERO_CLIP, m1)
    server.build_free_text_run_doc("Does #11 score the first goal?", HERO_CLIP, m2)
    assert FakeCodegen.total_calls == 2
    assert m1["run_id"] != m2["run_id"]


# --------------------------------------------------------------------------------------
# trust boundary: ungrounded runs are NOT cached (no poisoning), carry no run_id
# --------------------------------------------------------------------------------------

def test_ungrounded_run_is_not_cached(mock_codegen, hero_program):
    mock_codegen(enabled=False)  # codegen disabled -> every call resolves to _ungrounded
    m1: dict = {}
    d1 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP, m1)
    d2 = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert d1["program_source"] == "ungrounded"
    assert d2["program_source"] == "ungrounded"
    # NOT cached: store.get for this key is None (no poisoning, no permanent failure).
    key = run_cache.cache_key(HERO_Q, HERO_CLIP)
    assert server.get_store().get(key) is None
    # and an ungrounded run carries NO run_id (the UI must not offer a share link for it).
    assert "run_id" not in m1

    # a subsequent GROUNDED call for the same key is NOT short-circuited by a poisoned entry.
    install = mock_codegen(program=hero_program)  # re-enable codegen with a real program
    grounded = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert grounded["findings"]["grounded"] is True
    assert install.calls, "codegen WAS invoked (no poisoned cache short-circuit)"
    assert server.get_store().get(key) is not None  # now it caches the grounded run


def test_non_grounded_exits_carry_no_run_id(mock_codegen):
    # the four non-grounded exits must never carry a run_id. Codegen-disabled is one of them.
    m1: dict = {}
    mock_codegen(enabled=False)
    server.build_free_text_run_doc("how many players?", HERO_CLIP, m1)
    assert "run_id" not in m1
    # unknown clip exit (clip missing) likewise.
    m2: dict = {}
    doc2 = server.build_free_text_run_doc("anything", "no-such-clip", m2)
    assert doc2["program_source"] == "ungrounded"
    assert "run_id" not in m2


# --------------------------------------------------------------------------------------
# a cache hit must NOT consume the codegen budget
# --------------------------------------------------------------------------------------

def test_cache_hit_does_not_consume_codegen_budget(mock_codegen, hero_program, monkeypatch):
    mock_codegen(program=hero_program)
    # 1) warm the cache FIRST, while under budget (warming itself increments _codegen_calls).
    warm = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert warm["findings"]["grounded"] is True
    assert FakeCodegen.total_calls == 1

    # 2) now exhaust the budget. A cache hit must short-circuit BEFORE the budget check + the
    #    increment, so the repeat still grounds without billing.
    monkeypatch.setattr(server, "_codegen_calls", server.MAX_CODEGEN_CALLS)
    repeat = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)
    assert repeat["findings"]["grounded"] is True
    assert repeat["findings"] == warm["findings"]
    # still exactly one codegen call total (the hit billed nothing).
    assert FakeCodegen.total_calls == 1
    # the budget counter was NOT touched by the hit.
    assert server._codegen_calls == server.MAX_CODEGEN_CALLS


# --------------------------------------------------------------------------------------
# the permalink route replays a stored run cold (codegen disabled) — Phase 5 item 3
# --------------------------------------------------------------------------------------

def test_permalink_route_replays_stored_run(mock_codegen, hero_program):
    # warm a grounded free-text run and capture its run_id.
    mock_codegen(program=hero_program)
    warm_meta: dict = {}
    warm = server.build_free_text_run_doc(HERO_Q, HERO_CLIP, warm_meta)
    run_id = warm_meta["run_id"]
    assert run_cache.is_valid_key(run_id)

    # fetch it COLD with codegen disabled — the permalink path must BYPASS the free-text builder
    # (codegen.enabled() would return _ungrounded on a credential-less deploy).
    mock_codegen(enabled=False)
    resp = client.get("/api/run_doc", params={"run": run_id})
    assert resp.status_code == 200
    body = resp.json()
    # the run_doc envelope carries run_id + cached in `_meta`, the doc itself is schema-pure.
    assert body["_meta"]["cached"] is True
    assert body["_meta"]["run_id"] == run_id
    doc = {k: v for k, v in body.items() if k != "_meta"}
    assert not _schema_errors(doc)
    # program / findings match the warmed run byte-for-byte.
    assert doc["program"] == warm["program"]
    assert doc["findings"] == warm["findings"]
    assert doc["program_source"] == "live"


# --------------------------------------------------------------------------------------
# the SSE meta carries run_id + cached (false fresh, true on replay)
# --------------------------------------------------------------------------------------

def test_sse_meta_carries_run_id_and_cached(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    # first request: grounded free-text run -> meta has run_id + cached:false.
    resp1 = client.get("/api/run", params={"query_text": HERO_Q, "pace_ms": 0})
    meta1 = events(resp1.text)[0][1]
    assert "run_id" in meta1 and run_cache.is_valid_key(meta1["run_id"])
    assert meta1["cached"] is False

    # second identical request: served from the store -> cached:true, same run_id.
    resp2 = client.get("/api/run", params={"query_text": HERO_Q, "pace_ms": 0})
    meta2 = events(resp2.text)[0][1]
    assert meta2["cached"] is True
    assert meta2["run_id"] == meta1["run_id"]
    # exactly one codegen call across the two identical SSE requests.
    assert FakeCodegen.total_calls == 1


def test_ungrounded_sse_meta_has_cached_false_no_run_id(mock_codegen):
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    meta = events(resp.text)[0][1]
    assert meta["cached"] is False
    assert "run_id" not in meta  # no share link for an ungrounded run


# --------------------------------------------------------------------------------------
# permalink security + 404 placement
# --------------------------------------------------------------------------------------

def test_permalink_unknown_id_returns_404():
    # a well-formed (64-hex) id that is NOT in the store -> 404 (the MISS path, not the format guard).
    unknown = "a" * 64
    resp = client.get("/api/run_doc", params={"run": unknown})
    assert resp.status_code == 404
    # the SSE route 404s in the SYNC body too (before any StreamingResponse), not a 200 broken stream.
    resp_sse = client.get("/api/run", params={"run": unknown, "pace_ms": 0})
    assert resp_sse.status_code == 404


@pytest.mark.parametrize("bad", ["deadbeef", "../../etc/passwd", "g" * 64, "A" * 64, "a" * 63])
def test_permalink_malformed_id_rejected(bad, monkeypatch):
    # the store must never be touched for a malformed id (rejected at the format guard with 400).
    touched = {"get": 0}
    real_get = server.get_store().get

    def spy_get(key):
        touched["get"] += 1
        return real_get(key)
    monkeypatch.setattr(server.get_store(), "get", spy_get)

    resp = client.get("/api/run_doc", params={"run": bad})
    assert resp.status_code == 400
    assert touched["get"] == 0, "the store was touched for a malformed id (path-traversal risk)"


# --------------------------------------------------------------------------------------
# exactly-one-of-three resolution (run | query | query_text)
# --------------------------------------------------------------------------------------

def _status(fn):
    from fastapi import HTTPException
    try:
        fn()
        return 200
    except HTTPException as e:
        return e.status_code


def test_resolve_run_request_exactly_one_of_three():
    valid_id = "a" * 64
    # a bare ?run=<id> is ACCEPTED (the old two-way XOR would 400 it).
    kind, payload = server._resolve_run_request(None, None, None, valid_id)
    assert kind == "permalink" and payload == valid_id
    # run is mutually exclusive with query / query_text (two given -> 400).
    assert _status(lambda: server._resolve_run_request("hero-10-first-goal", None, None, valid_id)) == 400
    assert _status(lambda: server._resolve_run_request(None, "how many?", None, valid_id)) == 400
    # none given -> 400.
    assert _status(lambda: server._resolve_run_request(None, None, None, None)) == 400
    # a malformed run id -> 400 (format guard).
    assert _status(lambda: server._resolve_run_request(None, None, None, "deadbeef")) == 400
    # the existing two arms still resolve.
    assert server._resolve_run_request("hero-10-first-goal", None, None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None, None)[0] == "free"


# --------------------------------------------------------------------------------------
# corrupt-entry and invalid-program paths, end-to-end through the builder
# --------------------------------------------------------------------------------------

def test_corrupt_stored_entry_falls_back_to_codegen(mock_codegen, hero_program):
    # a corrupt on-disk entry must be a MISS on the ask path: the store self-heals
    # (deletes the bad file) and the question proceeds to a billed codegen call that
    # grounds — never streaming garbage from disk.
    mock_codegen(program=hero_program)
    store = server.get_store()
    key = run_cache.cache_key(HERO_Q, HERO_CLIP)
    path = store._path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("}{ corrupt")

    doc = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)

    assert FakeCodegen.total_calls == 1, "a corrupt entry must be a miss -> codegen runs"
    assert doc["findings"]["grounded"] is True
    repopulated = store.get(key)
    assert repopulated is not None and repopulated["findings"]["grounded"] is True


def test_invalid_program_run_is_not_cached(mock_codegen):
    # the invalid-program reject path (distinct from codegen-disabled) must not cache:
    # a program that fails validation never reaches the interpreter, so nothing
    # validated+grounded exists to store.
    invalid = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    mock_codegen(program=invalid)

    doc = server.build_free_text_run_doc(HERO_Q, HERO_CLIP)

    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert server.get_store().get(run_cache.cache_key(HERO_Q, HERO_CLIP)) is None
