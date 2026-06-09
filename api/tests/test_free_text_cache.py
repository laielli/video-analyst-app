"""
Integration tests for the free-text program cache + shareable permalink replays, over
build_free_text_run_doc and the permalink route. The headline contracts:

  - Two identical free-text questions fire EXACTLY ONE billed codegen call (the cost win).
  - A grounded run yields a stable `run_id` that, fetched cold (codegen disabled), replays the
    same run-doc.
  - Ungrounded/failed runs are NEVER cached as if successful (no poisoning).
  - A cache hit does NOT consume the per-process codegen budget.

The store + codegen-budget counter are isolated per test by the autouse fixtures in conftest.py
(`_isolate_run_store`, `_reset_codegen_budget`). The codegen seam is the `mock_codegen` fixture;
`FakeCodegen.total_calls` is the billed-call counter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
import run_cache
from conftest import FakeCodegen

client = TestClient(server.app)

API_DIR = Path(__file__).resolve().parent.parent
RD_SCHEMA = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())


def _schema_errors(doc):
    import jsonschema
    return list(jsonschema.Draft202012Validator(RD_SCHEMA).iter_errors(doc))


def events(resp_text: str):
    """Parse an SSE body 'event: X\\ndata: {...}\\n\\n' into [(name, dict), ...]."""
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
# the cost win: an identical repeat question is a pure cache hit (codegen called once)
# --------------------------------------------------------------------------------------

def test_identical_free_text_question_codegen_called_once(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    q, clip = "Does #10 score the first goal?", "single-goal"

    first = server.build_free_text_run_doc(q, clip)
    second = server.build_free_text_run_doc(q, clip)

    assert FakeCodegen.total_calls == 1, "the second identical question must be a pure cache hit"
    assert first["findings"] == second["findings"]
    assert first["program"] == second["program"]
    # both carry the same stable permalink id.
    assert first["run_id"] == second["run_id"]
    assert run_cache.is_valid_key(first["run_id"])
    # the second is flagged as a cache hit (the route reads this to set meta.cached).
    assert second.get("_cached") is True
    assert first.get("_cached") is None


def test_normalized_variants_share_cache_entry(mock_codegen, hero_program):
    # whitespace/case variant per the conservative Phase-1 normalizer -> one cache entry.
    mock_codegen(program=hero_program)
    server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    server.build_free_text_run_doc("  does #10 score the first goal?  ", "single-goal")
    assert FakeCodegen.total_calls == 1


def test_meaningfully_different_questions_do_not_collide(mock_codegen, hero_program):
    # Two DIFFERENT questions must each bill codegen and yield distinct run_ids (guards against
    # an over-aggressive normalizer merging questions whose answers differ).
    mock_codegen(program=hero_program)
    a = server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    b = server.build_free_text_run_doc("Does #7 score the first goal?", "single-goal")
    assert FakeCodegen.total_calls == 2
    assert a["run_id"] != b["run_id"]


# --------------------------------------------------------------------------------------
# no poisoning: ungrounded/failed runs are never cached
# --------------------------------------------------------------------------------------

def test_ungrounded_run_is_not_cached(mock_codegen, hero_program):
    # codegen disabled -> ungrounded. Calling twice must NOT short-circuit (nothing was stored).
    mock_codegen(enabled=False)
    q, clip = "how many players?", "single-goal"
    key = run_cache.cache_key(q, clip)

    doc1 = server.build_free_text_run_doc(q, clip)
    doc2 = server.build_free_text_run_doc(q, clip)
    assert doc1["program_source"] == "ungrounded"
    assert doc2["program_source"] == "ungrounded"
    # nothing was stored for this key (no poisoning).
    assert server.get_store().get(key) is None
    # the ungrounded doc carries NO run_id (only grounded runs are shareable).
    assert "run_id" not in doc1

    # a later GROUNDED call for the same key is NOT short-circuited by a poisoned entry: codegen
    # runs and the run now grounds + caches.
    mock_codegen(program=hero_program)
    FakeCodegen.total_calls = 0
    grounded = server.build_free_text_run_doc(q, clip)
    assert grounded["findings"]["grounded"] is True
    assert FakeCodegen.total_calls == 1
    assert server.get_store().get(key) is not None


def test_invalid_program_run_is_not_cached(mock_codegen):
    invalid = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}}]
    mock_codegen(program=invalid)
    q, clip = "anything", "single-goal"
    doc = server.build_free_text_run_doc(q, clip)
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["reason"] == "invalid-program"
    assert server.get_store().get(run_cache.cache_key(q, clip)) is None


def test_executed_but_ungrounded_run_is_not_cached(mock_codegen):
    # a VALID program that runs but doesn't ground (count over an empty detect window) is NOT cached.
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players?"}},
    ]
    mock_codegen(program=program)
    q, clip = "how many players?", "single-goal"
    doc = server.build_free_text_run_doc(q, clip)
    assert doc["findings"]["grounded"] is False
    assert server.get_store().get(run_cache.cache_key(q, clip)) is None


# --------------------------------------------------------------------------------------
# a cache hit does not consume the codegen budget
# --------------------------------------------------------------------------------------

def test_cache_hit_does_not_consume_codegen_budget(mock_codegen, hero_program, monkeypatch):
    mock_codegen(program=hero_program)
    q, clip = "Does #10 score the first goal?", "single-goal"

    # (1) warm the cache FIRST, under budget (warming itself increments _codegen_calls).
    warm = server.build_free_text_run_doc(q, clip)
    assert warm["findings"]["grounded"] is True
    assert server._codegen_calls == 1

    # (2) now pin the budget to the cap. A fresh question would degrade to ungrounded here.
    monkeypatch.setattr(server, "_codegen_calls", server.MAX_CODEGEN_CALLS)

    # (3) the repeat of the SAME question is a hit: grounded, AND the counter never moved.
    hit = server.build_free_text_run_doc(q, clip)
    assert hit["findings"]["grounded"] is True
    assert hit["program_source"] == "live"
    assert server._codegen_calls == server.MAX_CODEGEN_CALLS, "a cache hit must not burn budget"
    # and codegen.generate() was called exactly once total (warm only).
    assert FakeCodegen.total_calls == 1


# --------------------------------------------------------------------------------------
# permalink route: replay a stored run by id, cold (codegen disabled)
# --------------------------------------------------------------------------------------

def test_permalink_route_replays_stored_run(mock_codegen, hero_program):
    # warm a grounded free-text run and capture its run_id from the run_doc envelope.
    mock_codegen(program=hero_program)
    warm = client.get("/api/run_doc", params={"query_text": "Does #10 score the first goal?"})
    assert warm.status_code == 200
    warm_body = warm.json()
    run_id = warm_body["run_id"]
    assert run_cache.is_valid_key(run_id)
    assert warm_body["cached"] is False

    # now disable codegen and fetch the permalink COLD — the run branch must BYPASS codegen.
    mock_codegen(enabled=False)
    replay = client.get("/api/run_doc", params={"run": run_id})
    assert replay.status_code == 200
    body = replay.json()
    assert body["cached"] is True
    assert body["run_id"] == run_id
    # program + findings match the warmed run.
    assert body["program"] == warm_body["program"]
    assert body["findings"] == warm_body["findings"]
    # and the doc body (sans envelope fields) conforms to run_doc.schema.json.
    doc = {k: v for k, v in body.items() if k not in ("run_id", "cached")}
    assert not _schema_errors(doc)


def test_permalink_unknown_id_returns_404(mock_codegen):
    mock_codegen(enabled=False)
    unknown = "a" * 64  # well-formed but not in the store -> tests the MISS path, not the format guard
    resp = client.get("/api/run_doc", params={"run": unknown})
    assert resp.status_code == 404


@pytest.mark.parametrize("bad", ["deadbeef", "../../etc/passwd", "g" * 64, "A" * 64])
def test_permalink_malformed_id_rejected(bad, monkeypatch):
    # a malformed run id must 400 BEFORE the store is touched.
    touched = {"n": 0}
    real_get = server.get_store().get

    def spy(key):
        touched["n"] += 1
        return real_get(key)
    monkeypatch.setattr(server.get_store(), "get", spy)

    resp = client.get("/api/run_doc", params={"run": bad})
    assert resp.status_code == 400
    assert touched["n"] == 0, "store must not be touched for a malformed run id"


# --------------------------------------------------------------------------------------
# SSE meta carries run_id + cached
# --------------------------------------------------------------------------------------

def test_sse_meta_carries_run_id_and_cached(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    q = "Does #10 score the first goal?"

    first = client.get("/api/run", params={"query_text": q, "pace_ms": 0})
    assert first.status_code == 200
    meta1 = events(first.text)[0][1]
    assert run_cache.is_valid_key(meta1["run_id"])
    assert meta1["cached"] is False

    # the repeat stream is a cache hit: same run_id, cached=true.
    second = client.get("/api/run", params={"query_text": q, "pace_ms": 0})
    meta2 = events(second.text)[0][1]
    assert meta2["run_id"] == meta1["run_id"]
    assert meta2["cached"] is True
    assert FakeCodegen.total_calls == 1


def test_sse_meta_ungrounded_has_no_run_id(mock_codegen):
    # an ungrounded free-text run carries run_id=None (no share link) and cached=false.
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    meta = events(resp.text)[0][1]
    assert meta["run_id"] is None
    assert meta["cached"] is False


# --------------------------------------------------------------------------------------
# permalink replay over the SSE route too (full meta->step*->findings->done)
# --------------------------------------------------------------------------------------

def test_permalink_sse_replay_cold(mock_codegen, hero_program):
    mock_codegen(program=hero_program)
    warm = client.get("/api/run_doc", params={"query_text": "Does #10 score the first goal?"})
    run_id = warm.json()["run_id"]

    mock_codegen(enabled=False)  # cold replay path
    resp = client.get("/api/run", params={"run": run_id, "pace_ms": 0})
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    assert names[0] == "meta" and names[-1] == "done"
    meta = evs[0][1]
    assert meta["cached"] is True
    assert meta["run_id"] == run_id
    assert meta["program_source"] == "live"  # the stored doc kept its live provenance
    findings = next(d for n, d in evs if n == "findings")
    assert findings["grounded"] is True


# --------------------------------------------------------------------------------------
# _resolve_run_request: exactly-one-of-three (run | query | query_text)
# --------------------------------------------------------------------------------------

def _status(fn):
    try:
        fn()
        return 200
    except HTTPException as e:
        return e.status_code


def test_resolve_run_request_exactly_one_of_three():
    valid_run = "a" * 64
    # a bare ?run=<id> is now ACCEPTED (the old two-way XOR would have 400'd it).
    kind, payload = server._resolve_run_request(None, None, None, valid_run)
    assert kind == "permalink" and payload == valid_run

    # run is mutually exclusive with query / query_text -> 400 if two are given.
    assert _status(lambda: server._resolve_run_request("hero-10-first-goal", None, None, valid_run)) == 400
    assert _status(lambda: server._resolve_run_request(None, "how many?", None, valid_run)) == 400
    # all three -> 400.
    assert _status(lambda: server._resolve_run_request("hero-10-first-goal", "how many?", None, valid_run)) == 400
    # none -> 400 (unchanged).
    assert _status(lambda: server._resolve_run_request(None, None, None, None)) == 400
    # a malformed run id -> 400 (format guard).
    assert _status(lambda: server._resolve_run_request(None, None, None, "deadbeef")) == 400
    # the canned + free paths still resolve unchanged.
    assert server._resolve_run_request("hero-10-first-goal", None, None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None, None)[0] == "free"


def test_grounded_paths_attach_run_id_only_on_grounded(mock_codegen, hero_program):
    # the four NON-grounded exits must NOT carry a run_id.
    mock_codegen(enabled=False)
    assert "run_id" not in server.build_free_text_run_doc("q", "single-goal")  # codegen-disabled

    # unknown clip (defensive guard) -> ungrounded, no run_id.
    assert "run_id" not in server.build_free_text_run_doc("q", "no-such-clip")

    # the grounded exit DOES carry run_id.
    mock_codegen(program=hero_program)
    g = server.build_free_text_run_doc("Does #10 score the first goal?", "single-goal")
    assert "run_id" in g
