"""
SSE-stream tests over /api/run + the other endpoints, via FastAPI TestClient.

The streaming generator gen() yields meta -> step* -> findings -> done, or `error` on
exception; cancel re-raises CancelledError; the error path must never leak raw exception
text. gen() is a nested closure inside the run() route handler and is NOT importable, so the
info-leak and cancel tests drive the route coroutine directly and iterate resp.body_iterator.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import codegen
import server

client = TestClient(server.app)

HERO = "hero-10-first-goal"


def events(resp_text: str):
    """Parse 'event: X\\ndata: {...}\\n\\n' blocks into [(name, dict), ...]."""
    out = []
    for block in resp_text.strip().split("\n\n"):
        if not block.strip():
            continue
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((name, data))
    return out


# ---------------------------------------------------------------- event sequence

def test_run_event_sequence_canned():
    r = client.get(f"/api/run?query={HERO}&pace_ms=0")
    assert r.status_code == 200
    evs = events(r.text)
    names = [n for n, _ in evs]
    # meta first, done last, exactly one findings before done, the rest are steps.
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert names[-2] == "findings"
    meta = evs[0][1]
    total = meta["total_steps"]
    assert isinstance(total, int)
    step_names = names[1:-2]
    assert step_names == ["step"] * total
    assert names == ["meta"] + ["step"] * total + ["findings", "done"]


def test_run_meta_payload_shape():
    r = client.get(f"/api/run?query={HERO}&pace_ms=0")
    meta = events(r.text)[0][1]
    for key in ("query", "clip", "program", "total_steps", "pace_ms", "program_source"):
        assert key in meta, key
    assert meta["pace_ms"] == 0
    assert meta["query"] == "Does #10 score the first goal?"
    assert isinstance(meta["program"], list)
    # canned path with no Azure creds -> pinned program.
    assert meta["program_source"] in ("pinned", "live")


def test_run_done_event_payload():
    r = client.get(f"/api/run?query={HERO}&pace_ms=0")
    evs = events(r.text)
    assert evs[-1] == ("done", {"ok": True})


def test_run_free_text_ungrounded_stream(monkeypatch):
    # codegen disabled -> free-text streams meta -> findings(grounded=false) -> done; no `step`
    # because the ungrounded run-doc trace is empty.
    monkeypatch.setattr(codegen, "enabled", lambda: False)
    r = client.get("/api/run?query_text=Who%20scored?&pace_ms=0")
    assert r.status_code == 200
    evs = events(r.text)
    names = [n for n, _ in evs]
    assert names == ["meta", "findings", "done"]
    meta = evs[0][1]
    assert meta["program_source"] == "ungrounded"
    assert meta["total_steps"] == 0
    findings = evs[1][1]
    assert findings["grounded"] is False
    assert findings["reason"] == "codegen-disabled"


# ---------------------------------------------------------------- info-leak guard

@pytest.mark.asyncio
async def test_stream_error_never_leaks_exception_text(monkeypatch):
    """Drive the route coroutine directly (gen() is a non-importable nested closure). The forced
    exception must happen DURING doc['trace'] iteration — meta computes len(doc['trace']) BEFORE
    the loop, so the fake trace must __len__ to a positive int without raising and raise ONLY on
    iteration. Asserts meta arrives FIRST, then an `error` event with the fixed 'stream failed'
    message, and that the secret path appears NOWHERE in the body."""
    SECRET = "/srv/secret/azure-key-abc123"

    class ExplodingTrace:
        def __len__(self):
            return 3  # positive, so meta's len(doc['trace']) succeeds before the loop.

        def __iter__(self):
            raise RuntimeError(f"boom while reading {SECRET}")

    doc = {
        "query": "q", "clip": {"id": "t", "width": 1, "height": 1, "duration_ms": 1},
        "program": [], "trace": ExplodingTrace(),
        "findings": {"answer": None, "verdict": None, "grounded": False,
                     "reason": "x", "partial": True},
        "program_source": "pinned",
    }
    monkeypatch.setattr(server, "_doc_for_request", lambda *a, **k: doc)

    resp = await server.run(query=HERO, query_text=None, clip=None, pace_ms=0)
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)
    body = "".join(chunks)
    evs = events(body)
    names = [n for n, _ in evs]
    # (a) meta arrived first.
    assert names[0] == "meta"
    # (b) an error event with the EXACT fixed message follows.
    assert ("error", {"message": "stream failed"}) in evs
    assert names[-1] == "error"
    # (c) the secret appears nowhere in the streamed body.
    assert SECRET not in body
    assert "boom" not in body


@pytest.mark.asyncio
async def test_stream_cancel_reraises():
    """Proves ONLY that the generator re-raises CancelledError (does not swallow it into an
    `error` event). Drives the route coroutine and athrow()s CancelledError into the iterator."""
    resp = await server.run(query=HERO, query_text=None, clip=None, pace_ms=0)
    it = resp.body_iterator
    # Prime the generator so it is mid-stream, then throw cancellation in.
    first = await it.__anext__()
    assert first.startswith("event: meta")
    with pytest.raises(asyncio.CancelledError):
        await it.athrow(asyncio.CancelledError())


# ---------------------------------------------------------------- _sse unit

def test_sse_format_exact():
    assert server._sse("meta", {"a": 1}) == 'event: meta\ndata: {"a": 1}\n\n'


def test_sse_format_done():
    assert server._sse("done", {"ok": True}) == 'event: done\ndata: {"ok": true}\n\n'


# ---------------------------------------------------------------- /api/run_doc

def test_run_doc_canned_returns_full_doc():
    r = client.get(f"/api/run_doc?query={HERO}")
    assert r.status_code == 200
    doc = r.json()
    assert doc["query"] == "Does #10 score the first goal?"
    assert "program" in doc and "trace" in doc and "findings" in doc
    assert doc["program_source"] in ("pinned", "live")


def test_run_doc_free_text_ungrounded(monkeypatch):
    monkeypatch.setattr(codegen, "enabled", lambda: False)
    r = client.get("/api/run_doc?query_text=Who%20scored?")
    assert r.status_code == 200
    doc = r.json()
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False


# ---------------------------------------------------------------- /api/catalog

def test_catalog_drops_internal_fields():
    r = client.get("/api/catalog")
    assert r.status_code == 200
    body = r.json()
    assert "clips" in body and "queries" in body
    for clip in body["clips"]:
        assert "cache" not in clip  # internal path dropped
        assert "hint" not in clip   # internal prompt hint dropped
        for k in ("id", "label", "width", "height", "duration_ms"):
            assert k in clip, k
    q = body["queries"][0]
    assert set(q) == {"id", "text", "clip"}


# ---------------------------------------------------------------- /api/health

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body
    assert HERO in body["queries"]


# ---------------------------------------------------------------- _resolve_run_request

def test_resolve_neither_query_400():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        server._resolve_run_request(None, None, None)
    assert ei.value.status_code == 400


def test_resolve_both_query_400():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        server._resolve_run_request(HERO, "some text", None)
    assert ei.value.status_code == 400


def test_resolve_unknown_query_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        server._resolve_run_request("no-such-query", None, None)
    assert ei.value.status_code == 404


def test_resolve_over_length_query_text_400():
    from fastapi import HTTPException
    long_text = "x" * (server.MAX_QUERY_TEXT_LEN + 1)
    with pytest.raises(HTTPException) as ei:
        server._resolve_run_request(None, long_text, None)
    assert ei.value.status_code == 400


def test_resolve_unknown_clip_404():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        server._resolve_run_request(None, "valid text", "no-such-clip")
    assert ei.value.status_code == 404


def test_resolve_canned_ok():
    kind, payload = server._resolve_run_request(HERO, None, None)
    assert kind == "canned"
    assert payload["id"] == HERO


def test_resolve_free_defaults_clip():
    kind, payload = server._resolve_run_request(None, "free text", None)
    assert kind == "free"
    text, cid = payload
    assert text == "free text"
    # defaults to the sole v1 clip.
    import canned
    assert cid == next(iter(canned.CLIPS))


# ---------------------------------------------------------------- SSE response metadata

def test_run_response_headers():
    r = client.get(f"/api/run?query={HERO}&pace_ms=0")
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.headers["x-accel-buffering"] == "no"
