"""
SSE-stream tests over /api/run (and the sibling JSON endpoints) via FastAPI TestClient.

The /api/run route streams:  meta -> step* -> findings -> done   (or `error` on exception).
gen() is a NESTED CLOSURE inside the run() route handler — it is NOT importable, so the
info-leak and cancel tests drive the route coroutine directly (await server.run(...)) and
iterate resp.body_iterator. The canned/happy-path and metadata tests use TestClient, which
drives StreamingResponse synchronously.

asyncio_mode=auto (pytest.ini) collects the async cancel test without an explicit marker.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import server
import canned

client = TestClient(server.app)


def parse_events(text: str):
    """Parse an SSE body 'event: X\\ndata: {...}\\n\\n' into [(name, dict), ...]."""
    out = []
    for block in text.strip().split("\n\n"):
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


async def drain(body_iterator):
    """Collect raw chunks from a StreamingResponse body_iterator into one string."""
    chunks = []
    async for chunk in body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return "".join(chunks)


# ----------------------------------------------------------------------------------------
# Happy-path canned stream
# ----------------------------------------------------------------------------------------

def test_canned_stream_event_sequence():
    r = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    assert r.status_code == 200
    events = parse_events(r.text)
    names = [n for n, _ in events]
    # ordered: meta, step×N, findings, done
    assert names[0] == "meta"
    assert names[-2:] == ["findings", "done"]
    meta = events[0][1]
    total = meta["total_steps"]
    assert isinstance(total, int)
    step_names = [n for n, _ in events if n == "step"]
    assert len(step_names) == total
    # full ordered sequence
    assert names == ["meta"] + ["step"] * total + ["findings", "done"]
    assert events[-1][1] == {"ok": True}


def test_meta_payload_shape():
    r = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    _, meta = parse_events(r.text)[0]
    for key in ("query", "clip", "program", "total_steps", "pace_ms", "program_source"):
        assert key in meta, f"meta missing {key}"
    assert meta["pace_ms"] == 0
    assert meta["program_source"] in ("pinned", "live")
    assert isinstance(meta["program"], list) and meta["program"]


def test_free_text_ungrounded_stream(mock_codegen):
    # codegen disabled -> ungrounded run-doc with an empty trace, so NO step events.
    mock_codegen(enabled=False)
    r = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    events = parse_events(r.text)
    names = [n for n, _ in events]
    assert names == ["meta", "findings", "done"]  # no step (trace empty)
    meta = events[0][1]
    assert meta["program_source"] == "ungrounded"
    findings = next(d for n, d in events if n == "findings")
    assert findings["grounded"] is False


# ----------------------------------------------------------------------------------------
# Info-leak guard (drive the route coroutine; force failure DURING trace iteration)
# ----------------------------------------------------------------------------------------

async def test_stream_info_leak_guard(monkeypatch):
    """The error path must (a) let `meta` arrive first, (b) emit an `error` event whose message is
    EXACTLY 'stream failed', and (c) never leak the raw exception text. The failure must happen
    DURING trace iteration, not before: meta computes len(doc["trace"]) BEFORE the loop, so the
    fake trace must implement __len__ (positive int, no raise) and raise ONLY on iteration."""
    SECRET = "/secret/azure/path/leak-me-12345"

    class ExplodingTrace(list):
        def __len__(self):
            return 1  # positive int so meta's len(doc["trace"]) succeeds before the loop

        def __iter__(self):
            raise RuntimeError(f"boom at {SECRET}")

    real_doc_for_request = server._doc_for_request

    def fake_doc(*args, **kwargs):
        doc = real_doc_for_request(*args, **kwargs)
        doc["trace"] = ExplodingTrace()
        return doc

    monkeypatch.setattr(server, "_doc_for_request", fake_doc)

    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    body = await drain(resp.body_iterator)
    events = parse_events(body)
    names = [n for n, _ in events]
    # meta arrived FIRST, then an error (never findings/done after the explosion).
    assert names[0] == "meta"
    assert "error" in names
    err = next(d for n, d in events if n == "error")
    assert err == {"message": "stream failed"}
    # the secret appears NOWHERE in the body.
    assert SECRET not in body
    assert "secret" not in body


async def test_stream_info_leak_guard_bites_only_in_iteration(monkeypatch):
    """Anti-vacuity guard: confirm the failure we inject above is actually inside the trace
    loop. If __len__ raised (i.e. before the loop), meta itself would fail and the test would
    pass for the wrong reason. Here __len__ must NOT raise and meta must be emitted."""
    class ExplodingTrace(list):
        def __len__(self):
            return 1

        def __iter__(self):
            raise RuntimeError("iteration-only failure")

    real = server._doc_for_request

    def fake_doc(*a, **k):
        doc = real(*a, **k)
        doc["trace"] = ExplodingTrace()
        return doc

    monkeypatch.setattr(server, "_doc_for_request", fake_doc)
    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    events = parse_events(await drain(resp.body_iterator))
    # meta must be present (proves __len__ did not raise; failure was in iteration).
    assert events[0][0] == "meta"
    assert any(n == "error" for n, _ in events)
    # no step events emitted (iteration blew up immediately).
    assert not any(n == "step" for n, _ in events)


# ----------------------------------------------------------------------------------------
# Cancel / disconnect (proves ONLY the generator's CancelledError re-raise)
# ----------------------------------------------------------------------------------------

async def test_stream_cancel_reraises_not_swallowed():
    """Drive the generator directly and inject CancelledError into it. The gen() handler must
    RE-RAISE CancelledError (client navigated away) rather than swallow it into an `error` event.
    This proves ONLY the generator's re-raise behavior, not full ASGI disconnect handling."""
    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    it = resp.body_iterator
    # pull the first chunk (meta) so the generator is suspended inside the body.
    first = await it.__anext__()
    assert "event: meta" in (first if isinstance(first, str) else first.decode())
    with pytest.raises(asyncio.CancelledError):
        await it.athrow(asyncio.CancelledError)


# ----------------------------------------------------------------------------------------
# _sse unit
# ----------------------------------------------------------------------------------------

def test_sse_formats_event_block():
    out = server._sse("meta", {"a": 1, "b": "x"})
    assert out == 'event: meta\ndata: {"a": 1, "b": "x"}\n\n'


# ----------------------------------------------------------------------------------------
# /api/run_doc
# ----------------------------------------------------------------------------------------

def test_run_doc_canned_returns_full_doc():
    r = client.get("/api/run_doc", params={"query": "hero-10-first-goal"})
    assert r.status_code == 200
    doc = r.json()
    for key in ("schema_version", "query", "clip", "program", "trace", "findings", "program_source"):
        assert key in doc
    assert doc["program_source"] in ("pinned", "live")


def test_run_doc_free_text_is_ungrounded(mock_codegen):
    mock_codegen(enabled=False)
    r = client.get("/api/run_doc", params={"query_text": "how many?"})
    doc = r.json()
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False


# ----------------------------------------------------------------------------------------
# /api/catalog field projection
# ----------------------------------------------------------------------------------------

def test_catalog_projects_public_clip_fields_only():
    r = client.get("/api/catalog")
    body = r.json()
    assert "clips" in body and "queries" in body
    clip = body["clips"][0]
    for key in ("id", "label", "width", "height", "duration_ms"):
        assert key in clip
    # internal fields dropped.
    assert "cache" not in clip
    assert "hint" not in clip
    q = body["queries"][0]
    assert set(q) == {"id", "text", "clip"}


# ----------------------------------------------------------------------------------------
# /api/health
# ----------------------------------------------------------------------------------------

def test_health_reports_version_and_query_ids():
    r = client.get("/api/health")
    body = r.json()
    assert body["ok"] is True
    assert body["version"] == server.app.version
    assert "hero-10-first-goal" in body["queries"]
    assert body["queries"] == [q["id"] for q in canned.QUERIES]


# ----------------------------------------------------------------------------------------
# _resolve_run_request validation branches (no TestClient)
# ----------------------------------------------------------------------------------------

def test_resolve_run_request_validation_branches():
    from fastapi import HTTPException

    def status(fn):
        try:
            fn()
            return 200
        except HTTPException as e:
            return e.status_code

    # neither query nor query_text -> 400
    assert status(lambda: server._resolve_run_request(None, None, None)) == 400
    # both -> 400
    assert status(lambda: server._resolve_run_request("hero-10-first-goal", "x", None)) == 400
    # unknown query -> 404
    assert status(lambda: server._resolve_run_request("does-not-exist", None, None)) == 404
    # over-length query_text -> 400
    long = "x" * (server.MAX_QUERY_TEXT_LEN + 1)
    assert status(lambda: server._resolve_run_request(None, long, None)) == 400
    # unknown clip -> 404
    assert status(lambda: server._resolve_run_request(None, "how many?", "no-such-clip")) == 404
    # valid forms
    assert server._resolve_run_request("hero-10-first-goal", None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None)[0] == "free"


# ----------------------------------------------------------------------------------------
# SSE response metadata (anti-buffering headers)
# ----------------------------------------------------------------------------------------

def test_run_response_is_event_stream_with_anti_buffering_headers():
    with client.stream("GET", "/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no"
        r.read()  # drain so the stream closes cleanly
