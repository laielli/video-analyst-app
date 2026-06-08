"""
SSE-stream tests over /api/run via FastAPI TestClient + the surrounding HTTP endpoints
(/api/run_doc, /api/catalog, /api/health) and the _resolve_run_request validation helper.

The /api/run generator yields meta -> step* -> findings -> done, or an `error` event on an
exception during trace iteration (NEVER leaking raw exception text — info-leak guard), and
re-raises CancelledError on client disconnect. `gen()` is a NESTED CLOSURE inside the route
handler so it is NOT importable; the closure-specific tests drive it via `await server.run(...)`
and iterate `resp.body_iterator` directly.

Requires httpx (TestClient) + pytest-asyncio (asyncio_mode=auto in pytest.ini) for the async
cancel/info-leak tests.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import server
import canned

client = TestClient(server.app)


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


async def drain(body_iterator):
    """Collect a StreamingResponse body_iterator into a single decoded string."""
    chunks = []
    async for chunk in body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


# ---------------------------------------------------------------------------------------
# canned event sequence
# ---------------------------------------------------------------------------------------

def test_canned_event_sequence_is_meta_steps_findings_done():
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    # exactly meta, then N steps, then findings, then done.
    meta = evs[0][1]
    n = meta["total_steps"]  # an int — do NOT len() it
    assert isinstance(n, int)
    assert names == ["meta"] + ["step"] * n + ["findings", "done"]
    assert evs[-1][1] == {"ok": True}


def test_meta_payload_shape():
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    meta = events(resp.text)[0][1]
    for k in ("query", "clip", "program", "total_steps", "pace_ms", "program_source"):
        assert k in meta, f"meta missing {k}"
    assert meta["pace_ms"] == 0
    assert meta["program_source"] in ("pinned", "live")
    assert isinstance(meta["program"], list) and meta["program"]
    assert meta["total_steps"] == len(meta["program"])


# ---------------------------------------------------------------------------------------
# free-text ungrounded stream (codegen disabled)
# ---------------------------------------------------------------------------------------

def test_free_text_ungrounded_stream(mock_codegen):
    mock_codegen(enabled=False)  # no codegen -> ungrounded, empty trace
    resp = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    # empty trace -> no step events: meta -> findings -> done.
    assert names == ["meta", "findings", "done"]
    assert evs[0][1]["program_source"] == "ungrounded"
    findings = evs[1][1]
    assert findings["grounded"] is False
    assert findings["reason"] == "codegen-disabled"


# ---------------------------------------------------------------------------------------
# info-leak guard (drives the nested gen() closure via server.run)
# ---------------------------------------------------------------------------------------

class _RaisingTrace:
    """A `trace` that has a positive length (meta reads len() BEFORE the loop, must not raise)
    but raises a secret-bearing exception ONLY when iterated (the loop at server.py during
    trace iteration). This exercises the info-leak guard's except branch."""
    SECRET = "/secret/azure/key/path/leaked-token-xyz"

    def __len__(self):
        return 3  # positive int so meta's total_steps is sane and the loop is entered

    def __iter__(self):
        raise RuntimeError(self.SECRET)


@pytest.mark.asyncio
async def test_info_leak_guard_sends_fixed_message_no_secret(monkeypatch):
    """The error path must emit `error: {message: "stream failed"}` and NEVER the raw exception
    text. The forced failure happens DURING trace iteration (after meta), so meta arrives first
    and the secret never reaches the body."""
    real = server._doc_for_request

    def fake(query, query_text, clip):
        doc = real("hero-10-first-goal", None, None)
        doc["trace"] = _RaisingTrace()  # len() ok, iteration raises with a secret path
        return doc
    monkeypatch.setattr(server, "_doc_for_request", fake)

    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    body = await drain(resp.body_iterator)
    evs = events(body)
    names = [n for n, _ in evs]
    # meta FIRST, then the fixed error message.
    assert names[0] == "meta"
    assert ("error", {"message": "stream failed"}) in evs
    # the secret appears NOWHERE in the streamed body.
    assert _RaisingTrace.SECRET not in body
    assert "secret" not in body.lower()


# ---------------------------------------------------------------------------------------
# cancel / disconnect re-raises CancelledError (proves only the generator's re-raise)
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_reraises_not_swallowed_into_error():
    """Proves ONLY that the SSE generator re-raises asyncio.CancelledError (client disconnect)
    rather than swallowing it into an `error` event. Drives the nested gen() via server.run and
    throws CancelledError into its body_iterator."""
    # pass every param explicitly as a plain value — calling the route handler directly bypasses
    # FastAPI's Query() default resolution, so query_text/clip must be real None, not Query(None).
    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    it = resp.body_iterator
    # advance past meta so we're inside the generator.
    first = await it.__anext__()
    assert "event: meta" in (first.decode() if isinstance(first, (bytes, bytearray)) else first)
    with pytest.raises(asyncio.CancelledError):
        await it.athrow(asyncio.CancelledError())


# ---------------------------------------------------------------------------------------
# _sse unit
# ---------------------------------------------------------------------------------------

def test_sse_format_exact():
    out = server._sse("meta", {"a": 1})
    assert out == 'event: meta\ndata: {"a": 1}\n\n'


# ---------------------------------------------------------------------------------------
# /api/run_doc
# ---------------------------------------------------------------------------------------

def test_run_doc_canned_returns_full_doc():
    resp = client.get("/api/run_doc", params={"query": "hero-10-first-goal"})
    assert resp.status_code == 200
    doc = resp.json()
    for k in ("schema_version", "query", "clip", "program", "trace", "findings", "program_source"):
        assert k in doc
    assert doc["program_source"] in ("pinned", "live")


def test_run_doc_free_text_ungrounded(mock_codegen):
    mock_codegen(enabled=False)
    resp = client.get("/api/run_doc", params={"query_text": "how many players?"})
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False


# ---------------------------------------------------------------------------------------
# /api/catalog field projection
# ---------------------------------------------------------------------------------------

def test_catalog_drops_internal_fields():
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert "clips" in body and "queries" in body
    for clip in body["clips"]:
        # internal keys dropped.
        assert "cache" not in clip
        assert "hint" not in clip
        # public keys kept.
        for k in ("id", "label", "width", "height", "duration_ms"):
            assert k in clip
    for q in body["queries"]:
        assert set(q.keys()) == {"id", "text", "clip"}


# ---------------------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------------------

def test_health_lists_canned_queries():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["version"] == server.app.version
    assert "hero-10-first-goal" in body["queries"]
    assert body["queries"] == [q["id"] for q in canned.QUERIES]


# ---------------------------------------------------------------------------------------
# _resolve_run_request validation branches (plain helper, no TestClient)
# ---------------------------------------------------------------------------------------

def _status(fn):
    from fastapi import HTTPException
    try:
        fn()
        return 200
    except HTTPException as e:
        return e.status_code


def test_resolve_validation_branches():
    # neither query nor query_text -> 400
    assert _status(lambda: server._resolve_run_request(None, None, None)) == 400
    # both -> 400
    assert _status(lambda: server._resolve_run_request("hero-10-first-goal", "x", None)) == 400
    # unknown query -> 404
    assert _status(lambda: server._resolve_run_request("nope", None, None)) == 404
    # over-length query_text -> 400 (cap before codegen)
    over = "x" * (server.MAX_QUERY_TEXT_LEN + 1)
    assert _status(lambda: server._resolve_run_request(None, over, None)) == 400
    # unknown clip -> 404
    assert _status(lambda: server._resolve_run_request(None, "how many?", "no-such-clip")) == 404
    # happy paths
    assert server._resolve_run_request("hero-10-first-goal", None, None)[0] == "canned"
    assert server._resolve_run_request(None, "how many?", None)[0] == "free"


# ---------------------------------------------------------------------------------------
# SSE response metadata (content type + anti-buffering headers)
# ---------------------------------------------------------------------------------------

def test_sse_response_headers():
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
