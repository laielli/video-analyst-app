"""
SSE-stream tests over /api/run + the server's helper surface. gen() (the nested closure inside
the run() route) yields meta -> step* -> findings -> done, or error on exception; cancel re-raises
CancelledError; the error path must never leak raw exception text. FastAPI's TestClient drives the
StreamingResponse synchronously for the happy-path sequence; the cancel + info-leak tests drive the
route coroutine directly (gen() is not importable) and iterate resp.body_iterator.
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
    """Parse an SSE body ("event: X\\ndata: {...}\\n\\n") into [(name, dict), ...]."""
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


async def drain(body_iterator):
    """Collect a StreamingResponse body_iterator into a single decoded string."""
    chunks = []
    async for chunk in body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    return "".join(chunks)


# ---- happy-path event sequence ----------------------------------------------------------

def test_run_event_sequence_canned():
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    meta = dict(evs[0][1])
    n_steps = meta["total_steps"]
    assert isinstance(n_steps, int)
    # exact ordered sequence: meta, step*N, findings, done.
    assert names == ["meta"] + ["step"] * n_steps + ["findings", "done"]
    assert evs[-1] == ("done", {"ok": True})


def test_run_meta_payload_shape():
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    name, meta = events(resp.text)[0]
    assert name == "meta"
    for k in ("query", "clip", "program", "total_steps", "pace_ms", "program_source"):
        assert k in meta, f"meta missing {k}"
    assert meta["pace_ms"] == 0
    assert meta["program_source"] in ("pinned", "live")
    assert meta["total_steps"] == len(meta["program"])


def test_run_free_text_ungrounded_stream(mock_codegen):
    # codegen disabled -> ungrounded run-doc with an EMPTY trace, so NO step events fire.
    mock_codegen(enabled=False)
    resp = client.get("/api/run", params={"query_text": "how many players?", "pace_ms": 0})
    evs = events(resp.text)
    names = [n for n, _ in evs]
    assert names == ["meta", "findings", "done"]  # no step (trace empty)
    assert evs[0][1]["program_source"] == "ungrounded"
    assert evs[1][1]["grounded"] is False


# ---- info-leak guard --------------------------------------------------------------------

class _ExplodingTrace:
    """A `trace` that reports a positive length (so meta computes total_steps fine) but raises a
    secret-bearing exception ONLY when iterated — exercising the error path DURING the step loop,
    after meta has already been yielded."""
    SECRET = "/var/secret/azure-key-9f3a"

    def __len__(self):
        return 3  # positive int; meta's len(doc["trace"]) at meta-build time must not raise

    def __iter__(self):
        raise RuntimeError(f"boom at {self.SECRET}")


@pytest.mark.asyncio
async def test_info_leak_guard_meta_first_then_fixed_error(monkeypatch):
    """The failure must happen DURING trace iteration (after meta). Asserts: (a) meta arrived
    FIRST, (b) a fixed-message error event follows, (c) the secret appears NOWHERE in the body."""
    base = server._doc_for_request

    def doc_with_exploding_trace(*a, **k):
        doc = dict(base("hero-10-first-goal", None, None))
        doc["trace"] = _ExplodingTrace()
        return doc

    monkeypatch.setattr(server, "_doc_for_request", doc_with_exploding_trace)

    # Called directly (not via FastAPI), so the other Query() defaults must be passed as None.
    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    body = await drain(resp.body_iterator)
    evs = events(body)

    assert evs[0][0] == "meta", "meta must arrive before the failure"
    error_evs = [(n, d) for n, d in evs if n == "error"]
    assert error_evs, "an error event must follow"
    assert error_evs[0][1] == {"message": "stream failed"}  # fixed message, exactly
    assert _ExplodingTrace.SECRET not in body  # the secret never leaks


# ---- cancel / disconnect ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_reraises_not_swallowed():
    """Proves ONLY that the generator re-raises CancelledError (does not swallow it into an
    `error` event) when the client disconnects mid-stream. Requires pytest-asyncio + asyncio_mode
    auto. Drives the route coroutine directly because gen() is a non-importable nested closure."""
    # Called directly (not via FastAPI), so the other Query() defaults must be passed as None.
    resp = await server.run(query="hero-10-first-goal", query_text=None, clip=None, pace_ms=0)
    it = resp.body_iterator
    await it.__anext__()  # pull the first chunk (meta) so the generator is suspended in the loop
    with pytest.raises(asyncio.CancelledError):
        await it.athrow(asyncio.CancelledError)


# ---- _sse unit --------------------------------------------------------------------------

def test_sse_formats_event_block():
    out = server._sse("meta", {"a": 1, "b": "x"})
    assert out == 'event: meta\ndata: {"a": 1, "b": "x"}\n\n'


# ---- /api/run_doc -----------------------------------------------------------------------

def test_run_doc_canned_returns_full_doc():
    resp = client.get("/api/run_doc", params={"query": "hero-10-first-goal"})
    assert resp.status_code == 200
    doc = resp.json()
    for k in ("query", "clip", "program", "trace", "findings", "program_source"):
        assert k in doc
    assert doc["program_source"] in ("pinned", "live")


def test_run_doc_free_text_ungrounded(mock_codegen):
    mock_codegen(enabled=False)
    resp = client.get("/api/run_doc", params={"query_text": "anything"})
    doc = resp.json()
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False


# ---- /api/catalog -----------------------------------------------------------------------

def test_catalog_field_projection_drops_internal():
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    for c in body["clips"]:
        assert set(c) <= {"id", "label", "width", "height", "duration_ms"}
        # internal fields are dropped.
        assert "cache" not in c and "hint" not in c
        assert "id" in c
    # queries carry id/text/clip.
    for q in body["queries"]:
        assert set(q) == {"id", "text", "clip"}


# ---- /api/health ------------------------------------------------------------------------

def test_health_returns_ok_version_queries():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["version"] == server.app.version
    assert "hero-10-first-goal" in body["queries"]
    assert body["queries"] == [q["id"] for q in canned.QUERIES]


# ---- _resolve_run_request validation (no TestClient) ------------------------------------

def test_resolve_run_request_four_validation_branches():
    from fastapi import HTTPException

    def code(fn):
        try:
            fn()
            return 200
        except HTTPException as e:
            return e.status_code

    # neither query nor query_text -> 400.
    assert code(lambda: server._resolve_run_request(None, None, None)) == 400
    # both -> 400.
    assert code(lambda: server._resolve_run_request("hero-10-first-goal", "x", None)) == 400
    # unknown query -> 404.
    assert code(lambda: server._resolve_run_request("nope", None, None)) == 404
    # over-length query_text -> 400.
    long = "x" * (server.MAX_QUERY_TEXT_LEN + 1)
    assert code(lambda: server._resolve_run_request(None, long, None)) == 400
    # unknown clip -> 404.
    assert code(lambda: server._resolve_run_request(None, "x", "no-such-clip")) == 404


# ---- SSE response metadata --------------------------------------------------------------

def test_run_response_headers_anti_buffering():
    resp = client.get("/api/run", params={"query": "hero-10-first-goal", "pace_ms": 0})
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
