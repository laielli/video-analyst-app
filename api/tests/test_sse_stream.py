"""
SSE-stream tests over /api/run via FastAPI TestClient, plus _sse unit tests and the route
validation helper.

The stream contract (server.gen, nested in the run() route): meta -> step* -> findings -> done,
or `error` on exception; cancel re-raises CancelledError; the error path must NEVER leak raw
exception text. TestClient drives the StreamingResponse synchronously.

The cancel + info-leak tests drive the route coroutine directly (gen() is a nested closure,
not importable) and need pytest-asyncio (asyncio_mode=auto in pytest.ini).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server
import canned

client = TestClient(server.app)

HERO = "hero-10-first-goal"


def events(resp_text: str):
    """Parse 'event: X\\ndata: {...}\\n\\n' blocks -> [(name, dict), ...]."""
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
        if name is not None:
            out.append((name, data))
    return out


# --------------------------------------------------------------------------------------
# _sse unit
# --------------------------------------------------------------------------------------

def test_sse_format_exact():
    assert server._sse("meta", {"a": 1}) == 'event: meta\ndata: {"a": 1}\n\n'


# --------------------------------------------------------------------------------------
# Event sequence (canned)
# --------------------------------------------------------------------------------------

def test_canned_run_event_sequence():
    resp = client.get(f"/api/run?query={HERO}&pace_ms=0")
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    meta = evs[0][1]
    total = meta["total_steps"]  # an int — do NOT len()
    assert isinstance(total, int) and total > 0
    assert names == ["meta"] + ["step"] * total + ["findings", "done"]
    # done payload
    assert evs[-1] == ("done", {"ok": True})


def test_meta_payload_shape():
    resp = client.get(f"/api/run?query={HERO}&pace_ms=0")
    meta = events(resp.text)[0][1]
    for k in ("query", "clip", "program", "total_steps", "pace_ms", "program_source"):
        assert k in meta, k
    assert meta["pace_ms"] == 0
    assert meta["program_source"] in ("pinned", "live")
    assert isinstance(meta["program"], list) and len(meta["program"]) == meta["total_steps"]


def test_canned_findings_event_present():
    resp = client.get(f"/api/run?query={HERO}&pace_ms=0")
    evs = events(resp.text)
    findings = next(d for n, d in evs if n == "findings")
    assert "grounded" in findings


# --------------------------------------------------------------------------------------
# Free-text ungrounded stream (codegen disabled)
# --------------------------------------------------------------------------------------

def test_free_text_ungrounded_stream(mock_codegen):
    mock_codegen(enabled=False)
    resp = client.get("/api/run?query_text=how+many+players&pace_ms=0")
    assert resp.status_code == 200
    evs = events(resp.text)
    names = [n for n, _ in evs]
    # trace is empty -> no `step` events
    assert names == ["meta", "findings", "done"]
    assert evs[0][1]["program_source"] == "ungrounded"
    assert evs[0][1]["total_steps"] == 0
    findings = next(d for n, d in evs if n == "findings")
    assert findings["grounded"] is False


# --------------------------------------------------------------------------------------
# Info-leak guard — the error path must not forward exception text
# --------------------------------------------------------------------------------------

class _ExplodingTrace(list):
    """A trace that reports a positive length (meta computes len() BEFORE iterating) but raises a
    secret-bearing exception ONLY when iterated. This forces the failure DURING the trace loop,
    after `meta` has already been yielded, so the test proves the error is caught + sanitized."""
    SECRET = "/etc/secret/azure-key-9f3b.txt"

    def __len__(self):
        return 3  # positive so meta's total_steps is plausible and the loop is entered

    def __iter__(self):
        raise RuntimeError(f"boom leaking {self.SECRET}")


async def test_info_leak_guard_sanitizes_error(monkeypatch):
    """Drive the route coroutine directly (gen() is a nested closure, not importable). The forced
    exception happens DURING trace iteration, AFTER meta — proving: (a) meta arrives first,
    (b) a sanitized `error` event with message exactly 'stream failed' follows, (c) the secret
    appears NOWHERE in the streamed body."""
    doc = {
        "query": "q", "clip": {"id": "c", "width": 1, "height": 1, "duration_ms": 1},
        "program": [], "trace": _ExplodingTrace(), "findings": {"grounded": False},
        "program_source": "pinned",
    }
    monkeypatch.setattr(server, "_doc_for_request", lambda *a, **k: doc)

    # Call the route coroutine directly: pass every param explicitly so the unresolved
    # FastAPI Query(...) defaults aren't forwarded (the helper is bypassed via monkeypatch).
    resp = await server.run(query=HERO, query_text=None, clip=None, pace_ms=0)
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)
    body = "".join(chunks)

    evs = events(body)
    names = [n for n, _ in evs]
    assert names[0] == "meta"  # meta arrived FIRST
    assert ("error", {"message": "stream failed"}) in evs  # sanitized error follows
    assert _ExplodingTrace.SECRET not in body  # secret leaks NOWHERE
    assert "RuntimeError" not in body and "boom" not in body
    # the error path replaces done; no done after a failure
    assert "done" not in names


# --------------------------------------------------------------------------------------
# Cancel / disconnect — generator re-raises CancelledError (NOT swallowed into error)
# --------------------------------------------------------------------------------------

async def test_cancel_reraises_not_swallowed():
    """Proves ONLY that the generator re-raises CancelledError when the client disconnects mid
    stream, rather than swallowing it into an `error` event. Requires pytest-asyncio."""
    # Pass every param explicitly: calling the coroutine directly bypasses FastAPI's dependency
    # resolution, so the Query(...) defaults would otherwise be forwarded as Query objects.
    resp = await server.run(query=HERO, query_text=None, clip=None, pace_ms=0)
    it = resp.body_iterator
    first = await it.__anext__()  # prime: yields the meta frame
    assert first.startswith("event: meta")
    with pytest.raises(asyncio.CancelledError):
        await it.athrow(asyncio.CancelledError)


# --------------------------------------------------------------------------------------
# /api/run_doc
# --------------------------------------------------------------------------------------

def test_run_doc_returns_full_doc_with_program_source():
    resp = client.get(f"/api/run_doc?query={HERO}")
    assert resp.status_code == 200
    doc = resp.json()
    for k in ("query", "clip", "program", "trace", "findings", "program_source"):
        assert k in doc, k
    assert doc["program_source"] in ("pinned", "live")


def test_run_doc_free_text_ungrounded(mock_codegen):
    mock_codegen(enabled=False)
    resp = client.get("/api/run_doc?query_text=is+there+a+ball")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["program_source"] == "ungrounded"
    assert doc["findings"]["grounded"] is False


# --------------------------------------------------------------------------------------
# /api/catalog field projection
# --------------------------------------------------------------------------------------

def test_catalog_drops_internal_fields():
    resp = client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    clip = body["clips"][0]
    for k in ("id", "label", "width", "height", "duration_ms"):
        assert k in clip, k
    assert "cache" not in clip  # internal path dropped
    assert "hint" not in clip   # internal codegen hint dropped
    q = body["queries"][0]
    assert set(q) == {"id", "text", "clip"}


# --------------------------------------------------------------------------------------
# /api/health
# --------------------------------------------------------------------------------------

def test_health_reports_known_queries():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["version"] == server.app.version
    assert HERO in body["queries"]
    assert body["queries"] == [q["id"] for q in canned.QUERIES]


# --------------------------------------------------------------------------------------
# _resolve_run_request validation branches (no TestClient needed)
# --------------------------------------------------------------------------------------

def test_resolve_neither_query_nor_text_is_400():
    with pytest.raises(HTTPException) as exc:
        server._resolve_run_request(None, None, None)
    assert exc.value.status_code == 400


def test_resolve_both_query_and_text_is_400():
    with pytest.raises(HTTPException) as exc:
        server._resolve_run_request(HERO, "some text", None)
    assert exc.value.status_code == 400


def test_resolve_unknown_query_is_404():
    with pytest.raises(HTTPException) as exc:
        server._resolve_run_request("no-such-query", None, None)
    assert exc.value.status_code == 404


def test_resolve_overlong_query_text_is_400():
    over = "x" * (server.MAX_QUERY_TEXT_LEN + 1)
    with pytest.raises(HTTPException) as exc:
        server._resolve_run_request(None, over, None)
    assert exc.value.status_code == 400


def test_resolve_unknown_clip_is_404():
    with pytest.raises(HTTPException) as exc:
        server._resolve_run_request(None, "valid text", "no-such-clip")
    assert exc.value.status_code == 404


def test_resolve_canned_happy_path():
    kind, entry = server._resolve_run_request(HERO, None, None)
    assert kind == "canned"
    assert entry["id"] == HERO


def test_resolve_free_text_defaults_clip():
    kind, payload = server._resolve_run_request(None, "how many players", None)
    assert kind == "free"
    text, cid = payload
    assert text == "how many players"
    assert cid == next(iter(canned.CLIPS))


# --------------------------------------------------------------------------------------
# SSE response metadata (anti-buffering headers)
# --------------------------------------------------------------------------------------

def test_run_response_has_sse_headers():
    resp = client.get(f"/api/run?query={HERO}&pace_ms=0")
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
