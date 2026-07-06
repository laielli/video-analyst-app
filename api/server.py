"""
FastAPI + SSE wrapper around the interpreter. Streams a run-doc step-by-step so the UI can
render the glass-box run live (D-DR4 streamed authoring, D-DR7 ~1.2s/step, D-DR8 cause->effect).

Run (from api/):
    source .venv/bin/activate && pip install -r requirements.txt
    uvicorn server:app --reload --port 8000

Endpoints:
    GET /api/health                      liveness + known queries
    GET /api/catalog                     clips + canned queries (for the UI selectors)
    GET /api/run_doc?query=<id>          the full run-doc at once (debug / scrub)
    GET /api/run?query=<id>&pace_ms=1200 SSE: meta -> step* -> findings -> done

SSE event types (named):
    meta     {query, clip, program, total_steps, pace_ms}   render PROGRAM panel + stepper
    step     {<run-doc trace entry>}                         advance active line, draw overlays
    findings {<run-doc findings>}                            reveal the verdict
    done     {ok: true}
    error    {message}
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))            # interpreter package
sys.path.insert(0, str(API_DIR / "scripts"))  # strip_comments

from interpreter import Cache, Interpreter  # noqa: E402
from validate_program import strip_comments, validation_errors  # noqa: E402
from ocr_probe import load_dotenv  # noqa: E402
import canned  # noqa: E402
import codegen  # noqa: E402
import run_cache  # noqa: E402

# Load api/.env so `uvicorn server:app` picks up AZURE_OPENAI_* (codegen) without extra flags.
# No-op if the file is absent — codegen then stays disabled and runs fall back to pinned.
load_dotenv(API_DIR / ".env")

# Free-text guardrails (trust boundary / cost). A free-text query triggers a BILLED Azure
# codegen call on an unauthenticated GET, so we cap the query length (rejected with 400 BEFORE
# codegen) and bound codegen calls per process so an abuser can't run up unbounded paid calls.
MAX_QUERY_TEXT_LEN = 500
MAX_CODEGEN_CALLS = 200
_codegen_calls = 0


def _codegen_budget_left() -> bool:
    return _codegen_calls < MAX_CODEGEN_CALLS


# Content-addressed run cache (cost win + shareable permalinks). ONE store is constructed at
# startup from env; `build_free_text_run_doc` reads it before codegen and writes it only on a
# grounded run, and the `?run=<id>` permalink path replays a stored run by its content address.
# Exposed as a module-global so tests can monkeypatch it to a tmp_path-rooted FileRunStore
# (do NOT freeze it in a closure / default-arg — see api/run_cache.py + the test fixtures).
RUN_CACHE_DIR = os.environ.get("RUN_CACHE_DIR") or str(API_DIR / ".run_cache")
RUN_CACHE_MAX_ENTRIES = int(os.environ.get("RUN_CACHE_MAX_ENTRIES", "1000"))
_run_store: run_cache.RunStore = run_cache.FileRunStore(
    RUN_CACHE_DIR, max_entries=RUN_CACHE_MAX_ENTRIES
)


def get_store() -> run_cache.RunStore:
    """Accessor for the process-wide run store (tests monkeypatch `server._run_store`)."""
    return _run_store


# The run-doc schema is additionalProperties:false at the root, so run_id + cached can NOT live
# inside the doc — they ride a SEPARATE serve-meta dict (the SSE `meta` event / run_doc `_meta`
# envelope), built at serve time by the route. The builders return a SCHEMA-PURE doc and OPTIONALLY
# populate a caller-supplied `serve_meta` out-dict with {cached, run_id?}; a direct caller (e.g.
# the existing build_free_text_run_doc tests) gets the pure doc and ignores the out-param.
def _set_serve_meta(serve_meta: dict | None, *, cached: bool, run_id: str | None = None) -> None:
    if serve_meta is None:
        return
    serve_meta["cached"] = cached
    if run_id is not None:
        serve_meta["run_id"] = run_id


app = FastAPI(title="Glass-Box Video Analyst API", version="0.1.0")

# CORS: the Next.js client runs on a different origin — the dev server (localhost) or the
# deployed Static Web App. Dev localhost origins are always allowed (regex covers any port);
# production origins are supplied via ALLOWED_ORIGINS (comma-separated, e.g.
# "https://glassbox-web.azurestaticapps.net"). EventSource/SSE issues cross-origin GETs,
# so the deployed web origin MUST appear here or the live run stream is blocked.
_DEV_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def _parse_prod_origins(raw: str) -> list[str]:
    """Parse ALLOWED_ORIGINS (comma-separated) into a clean origin list.

    Whitespace is trimmed and blank entries dropped, so an unset/empty env var yields
    [] (dev-only CORS — the pre-deploy default) and sloppy deploy config (stray spaces,
    a blank entry, a trailing comma) never produces a bogus "" origin.
    """
    return [o.strip() for o in raw.split(",") if o.strip()]


_PROD_ORIGINS = _parse_prod_origins(os.environ.get("ALLOWED_ORIGINS", ""))
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS + _PROD_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _pinned_program(entry: dict) -> list[dict]:
    return strip_comments(json.loads(Path(entry["program"]).read_text()))["program"]


def _live_run_doc(entry: dict, cache: Cache) -> dict | None:
    """D-DR6 codegen path: ask Azure OpenAI to compile the question into a program, then accept
    it ONLY if it (a) validates structurally + semantically and (b) executes over the cache to a
    grounded (non-partial) answer. Any miss — no creds, API error, invalid program, cache gap —
    returns None and the caller silently falls back to the pinned known-good program. Execution
    stays replay-mode, so the live program runs deterministically; only codegen hits the network."""
    if not codegen.enabled():
        return None
    qid = entry["id"]
    try:
        program = codegen.AzureCodegen.from_env().generate(entry["text"])
    except Exception as e:  # noqa: BLE001 — any generation failure -> pinned fallback
        print(f"[codegen] '{qid}': generation failed ({e}); using pinned program", file=sys.stderr)
        return None

    # Apply the same numeric-bounds trust boundary the free-text path uses: resolve THIS
    # query's clip so validation rejects an out-of-range sample_frames before it can reach
    # Interpreter.run. clip_by_id(None) -> None -> bounds are skipped (back-compat), so an
    # unknown/missing clip degrades to the prior structure-only behavior rather than erroring.
    clip = canned.clip_by_id(entry.get("clip"))
    errors = validation_errors({"program": program}, clip=clip)
    if errors:
        print(f"[codegen] '{qid}': {len(errors)} validation error(s); using pinned. first: {errors[0]}",
              file=sys.stderr)
        return None

    try:
        doc = Interpreter(cache).run(program)
    except Exception as e:  # noqa: BLE001 — e.g. sampled window misses the cached frames
        print(f"[codegen] '{qid}': live program failed to execute ({e}); using pinned", file=sys.stderr)
        return None

    findings = doc.get("findings", {})
    if findings.get("partial") or findings.get("answer") in (None, "", "Unknown"):
        print(f"[codegen] '{qid}': live program produced no grounded answer; using pinned", file=sys.stderr)
        return None

    doc["program_source"] = "live"
    return doc


def build_run_doc(entry: dict) -> dict:
    """Resolve the program for this query (live codegen with D-DR6 fallback) and replay it over
    the cache -> run-doc. Replay is instant and free; with no Azure OpenAI creds this is exactly
    the prior behavior (the pinned program, tagged program_source='pinned')."""
    cache = Cache.load(entry["cache"])
    doc = _live_run_doc(entry, cache)
    if doc is not None:
        return doc
    doc = Interpreter(cache).run(_pinned_program(entry))
    doc["program_source"] = "pinned"
    return doc


# ---- free-text path (no pinned fallback) ------------------------------------------------

def _clip_for_doc(clip: dict) -> dict:
    """Project a canned CLIPS entry to ONLY the run-doc schema's clip keys (the schema's clip
    $def is additionalProperties:false, so `label`/`cache`/`hint` must be dropped)."""
    out = {k: clip[k] for k in ("id", "width", "height", "duration_ms") if k in clip}
    if "fps" in clip:
        out["fps"] = clip["fps"]
    return out


def _ungrounded(clip: dict, query_text: str, reason: str, program: list[dict] | None = None,
                trace: list[dict] | None = None) -> dict:
    """The honest 'couldn't ground this' run-doc (Q1 -> A). It carries EVERY key the SSE stream
    loop reads (query/clip/program/trace/findings) so the loop never KeyErrors, validates against
    the extended run_doc.schema.json, and tags program_source='ungrounded'. `reason` is a fixed
    enum (never raw exception text — info-leak guard). When a (validated) program ran but didn't
    ground, its program + trace are streamed so the user sees WHERE it ran dry (D-DR5)."""
    return {
        "schema_version": "1",
        "query": query_text,
        "clip": _clip_for_doc(clip),
        "program": program or [],
        "trace": trace or [],
        "findings": {"answer": None, "verdict": None, "grounded": False, "reason": reason, "partial": True},
        "program_source": "ungrounded",
    }


def build_free_text_run_doc(query_text: str, clip_id: str, serve_meta: dict | None = None) -> dict:
    """ViperGPT loop for a TYPED question, with the safety net removed: codegen -> validate ->
    execute over the clip's replay cache. There is NO pinned fallback, so every failure mode
    (no creds, codegen error, invalid program, execution error, no grounded answer) resolves to
    an explicit `_ungrounded` run-doc — never a hero-program substitution, never a 500. The clip
    is resolved by the caller (404 on unknown); here it is assumed valid.

    Returns a SCHEMA-PURE run-doc. If `serve_meta` is given it is populated with {cached, run_id?}
    for the route's SSE meta / run_doc envelope: `cached` true on a store hit; `run_id` present
    ONLY on the grounded path (a store hit, or a freshly grounded run) — never on the four
    non-grounded exits, so the UI never offers a share link to a run that was not stored."""
    global _codegen_calls
    _set_serve_meta(serve_meta, cached=False)  # default; the grounded/hit paths refine it below
    clip = canned.clip_by_id(clip_id)  # caller raises 404 if None; defensive guard below
    if clip is None:
        return _ungrounded({"id": clip_id, "width": 0, "height": 0, "duration_ms": 0},
                           query_text, "execution-error")

    # Content-addressed cache lookup — BEFORE codegen.enabled() and the budget check, so an
    # identical repeat question is served for FREE (no billed call, no budget burn). Only
    # validated + grounded runs are ever stored, so a hit is always a trustworthy run-doc;
    # store.get re-validates it against the schema and treats any bad entry as a miss.
    key = run_cache.cache_key(query_text, clip_id)
    hit = get_store().get(key)
    if hit is not None:
        # A hit is a previously validated + grounded run-doc — served free, no codegen, no budget
        # burn. The doc stays schema-pure; run_id + cached ride the serve_meta out-dict.
        _set_serve_meta(serve_meta, cached=True, run_id=key)
        return hit

    cache = Cache.load(clip["cache"])

    if not codegen.enabled():
        return _ungrounded(clip, query_text, "codegen-disabled")

    if not _codegen_budget_left():
        # Cost guard: out of per-process codegen budget — degrade honestly, don't bill more.
        return _ungrounded(clip, query_text, "codegen-error")

    try:
        _codegen_calls += 1
        program = codegen.AzureCodegen.from_env().generate(query_text, clip=clip)
    except Exception:  # noqa: BLE001 — any generation failure -> ungrounded (no exc text leaked)
        return _ungrounded(clip, query_text, "codegen-error")

    # Trust boundary: validate (structure + semantics + numeric bounds) BEFORE Interpreter.run.
    if validation_errors({"program": program}, clip=clip):
        return _ungrounded(clip, query_text, "invalid-program")

    try:
        doc = Interpreter(cache).run(program, query=query_text)
    except Exception:  # noqa: BLE001 — e.g. sampled window misses cached frames
        return _ungrounded(clip, query_text, "execution-error", program=program)

    f = doc.get("findings", {})
    if not f.get("grounded"):
        # Validated program ran but didn't ground: stream its trace so the failure is legible.
        # NOT cached (caching a failure would make a transient miss permanent + let an attacker
        # poison the cache with a cheap ungrounded question) and carries NO run_id.
        return _ungrounded(clip, query_text, f.get("reason") or "no-grounded-answer",
                           program=doc["program"], trace=doc["trace"])

    doc["program_source"] = "live"
    # Trust boundary: ONLY validated + grounded runs are cached. Store the schema-pure doc, then
    # surface run_id on the serve_meta (this run was freshly computed -> cached stays False).
    get_store().put(key, doc)
    _set_serve_meta(serve_meta, cached=False, run_id=key)
    return doc


def build_permalink_run_doc(run_id: str, serve_meta: dict | None = None) -> dict:
    """Replay a stored run by its content-addressed id (the permalink path). This BYPASSES
    build_free_text_run_doc entirely — no codegen.enabled() / budget / generate — so a shared
    link replays on a credential-less deploy (exactly the cold-fetch case permalinks exist for).
    The id is assumed format-validated by the caller; a store MISS raises HTTPException(404) in
    the SYNC route body (before any StreamingResponse) so a missing/expired id is a real 404,
    not a 200 with a broken stream. The replayed doc is schema-pure; run_id + cached=True ride
    the serve_meta out-dict."""
    doc = get_store().get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="unknown or expired run id")
    _set_serve_meta(serve_meta, cached=True, run_id=run_id)
    return doc


def _resolve_run_request(query: str | None, query_text: str | None, clip_id: str | None,
                         run: str | None = None):
    """Param validation shared by /api/run and /api/run_doc, factored into a plain helper so it
    is unit-testable without a TestClient. Returns ("canned", entry), ("free", (text, clip)), or
    ("permalink", run_id). Raises HTTPException(400) unless EXACTLY ONE of run/query/query_text is
    given (and the free-text cap / run-id format hold), and HTTPException(404) for an unknown
    query or clip. (A missing permalink id is a 404 raised later, in build_permalink_run_doc.)"""
    has_run = bool(run)
    has_query = bool(query)
    has_text = query_text is not None and query_text.strip() != ""
    # Exactly-one-of-three (rewritten from the prior two-way XOR so a bare ?run=<id> is accepted).
    if (has_run + has_query + has_text) != 1:
        raise HTTPException(status_code=400,
                            detail="provide exactly one of 'run', 'query', or 'query_text'")
    if has_run:
        # The run param is attacker-controlled and becomes a file-store lookup key — reject any
        # id that is not exactly a sha256 hex digest BEFORE touching the store (path-traversal /
        # info-leak guard: `run=../../etc/passwd` must 400 here, never reach the filesystem).
        if not run_cache.is_valid_key(run):
            raise HTTPException(status_code=400, detail="malformed run id")
        return "permalink", run
    if has_query:
        entry = canned.by_id(query)
        if not entry:
            raise HTTPException(status_code=404, detail=f"unknown query '{query}'")
        return "canned", entry
    # free text
    if len(query_text) > MAX_QUERY_TEXT_LEN:  # cap BEFORE any codegen call
        raise HTTPException(status_code=400, detail=f"query_text exceeds {MAX_QUERY_TEXT_LEN} chars")
    cid = clip_id or next(iter(canned.CLIPS))  # default to the sole v1 clip
    if canned.clip_by_id(cid) is None:
        raise HTTPException(status_code=404, detail=f"unknown clip '{cid}'")
    return "free", (query_text, cid)


def _doc_for_request(query: str | None, query_text: str | None, clip_id: str | None,
                     run: str | None = None, serve_meta: dict | None = None) -> dict:
    """Dispatch to the canned / free-text / permalink builder. Returns a SCHEMA-PURE run-doc; the
    optional `serve_meta` out-dict is populated with {cached, run_id?} for the route's SSE meta /
    run_doc envelope (canned runs leave it at the default cached=False, no run_id)."""
    # Calling the async route handler directly in tests (test_sse_stream's cancel/info-leak
    # tests) leaves any un-passed Query()-defaulted param as a FieldInfo sentinel, not None.
    # Coerce non-str values back to None so the exactly-one-of-three check stays correct.
    if not isinstance(run, str):
        run = None
    _set_serve_meta(serve_meta, cached=False)  # canned default; the builders refine it
    kind, payload = _resolve_run_request(query, query_text, clip_id, run)
    if kind == "canned":
        return build_run_doc(payload)
    if kind == "permalink":
        return build_permalink_run_doc(payload, serve_meta)
    text, cid = payload
    return build_free_text_run_doc(text, cid, serve_meta)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "queries": [q["id"] for q in canned.QUERIES]}


@app.get("/api/catalog")
def catalog():
    # Project clips to public fields only (drop internal `cache` path / `hint`); keep `label`
    # so the UI can show the clip context as a read-only label for the free-text input.
    pub = ("id", "label", "width", "height", "duration_ms")
    return {
        "clips": [{k: c[k] for k in pub if k in c} for c in canned.CLIPS.values()],
        "queries": [{"id": q["id"], "text": q["text"], "clip": q["clip"]} for q in canned.QUERIES],
    }


@app.get("/api/run_doc")
def run_doc(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
    run: str | None = Query(None),
):
    # A permalink miss must 404 HERE (sync body), before any response is built. _doc_for_request
    # -> build_permalink_run_doc raises HTTPException(404) which FastAPI surfaces as a 404.
    serve_meta: dict = {}
    doc = _doc_for_request(query, query_text, clip, run, serve_meta)
    # The doc stays schema-pure (run_doc.schema.json is additionalProperties:false); run_id /
    # cached ride alongside under `_meta`, never mixed into the doc keys the schema validates.
    return {**doc, "_meta": serve_meta}


@app.get("/api/run")
async def run(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
    run: str | None = Query(None),
    pace_ms: int = Query(1200, ge=0, le=10000),
):
    # Resolve + (for a permalink) 404-on-miss SYNCHRONOUSLY, before the StreamingResponse is
    # constructed — the SSE generator deliberately never errors mid-stream, so a 404 raised
    # inside gen() would be a 200 with a broken stream. _doc_for_request raises here on a miss.
    serve_meta: dict = {}
    doc = _doc_for_request(query, query_text, clip, run, serve_meta)

    async def gen():
        try:
            yield _sse("meta", {
                "query": doc["query"], "clip": doc["clip"],
                "program": doc["program"], "total_steps": len(doc["trace"]), "pace_ms": pace_ms,
                "program_source": doc.get("program_source", "pinned"),
                "cached": serve_meta.get("cached", False),
                **({"run_id": serve_meta["run_id"]} if "run_id" in serve_meta else {}),
            })
            for step in doc["trace"]:
                await asyncio.sleep(pace_ms / 1000)
                yield _sse("step", step)
            yield _sse("findings", doc["findings"])
            yield _sse("done", {"ok": True})
        except asyncio.CancelledError:  # client navigated away mid-stream
            raise
        except Exception as e:  # noqa: BLE001 — surface as an SSE error, don't 500 mid-stream
            # Never forward raw exception text (paths / Azure errors) to the client (info-leak
            # guard); log it server-side, send a fixed message.
            print(f"[run] stream error: {e}", file=sys.stderr)
            yield _sse("error", {"message": "stream failed"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
