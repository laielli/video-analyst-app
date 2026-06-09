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


# Content-addressed run-doc cache (api/run_cache.py): an identical repeat free-text question is
# served for free (no codegen, no budget burn) and becomes a shareable permalink (run_id == the
# cache key). ONE store is constructed here at startup from env so a future BlobRunStore can slot
# in. It is a module-global so tests can monkeypatch it to a tmp_path-rooted store (do NOT freeze
# it in a closure / default-arg — integration tests must not touch the real on-disk cache).
RUN_CACHE_DIR = os.environ.get("RUN_CACHE_DIR") or str(API_DIR / ".run_cache")
RUN_CACHE_MAX_ENTRIES = int(os.environ.get("RUN_CACHE_MAX_ENTRIES", "1000"))
_run_store: run_cache.RunStore = run_cache.FileRunStore(
    RUN_CACHE_DIR, max_entries=RUN_CACHE_MAX_ENTRIES
)


def get_store() -> run_cache.RunStore:
    """Accessor so callers read the CURRENT module-global store (tests monkeypatch _run_store)."""
    return _run_store


app = FastAPI(title="Glass-Box Video Analyst API", version="0.1.0")

# Dev CORS: the Next.js dev server runs on a different origin. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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


def build_free_text_run_doc(query_text: str, clip_id: str) -> dict:
    """ViperGPT loop for a TYPED question, with the safety net removed: codegen -> validate ->
    execute over the clip's replay cache. There is NO pinned fallback, so every failure mode
    (no creds, codegen error, invalid program, execution error, no grounded answer) resolves to
    an explicit `_ungrounded` run-doc — never a hero-program substitution, never a 500. The clip
    is resolved by the caller (404 on unknown); here it is assumed valid."""
    global _codegen_calls
    clip = canned.clip_by_id(clip_id)  # caller raises 404 if None; defensive guard below
    if clip is None:
        return _ungrounded({"id": clip_id, "width": 0, "height": 0, "duration_ms": 0},
                           query_text, "execution-error")

    # Cache lookup BEFORE codegen (and before even loading the replay cache): an identical repeat
    # question is served for free, with NO codegen call and NO budget burn (that is the whole cost
    # win). A hit is necessarily a validated+grounded run-doc (only the grounded branch below ever
    # calls store.put) and is re-validated against run_doc.schema.json on read, so a malformed/
    # stale entry is a miss, never a poison. The run_id / cached flags are carried OUT-OF-BAND
    # under a private "_serve" key (stripped by the route into meta) so the doc body stays
    # schema-pure (run_doc.schema.json is additionalProperties:false at the root) — _pop_serve_meta.
    key = run_cache.cache_key(query_text, clip_id)
    hit = get_store().get(key)
    if hit is not None:
        hit["_serve"] = {"run_id": key, "cached": True}
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
        # NOT cached — caching a failure would make a transient miss permanent and let an attacker
        # poison the cache with a cheap ungrounded question. No run_id either (no stored run).
        return _ungrounded(clip, query_text, f.get("reason") or "no-grounded-answer",
                           program=doc["program"], trace=doc["trace"])

    doc["program_source"] = "live"
    # TRUST BOUNDARY: store.put sits ONLY here — on the post-validation, grounded branch — so we
    # never persist an unvalidated/ungrounded run. The stored entry is the schema-pure doc; the
    # serve-meta (run_id, cached=False on this fresh run) is attached out-of-band afterwards.
    get_store().put(key, doc)
    doc["_serve"] = {"run_id": key, "cached": False}
    return doc


def _pop_serve_meta(doc: dict) -> dict:
    """Strip the out-of-band `_serve` key off a run-doc (mutating it back to schema-pure) and
    return the serve-meta the route surfaces: {run_id?, cached}. The run-doc schema is
    additionalProperties:false at the root, so `run_id`/`cached` can NEVER live inside the doc;
    they ride here, are derived at serve time, and are injected into the meta event /
    /api/run_doc response envelope by the route. A doc with no `_serve` (canned, ungrounded)
    yields {cached: False} and no run_id — exactly the four non-grounded exits the plan forbids
    from carrying a run_id."""
    serve = doc.pop("_serve", None)
    if serve is None:
        return {"cached": False}
    return serve


def replay_permalink_doc(run_id: str) -> dict:
    """The permalink branch: fetch a stored run-doc by id and serve it, BYPASSING
    build_free_text_run_doc entirely (so a credential-less deploy doesn't hit codegen.enabled()
    and degrade to 'codegen-disabled' on exactly the cold shared-link path this feature exists
    for). Raises HTTPException(400) on a malformed id (BEFORE touching the store) and
    HTTPException(404) on a miss — both in the SYNC body so the route can 404 before constructing
    a StreamingResponse (the SSE generator never errors mid-stream)."""
    if not run_cache.is_valid_key(run_id):
        # Attacker-controlled param becomes a store lookup key — reject anything that isn't a
        # clean 64-hex-char id BEFORE any file access (run=../../etc/passwd must be a 400).
        raise HTTPException(status_code=400, detail="malformed run id")
    doc = get_store().get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="unknown run id")
    doc["_serve"] = {"run_id": run_id, "cached": True}  # a replay is always served from the store
    return doc


def _resolve_run_request(query: str | None, query_text: str | None, clip_id: str | None,
                         run: str | None = None):
    """Param validation shared by /api/run and /api/run_doc, factored into a plain helper so it
    is unit-testable without a TestClient. Returns ("canned", entry), ("free", (text, clip)), or
    ("permalink", run_id). Raises HTTPException(400) unless EXACTLY ONE of run/query/query_text is
    given (and the free-text cap holds), and HTTPException(404) for an unknown query or clip.

    This is an exactly-one-of-THREE check (run | query | query_text), NOT the old two-way XOR —
    a bare ?run=<id> (no query, no query_text) must be ACCEPTED, not 400'd."""
    has_run = run is not None and run != ""
    has_query = bool(query)
    has_text = query_text is not None and query_text.strip() != ""
    if sum((has_run, has_query, has_text)) != 1:  # neither, or more than one
        raise HTTPException(status_code=400,
                            detail="provide exactly one of 'run', 'query', or 'query_text'")
    if has_run:
        # Format-validate the run id here too (defense in depth); the deeper 404/store lookup is
        # in replay_permalink_doc. A malformed id is a 400, never a file read.
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
                     run: str | None = None) -> dict:
    kind, payload = _resolve_run_request(query, query_text, clip_id, run)
    if kind == "canned":
        return build_run_doc(payload)
    if kind == "permalink":
        return replay_permalink_doc(payload)
    text, cid = payload
    return build_free_text_run_doc(text, cid)


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
    doc = _doc_for_request(query, query_text, clip, run)
    serve = _pop_serve_meta(doc)  # strip run_id/cached off the doc -> schema-pure body
    # Response envelope: the run-doc body stays schema-pure (all its keys at the top level so the
    # UI / existing callers read it unchanged), with run_id/cached surfaced alongside. run_id is
    # present only for grounded free-text runs and permalink replays; cached marks a store serve.
    return {**doc, "cached": serve.get("cached", False), **({"run_id": serve["run_id"]} if serve.get("run_id") else {})}


@app.get("/api/run")
async def run(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
    run: str | None = Query(None),
    pace_ms: int = Query(1200, ge=0, le=10000),
):
    doc = _doc_for_request(query, query_text, clip, run)
    serve = _pop_serve_meta(doc)  # 404 already raised in the sync body if it was a permalink miss

    async def gen():
        try:
            meta = {
                "query": doc["query"], "clip": doc["clip"],
                "program": doc["program"], "total_steps": len(doc["trace"]), "pace_ms": pace_ms,
                "program_source": doc.get("program_source", "pinned"),
                "cached": serve.get("cached", False),
            }
            if serve.get("run_id"):  # only grounded free-text runs + permalink replays carry one
                meta["run_id"] = serve["run_id"]
            yield _sse("meta", meta)
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
