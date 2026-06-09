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
from run_cache import FileRunStore, RunStore, cache_key, KEY_RE  # noqa: E402

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


# Free-text run cache (content-addressed on the normalized (query_text, clip) pair). An identical
# repeat question is served from here for FREE — no codegen call, no budget burn — and the same
# key is a stable permalink `run_id`. Constructed ONCE at startup from env so a future BlobRunStore
# can slot into the RunStore seam. Exposed as a module-global so tests can redirect it to tmp_path
# (do NOT freeze it in a closure/default-arg). RUN_CACHE_DIR defaults to a gitignored dir under api/.
RUN_CACHE_DIR = os.environ.get("RUN_CACHE_DIR") or str(API_DIR / ".run_cache")
RUN_CACHE_MAX_ENTRIES = int(os.environ.get("RUN_CACHE_MAX_ENTRIES", "1000"))
_run_store: RunStore = FileRunStore(Path(RUN_CACHE_DIR), max_entries=RUN_CACHE_MAX_ENTRIES)


def get_store() -> RunStore:
    """Accessor for the run-cache store (tests monkeypatch `server._run_store`)."""
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

    # Cache lookup BEFORE codegen.enabled() / budget / generate(): an identical repeat question is
    # served from the store for FREE (no codegen call, no `_codegen_calls` increment). A hit is
    # always a previously-validated+grounded run-doc (only that branch ever calls store.put), and
    # store.get re-validates it against run_doc.schema.json, so a hit is safe to stream verbatim.
    # The doc stays schema-pure; the route derives run_id/cached at serve time (run_id == the key).
    key = cache_key(query_text, clip_id)
    hit = get_store().get(key)
    if hit is not None:
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
        # poison the cache with a cheap ungrounded question.
        return _ungrounded(clip, query_text, f.get("reason") or "no-grounded-answer",
                           program=doc["program"], trace=doc["trace"])

    doc["program_source"] = "live"
    # ONLY cache a validated + grounded run (trust boundary). Store the schema-pure doc; the
    # run_id/cached flags are derived by the route at serve time, never inside the stored doc.
    get_store().put(key, doc)
    return doc


def build_permalink_run_doc(run_id: str) -> dict:
    """Replay a stored run by id (the permalink path). BYPASSES build_free_text_run_doc entirely:
    a permalink is store.get(run_id) -> serve the stored doc, so it works on a credential-less
    deploy (no codegen.enabled() check). The run_id is validated by the caller (^[0-9a-f]{64}$);
    a miss raises 404 in the SYNC route body before any StreamingResponse is built. The doc stays
    schema-pure; the route derives run_id/cached for the meta payload (run_id == the lookup key)."""
    doc = get_store().get(run_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="unknown or expired run id")
    return doc


def _free_text_cache_hit(query: str | None, query_text: str | None, clip: str | None) -> bool:
    """True iff a free-text request would be served from the store (so `cached: true` is honest).
    MUST be evaluated BEFORE the builder runs: the builder does store.put on a fresh grounded run,
    so a post-build store.get can't distinguish a hit from a just-stored miss. Never hashes
    un-capped input — an over-length question short-circuits here (it 400s in the builder anyway)."""
    if query or not (query_text and query_text.strip()):
        return False
    if len(query_text) > MAX_QUERY_TEXT_LEN:  # never key un-capped input (mirrors _resolve_run_request)
        return False
    cid = clip or next(iter(canned.CLIPS))
    if canned.clip_by_id(cid) is None:
        return False
    return get_store().get(cache_key(query_text, cid)) is not None


def _meta_extras(query: str | None, query_text: str | None, clip: str | None,
                 run: str | None, doc: dict, cached: bool) -> tuple[str | None, bool]:
    """Derive (run_id, cached) for the meta payload / run_doc envelope at SERVE time — never from
    inside the schema-pure doc. `cached` is passed in (computed pre-build). A run_id exists for a
    permalink replay and for a grounded free-text run (run_id == the content-address key, which the
    builder also stored). Canned runs and ungrounded free-text runs carry no run_id."""
    if run:  # permalink replay — always served from the store
        return run, True
    if query or not (query_text and query_text.strip()):
        return None, False  # canned / invalid free-text -> no run_id
    if doc.get("program_source") != "live":
        return None, cached  # ungrounded free-text -> no run_id (cached is False here)
    cid = clip or next(iter(canned.CLIPS))
    return cache_key(query_text, cid), cached


def _resolve_run_request(query: str | None, query_text: str | None, clip_id: str | None,
                         run: str | None = None):
    """Param validation shared by /api/run and /api/run_doc, factored into a plain helper so it
    is unit-testable without a TestClient. Returns ("canned", entry), ("free", (text, clip)), or
    ("permalink", run_id). Raises HTTPException(400) unless EXACTLY ONE of run/query/query_text is
    given (and the free-text cap / run-id format hold), and HTTPException(404) for an unknown query
    or clip. A bare ?run=<id> is accepted — the rewritten three-way check (was a two-way XOR)."""
    has_run = bool(run)
    has_query = bool(query)
    has_text = query_text is not None and query_text.strip() != ""
    if sum((has_run, has_query, has_text)) != 1:  # exactly one of three
        raise HTTPException(status_code=400,
                            detail="provide exactly one of 'run', 'query' or 'query_text'")
    if has_run:
        # The `run` param is attacker-controlled and becomes a file-store lookup key. Reject any id
        # that is not a sha256 hex digest BEFORE touching the store (defense in depth in the store
        # too) — so run=../../etc/passwd is a 400, never a file read.
        if not KEY_RE.match(run):
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
        return build_permalink_run_doc(payload)  # raises 404 here (sync, before StreamingResponse)
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
    # Compute the cache-hit flag BEFORE building (the builder stores fresh grounded runs). A
    # permalink (?run=<id>) or a free-text cache hit was served from the store; surface `cached` +
    # `run_id` in the response ENVELOPE only (the run-doc itself stays schema-pure / unwrapped).
    pre_cached = _free_text_cache_hit(query, query_text, clip)
    doc = _doc_for_request(query, query_text, clip, run)
    run_id, cached = _meta_extras(query, query_text, clip, run, doc, pre_cached)
    if run_id is not None:
        return {**doc, "run_id": run_id, "cached": cached}
    return doc


@app.get("/api/run")
async def run(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
    run: str | None = Query(None),
    pace_ms: int = Query(1200, ge=0, le=10000),
):
    # 404 a missing/expired permalink id in the SYNC body, BEFORE StreamingResponse — the SSE
    # generator never errors mid-stream (it would yield a 200 with a broken stream, not a 404).
    pre_cached = _free_text_cache_hit(query, query_text, clip)
    doc = _doc_for_request(query, query_text, clip, run)
    run_id, cached = _meta_extras(query, query_text, clip, run, doc, pre_cached)

    async def gen():
        try:
            yield _sse("meta", {
                "query": doc["query"], "clip": doc["clip"],
                "program": doc["program"], "total_steps": len(doc["trace"]), "pace_ms": pace_ms,
                "program_source": doc.get("program_source", "pinned"),
                "run_id": run_id, "cached": cached,
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
