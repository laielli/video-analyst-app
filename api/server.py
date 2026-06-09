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
from run_cache import FileRunStore, cache_key, is_valid_key  # noqa: E402

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


# Content-addressed run cache (the cost lever + permalink substrate). ONE store per process,
# constructed from env so tests can monkeypatch `_run_store` to a tmp_path-rooted FileRunStore
# (do NOT freeze it in a closure/default-arg — integration tests must redirect it off the real dir).
RUN_CACHE_DIR = os.environ.get("RUN_CACHE_DIR") or str(API_DIR / ".run_cache")
RUN_CACHE_MAX_ENTRIES = int(os.environ.get("RUN_CACHE_MAX_ENTRIES", "1000"))
_run_store: run_cache.RunStore = FileRunStore(RUN_CACHE_DIR, max_entries=RUN_CACHE_MAX_ENTRIES)


def get_store() -> run_cache.RunStore:
    """The cache accessor the request paths call. Reads the module global so a test can swap
    `server._run_store` to a tmp_path-rooted store without re-importing."""
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
    """ViperGPT loop for a TYPED question, with the safety net removed: cache lookup -> codegen ->
    validate -> execute over the clip's replay cache. There is NO pinned fallback, so every failure
    mode (no creds, codegen error, invalid program, execution error, no grounded answer) resolves
    to an explicit `_ungrounded` run-doc — never a hero-program substitution, never a 500. The clip
    is resolved by the caller (404 on unknown); here it is assumed valid.

    Cost lever: a content-addressed lookup runs BEFORE codegen, so an identical repeat question is
    served for free — no billed `generate()`, no budget burn, no `_codegen_calls` increment. Only
    a VALIDATED + GROUNDED run is cached (trust boundary respected): an ungrounded/failed run is
    NEVER stored, so a transient miss can't become permanent and the cache can't be poisoned with a
    cheap ungrounded question. `run_id` is attached on EXACTLY the grounded path (the one stored)."""
    global _codegen_calls
    clip = canned.clip_by_id(clip_id)  # caller raises 404 if None; defensive guard below
    if clip is None:
        return _ungrounded({"id": clip_id, "width": 0, "height": 0, "duration_ms": 0},
                           query_text, "execution-error")

    # Content-addressed cache lookup — BEFORE codegen.enabled()/budget/generate(). A hit is a pure
    # replay: no codegen call, no budget burn. The stored doc already passed validation+grounding
    # and is schema-validated on read (store.get fails-as-miss), so it streams safely. The returned
    # doc stays SCHEMA-PURE (no run_id/cached keys); the route attaches those envelope fields.
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
        return _ungrounded(clip, query_text, f.get("reason") or "no-grounded-answer",
                           program=doc["program"], trace=doc["trace"])

    doc["program_source"] = "live"
    # Cache ONLY the validated+grounded run. The doc stored (and returned) is schema-pure: run_id
    # and cached are NOT doc fields (run_doc.schema.json is additionalProperties:false) — the route
    # attaches them to the meta/response envelope. The permalink id IS this key (cache_key), so the
    # route re-derives it deterministically without reading it off the doc.
    get_store().put(key, doc)
    return doc


def build_permalink_run_doc(run_id: str) -> dict | None:
    """Replay a STORED run by id (the permalink path). BYPASSES build_free_text_run_doc entirely:
    a thin store.get, never codegen — so a cold shared link replays even on a credential-less
    deploy (the exact case permalinks exist for). The caller has already validated the id shape
    and 404s a miss; here a miss returns None so the route can raise before streaming. The returned
    doc is the schema-pure stored doc — the route attaches run_id/cached to the envelope."""
    return get_store().get(run_id)


def _resolve_run_request(query: str | None, query_text: str | None, clip_id: str | None,
                         run: str | None = None):
    """Param validation shared by /api/run and /api/run_doc, factored into a plain helper so it
    is unit-testable without a TestClient. Returns ("permalink", run_id) | ("canned", entry) |
    ("free", (text, clip)).

    EXACTLY-ONE-OF-THREE: `run` (permalink id) | `query` (canned id) | `query_text` (free text).
    A bare `?run=<id>` (no query/query_text) is now ACCEPTED — the prior two-way XOR would 400 it.
    Raises HTTPException(400) for none/multiple given (or an over-cap query_text, or a malformed
    run id) and HTTPException(404) for an unknown query or clip. A permalink MISS is NOT resolved
    here (the route 404s it before streaming) — only the id SHAPE is validated."""
    # `run` may arrive as FastAPI's Query(None) sentinel when server.run()/server.run_doc() are
    # called DIRECTLY (bypassing param resolution — the existing SSE-closure tests do this and do
    # not pass `run`). Treat anything that is not a non-empty str as absent, so a bare direct call
    # behaves like the pre-permalink two-way path.
    has_run = isinstance(run, str) and run != ""
    has_query = bool(query)
    has_text = query_text is not None and query_text.strip() != ""
    if sum((has_run, has_query, has_text)) != 1:  # exactly one of the three
        raise HTTPException(status_code=400,
                            detail="provide exactly one of 'run', 'query' or 'query_text'")
    if has_run:
        # SECURITY: the run param is attacker-controlled and becomes a file-store lookup key.
        # Reject anything not a 64-char sha256 hex with 400 BEFORE touching the store
        # (run=../../etc/passwd must be a 400, not a file read).
        if not is_valid_key(run):
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
    """Resolve a request to a run-doc, attaching the `run_id`/`cached` ENVELOPE fields here at the
    route layer (the builders return schema-pure docs; these two are NOT doc fields). The permalink
    arm BYPASSES build_free_text_run_doc (no codegen): a missing/expired id raises 404 HERE, in the
    sync caller, BEFORE any StreamingResponse is built — the SSE generator never errors mid-stream,
    so raising inside it would yield a broken 200, not a 404."""
    kind, payload = _resolve_run_request(query, query_text, clip_id, run)
    if kind == "permalink":
        doc = build_permalink_run_doc(payload)
        if doc is None:
            raise HTTPException(status_code=404, detail="unknown run id")
        doc["run_id"] = payload      # the id IS the cache key
        doc["cached"] = True         # a permalink replay is always served from the store
        return doc
    if kind == "canned":
        return build_run_doc(payload)  # canned runs mint no run_id (out of scope)
    text, cid = payload
    # `cached` is derived at serve time: a hit exists iff the store already holds this key BEFORE
    # the builder runs. `run_id` is attached only on the grounded path (the only one we store) —
    # the non-grounded exits carry neither, so the UI never offers a share link to an unstored run.
    key = cache_key(text, cid)
    was_hit = get_store().get(key) is not None
    doc = build_free_text_run_doc(text, cid)
    if doc.get("findings", {}).get("grounded") is True:
        doc["run_id"] = key
        doc["cached"] = was_hit
    return doc


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
    # The returned doc carries `run_id`/`cached` (when applicable) as the response-envelope fields;
    # the stored doc stays schema-pure (run_doc.schema.json is additionalProperties:false).
    return _doc_for_request(query, query_text, clip, run)


@app.get("/api/run")
async def run(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
    run: str | None = Query(None),
    pace_ms: int = Query(1200, ge=0, le=10000),
):
    # NOTE: the permalink/free-text resolution (incl. the 404 for a missing run id) happens HERE,
    # in the sync route body, BEFORE the StreamingResponse is built — never inside gen() (the SSE
    # generator deliberately never errors mid-stream, so a raise there would be a broken 200).
    doc = _doc_for_request(query, query_text, clip, run)

    async def gen():
        try:
            meta = {
                "query": doc["query"], "clip": doc["clip"],
                "program": doc["program"], "total_steps": len(doc["trace"]), "pace_ms": pace_ms,
                "program_source": doc.get("program_source", "pinned"),
                # Permalink/cache plumbing. `cached` is derived at serve time (default False); the
                # web "Copy run link" affordance + load-from-id read these from meta.
                "cached": bool(doc.get("cached", False)),
            }
            if doc.get("run_id"):  # present only on grounded free-text runs + permalink replays
                meta["run_id"] = doc["run_id"]
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
