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

# Load api/.env so `uvicorn server:app` picks up AZURE_OPENAI_* (codegen) without extra flags.
# No-op if the file is absent — codegen then stays disabled and runs fall back to pinned.
load_dotenv(API_DIR / ".env")

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

    errors = validation_errors({"program": program})
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


# --- Free-text path (the full ViperGPT loop, no pinned fallback) ----------------------------

# Cost/abuse guard (§Risks): cap the typed query length and bound paid codegen calls per process
# so an unauthenticated GET can't run up unbounded Azure spend or OOM the URL parser.
MAX_QUERY_TEXT_LEN = 500
MAX_CODEGEN_CALLS = int(os.environ.get("CODEGEN_CALL_BUDGET", "200"))
_codegen_calls = 0

# Fixed enums only — never forward raw exception text into the SSE payload (§Risks: info leakage).
UNGROUNDED_REASONS = {
    "codegen-disabled", "codegen-error", "invalid-program", "execution-error",
    "no-grounded-answer", "budget-exceeded",
}


def _ungrounded(clip: dict, query_text: str, reason: str, program: list[dict] | None = None,
                trace: list[dict] | None = None) -> dict:
    """A complete, schema-valid run-doc for the honest 'couldn't ground this' state (Q1 -> A).
    Carries ALL keys the SSE stream loop reads (query/clip/program/trace/findings) so the loop
    never KeyErrors. `program`/`trace` are empty when codegen failed before producing a program;
    when a validated program ran but didn't ground, its trace IS included so the UI shows where
    it ran dry. `reason` is a fixed enum (never an exception string)."""
    return {
        "schema_version": "1",
        "query": query_text,
        "clip": {k: clip[k] for k in ("id", "width", "height", "duration_ms") if k in clip},
        "program": program or [],
        "trace": trace or [],
        "findings": {"answer": None, "verdict": None, "partial": False,
                     "grounded": False, "reason": reason},
        "program_source": "ungrounded",
    }


def build_free_text_run_doc(query_text: str, clip_id: str) -> dict:
    """Compile a typed question for `clip_id` and replay it — the complete ViperGPT loop with the
    safety net removed. There is NO pinned program for a novel question, so every failure mode
    (codegen disabled/error, invalid program, execution error, no grounded answer) resolves to an
    explicit ungrounded run-doc, NEVER a hero-program substitution. The caller (route) has already
    resolved `clip_id` to a known clip and bounded `query_text` length."""
    global _codegen_calls
    clip = canned.clip_by_id(clip_id)  # route 404s before we get here if unknown
    cache = Cache.load(clip["cache"])

    if not codegen.enabled():
        return _ungrounded(clip, query_text, "codegen-disabled")
    if _codegen_calls >= MAX_CODEGEN_CALLS:
        return _ungrounded(clip, query_text, "budget-exceeded")

    try:
        _codegen_calls += 1
        program = codegen.AzureCodegen.from_env().generate(query_text, clip=clip)
    except Exception:  # noqa: BLE001 — any generation failure -> honest ungrounded (no fallback)
        return _ungrounded(clip, query_text, "codegen-error")

    # Trust boundary: no generated program executes unvalidated. Numeric bounds are enforced too
    # (a validated program can still OOM via sample_frames' range expansion) using the clip duration.
    if validation_errors({"program": program}, clip=clip):
        return _ungrounded(clip, query_text, "invalid-program", program=program)

    try:
        doc = Interpreter(cache).run(program, query=query_text)
    except Exception:  # noqa: BLE001 — e.g. the sampled window misses the cached frames
        return _ungrounded(clip, query_text, "execution-error", program=program)

    f = doc.get("findings", {})
    if f.get("grounded") is False or f.get("partial") or f.get("answer") in (None, "", "Unknown"):
        # The (validated) program ran but didn't ground — stream its trace so the UI shows where.
        return _ungrounded(clip, query_text, "no-grounded-answer",
                           program=program, trace=doc.get("trace", []))

    doc["program_source"] = "live"
    return doc


def resolve_run_request(query: str | None, query_text: str | None):
    """Param-check for /api/run and /api/run_doc: require EXACTLY ONE of `query` (canned) or
    `query_text` (free text). Returns ("canned"|"free", error_status, error_detail). Extracted as
    a plain helper so it unit-tests without a FastAPI TestClient (system python lacks fastapi)."""
    has_canned = query is not None and query != ""
    has_free = query_text is not None and query_text != ""
    if has_canned and has_free:
        return None, 400, "provide exactly one of 'query' or 'query_text', not both"
    if not has_canned and not has_free:
        return None, 400, "provide exactly one of 'query' or 'query_text'"
    if has_free:
        if len(query_text) > MAX_QUERY_TEXT_LEN:
            return None, 400, f"query_text exceeds {MAX_QUERY_TEXT_LEN} characters"
        return "free", None, None
    return "canned", None, None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "queries": [q["id"] for q in canned.QUERIES]}


@app.get("/api/catalog")
def catalog():
    return {
        "clips": list(canned.CLIPS.values()),
        "queries": [{"id": q["id"], "text": q["text"], "clip": q["clip"]} for q in canned.QUERIES],
    }


def _resolve_doc(query: str | None, query_text: str | None, clip: str) -> dict:
    """Shared route resolution: param-check, then build either the canned or the free-text
    run-doc. Raises HTTPException(400) on a bad param combo and HTTPException(404) on an unknown
    canned query or unknown clip. Used by both /api/run and /api/run_doc."""
    kind, status, detail = resolve_run_request(query, query_text)
    if status:
        raise HTTPException(status_code=status, detail=detail)
    if kind == "free":
        if canned.clip_by_id(clip) is None:
            raise HTTPException(status_code=404, detail=f"unknown clip '{clip}'")
        return build_free_text_run_doc(query_text, clip)
    entry = canned.by_id(query)
    if not entry:
        raise HTTPException(status_code=404, detail=f"unknown query '{query}'")
    return build_run_doc(entry)


@app.get("/api/run_doc")
def run_doc(query: str = Query(None), query_text: str = Query(None), clip: str = Query("single-goal")):
    return _resolve_doc(query, query_text, clip)


@app.get("/api/run")
async def run(
    query: str = Query(None),
    query_text: str = Query(None),
    clip: str = Query("single-goal"),
    pace_ms: int = Query(1200, ge=0, le=10000),
):
    doc = _resolve_doc(query, query_text, clip)

    async def gen():
        try:
            yield _sse("meta", {
                "query": doc["query"], "clip": doc["clip"],
                "program": doc["program"], "total_steps": len(doc["trace"]), "pace_ms": pace_ms,
                "program_source": doc.get("program_source", "pinned"),
            })
            for step in doc["trace"]:
                await asyncio.sleep(pace_ms / 1000)
                yield _sse("step", step)
            yield _sse("findings", doc["findings"])
            yield _sse("done", {"ok": True})
        except asyncio.CancelledError:  # client navigated away mid-stream
            raise
        except Exception as e:  # noqa: BLE001 — surface as an SSE error, don't 500 mid-stream
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
