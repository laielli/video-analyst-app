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


# ----------------------------------------------------------------------------------------------
# Free-text path (D-DR4): type a question -> live codegen -> validate -> replay -> grounded verdict
# OR an honest "couldn't ground this" state. There is NO pinned fallback here on purpose: a novel
# question has no hero program to substitute, so every failure mode resolves to an explicit
# ungrounded run-doc (program_source='ungrounded', findings.grounded=false), never a substitution.
# ----------------------------------------------------------------------------------------------

# Trust-boundary / abuse caps. Each distinct query_text triggers a billed Azure codegen call on an
# unauthenticated GET, so cap the input length and bound total codegen calls per process.
MAX_QUERY_TEXT_LEN = 500
CODEGEN_CALL_BUDGET = int(os.environ.get("CODEGEN_CALL_BUDGET", "200"))
_codegen_calls = 0

# Fixed reason enums for the ungrounded state — never raw exception text (info-leak guard).
UNGROUNDED_REASONS = {
    "codegen-disabled", "codegen-budget-exhausted", "codegen-error",
    "invalid-program", "execution-error", "no-grounded-answer",
}


class RequestError(Exception):
    """Carries an HTTP status for the route to translate (so the param-check helper is a plain,
    TestClient-free unit). status 400 = bad params, 404 = unknown clip."""
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def resolve_run_request(query: str | None, query_text: str | None, clip: str | None) -> dict:
    """Validate the run params and resolve the target — a plain helper so it is unit-testable
    without a FastAPI TestClient. Returns {"kind": "canned", "entry": ...} or
    {"kind": "free", "query_text": ..., "clip": ...}. Raises RequestError(400/404) otherwise.

    Exactly one of `query` / `query_text` must be present. Unknown query -> 404; unknown clip ->
    404; query_text over the length cap -> 400 (before any codegen call)."""
    has_query = query is not None and query != ""
    has_text = query_text is not None and query_text != ""
    if has_query and has_text:
        raise RequestError(400, "provide exactly one of 'query' or 'query_text', not both")
    if not has_query and not has_text:
        raise RequestError(400, "provide one of 'query' (canned) or 'query_text' (free text)")

    if has_query:
        entry = canned.by_id(query)
        if not entry:
            raise RequestError(404, f"unknown query '{query}'")
        return {"kind": "canned", "entry": entry}

    if len(query_text) > MAX_QUERY_TEXT_LEN:
        raise RequestError(400, f"query_text exceeds {MAX_QUERY_TEXT_LEN} chars")
    clip_id = clip or "single-goal"
    clip_obj = canned.clip_by_id(clip_id)
    if not clip_obj:
        raise RequestError(404, f"unknown clip '{clip_id}'")
    return {"kind": "free", "query_text": query_text, "clip": clip_obj}


def _ungrounded(clip: dict, query_text: str, reason: str,
                program: list[dict] | None = None, trace: list[dict] | None = None) -> dict:
    """A complete ungrounded run-doc (Q1 → A). Carries ALL keys the SSE stream loop reads
    (query/clip/program/trace/findings) so it never KeyErrors, with program_source='ungrounded'
    and findings.grounded=false. `reason` MUST be a fixed enum (never raw exception text)."""
    assert reason in UNGROUNDED_REASONS, f"ungrounded reason '{reason}' not in the fixed enum"
    clip_meta = {k: clip[k] for k in ("id", "width", "height", "duration_ms") if k in clip}
    return {
        "schema_version": "1",
        "query": query_text,
        "clip": clip_meta,
        "program": program or [],
        "trace": trace or [],
        "findings": {"answer": None, "verdict": None, "partial": False,
                     "grounded": False, "reason": reason},
        "program_source": "ungrounded",
    }


def build_free_text_run_doc(query_text: str, clip: dict) -> dict:
    """Compile a typed question against `clip`, validate it (the trust boundary), replay it over the
    clip's cache, and return either a grounded run-doc (program_source='live') or an honest
    ungrounded run-doc (program_source='ungrounded'). NEVER substitutes the hero program."""
    global _codegen_calls
    cache = Cache.load(clip["cache"])

    if not codegen.enabled():
        return _ungrounded(clip, query_text, "codegen-disabled")
    if _codegen_calls >= CODEGEN_CALL_BUDGET:
        return _ungrounded(clip, query_text, "codegen-budget-exhausted")

    _codegen_calls += 1
    try:
        program = codegen.AzureCodegen.from_env().generate(query_text, clip=clip)
    except Exception as e:  # noqa: BLE001 — never forward raw exception text to the client
        print(f"[codegen] free-text generation failed ({e})", file=sys.stderr)
        return _ungrounded(clip, query_text, "codegen-error")

    # Trust boundary: validate (with numeric bounds against this clip) before any execution.
    if validation_errors({"program": program}, clip=clip):
        return _ungrounded(clip, query_text, "invalid-program", program=program)

    try:
        doc = Interpreter(cache).run(program, query=query_text)
    except Exception as e:  # noqa: BLE001
        print(f"[codegen] free-text execution failed ({e})", file=sys.stderr)
        return _ungrounded(clip, query_text, "execution-error", program=program)

    f = doc.get("findings", {})
    if f.get("grounded") is False or f.get("partial") or f.get("answer") in (None, "", "Unknown"):
        # The (validated) program ran but didn't ground — stream its trace so the UI can show
        # WHERE it ran dry (D-DR5), but report honestly.
        return _ungrounded(clip, query_text, "no-grounded-answer",
                           program=program, trace=doc.get("trace", []))

    doc["program_source"] = "live"
    return doc


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version, "queries": [q["id"] for q in canned.QUERIES]}


@app.get("/api/catalog")
def catalog():
    return {
        "clips": [canned.public_clip(c) for c in canned.CLIPS.values()],
        "queries": [{"id": q["id"], "text": q["text"], "clip": q["clip"]} for q in canned.QUERIES],
    }


def _resolve_doc(query: str | None, query_text: str | None, clip: str | None) -> dict:
    """Shared route logic: param-check then build the canned or free-text run-doc. Raises
    HTTPException for bad params / unknown query / unknown clip."""
    try:
        req = resolve_run_request(query, query_text, clip)
    except RequestError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)
    if req["kind"] == "canned":
        return build_run_doc(req["entry"])
    return build_free_text_run_doc(req["query_text"], req["clip"])


@app.get("/api/run_doc")
def run_doc(query: str | None = Query(None), query_text: str | None = Query(None),
            clip: str | None = Query(None)):
    return _resolve_doc(query, query_text, clip)


@app.get("/api/run")
async def run(query: str | None = Query(None), query_text: str | None = Query(None),
              clip: str | None = Query(None), pace_ms: int = Query(1200, ge=0, le=10000)):
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
            # Info-leak guard: log the detail server-side, send a fixed generic message (never
            # raw exception text — paths / Azure errors) to the client.
            print(f"[run] stream error: {e}", file=sys.stderr)
            yield _sse("error", {"message": "stream failed"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
