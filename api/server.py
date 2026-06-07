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
    clip = canned.clip_by_id(entry.get("clip", ""))
    try:
        program = codegen.AzureCodegen.from_env().generate(entry["text"], clip=clip)
    except Exception as e:  # noqa: BLE001 — any generation failure -> pinned fallback
        print(f"[codegen] '{qid}': generation failed ({e}); using pinned program", file=sys.stderr)
        return None

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
    if (findings.get("grounded") is False or findings.get("partial")
            or findings.get("answer") in (None, "", "Unknown")):
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


# --- Free-text path (the full ViperGPT thesis) -------------------------------------------
# No pinned fallback on purpose: a novel question has no hero program, so every failure mode
# resolves to an explicit, honest "couldn't ground this" run-doc (Resolved Q1 -> A) — never a
# 404, a 500, or a silent hero substitution. Validation gates every generated program before
# Interpreter.run (the trust boundary), exactly as the canned path does.

QUERY_TEXT_MAX = 500  # cost/abuse cap: reject overlong typed questions before billing codegen.

# Fixed enum of ungrounded reasons. NEVER forward raw exception text into the SSE payload
# (info leakage — plan §Risks); the reason is always one of these strings.
_UNGROUNDED_REASONS = {
    "codegen-disabled", "codegen-error", "invalid-program",
    "execution-error", "no-grounded-answer",
}

# Per-process codegen-call budget: an unauthenticated GET can't run up unbounded paid Azure
# calls. Process-local + best-effort (not a distributed limiter); resets on restart.
_CODEGEN_BUDGET = int(os.environ.get("CODEGEN_CALL_BUDGET", "200"))
_codegen_calls = 0


def _clip_for_doc(clip: dict) -> dict:
    """The run-doc `clip` sub-object (run_doc.schema.json requires id/width/height/duration_ms)."""
    return {k: clip[k] for k in ("id", "width", "height", "duration_ms") if k in clip}


def _ungrounded(clip: dict, query_text: str, reason: str, program: list[dict] | None = None,
                trace: list[dict] | None = None) -> dict:
    """A complete, schema-valid ungrounded run-doc. Carries ALL keys the SSE stream reads
    (query/clip/program/trace/findings) so the stream loop never KeyErrors — empty program/trace
    for the codegen-disabled/error cases; the (validated) program + its trace when execution ran
    but didn't ground (so the UI shows WHERE it ran dry — extends D-DR5)."""
    assert reason in _UNGROUNDED_REASONS, f"unknown ungrounded reason '{reason}'"
    return {
        "schema_version": "1",
        "query": query_text,
        "clip": _clip_for_doc(clip),
        "program": program or [],
        "trace": trace or [],
        "findings": {"answer": None, "verdict": None, "partial": False,
                     "grounded": False, "reason": reason},
        "program_source": "ungrounded",
    }


def build_free_text_run_doc(query_text: str, clip_id: str) -> dict:
    """Compile a typed question into a program (clip-aware codegen), validate it (trust boundary),
    replay it over the clip's cache, and return either a grounded `live` run-doc or an honest
    `ungrounded` one. No pinned fallback: the hero program is NEVER substituted for free text.

    Caller is responsible for the 404 (unknown clip) and 400 (missing/overlong query_text)."""
    global _codegen_calls
    clip = canned.clip_by_id(clip_id)
    if clip is None:  # defensive; the route validates this first
        raise HTTPException(status_code=404, detail=f"unknown clip '{clip_id}'")
    cache = Cache.load(clip["cache"])

    if not codegen.enabled():
        return _ungrounded(clip, query_text, "codegen-disabled")
    if _codegen_calls >= _CODEGEN_BUDGET:
        return _ungrounded(clip, query_text, "codegen-error")  # budget exhausted -> honest miss

    try:
        _codegen_calls += 1
        program = codegen.AzureCodegen.from_env().generate(query_text, clip=clip)
    except Exception as e:  # noqa: BLE001 — fixed reason; never leak the exception text
        print(f"[free-text] codegen failed: {e}", file=sys.stderr)
        return _ungrounded(clip, query_text, "codegen-error")

    # Trust boundary: no generated program executes unvalidated (structure + semantics + bounds).
    if validation_errors({"program": program}, clip=clip):
        return _ungrounded(clip, query_text, "invalid-program")

    try:
        doc = Interpreter(cache).run(program, query=query_text)
    except Exception as e:  # noqa: BLE001 — e.g. a validated window misses the cached frames
        print(f"[free-text] execution failed: {e}", file=sys.stderr)
        return _ungrounded(clip, query_text, "execution-error",
                           program=strip_comments({"program": program})["program"])

    f = doc.get("findings", {})
    if f.get("grounded") is False or f.get("partial") or f.get("answer") in (None, "", "Unknown"):
        # Stream the (validated) program's trace so the user sees where it ran dry (D-DR5).
        return _ungrounded(clip, query_text, "no-grounded-answer",
                           program=doc.get("program", []), trace=doc.get("trace", []))

    doc["program_source"] = "live"
    return doc


def resolve_run_params(query: str | None, query_text: str | None, clip: str | None):
    """Param-validation for the run / run_doc routes (extracted so it is unit-testable without a
    FastAPI TestClient — plan BLOCKER). Returns ("canned", entry) or ("free", (text, clip_id)).
    Raises HTTPException(400) for the missing/both/empty/overlong cases and 404 for unknown
    query/clip. Exactly one of `query` / `query_text` must be present."""
    has_query = query is not None and query != ""
    has_text = query_text is not None and query_text.strip() != ""
    if has_query and has_text:
        raise HTTPException(status_code=400, detail="provide exactly one of 'query' or 'query_text', not both")
    if not has_query and not has_text:
        raise HTTPException(status_code=400, detail="provide one of 'query' (canned id) or 'query_text' (free text)")

    if has_query:
        entry = canned.by_id(query)
        if not entry:
            raise HTTPException(status_code=404, detail=f"unknown query '{query}'")
        return "canned", entry

    text = query_text.strip()
    if len(text) > QUERY_TEXT_MAX:
        raise HTTPException(status_code=400, detail=f"query_text exceeds {QUERY_TEXT_MAX} chars")
    clip_id = clip or "single-goal"
    if canned.clip_by_id(clip_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown clip '{clip_id}'")
    return "free", (text, clip_id)


def resolve_run_doc(query: str | None, query_text: str | None, clip: str | None) -> dict:
    """Resolve a run-doc for either path (canned or free text), applying param validation."""
    kind, payload = resolve_run_params(query, query_text, clip)
    if kind == "canned":
        return build_run_doc(payload)
    text, clip_id = payload
    return build_free_text_run_doc(text, clip_id)


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


@app.get("/api/run_doc")
def run_doc(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
):
    # Exactly one of query / query_text; canned auto-play (?query=<id>) still 200s (query is optional).
    return resolve_run_doc(query, query_text, clip)


@app.get("/api/run")
async def run(
    query: str | None = Query(None),
    query_text: str | None = Query(None),
    clip: str | None = Query(None),
    pace_ms: int = Query(1200, ge=0, le=10000),
):
    # Param validation (400/404) happens HERE, before the stream opens, so an unknown clip / bad
    # params surface as a normal HTTP status — not mid-stream. Once we have a doc, every state
    # (grounded OR ungrounded) streams the same meta -> step* -> findings -> done sequence.
    doc = resolve_run_doc(query, query_text, clip)

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
        except Exception:  # noqa: BLE001 — surface as an SSE error; never leak exception text
            yield _sse("error", {"message": "stream failed"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
