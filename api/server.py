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
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))            # interpreter package
sys.path.insert(0, str(API_DIR / "scripts"))  # strip_comments

from interpreter import Cache, Interpreter  # noqa: E402
from validate_program import strip_comments  # noqa: E402
import canned  # noqa: E402

app = FastAPI(title="Glass-Box Video Analyst API", version="0.1.0")

# Dev CORS: the Next.js dev server runs on a different origin. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET"],
    allow_headers=["*"],
)


def build_run_doc(entry: dict) -> dict:
    """Replay the canned program over its cache -> run-doc. Instant; no Azure calls."""
    program = strip_comments(json.loads(Path(entry["program"]).read_text()))["program"]
    return Interpreter(Cache.load(entry["cache"])).run(program)


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
def run_doc(query: str = Query(...)):
    entry = canned.by_id(query)
    if not entry:
        raise HTTPException(status_code=404, detail=f"unknown query '{query}'")
    return build_run_doc(entry)


@app.get("/api/run")
async def run(query: str = Query(...), pace_ms: int = Query(1200, ge=0, le=10000)):
    entry = canned.by_id(query)
    if not entry:
        raise HTTPException(status_code=404, detail=f"unknown query '{query}'")
    doc = build_run_doc(entry)

    async def gen():
        try:
            yield _sse("meta", {
                "query": doc["query"], "clip": doc["clip"],
                "program": doc["program"], "total_steps": len(doc["trace"]), "pace_ms": pace_ms,
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
