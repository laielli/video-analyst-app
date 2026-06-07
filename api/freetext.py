"""
Free-text query path — the full ViperGPT thesis, end to end.

A TYPED question is compiled by Azure OpenAI codegen into a live visual program, validated
(the trust boundary), and replayed over the clip's precomputed cache. Unlike the canned path
there is NO pinned fallback for a novel question, so EVERY failure mode resolves to an explicit,
honest *ungrounded* run-doc (Q1 -> A) — never a 500, never the hero program substituted.

This module deliberately has NO fastapi dependency: it raises its own RequestError, which the
FastAPI route layer (server.py) translates to HTTPException. That keeps the trust-boundary logic
importable + unit-testable under the system python the test suite runs in (fastapi and pytest do
not co-resolve in one interpreter — plan §Tests BLOCKER).
"""
from __future__ import annotations

import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))               # interpreter package
sys.path.insert(0, str(API_DIR / "scripts"))    # validate_program

from interpreter import Cache, Interpreter  # noqa: E402
from validate_program import validation_errors  # noqa: E402
import canned  # noqa: E402
import codegen  # noqa: E402

# Free-text guardrails (§Risks: cost/abuse on an unauthenticated GET).
MAX_QUERY_TEXT = 500            # reject longer typed questions BEFORE calling codegen.
CODEGEN_CALL_BUDGET = 200       # per-process cap on billed Azure codegen calls.
_codegen_calls = 0             # process-wide counter (free text only; canned uses pinned/live).

# Fixed-enum reasons a free-text query could not be grounded. NEVER raw exception text (paths /
# Azure errors) — those would leak into the SSE payload (§Risks: info leakage).
UNGROUNDED_REASONS = {
    "codegen-disabled", "codegen-error", "invalid-program", "execution-error", "no-grounded-answer",
}


class RequestError(Exception):
    """A bad/ambiguous run request (translated to HTTPException at the route boundary)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def reset_codegen_budget() -> None:
    """Reset the per-process codegen-call counter (test seam)."""
    global _codegen_calls
    _codegen_calls = 0


def _clip_meta(clip: dict) -> dict:
    """The run-doc `clip` block (run_doc.schema.json requires id/width/height/duration_ms)."""
    return {k: clip[k] for k in ("id", "width", "height", "duration_ms") if k in clip}


def ungrounded_run_doc(clip: dict, query_text: str, reason: str,
                       program: list[dict] | None = None, trace: list[dict] | None = None) -> dict:
    """A complete, schema-valid run-doc for a free-text query that could NOT be grounded. Carries
    ALL the keys the SSE stream loop reads (query/clip/program/trace/findings) so the loop never
    KeyErrors. `reason` is a FIXED enum (info-leakage guard). When codegen produced a validated
    program that simply didn't ground, pass its program + trace so the UI shows WHERE it ran dry
    (extends D-DR5); when codegen failed before producing one, program/trace are []."""
    assert reason in UNGROUNDED_REASONS, f"unknown ungrounded reason '{reason}'"
    return {
        "schema_version": "1",
        "query": query_text,
        "clip": _clip_meta(clip),
        "program": program or [],
        "trace": trace or [],
        "findings": {"answer": None, "verdict": None, "grounded": False,
                     "reason": reason, "partial": True},
        "program_source": "ungrounded",
    }


def build_free_text_run_doc(query_text: str, clip_id: str) -> dict:
    """Compile a typed question into a live program against `clip_id`'s replay cache, with NO
    pinned fallback. validation_errors (structural + semantic + NUMERIC bounds) is the trust
    boundary: no generated program reaches Interpreter.run unvalidated. Any miss -> an honest
    ungrounded run-doc. Raises RequestError(404) for an unknown clip (the route's one 404 site)."""
    global _codegen_calls
    clip = canned.clip_by_id(clip_id)
    if clip is None:
        raise RequestError(404, f"unknown clip '{clip_id}'")
    cache = Cache.load(clip["cache"])

    if not codegen.enabled():
        return ungrounded_run_doc(clip, query_text, "codegen-disabled")
    if _codegen_calls >= CODEGEN_CALL_BUDGET:
        # Cost/abuse guard: don't run up unbounded paid Azure calls on unauthenticated GETs.
        return ungrounded_run_doc(clip, query_text, "codegen-error")

    try:
        _codegen_calls += 1
        program = codegen.AzureCodegen.from_env().generate(query_text, clip=clip)
    except Exception as e:  # noqa: BLE001 — any generation failure -> ungrounded (no fallback)
        print(f"[free-text] codegen failed: {type(e).__name__}", file=sys.stderr)  # type only, no raw text
        return ungrounded_run_doc(clip, query_text, "codegen-error")

    # Trust boundary: structural + semantic + NUMERIC-BOUND validation BEFORE any execution.
    if validation_errors({"program": program}, clip=cache.clip):
        return ungrounded_run_doc(clip, query_text, "invalid-program", program=program)

    try:
        doc = Interpreter(cache).run(program, query=query_text)
    except Exception as e:  # noqa: BLE001 — e.g. sampled window misses the cached frames
        print(f"[free-text] execution failed: {type(e).__name__}", file=sys.stderr)
        return ungrounded_run_doc(clip, query_text, "execution-error", program=program)

    f = doc.get("findings", {})
    if f.get("grounded") is False or f.get("answer") in (None, "", "Unknown") or f.get("partial"):
        # Validated program ran but didn't ground -> stream its trace so the user sees where it ran
        # dry (the program IS in doc), tagged ungrounded with the no-grounded-answer reason.
        return ungrounded_run_doc(clip, query_text, "no-grounded-answer",
                                  program=doc.get("program", program), trace=doc.get("trace", []))

    doc["program_source"] = "live"
    return doc


def resolve_run_request(query: str | None, query_text: str | None):
    """Pure param-resolver for the /api/run + /api/run_doc routes: enforces 'exactly one of
    query / query_text' and the free-text length cap, returning ('canned', query_id) or
    ('free', query_text). Raises RequestError on a bad request. Extracted so it is unit-testable
    WITHOUT a FastAPI TestClient (plan §Tests BLOCKER)."""
    has_canned = bool(query)
    has_free = query_text is not None and query_text.strip() != ""
    if has_canned == has_free:
        raise RequestError(400, "provide exactly one of 'query' (canned) or 'query_text' (free text)")
    if has_canned:
        entry = canned.by_id(query)
        if not entry:
            raise RequestError(404, f"unknown query '{query}'")
        return "canned", entry
    # free text: reject overlong query BEFORE any codegen call (cost/abuse guard).
    if len(query_text) > MAX_QUERY_TEXT:
        raise RequestError(400, f"query_text exceeds {MAX_QUERY_TEXT} chars")
    return "free", query_text
