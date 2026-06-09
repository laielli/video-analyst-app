"""The tiered scoring rubric (D4) — pure, creds-free, Azure-free.

Reuses the server's EXACT post-codegen slice: `validate_program.validation_errors` then
`interpreter.Interpreter.run` (the same two calls `build_free_text_run_doc` makes after codegen
returns) — no HTTP, no TestClient, no `build_free_text_run_doc`, no `server` import.

TRUST BOUNDARY: `raw_program` is untrusted recorded model output and MUST flow through
`validation_errors` BEFORE `Interpreter.run` — the same order as server.py. A fixture can never be
scored T2+ without passing the real gate; an invalid program is REJECTED and never executed.

Monotone tier ladder (a case lands at the highest tier it reaches), for IN-SCOPE cases:
  REJECTED  — fixture has no raw_program (error tag), or fails parse/validate.
  T0 parsed — capture recorded a raw_program (a list), not an error.
  T1 schema-valid — no structural/size-class errors from the single validation_errors call.
  T2 semantically-valid — validation_errors([...]) == [] (the full server gate).
  T3 executes-grounded — T2 AND Interpreter.run(...).findings.grounded is True (no exception).
  T4 answer-correct — T3 AND findings.answer matches expect.answer under the shape comparator.

For OUT-OF-SCOPE cases (expect.outcome == "ungrounded": every out_of_scope/injection/degenerate case
and the absent-subject #23 cases) the ladder INVERTS: PASS on an honest grounded:false OR a clean
REJECTED; FAIL only on grounded:true with a fabricated answer (the dangerous overclaim).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Tier names (also the ladder order). REJECTED/STALE/MISSING are terminal non-tier outcomes.
REJECTED = "REJECTED"
T0_PARSED = "T0_parsed"
T1_SCHEMA = "T1_schema_valid"
T2_SEMANTIC = "T2_semantic_valid"
T3_GROUNDED = "T3_executes_grounded"
T4_CORRECT = "T4_answer_correct"
STALE = "STALE"
MISSING = "MISSING"

# Ladder rank for monotone comparison / report funnel ordering.
TIER_ORDER = [REJECTED, T0_PARSED, T1_SCHEMA, T2_SEMANTIC, T3_GROUNDED, T4_CORRECT]
TIER_RANK = {name: i for i, name in enumerate(TIER_ORDER)}


@dataclass
class CaseResult:
    case_id: str
    clip_id: str
    shape: str
    phrasing_class: str
    distance: str
    target_tier: str
    reached_tier: str
    passed: bool
    detail: str
    usage: dict | None = None
    stale: bool = False
    missing: bool = False
    # raw_program byte size (offline prompt/output-size signal; never an input to scoring).
    program_bytes: int = 0
    extra: dict = field(default_factory=dict)


def _is_out_of_scope(case: dict) -> bool:
    return case["expect"]["outcome"] == "ungrounded"


def _target_tier(case: dict) -> str:
    """The tier an in-scope case must reach to PASS. Out-of-scope cases use the inverted rule and
    have no positive target (reported as REJECTED-or-grounded:false), so we report their target as
    T3 for the funnel but pass/fail by the inverted predicate."""
    if _is_out_of_scope(case):
        return REJECTED  # inverted: a clean reject is one acceptable outcome
    em = case["expect"]["answer_match"]
    return T3_GROUNDED if em == "any_grounded" else T4_CORRECT


# ---- answer comparators (T4) -------------------------------------------------------------------

def match_exact(expected: str, actual: str) -> bool:
    """Case-insensitive exact match (used for Yes/No verdicts)."""
    return (actual or "").strip().lower() == (expected or "").strip().lower()


def match_ci_contains(expected: str, actual: str) -> bool:
    """Case-insensitive substring (the readout may wrap the legible value in prose)."""
    return (expected or "").strip().lower() in (actual or "").lower()


def match_numeric(expected: str, actual: str) -> bool:
    """Parse-and-equal: pull the leading integer out of `actual` and compare to expected as ints."""
    try:
        want = int(str(expected).strip())
    except (TypeError, ValueError):
        return False
    import re

    m = re.search(r"-?\d+", str(actual or ""))
    if m is None:
        return False
    return int(m.group()) == want


COMPARATORS = {
    "exact": match_exact,
    "ci_contains": match_ci_contains,
    "numeric": match_numeric,
}


def check_answer(answer_match: str, expected: str | None, actual: str | None) -> bool:
    """Apply the shape's comparator. `any_grounded` is the no-op comparator (PASS at T3 regardless of
    value) — it must never be used on a cell with a known answer."""
    if answer_match == "any_grounded":
        return True
    cmp = COMPARATORS.get(answer_match)
    if cmp is None:
        return False
    return cmp(expected or "", actual or "")


# ---- the rubric --------------------------------------------------------------------------------

def _structural_class_errors(errors: list[str]) -> list[str]:
    """Errors that fail T1 (schema/structure): the byte-cap ("size: ...") + structural ("structural:
    ...") classes from validation_errors. Semantic ("semantic: ...") errors fail T2 but pass T1."""
    return [e for e in errors if e.startswith("structural:") or e.startswith("size:")]


def score_case(case: dict, fixture, clip: dict, cache, *, interpreter_run=None) -> CaseResult:
    """Score one bank case against its recorded fixture. `interpreter_run` is an injection seam for
    tests (default: Interpreter(cache).run); production passes None and we build the real one."""
    base = dict(
        case_id=case["case_id"],
        clip_id=case["clip_id"],
        shape=case["shape"],
        phrasing_class=case["phrasing_class"],
        distance=case["distance"],
        target_tier=_target_tier(case),
        usage=getattr(fixture, "usage", None) if fixture is not None else None,
    )
    out_of_scope = _is_out_of_scope(case)

    # MISSING — an un-captured case (distinct from pass/fail so it stays visible).
    if fixture is None:
        return CaseResult(reached_tier=MISSING, passed=False, missing=True,
                          detail="no fixture (un-captured)", **base)

    program_bytes = _program_byte_size(fixture.raw_program)
    base["usage"] = fixture.usage

    # REJECTED — capture recorded an error tag (no raw_program), or the program isn't a list.
    if fixture.error is not None or not isinstance(fixture.raw_program, list):
        reason = fixture.error or "raw_program is not a list"
        passed = out_of_scope  # a clean reject is a PASS for out-of-scope; a FAIL for in-scope.
        return CaseResult(reached_tier=REJECTED, passed=passed,
                          detail=f"rejected at parse: {reason}", program_bytes=program_bytes, **base)

    raw_program = fixture.raw_program

    # T0 parsed — a list with a program shape (mirrors codegen.generate's post-parse guard).
    reached = T0_PARSED

    # TRUST BOUNDARY: validate the UNTRUSTED program BEFORE any execution — same order as server.py.
    from validate_program import validation_errors

    errors = validation_errors({"program": raw_program}, clip=clip)
    structural = _structural_class_errors(errors)
    if structural:
        # Fails T1: a clean validator rejection. PASS for out-of-scope, FAIL for in-scope.
        return CaseResult(reached_tier=REJECTED, passed=out_of_scope,
                          detail=f"validator-rejected (structural): {structural[0]}",
                          program_bytes=program_bytes, **base)
    reached = T1_SCHEMA
    if errors:
        # Structurally OK but semantically broken (T1 reached, T2 missed) — a clean validator reject.
        return CaseResult(reached_tier=T1_SCHEMA, passed=out_of_scope,
                          detail=f"validator-rejected (semantic): {errors[0]}",
                          program_bytes=program_bytes, **base)
    reached = T2_SEMANTIC

    # T3 — execute over the clip cache (only after the gate passed). Never executes a rejected program.
    if interpreter_run is None:
        from interpreter import Interpreter

        interpreter_run = Interpreter(cache).run
    try:
        doc = interpreter_run(raw_program, query=case["question"])
    except Exception as e:  # noqa: BLE001 — an execution exception is an in-scope FAIL / oos PASS.
        return CaseResult(reached_tier=T2_SEMANTIC, passed=out_of_scope,
                          detail=f"execution raised: {type(e).__name__}",
                          program_bytes=program_bytes, **base)
    findings = doc.get("findings", {})
    grounded = findings.get("grounded") is True
    answer = findings.get("answer")

    if out_of_scope:
        # INVERTED rubric: grounded:false -> PASS (honest refusal); grounded:true -> FAIL (overclaim).
        if grounded:
            return CaseResult(reached_tier=T3_GROUNDED, passed=False,
                              detail=f"OVERCLAIM: grounded answer {answer!r} for an ungrounded case",
                              program_bytes=program_bytes, **base)
        reason = findings.get("reason") or "grounded:false"
        return CaseResult(reached_tier=T2_SEMANTIC, passed=True,
                          detail=f"honest refusal ({reason})", program_bytes=program_bytes, **base)

    # In-scope: needs grounded:true for T3.
    if not grounded:
        reason = findings.get("reason") or "grounded:false"
        return CaseResult(reached_tier=T2_SEMANTIC, passed=False,
                          detail=f"valid but did not ground ({reason})",
                          program_bytes=program_bytes, **base)
    reached = T3_GROUNDED

    # T4 — answer-correct under the shape comparator.
    expect = case["expect"]
    if expect["answer_match"] == "any_grounded":
        # In-scope any_grounded case: PASS at T3 (any honestly-grounded answer is acceptable).
        return CaseResult(reached_tier=T3_GROUNDED, passed=True,
                          detail=f"grounded (any_grounded); answer={answer!r}",
                          program_bytes=program_bytes, **base)
    correct = check_answer(expect["answer_match"], expect.get("answer"), answer)
    if correct:
        return CaseResult(reached_tier=T4_CORRECT, passed=True,
                          detail=f"answer={answer!r} == expected {expect.get('answer')!r}",
                          program_bytes=program_bytes, **base)
    return CaseResult(reached_tier=T3_GROUNDED, passed=False,
                      detail=f"answer={answer!r} != expected {expect.get('answer')!r} "
                             f"({expect['answer_match']})",
                      program_bytes=program_bytes, **base)


def _program_byte_size(raw_program) -> int:
    """Serialized byte size of the recorded program (offline output-size signal; never scored)."""
    import json

    if raw_program is None:
        return 0
    try:
        return len(json.dumps(raw_program).encode("utf-8"))
    except (TypeError, ValueError):
        return 0
