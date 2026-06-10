"""
The tiered scoring rubric (D4), pure and creds-free.

A strict MONOTONE tier ladder — each tier presupposes the one below, so a case lands at exactly
the highest tier it reaches (or REJECTED / STALE / MISSING). The ladder reuses the server's EXACT
post-codegen slice: `validation_errors({"program": raw}, clip=clip)` (the trust boundary) then
`Interpreter(cache).run(raw, query=question)`. It NEVER imports `codegen`/Azure and NEVER imports
`server`.

Trust boundary (mirrors server.build_free_text_run_doc): `raw_program` is UNTRUSTED recorded model
output and MUST flow through `validation_errors` BEFORE `Interpreter.run`. A fixture can never be
scored T2+ without passing the real gate, and a validator-rejected program is never executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from validate_program import validation_errors

# Tier ladder (monotone). REJECTED/STALE/MISSING are non-tier outcomes.
T0_PARSED = "T0_parsed"
T1_SCHEMA = "T1_schema_valid"
T2_SEMANTIC = "T2_semantic_valid"
T3_GROUNDED = "T3_executes_grounded"
T4_CORRECT = "T4_answer_correct"
REJECTED = "REJECTED"
STALE = "STALE"
MISSING = "MISSING"

# Ordering for "reached at least tier X" comparisons in the report/threshold gate.
TIER_ORDER = [REJECTED, T0_PARSED, T1_SCHEMA, T2_SEMANTIC, T3_GROUNDED, T4_CORRECT]


@dataclass
class CaseResult:
    case_id: str
    target_tier: str
    reached_tier: str
    passed: bool
    detail: str = ""
    usage: None = None  # reserved; always None (Resolved #1)
    stale: bool = False
    missing: bool = False
    # Bookkeeping the report uses (clip/shape/class/distance copied off the case).
    clip_id: str = ""
    shape: str = ""
    phrasing_class: str = ""
    distance: str = "n/a"
    program_bytes: int = 0
    grounded: Optional[bool] = None
    overclaimed: bool = False  # an out-of-scope case that hallucinated a grounded answer
    _extra: dict = field(default_factory=dict)


# ---- answer comparators (T4) ------------------------------------------------------------

def _norm(s) -> str:
    return str(s).strip().lower()


def match_exact(expected: str, actual) -> bool:
    return _norm(expected) == _norm(actual)


def match_ci_contains(expected: str, actual) -> bool:
    return _norm(expected) in _norm(actual)


def match_numeric(expected: str, actual) -> bool:
    """Parse the first integer out of each side and compare. 'numeric("5","5 players")' matches."""
    import re

    def first_int(x):
        m = re.search(r"-?\d+", str(x))
        return int(m.group()) if m else None

    e, a = first_int(expected), first_int(actual)
    return e is not None and e == a


def match_any_grounded(expected, actual) -> bool:
    """The no-op comparator: PASS at T3 regardless of value. MUST NOT be used on a known-answer cell."""
    return True


COMPARATORS = {
    "exact": match_exact,
    "ci_contains": match_ci_contains,
    "numeric": match_numeric,
    "any_grounded": match_any_grounded,
}


def check_answer(answer_match: str, expected, actual) -> bool:
    return COMPARATORS[answer_match](expected, actual)


# ---- the ladder -------------------------------------------------------------------------

def _looks_like_program(raw) -> bool:
    """T0: a list with a program shape (each item a dict). Mirrors codegen.generate's post-parse
    guard (it requires `program` to be a list)."""
    return isinstance(raw, list) and all(isinstance(s, dict) for s in raw)


def _structural_only(errors: list[str]) -> bool:
    """True iff every error is structural-class (T1 fails). validation_errors prefixes structural
    errors with 'structural: ' and the byte-cap with 'size: '. Semantic errors are 'semantic: '."""
    return all(e.startswith("structural:") or e.startswith("size:") for e in errors)


def score_case(case: dict, fixture, clip: dict, cache) -> CaseResult:
    """Score one bank case against its recorded fixture, reusing the production gate. `fixture` is a
    fixtures.Fixture or None (MISSING). `clip`/`cache` are the resolved clip dict and loaded Cache.

    Tiers (in-scope):
      T0 parsed     -> fixture has a raw_program (not an error) with a program shape
      T1 schema     -> validation_errors has no structural/size errors
      T2 semantic   -> validation_errors == [] (the full trust boundary)
      T3 grounded   -> T2 AND Interpreter.run(...).findings.grounded is True (no exception)
      T4 correct    -> T3 AND findings.answer matches expect.answer under the comparator

    Out-of-scope (expect.outcome == 'ungrounded'): the ladder INVERTS — PASS iff the program is
    REJECTED or grounds out False (honest refusal); FAIL iff it grounds True (hallucination)."""
    expect = case["expect"]
    out_of_scope = expect["outcome"] == "ungrounded"
    # Target tier: in-scope grounded cases target T4 (or T3 for any_grounded); ungrounded target
    # the honest-refusal outcome (recorded as T3 grounded:false, scored inverted).
    if out_of_scope:
        target = T3_GROUNDED
    elif expect.get("answer_match") == "any_grounded":
        target = T3_GROUNDED
    else:
        target = T4_CORRECT

    base = dict(
        case_id=case["case_id"],
        target_tier=target,
        clip_id=case["clip_id"],
        shape=case["shape"],
        phrasing_class=case["phrasing_class"],
        distance=case.get("distance", "n/a"),
    )

    # MISSING: no fixture captured for this case.
    if fixture is None:
        return CaseResult(reached_tier=MISSING, passed=False, missing=True,
                          detail="no fixture captured", **base)

    # STALE handling is applied by the runner (it knows the current prompt_version); scoring still
    # produces a result so the report can show what the stale fixture WOULD score.
    raw = fixture.raw_program
    program_bytes = len(_json_bytes(raw)) if raw is not None else 0
    base["program_bytes"] = program_bytes  # type: ignore[assignment]

    # T0: capture recorded a raw_program (vs an `error` tag) with a program shape.
    if fixture.error is not None or not _looks_like_program(raw):
        # REJECTED / no-output. Out-of-scope: a clean no-output IS a pass (no fabricated answer).
        reached = REJECTED
        passed = out_of_scope
        detail = f"no parseable program (error={fixture.error})" if fixture.error else "raw_program not a program shape"
        return CaseResult(reached_tier=reached, passed=passed, detail=detail, **base)

    # Trust boundary: validate the UNTRUSTED program BEFORE any execution. ONE validation_errors
    # call computes both T1 and T2 (it runs the byte-cap + comment-strip before structural_errors
    # internally, so a separate structural_errors call would skip those guards — D4 note).
    errors = validation_errors({"program": raw}, clip=clip)
    if errors:
        # The full gate rejected it (it never reaches Interpreter.run). Highest *passed* tier:
        #   structural/size error -> failed structure, so the highest passed tier is T0 (parsed).
        #   semantic-only error   -> passed structure (T1) but failed the semantic gate (not T2).
        reached = T0_PARSED if _structural_only(errors) else T1_SCHEMA
        passed = out_of_scope  # a clean validator rejection is a PASS for out-of-scope cases
        return CaseResult(reached_tier=reached, passed=passed,
                          detail=f"validation: {errors[0]}", **base)

    # T2 reached: the full gate passed. Now execute over the cache (T3).
    from interpreter import Interpreter  # local import: keep scoring import-light, never Azure

    try:
        doc = Interpreter(cache).run(raw, query=case["question"])
    except Exception as e:  # noqa: BLE001
        # A validated program that still raises is a real defect; record T2 (it passed the gate).
        # For out-of-scope, a runtime rejection is still an honest non-answer -> PASS.
        return CaseResult(reached_tier=T2_SEMANTIC, passed=out_of_scope,
                          detail=f"execution-error: {type(e).__name__}", **base)

    findings = doc["findings"]
    grounded = bool(findings.get("grounded"))
    base["grounded"] = grounded  # type: ignore[assignment]

    if out_of_scope:
        # Inverted ladder: honest grounded:false -> PASS; grounded:true -> FAIL (hallucinated).
        if grounded:
            return CaseResult(reached_tier=T4_CORRECT, passed=False, overclaimed=True,
                              detail=f"HALLUCINATED grounded answer {findings.get('answer')!r}", **base)
        return CaseResult(reached_tier=T3_GROUNDED, passed=True,
                          detail=f"honest no-ground (reason={findings.get('reason')})", **base)

    # In-scope: T3 requires grounded:true.
    if not grounded:
        return CaseResult(reached_tier=T2_SEMANTIC, passed=False,
                          detail=f"validated but did not ground (reason={findings.get('reason')})", **base)

    # T3 reached. T4: compare the answer under the shape's comparator.
    actual = findings.get("answer")
    if check_answer(expect["answer_match"], expect.get("answer"), actual):
        detail = "" if expect["answer_match"] == "any_grounded" else f"answer={actual!r}"
        return CaseResult(reached_tier=T4_CORRECT, passed=True, detail=detail, **base)
    return CaseResult(reached_tier=T3_GROUNDED, passed=False,
                      detail=f"answer {actual!r} != expected {expect.get('answer')!r}", **base)


def _json_bytes(obj) -> bytes:
    import json

    try:
        return json.dumps(obj).encode("utf-8")
    except (TypeError, ValueError):
        return b""
