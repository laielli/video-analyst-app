"""
The tiered scoring rubric (D4) — pure and creds-free.

A strict monotone tier ladder, each tier presupposing the one below. A case lands at exactly the
highest tier it reaches (or REJECTED / STALE / MISSING). The ladder reuses the server's EXACT gate:
`validate_program.validation_errors` (the trust boundary) then `interpreter.Interpreter.run`. The
only harness-specific logic is the T4 answer comparators.

TRUST BOUNDARY: `raw_program` is UNTRUSTED recorded model output. It MUST flow through
`validation_errors` BEFORE `Interpreter.run` — the same order as server.build_free_text_run_doc.
A fixture can never be scored T2+ without passing the real validator. Never imports codegen/Azure.

For out-of-scope cases (expect.outcome == "ungrounded") the ladder INVERTS: an honest
`grounded == False` OR a clean REJECTED is a PASS; an executed `grounded == True` (a fabricated
answer) is the FAIL. This rewards calibrated refusal, which is the product's actual contract.
"""
from __future__ import annotations

from dataclasses import dataclass

# Tier names (monotone ladder) + the off-ladder outcomes.
T0_PARSED = "T0_parsed"
T1_SCHEMA = "T1_schema"
T2_SEMANTIC = "T2_semantic"
T3_GROUNDED = "T3_grounded"
T4_CORRECT = "T4_correct"
REJECTED = "REJECTED"   # failed parse/validate — for in-scope this is a FAIL, for out-of-scope a PASS
STALE = "STALE"         # fixture prompt_version != current — handled by the gate (D6)
MISSING = "MISSING"     # no fixture on disk — a distinct, visible outcome (not pass, not fail)

# The ordered ladder for "did it reach its target tier" comparisons.
TIER_ORDER = [REJECTED, T0_PARSED, T1_SCHEMA, T2_SEMANTIC, T3_GROUNDED, T4_CORRECT]


@dataclass
class CaseResult:
    case_id: str
    target_tier: str
    reached_tier: str
    passed: bool
    detail: str
    usage: dict | None = None
    stale: bool = False
    missing: bool = False


def _target_tier(case: dict) -> str:
    """The tier a case must reach to pass. Grounded cases with a known answer target T4; grounded
    cases scored `any_grounded` target T3; ungrounded (out-of-scope/honest-negative) cases have an
    inverted contract handled in score_case (target is informational here)."""
    expect = case["expect"]
    if expect["outcome"] == "ungrounded":
        return REJECTED  # informational: out-of-scope passes via the inverted path, not the ladder
    if expect.get("answer_match") == "any_grounded" or expect.get("answer") is None:
        return T3_GROUNDED
    return T4_CORRECT


def _answer_matches(expected: str | None, actual: str | None, mode: str) -> bool:
    """The four T4 comparators (D4). `any_grounded` is the no-op comparator (PASS at T3 regardless
    of the answer value) and MUST NOT be used on a cell with a known answer."""
    if mode == "any_grounded":
        return True
    if expected is None or actual is None:
        return False
    if mode == "exact":
        return str(actual).strip().lower() == str(expected).strip().lower()
    if mode == "ci_contains":
        return str(expected).strip().lower() in str(actual).strip().lower()
    if mode == "numeric":
        try:
            return float(str(actual).strip()) == float(str(expected).strip())
        except (TypeError, ValueError):
            # Fall back to digit extraction: "5 players" -> 5 (the count readout may wrap the value).
            import re

            m = re.search(r"-?\d+(?:\.\d+)?", str(actual))
            if m is None:
                return False
            try:
                return float(m.group()) == float(str(expected).strip())
            except (TypeError, ValueError):
                return False
    return False


def score_case(case: dict, fixture, clip: dict, cache) -> CaseResult:
    """Score one case against its recorded fixture (D4). `fixture` may be None (MISSING). `clip` is
    the resolved canned clip dict; `cache` is a loaded interpreter Cache. Pushes the untrusted
    `raw_program` through validate-then-execute (the production gate), then applies the comparator
    (or the inverted out-of-scope contract)."""
    from interpreter import Interpreter
    from validate_program import validation_errors

    expect = case["expect"]
    out_of_scope = expect["outcome"] == "ungrounded"
    target = _target_tier(case)

    # MISSING — a distinct outcome (not pass, not fail) so an un-captured case is visible.
    if fixture is None:
        return CaseResult(case["case_id"], target, MISSING, False, "no fixture on disk", missing=True)

    # error fixture (generate raised) or no raw_program -> no parsed output.
    if fixture.error is not None or fixture.raw_program is None:
        detail = f"no model output (error={fixture.error})"
        passed = out_of_scope  # a clean no-output is an honest refusal for out-of-scope
        return CaseResult(case["case_id"], target, REJECTED, passed, detail, usage=fixture.usage)

    raw = fixture.raw_program

    # T0 — parsed: a list with a program shape (mirrors codegen.generate's post-parse guard).
    if not isinstance(raw, list):
        passed = out_of_scope
        return CaseResult(case["case_id"], target, REJECTED, passed,
                          "raw_program is not a list", usage=fixture.usage)

    # ---- TRUST BOUNDARY: validate the untrusted program BEFORE any execution ----
    # One validation_errors call computes BOTH T1 and T2 (it runs the byte-cap + comment-strip
    # before structural_errors internally; calling structural_errors separately would skip those
    # guards and give T1 a divergent code path). T1 = no structural-class errors; T2 = empty list.
    errors = validation_errors({"program": raw}, clip=clip)
    # Recompute the structural slice for the T1/T2 split (cheap; on the comment-stripped doc the
    # server already passed through). A program that clears structure but fails semantics lands T1.
    structural = [e for e in errors if e.startswith("structural:") or e.startswith("size:")]

    if structural:
        # Did not even clear the schema/byte-cap -> REJECTED (below T1).
        passed = out_of_scope
        first = structural[0]
        return CaseResult(case["case_id"], target, REJECTED, passed,
                          f"rejected: {first}", usage=fixture.usage)

    if errors:
        # Cleared structure but failed semantics/bounds -> T1 (schema-valid, not semantically-valid).
        passed = out_of_scope  # for out-of-scope, a clean validator rejection is a PASS
        first = errors[0]
        return CaseResult(case["case_id"], target, T1_SCHEMA, passed,
                          f"semantic fail: {first}", usage=fixture.usage)

    # T2 — semantically-valid (the full server gate held).
    # T3 — execute over the clip cache (untrusted program, but now validated, exactly like server).
    try:
        doc = Interpreter(cache).run(raw, query=case["question"])
    except Exception as e:  # noqa: BLE001 — a validated program shouldn't raise, but be honest if it does
        # Execution error on a validated program: T2 is the highest tier reached.
        passed = out_of_scope  # an execution-error never fabricates an answer -> honest for OOS
        return CaseResult(case["case_id"], target, T2_SEMANTIC, passed,
                          f"execution-error: {type(e).__name__}", usage=fixture.usage)

    findings = doc.get("findings", {})
    grounded = bool(findings.get("grounded"))

    if out_of_scope:
        # INVERTED contract: honest grounded:false = PASS; grounded:true (fabricated) = FAIL.
        if grounded:
            return CaseResult(case["case_id"], target, T3_GROUNDED, False,
                              f"HALLUCINATED: grounded answer {findings.get('answer')!r} for an "
                              f"out-of-scope question", usage=fixture.usage)
        reason = findings.get("reason") or "ungrounded"
        return CaseResult(case["case_id"], target, T2_SEMANTIC, True,
                          f"honest no-ground (reason={reason})", usage=fixture.usage)

    if not grounded:
        # In-scope but didn't ground -> T2 (valid but missed the cache). For an in-scope case this
        # is below target (a FAIL relative to target), but it is an HONEST outcome, not a crash.
        reason = findings.get("reason") or "ungrounded"
        return CaseResult(case["case_id"], target, T2_SEMANTIC, False,
                          f"valid but ungrounded (reason={reason})", usage=fixture.usage)

    # T3 — executes-grounded. T4 — answer-correct under the shape's comparator.
    answer = findings.get("answer")
    mode = expect.get("answer_match", "any_grounded")
    if _answer_matches(expect.get("answer"), answer, mode):
        # any_grounded targets T3 and PASSES here; a known-answer case reaches T4.
        reached = T3_GROUNDED if mode == "any_grounded" else T4_CORRECT
        return CaseResult(case["case_id"], target, reached, True,
                          f"grounded answer {answer!r}", usage=fixture.usage)
    # Grounded but the answer is wrong -> T3 (below T4 target).
    return CaseResult(case["case_id"], target, T3_GROUNDED, False,
                      f"wrong answer: got {answer!r}, expected {expect.get('answer')!r} "
                      f"({mode})", usage=fixture.usage)


def reached_target(result: CaseResult) -> bool:
    """Whether a result reached (>=) its target tier on the ladder. Used by the report; the `passed`
    field already encodes the inverted out-of-scope contract."""
    try:
        return TIER_ORDER.index(result.reached_tier) >= TIER_ORDER.index(result.target_tier)
    except ValueError:
        return False
