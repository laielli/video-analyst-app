"""The tiered scoring rubric (D4) — pure, creds-free.

Reuses the server's EXACT post-codegen slice: `validate_program.validation_errors` (the trust
boundary) then `interpreter.Interpreter.run`. A recorded `raw_program` is UNTRUSTED model output
and MUST flow through validation_errors BEFORE Interpreter.run — the same order server.py uses, so
a fixture can never smuggle an unvalidated program into a "pass". Never imports codegen/Azure.

Monotone tier ladder (each tier presupposes the one below). For in-scope cases a case lands at the
highest tier it reaches; for out-of-scope/adversarial cases the ladder INVERTS (honest refusal or
clean rejection = PASS, a fabricated grounded answer = FAIL).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Tier ladder, lowest to highest. REJECTED/STALE/MISSING are non-tier outcomes.
REJECTED = "REJECTED"
STALE = "STALE"
MISSING = "MISSING"
T0_PARSED = "T0_parsed"
T1_SCHEMA = "T1_schema_valid"
T2_SEMANTIC = "T2_semantic_valid"
T3_GROUNDED = "T3_executes_grounded"
T4_CORRECT = "T4_answer_correct"

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
    expected_outcome: str = "grounded"
    expected_answer: str | None = None
    actual_answer: str | None = None
    grounded: bool | None = None
    raw_program_bytes: int = 0
    subject_note: str = ""
    flags: list[str] = field(default_factory=list)


# ---- answer-match comparators (T4) ---------------------------------------------------------

def _match_exact(expected: str, actual: str) -> bool:
    return expected.strip().casefold() == actual.strip().casefold()


def _match_ci_contains(expected: str, actual: str) -> bool:
    return expected.strip().casefold() in actual.casefold()


def _match_numeric(expected: str, actual: str) -> bool:
    """Parse the first integer out of each side and compare. `numeric("5", "5 players")` matches."""
    ai = _first_int(actual)
    ei = _first_int(expected)
    return ai is not None and ei is not None and ai == ei


def _first_int(s: str) -> int | None:
    import re

    m = re.search(r"-?\d+", s or "")
    return int(m.group()) if m else None


def answer_matches(answer_match: str, expected: str | None, actual: str | None) -> bool:
    """Dispatch the four comparators. `any_grounded` is the no-op (PASS regardless of value).
    A missing actual never matches a non-any_grounded expectation."""
    if answer_match == "any_grounded":
        return True
    if actual is None or expected is None:
        return False
    if answer_match == "exact":
        return _match_exact(expected, actual)
    if answer_match == "ci_contains":
        return _match_ci_contains(expected, actual)
    if answer_match == "numeric":
        return _match_numeric(expected, actual)
    raise ValueError(f"unknown answer_match mode: {answer_match!r}")


# ---- the rubric ----------------------------------------------------------------------------

def _byte_size(raw_program) -> int:
    import json

    try:
        return len(json.dumps(raw_program).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def score_case(case: dict, fixture, clip: dict, cache) -> CaseResult:
    """Score one case against its recorded fixture.

    `fixture` may be None (MISSING). `clip` is the resolved canned clip dict; `cache` is a loaded
    interpreter Cache. Tier computation reuses the production gate verbatim."""
    expect = case["expect"]
    expected_outcome = expect.get("outcome", "grounded")
    expected_answer = expect.get("answer")
    answer_match = expect["answer_match"]
    is_out_of_scope = expected_outcome == "ungrounded"
    # In-scope target: T4 when there is a concrete answer to check; T3 for any_grounded (PASS once
    # grounded, regardless of the answer value). Out-of-scope always targets T3 (inverted ladder).
    has_concrete_answer = expected_answer is not None and answer_match != "any_grounded"
    target_tier = T3_GROUNDED if (is_out_of_scope or not has_concrete_answer) else T4_CORRECT

    base = dict(
        case_id=case["case_id"],
        clip_id=case["clip_id"],
        shape=case["shape"],
        phrasing_class=case["phrasing_class"],
        distance=case["distance"],
        target_tier=target_tier,
        expected_outcome=expected_outcome,
        expected_answer=expected_answer,
    )

    # MISSING — distinct from pass/fail so an un-captured case is visible, not silent.
    if fixture is None:
        return CaseResult(reached_tier=MISSING, passed=False, missing=True,
                          detail="no fixture recorded (run capture)", **base)

    base["usage"] = fixture.usage
    base["raw_program_bytes"] = _byte_size(fixture.raw_program)

    # T0 / REJECTED — capture recorded a raw_program (vs an error tag), and it is a non-empty list.
    raw = fixture.raw_program
    if fixture.error is not None or not isinstance(raw, list) or not raw:
        return _finalize(base, REJECTED, is_out_of_scope, case,
                         detail=f"no parseable program (error={fixture.error})")

    # Trust boundary: validate BEFORE execute, exactly as server.py does. ONE validation_errors
    # call computes both T1 (no structural-class errors) and T2 (empty list). Calling
    # structural_errors separately would skip the byte-cap + comment-strip guards run internally.
    from validate_program import validation_errors

    errors = validation_errors({"program": raw}, clip=clip)
    structural = [e for e in errors if e.startswith("structural") or e.startswith("size")]
    if structural:
        # Stops below T1: malformed JSON shape / oversized doc.
        return _finalize(base, T0_PARSED, is_out_of_scope, case,
                         detail=f"structural: {structural[0]}")
    if errors:
        # Reached T1 (structurally valid) but failed semantic/caps/bounds → stops at T1.
        return _finalize(base, T1_SCHEMA, is_out_of_scope, case,
                         detail=f"semantic: {errors[0]}")

    # T2 reached: the full server gate passed. Execute (untrusted program already validated).
    from interpreter import Interpreter

    try:
        doc = Interpreter(cache).run(raw, query=case["question"])
    except Exception as e:  # noqa: BLE001 — any execution failure stops at T2 (validated but un-runnable)
        return _finalize(base, T2_SEMANTIC, is_out_of_scope, case,
                         detail=f"execution raised: {type(e).__name__}")

    findings = doc["findings"]
    grounded = bool(findings.get("grounded"))
    base["grounded"] = grounded
    actual_answer = findings.get("answer")
    base["actual_answer"] = actual_answer

    if not grounded:
        # T2 reached, did not ground (honest no-ground / cache miss).
        return _finalize(base, T2_SEMANTIC, is_out_of_scope, case,
                         detail=f"grounded=false (reason={findings.get('reason')})")

    # T3 reached: grounded. For temporal cases, log (not gate) whether the verdict names the
    # subject in the question.
    subject_note = _subject_note(case, findings)
    base["subject_note"] = subject_note

    if target_tier == T3_GROUNDED and not is_out_of_scope:
        # any_grounded in-scope cell: PASS at T3.
        return _finalize(base, T3_GROUNDED, is_out_of_scope, case,
                         detail="grounded (any_grounded target)")

    if is_out_of_scope:
        # Out-of-scope grounded == hallucination → handled in _finalize (FAIL).
        return _finalize(base, T3_GROUNDED, is_out_of_scope, case,
                         detail=f"grounded answer={actual_answer!r} (expected honest refusal)")

    # T4: answer-correct comparison.
    correct = answer_matches(answer_match, expected_answer, actual_answer)
    reached = T4_CORRECT if correct else T3_GROUNDED
    detail = ("answer correct" if correct
              else f"answer mismatch: got {actual_answer!r}, expected {expected_answer!r}")
    return _finalize(base, reached, is_out_of_scope, case, detail=detail)


def _finalize(base: dict, reached_tier: str, is_out_of_scope: bool, case: dict, detail: str) -> CaseResult:
    """Apply the pass/fail rule and build the CaseResult.

    In-scope: PASS iff reached_tier >= target_tier (REJECTED is below T0 → always a FAIL).
    Out-of-scope/adversarial: the ladder inverts — PASS iff cleanly REJECTED OR grounded is False;
    FAIL iff a program executed to a grounded answer (the dangerous overclaim)."""
    if is_out_of_scope:
        # PASS unless a program executed to a grounded answer. A REJECTED program, a
        # grounded:false run, or a never-grounded outcome are all the honest "I can't answer that".
        passed = base.get("grounded") is not True
    else:
        passed = TIER_RANK[reached_tier] >= TIER_RANK[base["target_tier"]]
    return CaseResult(reached_tier=reached_tier, passed=passed, detail=detail, **base)


def _subject_note(case: dict, findings: dict) -> str:
    """For temporal/first-goal cases, log whether the verdict string names the question's subject
    (e.g. '#7'). Logged, NOT gating — subject phrasing is the model's, not ground truth."""
    if case["shape"] != "first-goal":
        return ""
    import re

    q_nums = re.findall(r"#?(\d+)", case["question"])
    verdict = findings.get("verdict") or ""
    for n in q_nums:
        if n in verdict:
            return f"verdict names subject #{n}"
    return "verdict subject not confirmed (logged, not gating)"
