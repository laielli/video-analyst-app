"""Scoring tests — the heart of CI-safety. Proves the rubric reuses the production gate, runs
validate BEFORE execute (trust boundary), inverts for out-of-scope, and the comparators behave."""
from __future__ import annotations

import json

import canned
from eval import scoring
from eval.bank import derive_ground_truth
from eval.fixtures import ERROR_GENERATION_FAILED, Fixture
from interpreter import Cache, Interpreter
from validate_program import strip_comments

EXAMPLES = canned.API_DIR / "examples"
BERN_CACHE = str(EXAMPLES / "bernabeu-counter_cache.json")
HERO_CACHE = str(EXAMPLES / "hero_cache.json")


def _pinned(name):
    return strip_comments(json.loads((EXAMPLES / name).read_text()))["program"]


def _fixture(case_id, clip_id, raw_program, error=None):
    return Fixture(case_id=case_id, clip_id=clip_id, question="q", prompt_version="pv",
                   raw_program=raw_program, error=error, usage=None)


def _grounded_case(shape, answer, answer_match, clip_id="bernabeu-counter"):
    return {"case_id": "c", "clip_id": clip_id, "shape": shape, "phrasing_class": "paraphrase",
            "distance": "near", "question": "q",
            "expect": {"outcome": "grounded", "answer": answer, "answer_match": answer_match}}


def _oos_case(clip_id="bernabeu-counter", phrasing_class="out_of_scope"):
    return {"case_id": "c", "clip_id": clip_id, "shape": "none", "phrasing_class": phrasing_class,
            "distance": "n/a", "question": "q",
            "expect": {"outcome": "ungrounded", "answer": None, "answer_match": "any_grounded"}}


def test_scoring_tiers_monotone():
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)

    # REJECTED: an error-tag fixture (no raw_program).
    r = scoring.score_case(_grounded_case("count", "5", "numeric"),
                           _fixture("c", "bernabeu-counter", None, error=ERROR_GENERATION_FAILED),
                           clip, cache)
    assert r.reached_tier == scoring.REJECTED and r.passed is False

    # REJECTED (structural): a non-whitelisted op.
    bad_struct = [{"id": "x", "op": "exec", "args": {}},
                  {"id": "r", "op": "answer", "args": {"from": "x", "question": "q"}}]
    r = scoring.score_case(_grounded_case("count", "5", "numeric"),
                           _fixture("c", "bernabeu-counter", bad_struct), clip, cache)
    assert r.reached_tier == scoring.REJECTED and r.passed is False

    # T1 (schema-valid) but semantic fail: a valid step shape with no answer-last.
    no_answer = [{"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
    r = scoring.score_case(_grounded_case("count", "5", "numeric"),
                           _fixture("c", "bernabeu-counter", no_answer), clip, cache)
    assert r.reached_tier == scoring.T1_SCHEMA and r.passed is False

    # T2 (semantic-valid) but grounds:false: window misses the cache (count over an empty frame).
    misses = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 100, "end_ms": 200, "fps": 1}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "p"}},
        {"id": "r", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    r = scoring.score_case(_grounded_case("count", "5", "numeric"),
                           _fixture("c", "bernabeu-counter", misses), clip, cache)
    # An empty count grounds to "0" (a number) — still grounded, just wrong answer. So T3 not T4.
    assert r.reached_tier in (scoring.T2_SEMANTIC, scoring.T3_GROUNDED)

    # T4 (answer-correct): the pinned grounded programs over their real caches.
    cases_progs = [
        (_grounded_case("first-goal", "Yes", "exact", "single-goal"),
         _pinned("hero_program.json"), HERO_CACHE, "single-goal"),
        (_grounded_case("count", "5", "numeric"),
         _pinned("bernabeu-counter_count_program.json"), BERN_CACHE, "bernabeu-counter"),
        (_grounded_case("first-goal", "Yes", "exact"),
         _pinned("bernabeu-counter_first-goal-7_program.json"), BERN_CACHE, "bernabeu-counter"),
        (_grounded_case("scorer-number", "7", "ci_contains"),
         _pinned("bernabeu-counter_scorer-number_program.json"), BERN_CACHE, "bernabeu-counter"),
    ]
    for case, prog, cache_path, clip_id in cases_progs:
        c = canned.clip_by_id(clip_id)
        r = scoring.score_case(case, _fixture("c", clip_id, prog), c, Cache.load(cache_path))
        assert r.reached_tier == scoring.T4_CORRECT, (clip_id, case["shape"], r.detail)
        assert r.passed is True


def test_scoring_out_of_scope_inverted():
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)

    # An out-of-scope case whose program GROUNDS a fake answer -> FAIL (hallucination caught).
    hallucinate = _pinned("bernabeu-counter_count_program.json")
    r = scoring.score_case(_oos_case(), _fixture("c", "bernabeu-counter", hallucinate), clip, cache)
    assert r.passed is False, "a grounded answer for an ungrounded case is an overclaim FAIL"
    assert "OVERCLAIM" in r.detail

    # One that grounds out HONESTLY (filter on absent value) -> PASS.
    honest = _pinned("bernabeu-counter_first-goal-23_program.json")
    r = scoring.score_case(_oos_case(), _fixture("c", "bernabeu-counter", honest), clip, cache)
    assert r.passed is True

    # One cleanly REJECTED (non-whitelisted op) -> PASS.
    rejected = [{"id": "x", "op": "exec", "args": {}},
                {"id": "r", "op": "answer", "args": {"from": "x", "question": "q"}}]
    r = scoring.score_case(_oos_case(phrasing_class="injection"),
                           _fixture("c", "bernabeu-counter", rejected), clip, cache)
    assert r.passed is True and r.reached_tier == scoring.REJECTED


def test_scoring_pushes_untrusted_through_validator():
    """A step-bomb fixture is REJECTED and NEVER executed (trust-boundary assertion). Spy on the
    interpreter run callable: it must not be invoked for an invalid program."""
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)

    calls = []

    def spy_run(program, query=None):
        calls.append(program)
        return Interpreter(cache).run(program, query=query)

    # Oversized window (fails the numeric-bounds gate -> REJECTED before execute).
    bomb = [
        {"id": "f", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2000000000, "fps": 30}},
        {"id": "p", "op": "detect", "args": {"frames": "f", "classes": ["person"]}},
        {"id": "r", "op": "answer", "args": {"from": "p", "question": "q"}},
    ]
    r = scoring.score_case(_grounded_case("presence", "Yes", "exact"),
                           _fixture("c", "bernabeu-counter", bomb), clip, cache,
                           interpreter_run=spy_run)
    assert r.reached_tier == scoring.REJECTED
    assert calls == [], "an invalid program must NEVER reach the interpreter"


def test_answer_match_modes():
    assert scoring.match_numeric("5", "5 players") is True
    assert scoring.match_numeric("5", "found 5 on screen") is True
    assert scoring.match_numeric("5", "no digits here") is False
    assert scoring.match_exact("Yes", "yes") is True
    assert scoring.match_exact("Yes", "No") is False
    assert scoring.match_ci_contains("7", "Read: 7") is True
    assert scoring.match_ci_contains("7", "no number") is False
    assert scoring.check_answer("any_grounded", None, "whatever") is True
    assert scoring.check_answer("bogus", "5", "5") is False


def test_score_case_missing_fixture():
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)
    r = scoring.score_case(_grounded_case("count", "5", "numeric"), None, clip, cache)
    assert r.reached_tier == scoring.MISSING and r.missing is True and r.passed is False


def test_score_case_in_scope_wrong_answer_is_T3_fail():
    """A grounded program with the WRONG answer reaches T3 but fails T4."""
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)
    prog = _pinned("bernabeu-counter_count_program.json")  # grounds to "5"
    r = scoring.score_case(_grounded_case("count", "99", "numeric"),
                           _fixture("c", "bernabeu-counter", prog), clip, cache)
    assert r.reached_tier == scoring.T3_GROUNDED and r.passed is False


def test_in_scope_any_grounded_passes_at_T3():
    """The absent-subject #23 case is in-scope-shaped but expects ungrounded; an any_grounded
    in-scope grounded program passes at T3."""
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)
    case = _grounded_case("count", None, "any_grounded")
    prog = _pinned("bernabeu-counter_count_program.json")
    r = scoring.score_case(case, _fixture("c", "bernabeu-counter", prog), clip, cache)
    assert r.reached_tier == scoring.T3_GROUNDED and r.passed is True


def test_derive_ground_truth_values():
    assert derive_ground_truth(EXAMPLES / "bernabeu-counter_count_program.json", BERN_CACHE) == "5"
    assert derive_ground_truth(EXAMPLES / "hero_program.json", HERO_CACHE) == "Yes"


def test_scoring_execution_exception_is_handled():
    """A program that validates but raises at execution -> caught, scored as T2 (in-scope FAIL).
    Inject a raising interpreter_run after a real validating program."""
    clip = canned.clip_by_id("bernabeu-counter")
    cache = Cache.load(BERN_CACHE)
    prog = _pinned("bernabeu-counter_count_program.json")

    def boom(program, query=None):
        raise RuntimeError("kaboom")

    r = scoring.score_case(_grounded_case("count", "5", "numeric"),
                           _fixture("c", "bernabeu-counter", prog), clip, cache,
                           interpreter_run=boom)
    assert r.reached_tier == scoring.T2_SEMANTIC and r.passed is False
    assert "execution raised" in r.detail


def test_match_numeric_bad_expected():
    assert scoring.match_numeric("not-a-number", "5") is False


def test_program_byte_size_handles_unserializable():
    assert scoring._program_byte_size(None) == 0
    assert scoring._program_byte_size([{"x": object()}]) == 0
