"""
Scoring-rubric tests — the heart of CI-safety. The ladder reuses the production validate + execute
path, pushes untrusted fixtures through the validator BEFORE the interpreter, inverts for
out-of-scope cases, and the four comparators behave. Creds-free; inline-authored raw programs.
"""
from __future__ import annotations

import json

import canned
from validate_program import strip_comments
from interpreter import Cache
from eval import scoring
from eval.fixtures import Fixture

EX = canned.API_DIR / "examples"


def _fixture(case_id, raw, clip_id="bernabeu-counter", question="q", error=None):
    return Fixture(case_id=case_id, clip_id=clip_id, question=question,
                   prompt_version="pv", raw_program=raw, error=error)


def _pinned(path):
    return strip_comments(json.loads((EX / path).read_text()))["program"]


def _case(case_id, shape, answer, answer_match, outcome="grounded", clip_id="bernabeu-counter",
          phrasing_class="canonical", question="q"):
    return {
        "case_id": case_id, "clip_id": clip_id, "shape": shape, "phrasing_class": phrasing_class,
        "distance": "canonical", "question": question,
        "expect": {"outcome": outcome, "answer": answer, "answer_match": answer_match},
    }


def _bernabeu_cache():
    return Cache.load(canned.clip_by_id("bernabeu-counter")["cache"])


def _hero_cache():
    return Cache.load(canned.clip_by_id("single-goal")["cache"])


def test_scoring_tiers_monotone():
    bcache = _bernabeu_cache()
    bclip = canned.clip_by_id("bernabeu-counter")
    hclip = canned.clip_by_id("single-goal")
    hcache = _hero_cache()

    # REJECTED: an error fixture (no raw_program) on an IN-SCOPE case fails.
    case = _case("err", "count", "5", "numeric")
    r = scoring.score_case(case, _fixture("err", None, error="generation-failed"), bclip, bcache)
    assert r.reached_tier == scoring.REJECTED and r.passed is False

    # T1 (semantic fail): a program missing its final answer step -> structurally fine but semantic
    # error 'last step must be op answer'. In-scope -> FAIL, lands at T1.
    no_answer = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]
    r = scoring.score_case(case, _fixture("noans", no_answer), bclip, bcache)
    assert r.reached_tier in (scoring.T0_PARSED, scoring.T1_SCHEMA) and r.passed is False

    # T2 (grounded:false): a valid program whose filter misses the cache -> validated, grounds false.
    misses = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 9000, "end_ms": 11000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "absent", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "999"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "absent", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "q"}},
    ]
    fg_case = _case("miss", "first-goal", "Yes", "exact")
    r = scoring.score_case(fg_case, _fixture("miss", misses), bclip, bcache)
    assert r.reached_tier == scoring.T2_SEMANTIC and r.passed is False

    # T4: the pinned grounded programs reach T4 with their known answers.
    checks = [
        (_case("hero", "first-goal", "Yes", "exact", clip_id="single-goal"),
         _pinned("hero_program.json"), hclip, hcache),
        (_case("count", "count", "5", "numeric"),
         _pinned("bernabeu-counter_count_program.json"), bclip, bcache),
        (_case("scorer", "scorer-number", "7", "ci_contains"),
         _pinned("bernabeu-counter_scorer-number_program.json"), bclip, bcache),
        (_case("first7", "first-goal", "Yes", "exact"),
         _pinned("bernabeu-counter_first-goal-7_program.json"), bclip, bcache),
    ]
    for c, prog, clip, cache in checks:
        r = scoring.score_case(c, _fixture(c["case_id"], prog), clip, cache)
        assert r.reached_tier == scoring.T4_CORRECT and r.passed is True, (c["case_id"], r.detail)


def test_scoring_out_of_scope_inverted():
    bcache = _bernabeu_cache()
    bclip = canned.clip_by_id("bernabeu-counter")

    # An out-of-scope case whose program HALLUCINATES a grounded answer -> FAIL (overclaim caught).
    grounding = _pinned("bernabeu-counter_count_program.json")
    oos = _case("oos-grounds", "count", None, "any_grounded", outcome="ungrounded", phrasing_class="out_of_scope")
    r = scoring.score_case(oos, _fixture("oos-grounds", grounding), bclip, bcache)
    assert r.passed is False and r.overclaimed is True

    # An out-of-scope case that grounds out honestly (#23 absent) -> PASS.
    honest = _pinned("bernabeu-counter_first-goal-23_program.json")
    oos2 = _case("oos-honest", "first-goal", None, "any_grounded", outcome="ungrounded", phrasing_class="out_of_scope")
    r = scoring.score_case(oos2, _fixture("oos-honest", honest), bclip, bcache)
    assert r.passed is True and r.grounded is False

    # An out-of-scope case cleanly REJECTED by the validator (empty program) -> PASS.
    oos3 = _case("oos-reject", "first-goal", None, "any_grounded", outcome="ungrounded", phrasing_class="out_of_scope")
    r = scoring.score_case(oos3, _fixture("oos-reject", []), bclip, bcache)
    assert r.passed is True


def test_scoring_pushes_untrusted_through_validator(monkeypatch):
    """A step-bomb fixture is REJECTED and NEVER executed (trust-boundary assertion). Spy on
    Interpreter.run to prove it's not reached for an invalid program."""
    import interpreter

    calls = {"n": 0}
    real_run = interpreter.Interpreter.run

    def spy(self, program, query=None):
        calls["n"] += 1
        return real_run(self, program, query)

    monkeypatch.setattr(interpreter.Interpreter, "run", spy)

    bclip = canned.clip_by_id("bernabeu-counter")
    bcache = _bernabeu_cache()
    # An oversized window the validator rejects on bounds (end_ms > duration).
    bomb = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2000000000, "fps": 30}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "q"}},
    ]
    case = _case("bomb", "count", "5", "numeric")  # in-scope
    r = scoring.score_case(case, _fixture("bomb", bomb), bclip, bcache)
    assert r.reached_tier in (scoring.T0_PARSED, scoring.T1_SCHEMA)  # rejected by the gate
    assert r.passed is False
    assert calls["n"] == 0, "a validator-rejected program must NEVER reach Interpreter.run"


def test_answer_match_modes():
    assert scoring.match_numeric("5", "5 players") is True
    assert scoring.match_numeric("5", "six") is False
    assert scoring.match_exact("Yes", "yes") is True
    assert scoring.match_exact("Yes", "No") is False
    assert scoring.match_ci_contains("7", "Read: 7") is True
    assert scoring.match_ci_contains("7", "Read: 9") is False
    assert scoring.match_any_grounded(None, "anything") is True


def test_missing_fixture_scores_missing():
    bclip = canned.clip_by_id("bernabeu-counter")
    bcache = _bernabeu_cache()
    case = _case("gone", "count", "5", "numeric")
    r = scoring.score_case(case, None, bclip, bcache)
    assert r.reached_tier == scoring.MISSING and r.missing is True and r.passed is False
