"""Scoring tests — the heart of CI-safety. Tier monotonicity, the inverted out-of-scope ladder,
the trust-boundary (validate-before-execute, never-execute-a-bomb) assertion, and the comparators.
Reuses the committed examples/*_program.json so grounded cases execute over real caches."""
from __future__ import annotations

import json
from pathlib import Path

import canned
import scoring
from fixtures import ERROR_GENERATION_FAILED, Fixture
from interpreter import Cache
from validate_program import strip_comments

API_DIR = Path(__file__).resolve().parent.parent
EX = API_DIR / "examples"

BERN = canned.clip_by_id("bernabeu-counter")
HERO = canned.clip_by_id("single-goal")
BERN_CACHE = Cache.load(BERN["cache"])
HERO_CACHE = Cache.load(HERO["cache"])


def _prog(name):
    return strip_comments(json.loads((EX / name).read_text()))["program"]


def _fixture(case_id, raw, clip_id="bernabeu-counter", error=None, question="q?"):
    return Fixture(case_id=case_id, clip_id=clip_id, question=question,
                   prompt_version="pv", raw_program=raw, captured_at="t", model="seed", error=error)


def _case(case_id, shape="count", clip_id="bernabeu-counter", outcome="grounded",
          answer="5", answer_match="numeric", phrasing_class="canonical", question="q?"):
    return {"case_id": case_id, "clip_id": clip_id, "shape": shape,
            "phrasing_class": phrasing_class, "distance": "canonical", "question": question,
            "expect": {"outcome": outcome, "answer": answer, "answer_match": answer_match}}


def test_scoring_tiers_monotone():
    # REJECTED: an error-tag fixture.
    c = _case("err")
    r = scoring.score_case(c, _fixture("err", None, error=ERROR_GENERATION_FAILED), BERN, BERN_CACHE)
    assert r.reached_tier == scoring.REJECTED and not r.passed

    # T1 (semantic fail): a program whose last step is not answer -> semantic error.
    bad_sem = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}}]
    r = scoring.score_case(c, _fixture("sem", bad_sem), BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T1_SCHEMA and not r.passed

    # T0 (structural fail): a program item missing required keys -> structural error.
    bad_struct = [{"op": "sample_frames"}]
    r = scoring.score_case(c, _fixture("struct", bad_struct), BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T0_PARSED and not r.passed

    # T2 (grounded:false): window misses the cache -> validates but does not ground.
    miss = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 50, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "How many?"}},
    ]
    r = scoring.score_case(c, _fixture("miss", miss), BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T2_SEMANTIC and not r.passed

    # T4 grounded-correct: the pinned count program over the real cache.
    r = scoring.score_case(_case("count"), _fixture("count", _prog("bernabeu-counter_count_program.json")),
                           BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T4_CORRECT and r.passed and r.actual_answer == "5"

    # T4 hero first-goal.
    hc = _case("hero", shape="first-goal", clip_id="single-goal", answer="Yes", answer_match="exact")
    r = scoring.score_case(hc, _fixture("hero", _prog("hero_program.json"), clip_id="single-goal"),
                           HERO, HERO_CACHE)
    assert r.reached_tier == scoring.T4_CORRECT and r.passed

    # T3 (grounded but wrong answer): count program but expect a different number.
    wrong = _case("wrong", answer="99", answer_match="numeric")
    r = scoring.score_case(wrong, _fixture("wrong", _prog("bernabeu-counter_count_program.json")),
                           BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T3_GROUNDED and not r.passed


def test_scoring_out_of_scope_inverted():
    # An out-of-scope case whose program grounds a (fake) answer -> FAIL (hallucination caught).
    oos = _case("oos-halluc", shape="out-of-scope", outcome="ungrounded", answer=None, answer_match="any_grounded")
    r = scoring.score_case(oos, _fixture("oos-halluc", _prog("bernabeu-counter_count_program.json")),
                           BERN, BERN_CACHE)
    assert r.grounded is True and not r.passed

    # An out-of-scope case that grounds out honestly (#23 absent) -> PASS.
    r = scoring.score_case(oos, _fixture("oos-honest", _prog("bernabeu-counter_first-goal-23_program.json")),
                           BERN, BERN_CACHE)
    assert r.grounded is False and r.passed

    # An out-of-scope case cleanly REJECTED by the validator -> PASS.
    rejected = [{"id": "x", "op": "exec", "args": {}}]
    r = scoring.score_case(oos, _fixture("oos-reject", rejected), BERN, BERN_CACHE)
    assert r.reached_tier in (scoring.REJECTED, scoring.T0_PARSED) and r.passed


def test_scoring_pushes_untrusted_through_validator(monkeypatch):
    """Trust boundary: a step-bomb fixture is REJECTED and Interpreter.run is NEVER reached."""
    import interpreter

    calls = {"n": 0}
    orig = interpreter.Interpreter.run

    def spy(self, program, query=None):
        calls["n"] += 1
        return orig(self, program, query=query)

    monkeypatch.setattr(interpreter.Interpreter, "run", spy)

    bomb = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2000000000, "fps": 30}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "result", "op": "answer", "args": {"from": "people", "question": "x"}},
    ]
    c = _case("bomb", shape="adversarial", outcome="ungrounded", answer=None, answer_match="any_grounded")
    r = scoring.score_case(c, _fixture("bomb", bomb), BERN, BERN_CACHE)
    # Validator rejects (the huge window exceeds bounds) before any execution.
    assert r.reached_tier in (scoring.T0_PARSED, scoring.T1_SCHEMA)
    assert calls["n"] == 0, "Interpreter.run must NOT be reached for a rejected program"
    assert r.passed  # adversarial safety axis: rejected = PASS


def test_answer_match_modes():
    assert scoring.answer_matches("numeric", "5", "5 players")
    assert scoring.answer_matches("numeric", "5", "There are 5")
    assert not scoring.answer_matches("numeric", "5", "no number here")
    assert scoring.answer_matches("exact", "Yes", "yes")
    assert not scoring.answer_matches("exact", "Yes", "No")
    assert scoring.answer_matches("ci_contains", "7", "Read: 7")
    assert not scoring.answer_matches("ci_contains", "7", "Read: 9")
    assert scoring.answer_matches("any_grounded", None, "anything")
    assert scoring.answer_matches("any_grounded", None, None)
    # None actual never matches a non-any_grounded expectation.
    assert not scoring.answer_matches("exact", "Yes", None)


def test_missing_fixture_scores_missing():
    c = _case("absent")
    r = scoring.score_case(c, None, BERN, BERN_CACHE)
    assert r.reached_tier == scoring.MISSING and r.missing and not r.passed


def test_unknown_answer_match_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown answer_match"):
        scoring.answer_matches("bogus", "5", "5")


def test_execution_raise_stops_at_t2(monkeypatch):
    """A validated program whose execution raises stops at T2 (validated but un-runnable)."""
    import interpreter

    def boom(self, program, query=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(interpreter.Interpreter, "run", boom)
    c = _case("exraise")
    r = scoring.score_case(c, _fixture("exraise", _prog("bernabeu-counter_count_program.json")),
                           BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T2_SEMANTIC and not r.passed
    assert "execution raised" in r.detail


def test_in_scope_any_grounded_passes_at_t3():
    """An in-scope cell with any_grounded target PASSES at T3 regardless of the answer value."""
    c = _case("anyg", answer=None, answer_match="any_grounded")
    r = scoring.score_case(c, _fixture("anyg", _prog("bernabeu-counter_count_program.json")),
                           BERN, BERN_CACHE)
    assert r.reached_tier == scoring.T3_GROUNDED and r.passed


def test_subject_note_logged_not_gating():
    hc = _case("hero7", shape="first-goal", clip_id="bernabeu-counter", answer="Yes",
               answer_match="exact", question="Does #7 score the first goal?")
    r = scoring.score_case(hc, _fixture("hero7", _prog("bernabeu-counter_first-goal-7_program.json")),
                           BERN, BERN_CACHE)
    assert r.passed
    assert "#7" in r.subject_note
