"""
Scoring tests (D4) — the heart of CI-safety. Proves the rubric reuses the production gate, runs
validate BEFORE execute (the trust boundary), inverts for out-of-scope, and the comparators behave.
"""
from __future__ import annotations

import json

import pytest

import canned
from eval import scoring
from eval.fixtures import Fixture
from interpreter import Cache, Interpreter
from validate_program import strip_comments


def _clip_and_cache(clip_id):
    clip = canned.clip_by_id(clip_id)
    return clip, Cache.load(clip["cache"])


def _pinned(name):
    raw = json.loads(open(f"examples/{name}").read())
    return strip_comments(raw)["program"]


def _case(case_id="c", clip_id="bernabeu-counter", shape="count", outcome="grounded",
          answer="5", match="numeric", question="How many players are visible?"):
    return {"case_id": case_id, "clip_id": clip_id, "shape": shape, "phrasing_class": "canonical",
            "distance": "canonical", "question": question,
            "expect": {"outcome": outcome, "answer": answer, "answer_match": match}}


def _fixture(case, raw_program=None, error=None):
    return Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                   prompt_version="pv", raw_program=raw_program, error=error)


def test_scoring_tiers_monotone():
    """Feed programs that stop at each tier and assert the reached tier."""
    clip, cache = _clip_and_cache("bernabeu-counter")

    # error fixture -> REJECTED (no parsed output).
    case = _case()
    r = scoring.score_case(case, _fixture(case, error="generation-failed"), clip, cache)
    assert r.reached_tier == scoring.REJECTED and r.passed is False

    # structurally invalid (non-whitelisted op) -> REJECTED (below T1).
    bomb = [{"id": "f", "op": "exec", "args": {}}, {"id": "a", "op": "answer", "args": {"from": "f", "question": "x"}}]
    r = scoring.score_case(case, _fixture(case, raw_program=bomb), clip, cache)
    assert r.reached_tier == scoring.REJECTED

    # schema-valid but semantically broken (no answer last step) -> T1.
    no_answer = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 1}}]
    r = scoring.score_case(case, _fixture(case, raw_program=no_answer), clip, cache)
    assert r.reached_tier == scoring.T1_SCHEMA

    # valid but window misses the cache -> T2 (grounded:false), in-scope FAIL.
    miss = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 100, "end_ms": 200, "fps": 1}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "people"}},
            {"id": "result", "op": "answer", "args": {"from": "n", "question": "x"}}]
    r = scoring.score_case(case, _fixture(case, raw_program=miss), clip, cache)
    assert r.reached_tier == scoring.T2_SEMANTIC and r.passed is False

    # pinned count program -> T4 with the known answer '5'.
    r = scoring.score_case(case, _fixture(case, raw_program=_pinned("bernabeu-counter_count_program.json")),
                           clip, cache)
    assert r.reached_tier == scoring.T4_CORRECT and r.passed is True


@pytest.mark.parametrize("name,clip_id,shape,answer,match", [
    ("hero_program.json", "single-goal", "first-goal", "Yes", "exact"),
    ("bernabeu-counter_count_program.json", "bernabeu-counter", "count", "5", "numeric"),
    ("bernabeu-counter_first-goal-7_program.json", "bernabeu-counter", "first-goal", "Yes", "exact"),
    ("bernabeu-counter_scorer-number_program.json", "bernabeu-counter", "scorer-number", "7", "ci_contains"),
])
def test_pinned_programs_score_t4(name, clip_id, shape, answer, match):
    """The four pinned grounded programs score T4 with their known answers."""
    clip, cache = _clip_and_cache(clip_id)
    case = _case(clip_id=clip_id, shape=shape, answer=answer, match=match, question="q")
    r = scoring.score_case(case, _fixture(case, raw_program=_pinned(name)), clip, cache)
    assert r.reached_tier == scoring.T4_CORRECT and r.passed is True


def test_absent_subject_23_passes_on_grounded_false():
    """The #23 absent-subject program grounds out honestly -> PASS (ungrounded contract)."""
    clip, cache = _clip_and_cache("bernabeu-counter")
    case = _case(case_id="fg23", shape="first-goal", outcome="ungrounded", answer=None,
                 match="any_grounded", question="Does #23 score the first goal?")
    r = scoring.score_case(case, _fixture(case, raw_program=_pinned("bernabeu-counter_first-goal-23_program.json")),
                           clip, cache)
    assert r.passed is True and scoring.score_case is not None
    assert r.reached_tier == scoring.T2_SEMANTIC  # grounded:false lands at T2 (the inverted PASS)


def test_scoring_out_of_scope_inverted():
    """Out-of-scope: a hallucinated grounded answer FAILS; honest no-ground / clean reject PASS."""
    clip, cache = _clip_and_cache("bernabeu-counter")
    oos = _case(case_id="oos", outcome="ungrounded", answer=None, match="any_grounded",
                question="What's the weather?")

    # Grounds a real answer for an out-of-scope question -> hallucination -> FAIL.
    r = scoring.score_case(oos, _fixture(oos, raw_program=_pinned("bernabeu-counter_count_program.json")),
                           clip, cache)
    assert r.passed is False and "HALLUCINATED" in r.detail

    # Grounds out honestly (window misses cache) -> PASS.
    miss = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 100, "end_ms": 200, "fps": 1}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "people"}},
            {"id": "result", "op": "answer", "args": {"from": "n", "question": "x"}}]
    r = scoring.score_case(oos, _fixture(oos, raw_program=miss), clip, cache)
    assert r.passed is True

    # Cleanly rejected by the validator -> PASS.
    bomb = [{"id": "f", "op": "exec", "args": {}}, {"id": "a", "op": "answer", "args": {"from": "f", "question": "x"}}]
    r = scoring.score_case(oos, _fixture(oos, raw_program=bomb), clip, cache)
    assert r.passed is True and r.reached_tier == scoring.REJECTED


def test_scoring_pushes_untrusted_through_validator(monkeypatch):
    """A bomb fixture is REJECTED and Interpreter.run is NEVER called (trust boundary)."""
    clip, cache = _clip_and_cache("bernabeu-counter")
    # An OOM-window program: rejected by the numeric-bounds gate before any execution.
    bomb = [{"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "n", "op": "count", "args": {"items": "people"}},
            {"id": "result", "op": "answer", "args": {"from": "n", "question": "x"}}]
    called = {"n": 0}
    real_run = Interpreter.run

    def spy(self, program, query=None):
        called["n"] += 1
        return real_run(self, program, query)

    monkeypatch.setattr(Interpreter, "run", spy)
    case = _case()
    r = scoring.score_case(case, _fixture(case, raw_program=bomb), clip, cache)
    assert r.reached_tier == scoring.REJECTED
    assert called["n"] == 0, "an invalid program must never reach Interpreter.run (trust boundary)"


def test_answer_match_modes():
    f = scoring._answer_matches
    assert f("5", "5 players found", "numeric") is True
    assert f("5", "five", "numeric") is False
    assert f("Yes", "yes", "exact") is True
    assert f("Yes", "No", "exact") is False
    assert f("7", "Read: 7", "ci_contains") is True
    assert f("7", "no number", "ci_contains") is False
    assert f("anything", "any value", "any_grounded") is True
    assert f("5", None, "numeric") is False


def test_missing_fixture_is_visible():
    clip, cache = _clip_and_cache("bernabeu-counter")
    case = _case()
    r = scoring.score_case(case, None, clip, cache)
    assert r.missing is True and r.reached_tier == scoring.MISSING and r.passed is False
