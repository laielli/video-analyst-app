"""Scoring tests — the heart of CI-safety. Proves the tier ladder is monotone, the out-of-scope
ladder inverts, untrusted fixtures flow through the validator BEFORE the interpreter, and the
four answer comparators behave. Creds-free; reuses the committed examples/*_program.json and the
seed fixtures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = API_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import canned  # noqa: E402
import scoring  # noqa: E402
from fixtures import Fixture  # noqa: E402
from interpreter import Cache  # noqa: E402
from validate_program import strip_comments  # noqa: E402


def _pinned(name: str) -> list:
    raw = json.loads((API_DIR / "examples" / name).read_text())
    return strip_comments(raw)["program"]


def _fixture(case_id, clip_id, question, raw_program, error=None):
    return Fixture(case_id=case_id, clip_id=clip_id, question=question,
                   prompt_version="pv", captured_at="t", model="m",
                   raw_program=raw_program, usage=None, error=error)


def _case(case_id, clip_id, shape, question, outcome, answer, answer_match, pc="canonical"):
    return {
        "case_id": case_id, "clip_id": clip_id, "shape": shape, "phrasing_class": pc,
        "distance": "canonical", "question": question,
        "expect": {"outcome": outcome, "answer": answer, "answer_match": answer_match},
    }


def _score(case, fixture, is_fresh=True):
    clip = canned.clip_by_id(case["clip_id"])
    cache = Cache.load(clip["cache"]) if clip else None
    return scoring.score_case(case, fixture, clip, cache, is_fresh=is_fresh)


# ---- tier ladder monotonicity ------------------------------------------------------------

def test_scoring_tiers_monotone():
    ber = "bernabeu-counter"

    # REJECTED: an error fixture (unparseable) -> no raw_program.
    c = _case("err", ber, "count", "How many players?", "grounded", "5", "numeric")
    r = _score(c, _fixture("err", ber, "q", None, error="unparseable"))
    assert r.reached_tier == scoring.REJECTED and r.passed is False  # in-scope REJECTED = FAIL

    # T1 only: a program that is structurally valid but semantically broken (last step not answer).
    bad_semantic = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]
    c = _case("t1", ber, "count", "How many players?", "grounded", "5", "numeric")
    r = _score(c, _fixture("t1", ber, "q", bad_semantic))
    assert r.reached_tier == scoring.T1 and r.passed is False

    # T2 only: a valid program whose window MISSES the cache -> grounds:false.
    miss = _pinned("bernabeu-counter_count_program.json")
    miss = json.loads(json.dumps(miss))
    miss[0]["args"] = {"start_ms": 100, "end_ms": 200, "fps": 8}  # no detections there
    c = _case("t2", ber, "count", "How many players?", "grounded", "5", "numeric")
    r = _score(c, _fixture("t2", ber, "q", miss))
    assert r.reached_tier == scoring.T2 and r.grounded is False and r.passed is False

    # T4: the four pinned grounded programs reach T4 with their known answers.
    for case_id, clip_id, shape, q, ans, match, prog in (
        ("hero", "single-goal", "first-goal", "Does #10 score the first goal?", "Yes", "exact", "hero_program.json"),
        ("count", ber, "count", "How many players?", "5", "numeric", "bernabeu-counter_count_program.json"),
        ("seven", ber, "first-goal", "Does #7 score the first goal?", "Yes", "exact", "bernabeu-counter_first-goal-7_program.json"),
        ("scorer", ber, "scorer-number", "What number does the scorer wear?", "7", "ci_contains", "bernabeu-counter_scorer-number_program.json"),
    ):
        c = _case(case_id, clip_id, shape, q, "grounded", ans, match)
        r = _score(c, _fixture(case_id, clip_id, q, _pinned(prog)))
        assert r.reached_tier == scoring.T4, f"{case_id} reached {r.reached_tier}"
        assert r.passed is True


def test_scoring_t3_only_when_answer_wrong():
    """A grounded program whose answer is wrong reaches T3, not T4."""
    ber = "bernabeu-counter"
    c = _case("wrong", ber, "count", "How many players?", "grounded", "99", "numeric")
    r = _score(c, _fixture("wrong", ber, "q", _pinned("bernabeu-counter_count_program.json")))
    assert r.reached_tier == scoring.T3 and r.passed is False


# ---- out-of-scope inversion --------------------------------------------------------------

def test_scoring_out_of_scope_inverted():
    ber = "bernabeu-counter"

    # An out-of-scope case whose program GROUNDS a (fake) answer -> FAIL (hallucination caught).
    c = _case("oos-grounded", ber, "out-of-scope", "What's the weather?", "ungrounded", None, "any_grounded", pc="out_of_scope")
    r = _score(c, _fixture("oos-grounded", ber, "q", _pinned("bernabeu-counter_count_program.json")))
    assert r.grounded is True and r.passed is False

    # One that grounds out honestly (#23-style non-grounding program) -> PASS.
    c = _case("oos-honest", ber, "out-of-scope", "What's the weather?", "ungrounded", None, "any_grounded", pc="out_of_scope")
    r = _score(c, _fixture("oos-honest", ber, "q", _pinned("bernabeu-counter_first-goal-23_program.json")))
    assert r.grounded is False and r.passed is True

    # One cleanly REJECTED by the validator (non-whitelisted op) -> PASS.
    bomb = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "x", "op": "exec", "args": {"frames": "frames"}},
        {"id": "result", "op": "answer", "args": {"from": "x", "question": "What's the weather?"}},
    ]
    c = _case("oos-rejected", ber, "out-of-scope", "What's the weather?", "ungrounded", None, "any_grounded", pc="out_of_scope")
    r = _score(c, _fixture("oos-rejected", ber, "q", bomb))
    assert r.reached_tier == scoring.REJECTED and r.passed is True


# ---- the trust-boundary assertion --------------------------------------------------------

def test_scoring_pushes_untrusted_through_validator():
    """A step-bomb fixture is REJECTED and NEVER executed — the trust-boundary assertion. Spy on
    the interpreter run_fn to prove it is never called when validation fails."""
    ber = "bernabeu-counter"
    executed = {"count": 0}

    def spy_run(program, cache, question):
        executed["count"] += 1
        from interpreter import Interpreter
        return Interpreter(cache).run(program, query=question)

    # An oversized window: validator rejects on numeric bounds / frame cap BEFORE execute.
    bomb = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2000000000, "fps": 30}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "How many?"}},
    ]
    c = _case("bomb", ber, "count", "How many?", "grounded", "5", "numeric")
    clip = canned.clip_by_id(ber)
    cache = Cache.load(clip["cache"])
    r = scoring.score_case(c, _fixture("bomb", ber, "q", bomb), clip, cache, run_fn=spy_run)
    assert r.reached_tier == scoring.REJECTED
    assert executed["count"] == 0, "a rejected program must NEVER be executed (trust boundary)"


# ---- comparators -------------------------------------------------------------------------

def test_answer_match_modes():
    assert scoring.match_numeric("5", "5 players") is True
    assert scoring.match_numeric("5", "five players") is False
    assert scoring.match_exact("Yes", "yes") is True
    assert scoring.match_exact("Yes", "No") is False
    assert scoring.match_ci_contains("7", "Read: 7") is True
    assert scoring.match_ci_contains("7", "no digits") is False
    assert scoring.match_any_grounded(None, "anything") is True
    # dispatcher
    assert scoring.check_answer("numeric", "5", "5 found") is True
    assert scoring.check_answer("exact", "Yes", "No") is False
