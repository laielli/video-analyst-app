"""
Phase 0 (from /codex consult): the interpreter's answer path is generalized so a free-text
question is answered to the question that was ACTUALLY asked (or grounds out honestly), instead
of always emitting the hardcoded #10 first-scorer verdict.

These tests run small DSL programs through the real Interpreter over the committed hero cache,
exercising op_answer's dispatch on the `from` binding kind (ordered / number / texts / detections)
and the honesty regression: a counting question NEVER emits "#10 scored the first goal".

Pattern mirror: test_precompute.py (Cache.load + Interpreter().run over the hero cache).
"""
from __future__ import annotations

from pathlib import Path

from interpreter import Cache, Interpreter

API_DIR = Path(__file__).resolve().parent.parent
HERO_CACHE = API_DIR / "examples" / "hero_cache.json"


def _cache():
    return Cache.load(HERO_CACHE)


# Reusable program prefix: sample the hero window, detect people, crop jerseys, read text.
def _prefix():
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
    ]


def _run(program, query):
    return Interpreter(_cache()).run(program, query=query)


# --------------------------------------------------------------------------------------
# test_count_question_answers_count_not_hero_verdict — the regression for the hardcoded bug
# --------------------------------------------------------------------------------------

def test_count_question_answers_count_not_hero_verdict():
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4625, "end_ms": 4626, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many players are visible?"}},
    ]
    doc = _run(program, "how many players are visible?")
    f = doc["findings"]
    assert f["grounded"] is True
    # the hero cache has 5 person detections at 4625; the answer is about the count.
    assert f["answer"] == "5"
    # NEVER the hardcoded hero verdict.
    assert "#10 scored the first goal" not in (f["verdict"] or "")
    assert "10" not in (f["answer"] or "") or f["answer"] == "5"  # answer is the count, not a jersey


# --------------------------------------------------------------------------------------
# test_temporal_answer_derives_subject_from_filter — subject is derived, not literal #10
# --------------------------------------------------------------------------------------

def test_temporal_answer_derives_subject_from_filter():
    # "did #10 score first?" -> filter text == 10 -> subject derived as #10 from the FILTER's
    # matched jersey value, NOT a hardcoded literal. The verdict must name #10 because the filter
    # matched #10 (the load-bearing derivation), and the same code path with a different filter
    # value would name that value instead.
    program10 = _prefix() + [
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "did #10 score first?"}},
    ]
    doc10 = _run(program10, "did #10 score first?")
    f10 = doc10["findings"]
    assert f10["grounded"] is True
    assert f10["verdict"] == "Yes — #10 scored the first goal"  # subject DERIVED from filter == 10

    # Cross-check the derivation is NOT a literal: filtering on a number NOT in the cache (#7)
    # matches nobody, so op_temporal_order produces an empty subject (subject_label is None).
    # Even though a first goal exists, we never located #7 — so the answer step must GROUND OUT
    # honestly (Q1 -> A) rather than emit a confident "No" about a player we couldn't find, and
    # it must NEVER fabricate a #10 claim.
    program7 = _prefix() + [
        {"id": "sevens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "7"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "sevens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "did #7 score first?"}},
    ]
    doc7 = _run(program7, "did #7 score first?")
    f7 = doc7["findings"]
    assert f7["grounded"] is False
    assert f7["answer"] is None and f7["verdict"] is None
    assert f7["reason"] == "no-grounded-answer"


# --------------------------------------------------------------------------------------
# test_hero_question_still_grounds_to_ten — dedicated regression guard
# --------------------------------------------------------------------------------------

def test_hero_question_still_grounds_to_ten():
    # Generalizing op_answer (Phase 0) must NOT regress the canonical hero demo. This is the
    # standalone guard for that: it pins the hero question's grounded #10 verdict on its own, so
    # the guarantee survives even if test_temporal_answer_derives_subject_from_filter (whose
    # purpose is the DERIVATION, not the demo) is later refactored. Mirrors the dedicated hero
    # checks #8/#12 carried (the embedded assertion above already covers it; this makes it durable).
    program = _prefix() + [
        {"id": "tens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "10"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "tens", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #10 score the first goal?"}},
    ]
    doc = _run(program, "Does #10 score the first goal?")
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["verdict"] == "Yes — #10 scored the first goal"


# --------------------------------------------------------------------------------------
# test_answer_on_unsupported_kind_is_ungrounded — honest grounding-out (ties to Q1=A)
# --------------------------------------------------------------------------------------

def test_answer_on_unsupported_kind_is_ungrounded():
    # An empty count (count over an empty detection set) -> 0 -> ungrounded, not a fake verdict.
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 100, "end_ms": 200, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},  # no cache @100
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]
    doc = _run(program, "how many?")
    f = doc["findings"]
    assert f["grounded"] is False
    assert f["verdict"] is None and f["answer"] is None
    assert f["reason"] == "no-grounded-answer"


# --------------------------------------------------------------------------------------
# test_text_readout_question — "what number is the scorer wearing?" returns the read value
# --------------------------------------------------------------------------------------

def test_text_readout_question():
    program = _prefix() + [
        {"id": "result", "op": "answer", "args": {"from": "numbers", "question": "what number is the scorer wearing?"}},
    ]
    doc = _run(program, "what number is the scorer wearing?")
    f = doc["findings"]
    assert f["grounded"] is True
    # the cache pins p_messi's jersey OCR to "10".
    assert "10" in f["answer"]
    assert "#10 scored the first goal" not in (f["verdict"] or "")


# --------------------------------------------------------------------------------------
# test_existence_question — "is there a referee?" over detections answers by presence
# --------------------------------------------------------------------------------------

def test_existence_question_present():
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4625, "end_ms": 4626, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "result", "op": "answer", "args": {"from": "people", "question": "is there a person?"}},
    ]
    doc = _run(program, "is there a person?")
    f = doc["findings"]
    assert f["grounded"] is True
    assert f["answer"] == "Yes"


# --------------------------------------------------------------------------------------
# test_scene_question — describe_scene -> answer grounds to the committed scene caption
# --------------------------------------------------------------------------------------

def test_scene_question_grounds_to_caption():
    program = [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 4000, "end_ms": 4626, "fps": 8}},
        {"id": "scene", "op": "describe_scene", "args": {"frames": "frames"}},
        {"id": "result", "op": "answer", "args": {"from": "scene", "question": "what is happening?"}},
    ]
    doc = _run(program, "what is happening?")
    f = doc["findings"]
    assert f["grounded"] is True
    # the hero cache's caption slice at 4625ms.
    assert "player" in (f["answer"] or "").lower()
    # the scene answer is NOT a fabricated Yes/No nor the hero verdict.
    assert f["answer"] not in ("Yes", "No")
    assert "#10 scored the first goal" not in (f["verdict"] or "")
