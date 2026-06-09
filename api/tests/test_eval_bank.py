"""Bank tests — schema conformance, cell coverage, canonical anchoring, derived ground truth,
prompt-version invalidation. The drift guard (schema maxLength == server.MAX_QUERY_TEXT_LEN)
lives HERE (tests import server freely; the eval library never does)."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

import bank
import canned

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def test_question_bank_conforms_to_schema():
    bank_obj = json.loads((EVAL_DIR / "question_bank.json").read_text())
    schema = json.loads((EVAL_DIR / "question_bank.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(bank_obj)

    cases = bank_obj["cases"]
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "case_ids must be unique"
    for c in cases:
        assert canned.clip_by_id(c["clip_id"]) is not None, c["clip_id"]


def test_schema_maxlength_matches_server_query_cap():
    """Drift guard: the static schema literal must equal server.MAX_QUERY_TEXT_LEN."""
    import server

    schema = json.loads((EVAL_DIR / "question_bank.schema.json").read_text())
    literal = schema["$defs"]["case"]["properties"]["question"]["maxLength"]
    assert literal == server.MAX_QUERY_TEXT_LEN


def test_load_bank_raises_on_schema_failure(tmp_path):
    bad = {"bank_version": 1, "prompt_version": "x", "cases": [{"case_id": "Bad ID!"}]}
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="fails schema"):
        bank.load_bank(p)


def test_load_bank_raises_on_duplicate_case_id(tmp_path):
    dup = {
        "bank_version": 1, "prompt_version": "x",
        "cases": [
            {"case_id": "a", "clip_id": "single-goal", "shape": "presence",
             "phrasing_class": "canonical", "distance": "canonical", "question": "x?",
             "expect": {"outcome": "grounded", "answer": "Yes", "answer_match": "exact"}},
            {"case_id": "a", "clip_id": "single-goal", "shape": "presence",
             "phrasing_class": "canonical", "distance": "canonical", "question": "y?",
             "expect": {"outcome": "grounded", "answer": "Yes", "answer_match": "exact"}},
        ],
    }
    p = tmp_path / "bank.json"
    p.write_text(json.dumps(dup))
    with pytest.raises(ValueError, match="duplicate case_id"):
        bank.load_bank(p)


def test_bank_covers_every_live_cell():
    """Each live (clip x shape) cell has >=1 anchor + >=3 paraphrase; out-of-scope/adversarial/
    degenerate sets are non-empty."""
    cases = bank.load_bank()["cases"]
    live_cells = [
        ("bernabeu-counter", "count"), ("bernabeu-counter", "scorer-number"),
        ("bernabeu-counter", "first-goal"), ("bernabeu-counter", "presence"),
        ("single-goal", "first-goal"), ("single-goal", "presence"),
    ]
    for clip, shape in live_cells:
        cell = [c for c in cases
                if c["clip_id"] == clip and c["shape"] == shape
                and c["phrasing_class"] in ("canonical", "paraphrase")]
        anchors = [c for c in cell if c["phrasing_class"] == "canonical"]
        paraphrases = [c for c in cell if c["phrasing_class"] == "paraphrase"]
        assert len(anchors) >= 1, (clip, shape)
        assert len(paraphrases) >= 3, (clip, shape, len(paraphrases))

    assert any(c["shape"] == "out-of-scope" for c in cases)
    assert any(c["shape"] == "adversarial" for c in cases)
    assert any(c["shape"] == "degenerate" for c in cases)


def test_canonical_matches_canned_queries():
    """Every canonical case whose cell has a canned.QUERIES entry matches that text verbatim
    (presence cells anchor to the derived program question, so they are exempt)."""
    cases = bank.load_bank()["cases"]
    canned_texts = {q["text"] for q in canned.QUERIES}
    for c in cases:
        if c["phrasing_class"] == "canonical" and c["shape"] != "presence":
            assert c["question"] in canned_texts, c["case_id"]


def test_count_ground_truth_derived_from_pinned():
    """The bank's count answer equals derive_ground_truth() over the pinned count program."""
    derived = bank.derive_ground_truth(
        "examples/bernabeu-counter_count_program.json",
        "examples/bernabeu-counter_cache.json",
    )
    cases = bank.load_bank()["cases"]
    canonical = next(c for c in cases if c["case_id"] == "bernabeu-count-canonical")
    assert canonical["expect"]["answer"] == derived


def test_presence_ground_truth_derived_from_pinned():
    for clip, prog, cache, cid in [
        ("bernabeu-counter", "examples/bernabeu-counter_presence_program.json",
         "examples/bernabeu-counter_cache.json", "bernabeu-presence-anchor"),
        ("single-goal", "examples/hero_presence_program.json",
         "examples/hero_cache.json", "hero-presence-anchor"),
    ]:
        derived = bank.derive_ground_truth(prog, cache)
        cases = bank.load_bank()["cases"]
        anchor = next(c for c in cases if c["case_id"] == cid)
        assert anchor["expect"]["answer"] == derived


def test_prompt_version_changes_on_prompt_edit(monkeypatch):
    import codegen

    pv = bank.prompt_version(None)
    assert isinstance(pv, str) and len(pv) == 16
    int(pv, 16)  # is hex

    monkeypatch.setattr(codegen, "SYSTEM_PROMPT", codegen.SYSTEM_PROMPT + "\nNEW RULE.")
    assert bank.prompt_version(None) != pv


def test_load_cases_returns_list():
    cases = bank.load_cases()
    assert isinstance(cases, list) and len(cases) == 60


def test_filter_cases_by_axes():
    cases = bank.load_bank()["cases"]
    counts = bank.filter_cases(cases, shape="count")
    assert counts and all(c["shape"] == "count" for c in counts)
    bern = bank.filter_cases(cases, clip="bernabeu-counter")
    assert bern and all(c["clip_id"] == "bernabeu-counter" for c in bern)
    only_id = bank.filter_cases(cases, only="bernabeu-count-canonical")
    assert len(only_id) == 1
    only_shape = bank.filter_cases(cases, only="count")
    assert only_shape == counts
    canon = bank.filter_cases(cases, phrasing_class="canonical")
    assert canon and all(c["phrasing_class"] == "canonical" for c in canon)
