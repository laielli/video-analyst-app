"""
Question-bank tests — the bank is schema-valid, anchored to canned.QUERIES, covers every live
cell, and its derived ground truth can't drift from the cache. Creds-free.
"""
from __future__ import annotations

import json
from collections import defaultdict

import canned
import codegen
import server  # tests may import server freely; the eval LIBRARY never does (the drift guard)
from eval import bank
from eval.bank import BANK_PATH, SCHEMA_PATH


def test_question_bank_conforms_to_schema():
    import jsonschema

    bank_obj = json.loads(BANK_PATH.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(bank_obj)

    cases = bank_obj["cases"]
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "case_ids must be unique"
    for c in cases:
        assert canned.clip_by_id(c["clip_id"]) is not None, f"unknown clip_id {c['clip_id']}"
        assert c["shape"] in {"first-goal", "count", "scorer-number", "presence"}
        assert c["phrasing_class"] in {"canonical", "paraphrase", "out_of_scope", "injection", "degenerate"}


def test_schema_maxlength_matches_route_cap():
    """The bank schema's question maxLength is a literal (a static JSON Schema can't import
    server.MAX_QUERY_TEXT_LEN). This drift guard — in the TEST, which imports server freely —
    asserts the literal equals the route cap, so no bank case can be one the route would reject."""
    schema = json.loads(SCHEMA_PATH.read_text())
    literal = schema["$defs"]["case"]["properties"]["question"]["maxLength"]
    assert literal == server.MAX_QUERY_TEXT_LEN


def test_bank_covers_every_live_cell():
    """Each live (clip × shape) cell has >=1 anchor + >=3 paraphrase; oos/injection/degenerate
    sets are non-empty."""
    cases = bank.load_cases()
    by_cell = defaultdict(lambda: defaultdict(int))
    for c in cases:
        by_cell[(c["clip_id"], c["shape"])][c["phrasing_class"]] += 1

    live_cells = [
        ("single-goal", "first-goal"),
        ("single-goal", "presence"),
        ("bernabeu-counter", "count"),
        ("bernabeu-counter", "scorer-number"),
        ("bernabeu-counter", "first-goal"),
    ]
    for cell in live_cells:
        anchor = by_cell[cell]["canonical"]
        paraphrase = by_cell[cell]["paraphrase"]
        assert anchor >= 1, f"{cell} needs >=1 anchor case"
        assert paraphrase >= 3, f"{cell} needs >=3 paraphrase cases, has {paraphrase}"

    classes = defaultdict(int)
    for c in cases:
        classes[c["phrasing_class"]] += 1
    assert classes["out_of_scope"] >= 1
    assert classes["injection"] >= 1
    assert classes["degenerate"] >= 1


def test_canonical_matches_canned_queries():
    """Every canonical case's text matches a canned.QUERIES entry (anchors the bank to known-good).
    Presence cells have no canned.QUERIES entry, so the rule applies only to non-presence canonicals."""
    canned_texts = {q["text"] for q in canned.QUERIES}
    for c in bank.load_cases():
        if c["phrasing_class"] == "canonical" and c["shape"] != "presence":
            assert c["question"] in canned_texts, f"{c['case_id']} canonical not in canned.QUERIES"


def test_count_ground_truth_derived_from_pinned():
    """derive_ground_truth over the pinned count + presence programs equals the bank's expect.answer
    — the answer key can't drift from the cache."""
    EX = canned.API_DIR / "examples"
    bernabeu_cache = str(EX / "bernabeu-counter_cache.json")
    hero_cache = str(EX / "hero_cache.json")

    count_answer = bank.derive_ground_truth(EX / "bernabeu-counter_count_program.json", bernabeu_cache)
    count_case = next(c for c in bank.load_cases() if c["case_id"] == "bernabeu-count-canonical")
    assert count_case["expect"]["answer"] == count_answer == "5"

    presence_b = bank.derive_ground_truth(EX / "bernabeu-counter_presence_program.json", bernabeu_cache)
    presence_h = bank.derive_ground_truth(EX / "hero_presence_program.json", hero_cache)
    assert presence_b == presence_h == "Yes"
    for cid in ("single-goal-presence-anchor",):
        case = next(c for c in bank.load_cases() if c["case_id"] == cid)
        assert case["expect"]["answer"] == "Yes"


def test_prompt_version_is_stable_16hex_and_changes_on_edit(monkeypatch):
    pv = bank.prompt_version(None)
    assert isinstance(pv, str) and len(pv) == 16
    assert all(ch in "0123456789abcdef" for ch in pv)
    # Editing the prompt changes the version (the invalidation signal works).
    monkeypatch.setattr(codegen, "SYSTEM_PROMPT", codegen.SYSTEM_PROMPT + "\nEXTRA RULE")
    assert bank.prompt_version(None) != pv


def test_load_bank_raises_on_schema_failure(tmp_path):
    bad = tmp_path / "bad_bank.json"
    bad.write_text(json.dumps({"bank_version": 1, "prompt_version": "x", "cases": [{"case_id": "x"}]}))
    try:
        bank.load_bank(bad)
        assert False, "expected ValueError on schema failure"
    except ValueError as e:
        assert "fails schema" in str(e)
