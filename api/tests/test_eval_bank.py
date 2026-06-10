"""Bank tests — the question bank conforms to its schema, anchors to known-good, and its derived
ground truth can't drift from the cache. Creds-free; uses the committed bank.

The eval modules import each other as top-level (`import bank`), so the eval/ dir must be on
sys.path. pytest's pythonpath already has `.` and `scripts`; add `eval` here (the modules' own
import contract)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = API_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import bank  # noqa: E402
import canned  # noqa: E402

LIVE_CELLS = {
    ("single-goal", "first-goal"),
    ("single-goal", "presence"),
    ("bernabeu-counter", "count"),
    ("bernabeu-counter", "scorer-number"),
    ("bernabeu-counter", "first-goal"),
    ("bernabeu-counter", "presence"),
}


def test_question_bank_conforms_to_schema():
    import jsonschema

    doc = json.loads((EVAL_DIR / "question_bank.json").read_text())
    schema = json.loads((EVAL_DIR / "question_bank.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(doc)  # raises on failure

    cases = doc["cases"]
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "case_ids must be unique"
    for c in cases:
        assert canned.clip_by_id(c["clip_id"]) is not None, f"unknown clip {c['clip_id']}"


def test_schema_maxlength_matches_server_query_cap():
    """The drift guard lives HERE (tests import server freely); the eval library never imports
    server. The schema's literal question maxLength must equal server.MAX_QUERY_TEXT_LEN so no
    bank case can be one the route would reject."""
    import server

    schema = json.loads((EVAL_DIR / "question_bank.schema.json").read_text())
    max_len = schema["$defs"]["case"]["properties"]["question"]["maxLength"]
    assert max_len == server.MAX_QUERY_TEXT_LEN


def test_bank_covers_every_live_cell():
    cases = bank.load_bank()
    for clip_id, shape in LIVE_CELLS:
        cell = [c for c in cases if c["clip_id"] == clip_id and c["shape"] == shape
                and c["phrasing_class"] in ("canonical", "paraphrase")]
        anchors = [c for c in cell if c["phrasing_class"] == "canonical"]
        paraphrases = [c for c in cell if c["phrasing_class"] == "paraphrase"]
        assert len(anchors) >= 1, f"cell {(clip_id, shape)} needs >=1 anchor"
        assert len(paraphrases) >= 3, f"cell {(clip_id, shape)} needs >=3 paraphrase"
    # adversarial / out-of-scope / degenerate sets are non-empty.
    assert any(c["phrasing_class"] == "out_of_scope" for c in cases)
    assert any(c["phrasing_class"] == "injection" for c in cases)
    assert any(c["phrasing_class"] == "degenerate" for c in cases)


def test_canonical_matches_canned_queries():
    """Every canonical case with a canned.QUERIES counterpart is verbatim from it. Presence
    anchors have no canned.QUERIES entry (D1 caveat); they anchor to the derived program text and
    are exempt from this check."""
    cases = bank.load_bank()
    canned_texts = {q["text"] for q in canned.QUERIES}
    for c in cases:
        if c["phrasing_class"] == "canonical" and c["shape"] != "presence":
            assert c["question"] in canned_texts, f"{c['case_id']} not verbatim canned.QUERIES"


def test_count_ground_truth_derived_from_pinned():
    """The count answer key is DERIVED from the pinned program over the committed cache, never
    hand-typed — so it can't drift from the cache."""
    ber = canned.clip_by_id("bernabeu-counter")
    derived = bank.derive_ground_truth(
        API_DIR / "examples" / "bernabeu-counter_count_program.json", ber["cache"])
    cases = bank.load_bank()
    canonical_count = next(c for c in cases if c["case_id"] == "bernabeu-count-canonical")
    assert canonical_count["expect"]["answer"] == derived == "5"


def test_presence_ground_truth_derived_from_pinned():
    """Presence cells are live (D1 caveat resolved by authoring pinned presence programs); their
    answer is derived, never hand-typed."""
    for clip_id, prog in (("single-goal", "single-goal_presence_program.json"),
                          ("bernabeu-counter", "bernabeu-counter_presence_program.json")):
        clip = canned.clip_by_id(clip_id)
        derived = bank.derive_ground_truth(API_DIR / "examples" / prog, clip["cache"])
        anchor = next(c for c in bank.load_bank()
                      if c["clip_id"] == clip_id and c["shape"] == "presence"
                      and c["phrasing_class"] == "canonical")
        assert anchor["expect"]["answer"] == derived == "Yes"


def test_bank_prompt_version_field_is_current():
    """The bank's top-level `prompt_version` field is provenance-only (replay compares fixtures
    against the live per-clip hash, not this field), but `_build_meta` copies it verbatim into
    reports — so it must not lie. Guard it against silent drift: it must equal the live
    `bank.prompt_version(None)` hash. This also forces the field to be updated in the SAME commit
    as any SYSTEM_PROMPT edit (which is what changes the hash)."""
    doc = json.loads((EVAL_DIR / "question_bank.json").read_text())
    assert doc["prompt_version"] == bank.prompt_version(None), (
        "bank.prompt_version field is stale — recompute it via bank.prompt_version(None) after "
        "the SYSTEM_PROMPT edit and update question_bank.json (never hand-copy a hash)"
    )


def test_prompt_version_changes_on_prompt_edit(monkeypatch):
    import codegen

    v1 = bank.prompt_version(None)
    assert isinstance(v1, str) and len(v1) == 16
    assert all(ch in "0123456789abcdef" for ch in v1)

    monkeypatch.setattr(codegen, "SYSTEM_PROMPT", codegen.SYSTEM_PROMPT + "\nEXTRA RULE")
    v2 = bank.prompt_version(None)
    assert v2 != v1, "a SYSTEM_PROMPT edit must change the prompt_version stamp"


def test_filter_cases_by_clip_shape_and_only():
    cases = bank.load_bank()
    only_count = bank.filter_cases(cases, only="count")
    assert only_count and all(c["shape"] == "count" for c in only_count)
    only_clip = bank.filter_cases(cases, clip="single-goal")
    assert only_clip and all(c["clip_id"] == "single-goal" for c in only_clip)
    one = bank.filter_cases(cases, only="bernabeu-count-canonical")
    assert len(one) == 1 and one[0]["case_id"] == "bernabeu-count-canonical"
