"""
Bank tests — the always-on hard gate on the question bank + its provenance (D1, D4).

These import `server` freely (tests may; the eval LIBRARY never does) to assert the schema's
maxLength literal tracks server.MAX_QUERY_TEXT_LEN.
"""
from __future__ import annotations

import json
from collections import Counter

import pytest

import canned
from eval import bank


def test_question_bank_conforms_to_schema():
    """Every case validates; case_ids unique; clip_ids resolve; enums hold."""
    cases = bank.load_bank()  # load_bank raises on a schema failure
    assert len(cases) >= 50

    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "case_ids must be unique"

    for c in cases:
        assert canned.clip_by_id(c["clip_id"]) is not None, f"unknown clip_id {c['clip_id']}"
        assert c["shape"] in {"first-goal", "count", "scorer-number", "presence"}
        assert c["phrasing_class"] in {"canonical", "paraphrase", "out_of_scope", "injection", "degenerate"}
        assert c["distance"] in {"canonical", "near", "far", "n/a"}
        assert c["expect"]["outcome"] in {"grounded", "ungrounded"}


def test_bank_schema_maxlength_matches_server_query_cap():
    """The bank schema's `maxLength` literal must equal server.MAX_QUERY_TEXT_LEN — the drift guard
    lives HERE (a test may import server; the eval library must not)."""
    import server

    schema = json.loads((bank.BANK_SCHEMA_PATH).read_text())
    cap = schema["$defs"]["case"]["properties"]["question"]["maxLength"]
    assert cap == server.MAX_QUERY_TEXT_LEN


def test_bank_covers_every_live_cell():
    """Each live (clip × shape) cell has its anchor + >=3 paraphrases; the out-of-scope and
    adversarial sets are non-empty."""
    cases = bank.load_bank()
    live_cells = {
        ("single-goal", "first-goal"),
        ("single-goal", "presence"),
        ("bernabeu-counter", "count"),
        ("bernabeu-counter", "scorer-number"),
        ("bernabeu-counter", "first-goal"),
        ("bernabeu-counter", "presence"),
    }
    for clip_id, shape in live_cells:
        cell = [c for c in cases if c["clip_id"] == clip_id and c["shape"] == shape
                and c["phrasing_class"] in ("canonical", "paraphrase")]
        anchors = [c for c in cell if c["phrasing_class"] == "canonical"]
        paraphrases = [c for c in cell if c["phrasing_class"] == "paraphrase"]
        assert len(anchors) >= 1, f"{clip_id}/{shape} needs an anchor"
        assert len(paraphrases) >= 3, f"{clip_id}/{shape} needs >=3 paraphrases, has {len(paraphrases)}"

    by_class = Counter(c["phrasing_class"] for c in cases)
    assert by_class["out_of_scope"] >= 1
    assert by_class["injection"] >= 1
    assert by_class["degenerate"] >= 1


def test_canonical_matches_canned_queries():
    """Every `canonical` case whose cell has a canned.QUERIES entry matches that text verbatim.
    Presence canonicals anchor to the derived presence program's question (no canned entry)."""
    cases = bank.load_bank()
    canned_texts = {q["text"] for q in canned.QUERIES}
    for c in cases:
        if c["phrasing_class"] != "canonical":
            continue
        if c["shape"] == "presence":
            continue  # presence anchors to the derived program text (D1 caveat), not canned.QUERIES
        assert c["question"] in canned_texts, f"canonical {c['case_id']} not in canned.QUERIES"


def test_count_ground_truth_derived_from_pinned():
    """The count expect.answer is DERIVED from the pinned count program, not hand-typed."""
    derived = bank.derive_ground_truth(
        "examples/bernabeu-counter_count_program.json",
        "examples/bernabeu-counter_cache.json",
    )
    cases = bank.load_bank()
    count_canonical = next(c for c in cases if c["case_id"] == "bernabeu-count-canonical")
    assert count_canonical["expect"]["answer"] == derived == "5"


def test_presence_ground_truth_derived_from_pinned():
    """Presence expect.answer is derived from the pinned presence programs (D1 caveat)."""
    for clip_id, prog in [
        ("single-goal", "examples/hero_presence_program.json"),
        ("bernabeu-counter", "examples/bernabeu-counter_presence_program.json"),
    ]:
        clip = canned.clip_by_id(clip_id)
        derived = bank.derive_ground_truth(prog, clip["cache"])
        case = next(c for c in bank.load_bank()
                    if c["clip_id"] == clip_id and c["shape"] == "presence"
                    and c["phrasing_class"] == "canonical")
        assert case["expect"]["answer"] == derived == "Yes"


def test_prompt_version_changes_on_prompt_edit(monkeypatch):
    """prompt_version is a stable 16-hex string; editing the prompt changes it."""
    import codegen

    pv = bank.prompt_version(None)
    assert isinstance(pv, str) and len(pv) == 16
    assert all(ch in "0123456789abcdef" for ch in pv)
    # Stable across calls.
    assert bank.prompt_version(None) == pv
    # An edit to the prompt surface changes the hash (the invalidation signal works).
    monkeypatch.setattr(codegen, "SYSTEM_PROMPT", codegen.SYSTEM_PROMPT + "\nEXTRA RULE.")
    assert bank.prompt_version(None) != pv


def test_load_bank_raises_on_schema_failure(tmp_path):
    """A malformed bank fails loudly (mirrors the repo's schema discipline)."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bank_version": 1, "prompt_version": "x", "cases": [{"case_id": "BAD ID"}]}))
    with pytest.raises(ValueError, match="fails its schema"):
        bank.load_bank(bad)


def test_filter_cases_by_only_shape_clip():
    cases = bank.load_bank()
    counts = bank.filter_cases(cases, only="count")
    assert counts and all(c["shape"] == "count" for c in counts)
    by_clip = bank.filter_cases(cases, clip="single-goal")
    assert by_clip and all(c["clip_id"] == "single-goal" for c in by_clip)
    by_id = bank.filter_cases(cases, only="bernabeu-count-canonical")
    assert len(by_id) == 1 and by_id[0]["case_id"] == "bernabeu-count-canonical"
