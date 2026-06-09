"""Bank tests — the bank is schema-valid, covers every live cell, anchors to canned.QUERIES, and
its derived ground truth can't drift from the cache. The schema/maxLength drift guard lives HERE
(the eval library never imports server)."""
from __future__ import annotations

from collections import Counter

import jsonschema
import pytest

import canned
import server  # tests may import server freely; the eval library must not.
from eval import bank


def test_question_bank_conforms_to_schema():
    b = bank.load_bank()
    schema = __import__("json").loads(bank.BANK_SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(b)  # raises on any malformed case
    cases = b["cases"]
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)), "case_ids must be unique"
    for c in cases:
        assert canned.clip_by_id(c["clip_id"]) is not None, c["clip_id"]


def test_schema_maxlength_matches_server_query_cap():
    """The schema's question maxLength is a literal; assert it equals server.MAX_QUERY_TEXT_LEN so a
    bank case can never be one the route would reject. The drift guard lives in the test."""
    schema = __import__("json").loads(bank.BANK_SCHEMA_PATH.read_text())
    literal = schema["$defs"]["case"]["properties"]["question"]["maxLength"]
    assert literal == server.MAX_QUERY_TEXT_LEN


def test_bank_covers_every_live_cell():
    cases = bank.load_cases()
    # Live groundable cells (clip x shape) the caches can ground.
    live_cells = {
        ("single-goal", "first-goal"),
        ("single-goal", "presence"),
        ("bernabeu-counter", "count"),
        ("bernabeu-counter", "scorer-number"),
        ("bernabeu-counter", "first-goal"),
        ("bernabeu-counter", "presence"),
    }
    anchors = Counter()
    paraphrases = Counter()
    for c in cases:
        key = (c["clip_id"], c["shape"])
        if c["phrasing_class"] == "canonical":
            anchors[key] += 1
        elif c["phrasing_class"] == "paraphrase":
            paraphrases[key] += 1
    for cell in live_cells:
        assert anchors[cell] >= 1, f"{cell} has no anchor case"
        assert paraphrases[cell] >= 3, f"{cell} has < 3 paraphrases ({paraphrases[cell]})"
    # Non-empty adversarial axes.
    classes = Counter(c["phrasing_class"] for c in cases)
    assert classes["out_of_scope"] >= 1
    assert classes["injection"] >= 1
    assert classes["degenerate"] >= 1


def test_canonical_matches_canned_queries():
    """Every non-presence canonical case text is verbatim from canned.QUERIES (anchored to a
    known-good). Presence canonicals anchor to the derived program's question text (D1 caveat)."""
    qtexts = {q["text"] for q in canned.QUERIES}
    for c in bank.load_cases():
        if c["phrasing_class"] == "canonical" and c["shape"] != "presence":
            assert c["question"] in qtexts, (c["case_id"], c["question"])


def test_count_ground_truth_derived_from_pinned():
    """The count expect.answer equals derive_ground_truth() over the pinned count program — the
    answer key can't drift from the cache."""
    derived = bank.derive_ground_truth(
        canned.API_DIR / "examples" / "bernabeu-counter_count_program.json",
        canned.API_DIR / "examples" / "bernabeu-counter_cache.json",
    )
    count_cases = [c for c in bank.load_cases() if c["shape"] == "count"]
    assert count_cases
    for c in count_cases:
        assert c["expect"]["answer"] == derived, (c["case_id"], c["expect"]["answer"], derived)


def test_prompt_version_changes_on_prompt_edit(monkeypatch):
    """prompt_version is a stable 16-hex string; editing SYSTEM_PROMPT changes it (invalidation works)."""
    clip = canned.clip_by_id("bernabeu-counter")
    pv1 = bank.prompt_version(clip)
    assert len(pv1) == 16 and all(ch in "0123456789abcdef" for ch in pv1)
    assert bank.prompt_version(clip) == pv1  # stable

    import codegen
    monkeypatch.setattr(codegen, "SYSTEM_PROMPT", codegen.SYSTEM_PROMPT + "\nEXTRA RULE.")
    pv2 = bank.prompt_version(clip)
    assert pv2 != pv1, "a prompt edit must change the prompt_version stamp"


def test_filter_cases_by_clip_shape_only():
    cases = bank.load_cases()
    only_bern = bank.filter_cases(cases, clip="bernabeu-counter")
    assert all(c["clip_id"] == "bernabeu-counter" for c in only_bern)
    only_count = bank.filter_cases(cases, shape="count")
    assert all(c["shape"] == "count" for c in only_count)
    # `only` token matches case_id / shape / clip_id.
    one = bank.filter_cases(cases, only="bernabeu-count-canonical")
    assert [c["case_id"] for c in one] == ["bernabeu-count-canonical"]
    by_shape = bank.filter_cases(cases, only="presence")
    assert by_shape and all(c["shape"] == "presence" for c in by_shape)
    by_class = bank.filter_cases(cases, phrasing_class="injection")
    assert all(c["phrasing_class"] == "injection" for c in by_class)


def test_load_bank_raises_on_schema_failure(tmp_path):
    bad = tmp_path / "bad_bank.json"
    bad.write_text('{"bank_version": 1, "prompt_version": null, "cases": [{"case_id": "x"}]}')
    with pytest.raises(jsonschema.ValidationError):
        bank.load_bank(bad)
