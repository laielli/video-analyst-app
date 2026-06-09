"""Report tests (D5): md+json agree, the tables/funnel render, diff-stable rendering."""
from __future__ import annotations

import json

from eval import report, scoring
from eval.scoring import CaseResult


def _cases_by_id():
    return {
        "a-grounded": {"case_id": "a-grounded", "clip_id": "bernabeu-counter", "shape": "count",
                       "phrasing_class": "canonical", "distance": "canonical",
                       "expect": {"outcome": "grounded", "answer": "5", "answer_match": "numeric"}},
        "b-grounded-wrong": {"case_id": "b-grounded-wrong", "clip_id": "single-goal", "shape": "first-goal",
                             "phrasing_class": "paraphrase", "distance": "far",
                             "expect": {"outcome": "grounded", "answer": "Yes", "answer_match": "exact"}},
        "c-oos": {"case_id": "c-oos", "clip_id": "bernabeu-counter", "shape": "count",
                  "phrasing_class": "out_of_scope", "distance": "n/a",
                  "expect": {"outcome": "ungrounded", "answer": None, "answer_match": "any_grounded"}},
        "d-injection": {"case_id": "d-injection", "clip_id": "single-goal", "shape": "presence",
                        "phrasing_class": "injection", "distance": "n/a",
                        "expect": {"outcome": "ungrounded", "answer": None, "answer_match": "any_grounded"}},
    }


def _results():
    return [
        CaseResult("a-grounded", scoring.T4_CORRECT, scoring.T4_CORRECT, True, "grounded '5'"),
        CaseResult("b-grounded-wrong", scoring.T4_CORRECT, scoring.T3_GROUNDED, False, "wrong answer: 'No'"),
        CaseResult("c-oos", scoring.REJECTED, scoring.T2_SEMANTIC, True, "honest no-ground"),
        CaseResult("d-injection", scoring.REJECTED, scoring.REJECTED, True, "rejected"),
    ]


def _meta():
    return {"bank_version": 1, "prompt_version": "abc", "captured_at": "seed", "model": "seed",
            "mode": "replay", "prompt_sizes": {"bernabeu-counter": 3017}, "total_raw_bytes": 100,
            "est_cost": "~$0.36"}


def test_report_markdown_and_json_agree():
    md, j = report.build_report(_results(), _cases_by_id(), _meta())
    # JSON funnel matches the markdown headline.
    assert j["summary"]["funnel"][scoring.T4_CORRECT]["reached"] == 1  # only a-grounded
    assert j["summary"]["funnel"][scoring.T2_SEMANTIC]["reached"] == 2  # a + b reach T2+
    assert "Tier funnel" in md
    assert "Per-shape" in md and "Per-clip" in md and "Per-bucket" in md
    # A failure row for the sub-target case.
    assert "b-grounded-wrong" in md
    # OOS + adversarial lines.
    assert j["summary"]["out_of_scope_honest"]["passed"] == 2  # c + d are ungrounded-outcome
    assert j["summary"]["adversarial_safe"]["passed"] == 1     # only d is injection/degenerate


def test_report_is_diff_stable():
    md1, j1 = report.build_report(_results(), _cases_by_id(), _meta())
    md2, j2 = report.build_report(list(reversed(_results())), _cases_by_id(), _meta())
    assert md1 == md2  # sorted cases -> identical regardless of input order
    assert json.dumps(j1) == json.dumps(j2)


def test_report_escapes_hostile_strings():
    """A control/bidi char in a detail is escaped repr-style, not rendered raw."""
    cbid = _cases_by_id()
    cbid["x"] = {"case_id": "x", "clip_id": "single-goal", "shape": "presence",
                 "phrasing_class": "injection", "distance": "n/a",
                 "expect": {"outcome": "ungrounded", "answer": None, "answer_match": "any_grounded"}}
    results = _results() + [CaseResult("x", scoring.REJECTED, scoring.REJECTED, False, "bad ‮ rtl")]
    md, _ = report.build_report(results, cbid, _meta())
    assert "\\u202e" in md and "‮" not in md


def test_write_report_atomic(tmp_path):
    md, j = report.build_report(_results(), _cases_by_id(), _meta())
    md_path, json_path = tmp_path / "report.md", tmp_path / "report.json"
    report.write_report(md, j, md_path, json_path)
    assert md_path.read_text() == md
    assert json.loads(json_path.read_text()) == j
    assert not list(tmp_path.glob("*.tmp"))
