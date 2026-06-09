"""Report tests — md/json agreement, diff-stability, failure rows, hostile-string escaping."""
from __future__ import annotations

import json

import report as report_mod
import scoring
from scoring import CaseResult


def _result(case_id, shape="count", reached=scoring.T4_CORRECT, passed=True,
            outcome="grounded", clip="bernabeu-counter", phrasing="canonical", distance="canonical",
            detail="ok", grounded=True):
    target = scoring.T4_CORRECT if outcome == "grounded" else scoring.T3_GROUNDED
    return CaseResult(
        case_id=case_id, clip_id=clip, shape=shape, phrasing_class=phrasing, distance=distance,
        target_tier=target, reached_tier=reached, passed=passed, detail=detail,
        expected_outcome=outcome, grounded=grounded,
    )


def _meta():
    return {"bank_version": 1, "prompt_version": "abc", "captured_at": "t", "model": "seed",
            "mode": "replay", "prompt_sizes": {"x": 100}, "total_program_bytes": 42,
            "cost_estimate": "~$0.40", "n_fixtures": 3}


def test_report_markdown_and_json_agree():
    results = [
        _result("a-pass", reached=scoring.T4_CORRECT, passed=True),
        _result("b-subtarget", reached=scoring.T3_GROUNDED, passed=False, detail="answer mismatch: got '4'"),
        _result("c-oos", shape="out-of-scope", outcome="ungrounded", reached=scoring.T2_SEMANTIC,
                passed=True, grounded=False),
    ]
    md, j = report_mod.build_report(results, _meta())
    # md contains the funnel + per-shape + per-clip tables + the failure row.
    assert "Tier funnel" in md
    assert "Per-shape" in md
    assert "Per-clip" in md
    assert "b-subtarget" in md
    # json carries the same per-tier counts.
    assert j["summary"]["total_cases"] == 3
    assert j["summary"]["passed"] == 2
    # cases sorted by case_id.
    assert [c["case_id"] for c in j["cases"]] == ["a-pass", "b-subtarget", "c-oos"]


def test_report_is_diff_stable():
    results = [_result("a"), _result("b", reached=scoring.T3_GROUNDED, passed=False)]
    md1, j1 = report_mod.build_report(results, _meta())
    md2, j2 = report_mod.build_report(list(reversed(results)), _meta())
    assert md1 == md2  # sorted, fixed-rate rendering -> byte-identical
    assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)


def test_report_escapes_hostile_strings():
    r = _result("adv", shape="adversarial", outcome="ungrounded", reached=scoring.REJECTED,
                passed=False, detail="weird ‮ bidi \x00 null", grounded=None)
    md, _ = report_mod.build_report([r], _meta())
    # The raw control/bidi chars must not appear verbatim in the rendered markdown.
    assert "‮" not in md
    assert "\x00" not in md


def test_report_shouts_on_stale_and_missing():
    r1 = _result("stale", passed=False)
    r1.stale = True
    r2 = _result("missing", passed=False)
    r2.missing = True
    md, j = report_mod.build_report([r1, r2], {**_meta(), "n_fixtures": 1})
    assert "re-capture needed" in md
    assert j["summary"]["stale"] == 1
    assert j["summary"]["missing"] == 1


def test_write_report_atomic(tmp_path):
    md, j = report_mod.build_report([_result("a")], _meta())
    mdp, jp = tmp_path / "r.md", tmp_path / "r.json"
    report_mod.write_report(md, j, mdp, jp)
    assert mdp.read_text() == md
    assert json.loads(jp.read_text())["summary"]["total_cases"] == 1
    assert not list(tmp_path.glob(".*.tmp"))
