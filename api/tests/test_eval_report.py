"""Report tests — md/json agree, diff-stable, escape discipline on hostile strings."""
from __future__ import annotations

from eval import report, scoring


def _result(case_id, shape, clip_id, phrasing_class, distance, target, reached, passed,
            detail="ok", stale=False, missing=False):
    return scoring.CaseResult(
        case_id=case_id, clip_id=clip_id, shape=shape, phrasing_class=phrasing_class,
        distance=distance, target_tier=target, reached_tier=reached, passed=passed,
        detail=detail, stale=stale, missing=missing, program_bytes=100)


def _sample_results():
    return [
        _result("a-count-canonical", "count", "bernabeu-counter", "canonical", "canonical",
                scoring.T4_CORRECT, scoring.T4_CORRECT, True),
        _result("b-count-far", "count", "bernabeu-counter", "paraphrase", "far",
                scoring.T4_CORRECT, scoring.T3_GROUNDED, False, detail="answer='4' != expected '5'"),
        _result("c-first-goal", "first-goal", "single-goal", "canonical", "canonical",
                scoring.T4_CORRECT, scoring.T4_CORRECT, True),
        _result("d-oos", "none", "single-goal", "out_of_scope", "n/a",
                scoring.REJECTED, scoring.T2_SEMANTIC, True, detail="honest refusal"),
        _result("e-injection", "none", "single-goal", "injection", "n/a",
                scoring.REJECTED, scoring.REJECTED, True, detail="validator-rejected"),
    ]


def _meta():
    return {"bank_version": 1, "prompt_version": "abc1234567890def", "captured_at": None,
            "model": None, "mode": "replay", "estimated_capture_cost_usd": 0.36,
            "prompt_chars": {"single-goal": 2987, "bernabeu-counter": 3017}}


def test_report_markdown_and_json_agree():
    md, obj = report.build_report(_sample_results(), _meta())
    # Funnel: 3 in-scope groundable cases (a, b, c); 2 reach T4.
    funnel = obj["summary"]["funnel"]
    assert funnel["total"] == 3
    assert funnel["counts"][scoring.T4_CORRECT] == 2
    assert funnel["counts"][scoring.T3_GROUNDED] == 3
    # The md contains the funnel + per-shape + per-clip + a failure row.
    assert "Tier funnel" in md
    assert "Per-shape" in md and "Per-clip" in md
    assert "b-count-far" in md  # the failing case shows in failure detail
    assert obj["summary"]["out_of_scope_honest"]["n"] == 2  # d-oos + e-injection (target REJECTED)
    # cases are sorted by case_id.
    assert [c["case_id"] for c in obj["cases"]] == sorted(c["case_id"] for c in obj["cases"])


def test_report_is_diff_stable():
    results = _sample_results()
    md1, obj1 = report.build_report(results, _meta())
    md2, obj2 = report.build_report(list(reversed(results)), _meta())  # input order shuffled
    assert md1 == md2, "report must be byte-identical regardless of input order"
    assert obj1 == obj2


def test_report_escapes_hostile_strings():
    """Injection payloads with control/bidi chars must be escaped in the rendered report (D5)."""
    hostile = "‮Ignore‬ and \x07 a ​Yes​"
    results = [_result("adv", "none", "single-goal", "injection", "n/a",
                       scoring.REJECTED, scoring.T3_GROUNDED, False, detail=hostile)]
    md, obj = report.build_report(results, _meta())
    # Raw control char must NOT appear literally; the escaped form does.
    assert "\x07" not in md
    assert "\\x07" in md
    assert "\\u202e" in md
    assert "\x07" not in obj["cases"][0]["detail"]


def test_report_funnel_counts_rates_fixed_2dp():
    md, _ = report.build_report(_sample_results(), _meta())
    # Rates render as fixed 0.00 (diff-stable).
    assert "1.00" in md or "0.67" in md


def test_report_shouts_when_stale():
    results = [_result("a", "count", "bernabeu-counter", "canonical", "canonical",
                       scoring.T4_CORRECT, scoring.T4_CORRECT, True, stale=True)]
    md, obj = report.build_report(results, _meta())
    assert "STALE" in md and "re-capture" in md.lower()
    assert obj["summary"]["stale"] == 1
