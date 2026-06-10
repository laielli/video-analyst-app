"""Report tests — md + json agree, the funnel/tables/failure rows render, and re-rendering the
same report is byte-identical (diff-stable). Creds-free, pure over synthetic CaseResults."""
from __future__ import annotations

from eval import report as report_mod
from eval import scoring
from eval.scoring import CaseResult


def _results():
    return [
        CaseResult(case_id="b-count-canonical", target_tier=scoring.T4_CORRECT,
                   reached_tier=scoring.T4_CORRECT, passed=True, clip_id="bernabeu-counter",
                   shape="count", phrasing_class="canonical", distance="canonical",
                   grounded=True, program_bytes=200, detail="answer='5'"),
        CaseResult(case_id="b-count-far", target_tier=scoring.T4_CORRECT,
                   reached_tier=scoring.T3_GROUNDED, passed=False, clip_id="bernabeu-counter",
                   shape="count", phrasing_class="paraphrase", distance="far",
                   grounded=True, program_bytes=210, detail="answer='4' != expected '5'"),
        CaseResult(case_id="h-first-goal", target_tier=scoring.T4_CORRECT,
                   reached_tier=scoring.T4_CORRECT, passed=True, clip_id="single-goal",
                   shape="first-goal", phrasing_class="canonical", distance="canonical",
                   grounded=True, program_bytes=300),
        CaseResult(case_id="oos-weather", target_tier=scoring.T3_GROUNDED,
                   reached_tier=scoring.T3_GROUNDED, passed=True, clip_id="single-goal",
                   shape="first-goal", phrasing_class="out_of_scope", distance="n/a",
                   grounded=False),
        CaseResult(case_id="adv-inject", target_tier=scoring.T3_GROUNDED,
                   reached_tier=scoring.T0_PARSED, passed=True, clip_id="single-goal",
                   shape="first-goal", phrasing_class="injection", distance="n/a"),
    ]


def _meta():
    return {"bank_version": 1, "prompt_version": "abc123", "captured_at": "2026-06-09T00:00:00Z",
            "model": "gpt-4o", "mode": "replay", "clip_prompt_chars": {"single-goal": 2987},
            "est_cost": "~$0.30"}


def test_report_markdown_and_json_agree():
    md, j = report_mod.build_report(_results(), _meta())
    s = j["summary"]
    # 3 in-scope (canonical/paraphrase): 2 grounded, 1 fails T4 -> T4 count = 2.
    assert s["funnel"][scoring.T4_CORRECT] == 2
    assert s["funnel"][scoring.T3_GROUNDED] == 3
    assert s["out_of_scope_honest"] == 1
    assert s["adversarial_safe"] == 1
    # md contains the funnel line + tables + a failure row for the sub-target case.
    assert "Tier funnel (in-scope)" in md
    assert "T4_correct 2/3" in md
    assert "Per-shape" in md and "Per-clip" in md and "paraphrase-distance" in md
    assert "b-count-far" in md  # the failure row


def test_report_is_diff_stable():
    md1, j1 = report_mod.build_report(_results(), _meta())
    md2, j2 = report_mod.build_report(list(reversed(_results())), _meta())
    assert md1 == md2, "report must be byte-identical regardless of input order (sorted by case_id)"
    import json
    assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)


def test_report_escapes_hostile_strings():
    """Control/bidi chars in a failing case's detail are repr-escaped (D5 escape discipline)."""
    r = CaseResult(case_id="rtl", target_tier=scoring.T3_GROUNDED, reached_tier=scoring.T0_PARSED,
                   passed=False, clip_id="single-goal", shape="first-goal",
                   phrasing_class="injection", detail="payload ‮ reversed")
    md, _ = report_mod.build_report([r], _meta())
    assert "‮" not in md  # the raw bidi char must not appear unescaped
    assert "\\u202e" in md


def test_report_shouts_on_uniform_stale():
    rs = [CaseResult(case_id="c1", target_tier=scoring.T4_CORRECT, reached_tier=scoring.T4_CORRECT,
                     passed=True, clip_id="single-goal", shape="first-goal",
                     phrasing_class="canonical", stale=True)]
    md, j = report_mod.build_report(rs, _meta())
    assert j["summary"]["stale"] == 1
    assert "ALL FIXTURES STALE" in md
