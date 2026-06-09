"""Report tests — md + json agree, and the report is diff-stable (byte-identical on re-render).
Creds-free; builds synthetic CaseResult lists."""
from __future__ import annotations

import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = API_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import report as report_mod  # noqa: E402
import scoring  # noqa: E402


def _result(case_id, reached, passed, **kw):
    return scoring.CaseResult(
        case_id=case_id, target_tier=kw.get("target_tier", scoring.T4),
        reached_tier=reached, passed=passed, detail=kw.get("detail", "d"),
        shape=kw.get("shape", "count"), clip_id=kw.get("clip_id", "bernabeu-counter"),
        phrasing_class=kw.get("phrasing_class", "canonical"), distance=kw.get("distance", "canonical"),
        out_of_scope=kw.get("out_of_scope", False), safety_axis=kw.get("safety_axis", False),
        stale=kw.get("stale", False), missing=kw.get("missing", False),
    )


def _sample_results():
    return [
        _result("a-pass", scoring.T4, True),
        _result("b-t3", scoring.T3, False, detail="T3 only (wrong answer)"),
        _result("c-oos", scoring.T2, True, out_of_scope=True, phrasing_class="out_of_scope",
                shape="out-of-scope", target_tier=scoring.T3),
        _result("d-adv", scoring.REJECTED, True, out_of_scope=True, safety_axis=True,
                phrasing_class="injection", shape="adversarial", target_tier=scoring.T3),
    ]


META = {"bank_version": 1, "prompt_version": "pv", "model": "seed", "captured_at": "seed",
        "prompt_sizes": {"bernabeu-counter": 3017}, "program_bytes": 1000,
        "estimated_capture_cost_usd": 0.38}


def test_report_markdown_and_json_agree():
    md, j = report_mod.build_report(_sample_results(), META)
    # funnel counts present and consistent: 2 in-scope cases, 1 reaches T4.
    assert j["summary"]["in_scope"] == 2
    assert j["summary"]["funnel"][scoring.T4] == 1
    assert j["summary"]["funnel"][scoring.T2] == 2
    assert j["summary"]["out_of_scope_honest"] == {"passed": 2, "total": 2}
    assert j["summary"]["adversarial_safe"] == {"passed": 1, "total": 1}
    # md carries the funnel line + per-shape and per-clip tables + a failure row.
    assert "Tier funnel" in md
    assert "## Per-shape" in md and "## Per-clip" in md and "Per-bucket" in md
    assert "b-t3" in md  # the sub-target case appears in failure detail
    assert "T3 only" in md


def test_report_is_diff_stable():
    md1, j1 = report_mod.build_report(_sample_results(), META)
    # shuffle the input order; output must be identical (sorted cases, fixed 0.00 rates).
    shuffled = list(reversed(_sample_results()))
    md2, j2 = report_mod.build_report(shuffled, META)
    assert md1 == md2
    assert j1 == j2


def test_report_escapes_hostile_strings():
    """Control/bidi chars in a case detail are escaped (repr-style) so they can't garble logs."""
    hostile_detail = "answer " + chr(0x202E) + "evil" + chr(0x07)  # RLO override + BEL control char
    r = _result("hostile", scoring.T3, False, detail=hostile_detail)
    _, j = report_mod.build_report([r], META)
    hostile = next(c for c in j["cases"] if c["case_id"] == "hostile")
    assert chr(0x07) not in hostile["detail"]  # raw BEL must be escaped, not present
    assert "\\x07" in hostile["detail"]
    assert "\\u202e" in hostile["detail"]
