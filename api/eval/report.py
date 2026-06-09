"""
Report artifact (D5): machine `report.json` (the diffable source of truth) + human `report.md`
(generated FROM the json so they never disagree). Both regenerate fully each run (no append) and
are deterministic given the fixtures: cases sorted by `case_id`, rates fixed to `0.00`, so
re-rendering the same report is byte-identical.

Escape discipline: case `question` text and echoed answers are rendered repr-style (control/bidi
chars escaped) — the bank deliberately carries hostile strings (injection payloads, RTL noise),
and raw rendering could garble CI logs and diffs.

Pure formatting over the CaseResult list + a meta dict. No scoring logic, no IO of its own beyond
the atomic write helper.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import scoring
from .scoring import (
    T0_PARSED,
    T1_SCHEMA,
    T2_SEMANTIC,
    T3_GROUNDED,
    T4_CORRECT,
    TIER_ORDER,
)

# The tier-funnel rungs, in monotone order (REJECTED/STALE/MISSING are off-ladder, reported apart).
FUNNEL_TIERS = [T0_PARSED, T1_SCHEMA, T2_SEMANTIC, T3_GROUNDED, T4_CORRECT]

DETERMINISM_FOOTER = (
    "_Determinism: scores reflect the FROZEN capture, not a fresh call. `temperature=0` does not "
    "guarantee bit-identical gpt-4o output; replay scores recorded `raw_program` over committed "
    "caches, so the gate never flakes. A re-capture shift surfaces as a `report.json` diff._"
)


def _esc(s) -> str:
    """repr-style escape so hostile strings (control/bidi/unicode) never garble logs or diffs."""
    if s is None:
        return ""
    return repr(str(s))[1:-1]  # drop the surrounding quotes repr() adds


def _rate(n: int, d: int) -> str:
    return f"{(n / d):.2f}" if d else "0.00"


def _reached_at_least(reached: str, tier: str) -> bool:
    """Whether `reached` is at or above `tier` on the ladder (off-ladder STALE/MISSING never are)."""
    if reached not in TIER_ORDER or tier not in TIER_ORDER:
        return False
    return TIER_ORDER.index(reached) >= TIER_ORDER.index(tier)


def _in_scope(case: dict) -> bool:
    return case["expect"]["outcome"] == "grounded"


def build_report(results: list, cases_by_id: dict, meta: dict) -> tuple[str, dict]:
    """Build (markdown_str, json_obj) from the CaseResult list. `cases_by_id` maps case_id -> bank
    case (for shape/clip/phrasing rollups). `meta` carries header provenance."""
    results = sorted(results, key=lambda r: r.case_id)

    in_scope_results = [r for r in results if _in_scope(cases_by_id[r.case_id])]
    oos_results = [r for r in results
                   if cases_by_id[r.case_id]["expect"]["outcome"] == "ungrounded"]
    adversarial = [r for r in results
                   if cases_by_id[r.case_id]["phrasing_class"] in ("injection", "degenerate")]

    n_total = len(results)
    n_stale = sum(1 for r in results if r.stale)
    n_missing = sum(1 for r in results if r.missing)

    # Tier funnel over in-scope cases (the headline). Each rung counts in-scope cases reaching it.
    funnel = {}
    n_in = len(in_scope_results)
    for tier in FUNNEL_TIERS:
        funnel[tier] = sum(1 for r in in_scope_results if _reached_at_least(r.reached_tier, tier))

    # Out-of-scope honest-refusal rate + adversarial-safe rate (their own lines).
    oos_pass = sum(1 for r in oos_results if r.passed)
    adv_pass = sum(1 for r in adversarial if r.passed)

    per_shape = _rollup(in_scope_results, cases_by_id, "shape")
    per_clip = _rollup(in_scope_results, cases_by_id, "clip_id")
    per_bucket = _rollup_bucket(results, cases_by_id)

    summary = {
        "total_cases": n_total,
        "in_scope_cases": n_in,
        "out_of_scope_cases": len(oos_results),
        "adversarial_cases": len(adversarial),
        "stale": n_stale,
        "missing": n_missing,
        "funnel": {t: {"reached": funnel[t], "of": n_in, "rate": _rate(funnel[t], n_in)}
                   for t in FUNNEL_TIERS},
        "out_of_scope_honest": {"passed": oos_pass, "of": len(oos_results),
                                "rate": _rate(oos_pass, len(oos_results))},
        "adversarial_safe": {"passed": adv_pass, "of": len(adversarial),
                             "rate": _rate(adv_pass, len(adversarial))},
    }

    json_cases = [
        {
            "case_id": r.case_id,
            "target_tier": r.target_tier,
            "reached_tier": r.reached_tier,
            "passed": r.passed,
            "detail": r.detail,
            "usage": r.usage,
            "stale": r.stale,
            "missing": r.missing,
        }
        for r in results
    ]

    json_obj = {
        "meta": meta,
        "summary": summary,
        "per_shape": per_shape,
        "per_clip": per_clip,
        "per_bucket": per_bucket,
        "cases": json_cases,
    }

    md = _render_markdown(json_obj, results, cases_by_id)
    return md, json_obj


def _rollup(results: list, cases_by_id: dict, key: str) -> dict:
    """Per-{shape|clip} T2/T3/T4 rates over the given (in-scope) results."""
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(cases_by_id[r.case_id][key], []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        d = len(rs)
        out[g] = {
            "of": d,
            "T2_semantic": _rate(sum(1 for r in rs if _reached_at_least(r.reached_tier, T2_SEMANTIC)), d),
            "T3_grounded": _rate(sum(1 for r in rs if _reached_at_least(r.reached_tier, T3_GROUNDED)), d),
            "T4_correct": _rate(sum(1 for r in rs if _reached_at_least(r.reached_tier, T4_CORRECT)), d),
        }
    return out


def _rollup_bucket(results: list, cases_by_id: dict) -> dict:
    """Per paraphrase-distance / phrasing bucket: pass-rate. Buckets: canonical, near, far,
    out_of_scope, injection, degenerate (the cut that localizes brittleness to phrasing drift)."""
    def bucket(case: dict) -> str:
        pc = case["phrasing_class"]
        if pc == "canonical":
            return "canonical"
        if pc == "paraphrase":
            return f"paraphrase-{case['distance']}"
        return pc

    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(bucket(cases_by_id[r.case_id]), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        d = len(rs)
        out[g] = {"of": d, "pass_rate": _rate(sum(1 for r in rs if r.passed), d)}
    return out


def _render_markdown(j: dict, results: list, cases_by_id: dict) -> str:
    meta, s = j["meta"], j["summary"]
    lines: list[str] = []
    lines.append("# Codegen eval report")
    lines.append("")
    lines.append(f"- bank_version: `{meta.get('bank_version')}`")
    lines.append(f"- prompt_version (current): `{meta.get('prompt_version')}`")
    lines.append(f"- captured_at: `{meta.get('captured_at')}`")
    lines.append(f"- model: `{meta.get('model')}`")
    lines.append(f"- mode: `{meta.get('mode')}`")
    lines.append(f"- total cases: {s['total_cases']} "
                 f"(in-scope {s['in_scope_cases']}, out-of-scope {s['out_of_scope_cases']}, "
                 f"adversarial {s['adversarial_cases']})")
    lines.append(f"- estimated capture cost: {meta.get('est_cost', 'n/a')} "
                 f"(offline estimate — `usage` is null per Resolved #1)")
    for clip_id, size in sorted((meta.get("prompt_sizes") or {}).items()):
        lines.append(f"- prompt size [{clip_id}]: {size} chars")
    lines.append(f"- total raw_program bytes: {meta.get('total_raw_bytes', 0)}")
    lines.append(f"- STALE fixtures: {s['stale']}  ·  MISSING fixtures: {s['missing']}")
    if s["stale"] and s["stale"] == s["total_cases"]:
        lines.append("")
        lines.append("> **FIXTURES STALE — re-capture needed.** All fixtures were captured against a "
                     "different prompt version. The gate is advisory (D6 uniform-stale bypass).")
    elif s["stale"]:
        lines.append("")
        lines.append("> **PARTIAL STALENESS — re-capture needed.** Some fixtures drifted from the "
                     "current prompt; stale fixtures count as failures beyond `max_stale` (D6).")
    lines.append("")

    # Tier funnel headline.
    funnel_parts = []
    for t in FUNNEL_TIERS:
        f = s["funnel"][t]
        funnel_parts.append(f"{t} {f['reached']}/{f['of']}")
    lines.append("## Tier funnel (in-scope)")
    lines.append("")
    lines.append(f"`{' → '.join(funnel_parts)}`")
    lines.append("")
    oos = s["out_of_scope_honest"]
    adv = s["adversarial_safe"]
    lines.append(f"- out-of-scope honest-refusal: {oos['passed']}/{oos['of']} (rate {oos['rate']})")
    lines.append(f"- adversarial-safe: {adv['passed']}/{adv['of']} (rate {adv['rate']})")
    lines.append("")

    # Per-shape table.
    lines.append("## Per-shape (in-scope)")
    lines.append("")
    lines.append("| shape | n | T2 | T3 | T4 |")
    lines.append("|---|---|---|---|---|")
    for shape, row in j["per_shape"].items():
        lines.append(f"| {shape} | {row['of']} | {row['T2_semantic']} | {row['T3_grounded']} | {row['T4_correct']} |")
    lines.append("")

    # Per-clip table.
    lines.append("## Per-clip (in-scope)")
    lines.append("")
    lines.append("| clip | n | T2 | T3 | T4 |")
    lines.append("|---|---|---|---|---|")
    for clip_id, row in j["per_clip"].items():
        lines.append(f"| {clip_id} | {row['of']} | {row['T2_semantic']} | {row['T3_grounded']} | {row['T4_correct']} |")
    lines.append("")

    # Per-bucket / paraphrase-distance table.
    lines.append("## Per-bucket (phrasing / paraphrase distance)")
    lines.append("")
    lines.append("| bucket | n | pass rate |")
    lines.append("|---|---|---|")
    for bucket, row in j["per_bucket"].items():
        lines.append(f"| {bucket} | {row['of']} | {row['pass_rate']} |")
    lines.append("")

    # Failure detail: every case below its target tier (or failed), one line.
    failures = [r for r in results if not r.passed or not scoring.reached_target(r) or r.stale or r.missing]
    lines.append("## Failure detail")
    lines.append("")
    if not failures:
        lines.append("_No cases below target tier._")
    else:
        lines.append("| case_id | reached | passed | flag | detail |")
        lines.append("|---|---|---|---|---|")
        for r in failures:
            flag = "STALE" if r.stale else ("MISSING" if r.missing else "")
            lines.append(f"| `{r.case_id}` | {r.reached_tier} | {r.passed} | {flag} | {_esc(r.detail)} |")
    lines.append("")
    lines.append(DETERMINISM_FOOTER)
    lines.append("")
    return "\n".join(lines)


def write_report(md: str, json_obj: dict, md_path: Path, json_path: Path) -> None:
    """Atomically write both artifacts (mkstemp + os.replace)."""
    _atomic_write(md_path, md)
    _atomic_write(json_path, json.dumps(json_obj, indent=2, ensure_ascii=False) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
