"""Report artifact (D5): a machine `report.json` + a human `report.md` generated FROM the json so
they never disagree. Both are regenerated fully each run (no append), cases sorted by `case_id`,
rates fixed to 0.00 — so re-rendering the same results is byte-identical (diff-stable).

Escape discipline: the bank deliberately contains hostile strings (injection payloads, RTL/unicode
noise). Render case `question` text and any echoed answer with control/bidi chars escaped (repr-style)
so they can't garble CI logs/diffs.
"""
from __future__ import annotations

import json

from scoring import (
    MISSING,
    REJECTED,
    STALE,
    T0_PARSED,
    T1_SCHEMA,
    T2_SEMANTIC,
    T3_GROUNDED,
    T4_CORRECT,
    TIER_RANK,
    CaseResult,
)

FUNNEL_TIERS = [T0_PARSED, T1_SCHEMA, T2_SEMANTIC, T3_GROUNDED, T4_CORRECT]

DETERMINISM_FOOTER = (
    "Scores reflect the frozen capture, not a fresh call. `temperature=0` does not guarantee "
    "bit-identical gpt-4o output; replay is deterministic over the recorded `raw_program` + fixed "
    "caches. A re-capture shift surfaces as a `report.json` diff, never silent. Hostile question "
    "text is escaped repr-style (the bank contains injection/unicode payloads)."
)


def _esc(s) -> str:
    """repr-style escape of control/bidi chars so a hostile string can't garble logs/diffs."""
    if s is None:
        return ""
    return repr(str(s))[1:-1]


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _fmt_rate(num: int, den: int) -> str:
    return f"{(num / den if den else 0.0):.2f}"


def _is_in_scope(r: CaseResult) -> bool:
    return r.expected_outcome == "grounded"


def _reached_at_least(r: CaseResult, tier: str) -> bool:
    if r.reached_tier in (MISSING, STALE):
        return False
    return TIER_RANK.get(r.reached_tier, -1) >= TIER_RANK[tier]


def build_report(results: list[CaseResult], meta: dict) -> tuple[str, dict]:
    """Return (markdown_str, json_obj). `meta` carries bank_version, prompt_version, captured_at,
    model, mode, prompt sizes, stale/missing counts, cost estimate."""
    results = sorted(results, key=lambda r: r.case_id)
    in_scope = [r for r in results if _is_in_scope(r)]
    out_of_scope = [r for r in results if not _is_in_scope(r) and r.shape != "adversarial"]
    adversarial = [r for r in results if r.shape == "adversarial"]

    # --- funnel over in-scope cases (each tier counts cases that reached >= that tier) ---
    funnel = {tier: sum(1 for r in in_scope if _reached_at_least(r, tier)) for tier in FUNNEL_TIERS}
    in_total = len(in_scope)

    summary = {
        "total_cases": len(results),
        "in_scope": in_total,
        "out_of_scope": len(out_of_scope),
        "adversarial": len(adversarial),
        "stale": sum(1 for r in results if r.stale),
        "missing": sum(1 for r in results if r.missing),
        "funnel": {tier: {"reached": funnel[tier], "rate": _rate(funnel[tier], in_total)} for tier in FUNNEL_TIERS},
        "out_of_scope_honest": _rate(sum(1 for r in out_of_scope if r.passed), len(out_of_scope)),
        "adversarial_safe": _rate(sum(1 for r in adversarial if r.passed), len(adversarial)),
        "passed": sum(1 for r in results if r.passed),
    }

    per_shape = _group_rates(results, key=lambda r: r.shape)
    per_clip = _group_rates(results, key=lambda r: r.clip_id)
    per_bucket = _group_rates(results, key=_bucket)

    case_records = [
        {
            "case_id": r.case_id,
            "clip_id": r.clip_id,
            "shape": r.shape,
            "phrasing_class": r.phrasing_class,
            "distance": r.distance,
            "target_tier": r.target_tier,
            "reached_tier": r.reached_tier,
            "passed": r.passed,
            "detail": r.detail,
            "usage": r.usage,
            "stale": r.stale,
            "missing": r.missing,
            "expected_answer": r.expected_answer,
            "actual_answer": r.actual_answer,
            "raw_program_bytes": r.raw_program_bytes,
        }
        for r in results
    ]

    report_json = {
        "summary": summary,
        "meta": meta,
        "per_shape": per_shape,
        "per_clip": per_clip,
        "per_bucket": per_bucket,
        "cases": case_records,
    }

    md = _render_markdown(results, summary, meta, per_shape, per_clip, per_bucket)
    return md, report_json


def _bucket(r: CaseResult) -> str:
    """The paraphrase-distance / class bucket for the brittleness cut."""
    if r.phrasing_class == "canonical":
        return "in_scope_canonical"
    if r.phrasing_class == "paraphrase":
        return f"in_scope_{r.distance}"
    return r.phrasing_class  # out_of_scope | injection | degenerate


def _group_rates(results: list[CaseResult], key) -> dict:
    groups: dict[str, list[CaseResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for name in sorted(groups):
        rs = groups[name]
        in_scope = [r for r in rs if _is_in_scope(r)]
        n_is = len(in_scope)
        out[name] = {
            "n": len(rs),
            "passed": _rate(sum(1 for r in rs if r.passed), len(rs)),
            "T2": _rate(sum(1 for r in in_scope if _reached_at_least(r, T2_SEMANTIC)), n_is),
            "T3": _rate(sum(1 for r in in_scope if _reached_at_least(r, T3_GROUNDED)), n_is),
            "T4": _rate(sum(1 for r in in_scope if _reached_at_least(r, T4_CORRECT)), n_is),
        }
    return out


def _render_markdown(results, summary, meta, per_shape, per_clip, per_bucket) -> str:
    L: list[str] = []
    L.append("# Codegen eval report")
    L.append("")
    L.append(f"- bank_version: `{meta.get('bank_version')}`")
    L.append(f"- prompt_version (current): `{meta.get('prompt_version')}`")
    L.append(f"- captured_at: `{meta.get('captured_at', 'n/a')}`")
    L.append(f"- model: `{meta.get('model', 'n/a')}`")
    L.append(f"- mode: `{meta.get('mode', 'replay')}`")
    L.append(f"- total cases: {summary['total_cases']} "
             f"(in-scope {summary['in_scope']}, out-of-scope {summary['out_of_scope']}, "
             f"adversarial {summary['adversarial']})")
    L.append(f"- estimated capture cost: {meta.get('cost_estimate', 'n/a')} "
             f"(offline estimate; `usage` is null per Resolved #1)")
    sizes = meta.get("prompt_sizes", {})
    if sizes:
        L.append("- prompt size (chars): " + ", ".join(f"{k}={v}" for k, v in sorted(sizes.items())))
    L.append(f"- total raw_program bytes: {meta.get('total_program_bytes', 0)}")
    stale_n, missing_n = summary["stale"], summary["missing"]
    if stale_n or missing_n:
        L.append(f"- **STALE fixtures: {stale_n}; MISSING fixtures: {missing_n} — re-capture needed**")
    else:
        L.append(f"- STALE fixtures: {stale_n}; MISSING fixtures: {missing_n}")
    L.append("")

    # Funnel
    L.append("## Tier funnel (in-scope)")
    L.append("")
    funnel = summary["funnel"]
    parts = [f"{tier} {funnel[tier]['reached']}/{summary['in_scope']}" for tier in FUNNEL_TIERS]
    L.append("`" + " -> ".join(parts) + "`")
    L.append("")
    L.append(f"- out-of-scope honest-refusal rate: {summary['out_of_scope_honest']:.2f}")
    L.append(f"- adversarial safe rate: {summary['adversarial_safe']:.2f}")
    L.append("")

    L.append("## Per-shape (in-scope tier rates)")
    L.append("")
    L.append(_table(per_shape, "shape"))
    L.append("")
    L.append("## Per-clip (in-scope tier rates)")
    L.append("")
    L.append(_table(per_clip, "clip"))
    L.append("")
    L.append("## Per-bucket / paraphrase distance")
    L.append("")
    L.append(_table(per_bucket, "bucket"))
    L.append("")

    # Failure detail
    failures = [r for r in results if not r.passed]
    L.append(f"## Failure detail ({len(failures)})")
    L.append("")
    if failures:
        L.append("| case_id | tier reached | detail | flag |")
        L.append("|---------|--------------|--------|------|")
        for r in sorted(failures, key=lambda r: r.case_id):
            flag = "STALE" if r.stale else ("MISSING" if r.missing else "")
            L.append(f"| {r.case_id} | {r.reached_tier} | {_esc(r.detail)} | {flag} |")
    else:
        L.append("_none — every case met its target tier._")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"_{DETERMINISM_FOOTER}_")
    L.append("")
    return "\n".join(L)


def _table(grouped: dict, label: str) -> str:
    rows = [f"| {label} | n | passed | T2 | T3 | T4 |", "|---|---|---|---|---|---|"]
    for name in sorted(grouped):
        g = grouped[name]
        rows.append(f"| {name} | {g['n']} | {g['passed']:.2f} | {g['T2']:.2f} | {g['T3']:.2f} | {g['T4']:.2f} |")
    return "\n".join(rows)


def write_report(md: str, report_json: dict, md_path, json_path) -> None:
    """Write both artifacts atomically (sorted keys for byte-stable json diffs)."""
    import os
    import tempfile
    from pathlib import Path

    for path, payload in ((Path(md_path), md), (Path(json_path), json.dumps(report_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
