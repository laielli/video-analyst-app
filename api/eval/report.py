"""
Report artifact (D5). Replay writes two artifacts: report.json (machine, diffable source of truth)
and report.md (human, generated FROM the JSON so they never disagree). Both are regenerated fully
each run (no append), cases sorted by case_id, rates fixed to 0.00 — so re-rendering the same
report is byte-identical (diff-stable).

Pure formatting over a list of scoring.CaseResult. No IO except write_report (atomic).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .scoring import (
    MISSING,
    STALE,
    T0_PARSED,
    T1_SCHEMA,
    T2_SEMANTIC,
    T3_GROUNDED,
    T4_CORRECT,
    TIER_ORDER,
)

DETERMINISM_FOOTER = (
    "Scores reflect the FROZEN capture (recorded gpt-4o output), not a fresh model call. "
    "temperature=0 does not guarantee bit-identical output; replay is deterministic over the "
    "committed fixtures + caches. Re-capture on a prompt edit and review the report.json diff."
)

ESCAPE_NOTE = (
    "Question/answer text is rendered repr-escaped: the bank deliberately contains hostile strings "
    "(injection payloads, RTL/unicode noise) and raw rendering could garble CI logs and diffs."
)


def _rate(num: int, den: int) -> float:
    return round(num / den, 2) if den else 0.0


def _fmt(r: float) -> str:
    return f"{r:.2f}"


def _reached_at_least(reached: str, tier: str) -> bool:
    """True iff `reached` is at least `tier` on the monotone ladder. STALE/MISSING are below all."""
    if reached in (STALE, MISSING):
        return False
    try:
        return TIER_ORDER.index(reached) >= TIER_ORDER.index(tier)
    except ValueError:
        return False


def _esc(s) -> str:
    """repr-escape control/bidi chars (D5 escape discipline) without the surrounding quotes."""
    return repr(str(s))[1:-1]


def build_report(results, meta: dict):
    """Return (markdown_str, json_obj). `results` is a list of scoring.CaseResult; `meta` carries
    bank_version / prompt_version / model / mode / clip prompt sizes / capture timestamp / cost."""
    results = sorted(results, key=lambda r: r.case_id)
    total = len(results)
    in_scope = [r for r in results if r.phrasing_class in ("canonical", "paraphrase")]
    out_of_scope = [r for r in results if r.phrasing_class == "out_of_scope"]
    injection = [r for r in results if r.phrasing_class == "injection"]
    degenerate = [r for r in results if r.phrasing_class == "degenerate"]

    stale_n = sum(1 for r in results if r.stale)
    missing_n = sum(1 for r in results if r.missing)
    total_program_bytes = sum(r.program_bytes for r in results)

    # Tier funnel over IN-SCOPE cases (the headline). Each tier counts cases that reached >= it.
    funnel_tiers = [T0_PARSED, T1_SCHEMA, T2_SEMANTIC, T3_GROUNDED, T4_CORRECT]
    funnel = {t: sum(1 for r in in_scope if _reached_at_least(r.reached_tier, t)) for t in funnel_tiers}

    oos_honest = sum(1 for r in out_of_scope if r.passed)
    adv_safe = sum(1 for r in injection if r.passed)
    degen_safe = sum(1 for r in degenerate if r.passed)

    # Per-shape / per-clip / per-bucket tables.
    per_shape = _group_rates(in_scope, key=lambda r: r.shape)
    per_clip = _group_rates(in_scope, key=lambda r: r.clip_id)
    per_bucket = _bucket_rates(results)

    summary = {
        "total_cases": total,
        "in_scope": len(in_scope),
        "out_of_scope": len(out_of_scope),
        "injection": len(injection),
        "degenerate": len(degenerate),
        "stale": stale_n,
        "missing": missing_n,
        "total_program_bytes": total_program_bytes,
        "funnel": {t: funnel[t] for t in funnel_tiers},
        "funnel_denominator": len(in_scope),
        "out_of_scope_honest": oos_honest,
        "out_of_scope_honest_rate": _rate(oos_honest, len(out_of_scope)),
        "adversarial_safe": adv_safe,
        "adversarial_safe_rate": _rate(adv_safe, len(injection)),
        "degenerate_safe": degen_safe,
        "degenerate_safe_rate": _rate(degen_safe, len(degenerate)),
        # Tier RATES the threshold gate keys on (in-scope denominator).
        "T2_semantic_valid": _rate(funnel[T2_SEMANTIC], len(in_scope)),
        "T3_executes_grounded": _rate(funnel[T3_GROUNDED], len(in_scope)),
        "T4_answer_correct": _rate(funnel[T4_CORRECT], len(in_scope)),
    }

    json_obj = {
        "meta": meta,
        "summary": summary,
        "per_shape": per_shape,
        "per_clip": per_clip,
        "per_bucket": per_bucket,
        "cases": [_case_record(r) for r in results],
    }
    md = _render_md(json_obj, results, meta)
    return md, json_obj


def _case_record(r) -> dict:
    return {
        "case_id": r.case_id,
        "clip_id": r.clip_id,
        "shape": r.shape,
        "phrasing_class": r.phrasing_class,
        "distance": r.distance,
        "target_tier": r.target_tier,
        "reached_tier": r.reached_tier,
        "passed": r.passed,
        "stale": r.stale,
        "missing": r.missing,
        "overclaimed": r.overclaimed,
        "program_bytes": r.program_bytes,
        "usage": r.usage,
        "detail": r.detail,
    }


def _group_rates(results, key):
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for g in sorted(groups):
        rs = groups[g]
        den = len(rs)
        out[g] = {
            "count": den,
            "T2": _rate(sum(1 for r in rs if _reached_at_least(r.reached_tier, T2_SEMANTIC)), den),
            "T3": _rate(sum(1 for r in rs if _reached_at_least(r.reached_tier, T3_GROUNDED)), den),
            "T4": _rate(sum(1 for r in rs if _reached_at_least(r.reached_tier, T4_CORRECT)), den),
        }
    return out


def _bucket_rates(results):
    """Per-bucket / per-paraphrase-distance: canonical, near, far, out_of_scope, injection,
    degenerate. The pass RATE answers 'does validity degrade as phrasing drifts from canned?'."""
    def bucket(r):
        if r.phrasing_class in ("out_of_scope", "injection", "degenerate"):
            return r.phrasing_class
        return r.distance if r.distance in ("canonical", "near", "far") else "other"

    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(bucket(r), []).append(r)
    order = ["canonical", "near", "far", "out_of_scope", "injection", "degenerate", "other"]
    out = {}
    for g in order:
        if g not in groups:
            continue
        rs = groups[g]
        out[g] = {"count": len(rs), "pass_rate": _rate(sum(1 for r in rs if r.passed), len(rs))}
    return out


def _render_md(j: dict, results, meta: dict) -> str:
    s = j["summary"]
    f = s["funnel"]
    den = s["funnel_denominator"]
    lines = []
    lines.append("# Codegen eval report")
    lines.append("")
    lines.append(f"- bank_version: {meta.get('bank_version')}")
    lines.append(f"- prompt_version: {meta.get('prompt_version')}")
    lines.append(f"- captured_at: {meta.get('captured_at')}")
    lines.append(f"- model: {meta.get('model')}")
    lines.append(f"- mode: {meta.get('mode')}")
    lines.append(f"- total cases: {s['total_cases']} (in-scope {s['in_scope']}, "
                 f"out-of-scope {s['out_of_scope']}, injection {s['injection']}, degenerate {s['degenerate']})")
    lines.append(f"- estimated capture cost (offline): {meta.get('est_cost', 'n/a')}  "
                 "(usage is null — Resolved #1; estimate only)")
    for clip_id, size in sorted((meta.get("clip_prompt_chars") or {}).items()):
        lines.append(f"- prompt size [{clip_id}]: {size} chars")
    lines.append(f"- total raw_program bytes: {s['total_program_bytes']}")
    lines.append(f"- STALE fixtures: {s['stale']}    MISSING fixtures: {s['missing']}")
    if s["stale"] and s["stale"] == s["total_cases"]:
        lines.append("- **ALL FIXTURES STALE — re-capture needed** (gate downgraded to advisory).")
    elif s["stale"]:
        lines.append("- **PARTIAL STALENESS — re-capture or fix the doctored fixture(s)** "
                     "(stale fixtures count as failures beyond max_stale).")
    lines.append("")
    lines.append("## Tier funnel (in-scope)")
    lines.append("")
    lines.append(
        f"`T0_parsed {f[T0_PARSED]}/{den} -> T1_schema {f[T1_SCHEMA]}/{den} -> "
        f"T2_semantic {f[T2_SEMANTIC]}/{den} -> T3_grounded {f[T3_GROUNDED]}/{den} -> "
        f"T4_correct {f[T4_CORRECT]}/{den}`"
    )
    lines.append("")
    lines.append(f"- out-of-scope honest-refusal: {s['out_of_scope_honest']}/{s['out_of_scope']} "
                 f"(rate {_fmt(s['out_of_scope_honest_rate'])})")
    lines.append(f"- adversarial/injection safe: {s['adversarial_safe']}/{s['injection']} "
                 f"(rate {_fmt(s['adversarial_safe_rate'])})")
    lines.append(f"- degenerate safe: {s['degenerate_safe']}/{s['degenerate']} "
                 f"(rate {_fmt(s['degenerate_safe_rate'])})")
    lines.append("")
    lines.append("## Per-shape (in-scope T2/T3/T4 rates)")
    lines.append("")
    lines.append("| shape | n | T2 | T3 | T4 |")
    lines.append("|---|---|---|---|---|")
    for shape in sorted(j["per_shape"]):
        v = j["per_shape"][shape]
        lines.append(f"| {shape} | {v['count']} | {_fmt(v['T2'])} | {_fmt(v['T3'])} | {_fmt(v['T4'])} |")
    lines.append("")
    lines.append("## Per-clip (in-scope T2/T3/T4 rates)")
    lines.append("")
    lines.append("| clip | n | T2 | T3 | T4 |")
    lines.append("|---|---|---|---|---|")
    for clip_id in sorted(j["per_clip"]):
        v = j["per_clip"][clip_id]
        lines.append(f"| {clip_id} | {v['count']} | {_fmt(v['T2'])} | {_fmt(v['T3'])} | {_fmt(v['T4'])} |")
    lines.append("")
    lines.append("## Per-bucket / paraphrase-distance (pass rate)")
    lines.append("")
    lines.append("| bucket | n | pass rate |")
    lines.append("|---|---|---|")
    for bucket, v in j["per_bucket"].items():
        lines.append(f"| {bucket} | {v['count']} | {_fmt(v['pass_rate'])} |")
    lines.append("")
    lines.append("## Failure / below-target detail")
    lines.append("")
    failures = [r for r in results if (not r.passed) or r.stale or r.missing]
    if not failures:
        lines.append("_All cases passed at or above their target tier._")
    else:
        lines.append("| case_id | reached | detail | flag |")
        lines.append("|---|---|---|---|")
        for r in failures:
            flag = "STALE" if r.stale else ("MISSING" if r.missing else ("OVERCLAIM" if r.overclaimed else ""))
            lines.append(f"| {_esc(r.case_id)} | {r.reached_tier} | {_esc(r.detail)} | {flag} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_{ESCAPE_NOTE}_")
    lines.append("")
    lines.append(f"_{DETERMINISM_FOOTER}_")
    lines.append("")
    return "\n".join(lines)


def write_report(md: str, json_obj: dict, md_path: Path, json_path: Path) -> None:
    """Atomically write both artifacts (mkstemp + os.replace). json sorted-keys + trailing newline
    for byte-stable diffs."""
    _atomic_write(md_path, md)
    _atomic_write(json_path, json.dumps(json_obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
