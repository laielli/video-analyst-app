"""Report artifact (D5): build_report(results, meta) -> (markdown_str, json_obj).

Two artifacts: report.json (machine, diffable source of truth) + report.md (human, generated FROM
the JSON so they never disagree). Both regenerated fully each run (no append). Cases sorted by
case_id and rates fixed to 0.00 so re-rendering the same report is byte-identical (diff-stable).

ESCAPE DISCIPLINE: the bank deliberately carries hostile strings (injection payloads, RTL/unicode
noise). Render every case `question` and echoed answer with control/bidi chars escaped (repr-style)
so a raw payload can't garble CI logs or diffs. See the eval README.
"""
from __future__ import annotations

from .scoring import (
    REJECTED,
    T0_PARSED,
    T1_SCHEMA,
    T2_SEMANTIC,
    T3_GROUNDED,
    T4_CORRECT,
    TIER_RANK,
)

# The funnel tiers, in ladder order, with the report label.
FUNNEL_TIERS = [
    (T0_PARSED, "T0_parsed"),
    (T1_SCHEMA, "T1_schema"),
    (T2_SEMANTIC, "T2_semantic"),
    (T3_GROUNDED, "T3_grounded"),
    (T4_CORRECT, "T4_correct"),
]

DETERMINISM_FOOTER = (
    "_Determinism: scores reflect the FROZEN capture, not a fresh call. temperature=0 does not "
    "guarantee bit-identical gpt-4o output; this report scores recorded fixtures. Re-capture shifts "
    "surface as a report.json diff, never a silent change._"
)


def _esc(s) -> str:
    """repr-style escape of control/bidi/unicode noise, then strip the surrounding quotes repr adds.
    Keeps hostile question/answer text from garbling logs and diffs (D5 escape discipline)."""
    r = repr("" if s is None else str(s))
    return r[1:-1] if len(r) >= 2 else r


def _rate(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _fmt(rate: float) -> str:
    return f"{rate:.2f}"


def _reached_at_least(result, tier: str) -> bool:
    return TIER_RANK.get(result.reached_tier, -1) >= TIER_RANK[tier]


def _is_oos_distance(result) -> bool:
    # out_of_scope/injection/degenerate cases carry distance "n/a"; absent-subject in-scope cases
    # are tagged first-goal/canonical|paraphrase with a real distance but expect ungrounded.
    return result.phrasing_class in ("out_of_scope", "injection", "degenerate")


def _funnel(results) -> dict:
    """Tier funnel counts over IN-SCOPE-groundable cases (canonical/paraphrase that expect grounded).
    Out-of-scope and absent-subject cases are reported on their own lines, not in the funnel."""
    funnel_cases = [r for r in results if r.target_tier in (T3_GROUNDED, T4_CORRECT)
                    and not _is_oos_distance(r)]
    total = len(funnel_cases)
    counts = {}
    for tier, _label in FUNNEL_TIERS:
        counts[tier] = sum(1 for r in funnel_cases if _reached_at_least(r, tier))
    return {"total": total, "counts": counts}


def _group_rates(results, key) -> dict:
    """Per-group T2/T3/T4 reach rates over in-scope-groundable cases grouped by key(result)."""
    groups: dict = {}
    for r in results:
        if r.target_tier not in (T3_GROUNDED, T4_CORRECT) or _is_oos_distance(r):
            continue
        groups.setdefault(key(r), []).append(r)
    out = {}
    for g, rs in groups.items():
        den = len(rs)
        out[g] = {
            "n": den,
            "T2": _rate(sum(1 for r in rs if _reached_at_least(r, T2_SEMANTIC)), den),
            "T3": _rate(sum(1 for r in rs if _reached_at_least(r, T3_GROUNDED)), den),
            "T4": _rate(sum(1 for r in rs if _reached_at_least(r, T4_CORRECT)), den),
        }
    return out


def _oos_rate(results) -> dict:
    """Honest-refusal rate over out_of_scope/degenerate cases + the absent-subject in-scope cases."""
    oos = [r for r in results if r.target_tier == REJECTED]
    den = len(oos)
    return {"n": den, "rate": _rate(sum(1 for r in oos if r.passed), den)}


def _adversarial_rate(results) -> dict:
    adv = [r for r in results if r.phrasing_class == "injection"]
    den = len(adv)
    return {"n": den, "rate": _rate(sum(1 for r in adv if r.passed), den)}


def build_report(results, meta: dict) -> tuple[str, dict]:
    """Build (markdown, json_obj) from a CaseResult list + meta. Pure; deterministic given inputs."""
    results = sorted(results, key=lambda r: r.case_id)
    funnel = _funnel(results)
    per_shape = _group_rates(results, lambda r: r.shape)
    per_clip = _group_rates(results, lambda r: r.clip_id)
    per_bucket = _bucket_rates(results)
    oos = _oos_rate(results)
    adv = _adversarial_rate(results)
    stale_n = sum(1 for r in results if r.stale)
    missing_n = sum(1 for r in results if r.missing)

    json_obj = {
        "summary": {
            "bank_version": meta.get("bank_version"),
            "prompt_version": meta.get("prompt_version"),
            "captured_at": meta.get("captured_at"),
            "model": meta.get("model"),
            "mode": meta.get("mode", "replay"),
            "total_cases": len(results),
            "funnel": funnel,
            "out_of_scope_honest": oos,
            "adversarial_safe": adv,
            "stale": stale_n,
            "missing": missing_n,
            "estimated_capture_cost_usd": meta.get("estimated_capture_cost_usd"),
            "prompt_chars": meta.get("prompt_chars", {}),
            "total_program_bytes": sum(r.program_bytes for r in results),
        },
        "per_shape": per_shape,
        "per_clip": per_clip,
        "per_bucket": per_bucket,
        "cases": [
            {
                "case_id": r.case_id,
                "clip_id": r.clip_id,
                "shape": r.shape,
                "phrasing_class": r.phrasing_class,
                "distance": r.distance,
                "target_tier": r.target_tier,
                "reached_tier": r.reached_tier,
                "passed": r.passed,
                "detail": _esc(r.detail),
                "usage": r.usage,
                "stale": r.stale,
                "missing": r.missing,
                "program_bytes": r.program_bytes,
            }
            for r in results
        ],
    }
    md = _render_md(results, json_obj, funnel, per_shape, per_clip, per_bucket, oos, adv,
                    stale_n, missing_n, meta)
    return md, json_obj


def _bucket_rates(results) -> dict:
    """Per-paraphrase-distance / bucket breakdown: canonical | near | far | out_of_scope | injection."""
    def bucket(r):
        if r.phrasing_class in ("out_of_scope", "degenerate"):
            return "out_of_scope"
        if r.phrasing_class == "injection":
            return "injection"
        return r.distance  # canonical | near | far

    groups: dict = {}
    for r in results:
        groups.setdefault(bucket(r), []).append(r)
    out = {}
    for g, rs in groups.items():
        den = len(rs)
        out[g] = {"n": den, "pass_rate": _rate(sum(1 for r in rs if r.passed), den)}
    return out


def _render_md(results, json_obj, funnel, per_shape, per_clip, per_bucket, oos, adv,
               stale_n, missing_n, meta) -> str:
    s = json_obj["summary"]
    lines: list[str] = []
    lines.append("# Codegen eval report")
    lines.append("")
    lines.append(f"- bank_version: `{s['bank_version']}`")
    lines.append(f"- prompt_version: `{s['prompt_version']}`")
    lines.append(f"- captured_at: `{s['captured_at']}`")
    lines.append(f"- model: `{s['model']}`")
    lines.append(f"- mode: `{s['mode']}`")
    lines.append(f"- total cases: {s['total_cases']}")
    cost = s.get("estimated_capture_cost_usd")
    lines.append(f"- estimated capture cost (offline): "
                 f"{'$' + format(cost, '.2f') if cost is not None else 'n/a'} "
                 f"(usage is null per Resolved #1)")
    pc = s.get("prompt_chars") or {}
    if pc:
        lines.append("- prompt size (chars): " + ", ".join(f"{k}={v}" for k, v in sorted(pc.items())))
    lines.append(f"- total raw_program bytes: {s['total_program_bytes']}")
    lines.append(f"- STALE fixtures: {stale_n} | MISSING fixtures: {missing_n}")
    if stale_n:
        lines.append("")
        lines.append("> **FIXTURES STALE — re-capture needed.** Some/all fixtures were captured "
                     "against a different prompt_version than the current prompt.")
    lines.append("")

    # Tier funnel (headline).
    total = funnel["total"]
    funnel_str = " -> ".join(f"{label} {funnel['counts'][tier]}/{total}"
                             for tier, label in FUNNEL_TIERS)
    lines.append("## Tier funnel (in-scope groundable)")
    lines.append("")
    lines.append(funnel_str if total else "_(no in-scope groundable cases)_")
    lines.append("")
    lines.append(f"- out-of-scope honest-refusal rate: {_fmt(oos['rate'])} ({oos['n']} cases)")
    lines.append(f"- adversarial-safe rate: {_fmt(adv['rate'])} ({adv['n']} cases)")
    lines.append("")

    # Per-shape table.
    lines.append("## Per-shape (T2 / T3 / T4 reach rates)")
    lines.append("")
    lines.append("| shape | n | T2 | T3 | T4 |")
    lines.append("|---|---|---|---|---|")
    for shape in sorted(per_shape):
        g = per_shape[shape]
        lines.append(f"| {shape} | {g['n']} | {_fmt(g['T2'])} | {_fmt(g['T3'])} | {_fmt(g['T4'])} |")
    lines.append("")

    # Per-clip table.
    lines.append("## Per-clip (T2 / T3 / T4 reach rates)")
    lines.append("")
    lines.append("| clip | n | T2 | T3 | T4 |")
    lines.append("|---|---|---|---|---|")
    for clip in sorted(per_clip):
        g = per_clip[clip]
        lines.append(f"| {clip} | {g['n']} | {_fmt(g['T2'])} | {_fmt(g['T3'])} | {_fmt(g['T4'])} |")
    lines.append("")

    # Per-bucket / paraphrase-distance table.
    lines.append("## Per-bucket / paraphrase-distance (pass rate)")
    lines.append("")
    lines.append("| bucket | n | pass rate |")
    lines.append("|---|---|---|")
    for bucket in sorted(per_bucket):
        g = per_bucket[bucket]
        lines.append(f"| {bucket} | {g['n']} | {_fmt(g['pass_rate'])} |")
    lines.append("")

    # Failure detail: every case below its target tier (or a failing out-of-scope/adversarial case).
    failures = [r for r in results if not r.passed]
    lines.append(f"## Failure detail ({len(failures)} case(s) below target)")
    lines.append("")
    if failures:
        lines.append("| case_id | reached | target | detail | flag |")
        lines.append("|---|---|---|---|---|")
        for r in failures:
            flag = "STALE" if r.stale else ("MISSING" if r.missing else "")
            lines.append(f"| {r.case_id} | {r.reached_tier} | {r.target_tier} | "
                         f"{_esc(r.detail)} | {flag} |")
    else:
        lines.append("_(no failures)_")
    lines.append("")
    lines.append(DETERMINISM_FOOTER)
    lines.append("")
    return "\n".join(lines)
