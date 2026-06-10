"""Report artifact (D5). Replay writes report.json (machine, the diffable source of truth) and
report.md (human, generated FROM the json so they never disagree). Both regenerate fully each run
(no append) and are deterministic given the fixtures: cases sorted by case_id, rates formatted to
a fixed 0.00, so re-rendering the same report is byte-identical.

Pure formatting over the CaseResult list. Escape discipline: case `question` text and any echoed
answer are rendered with control/bidi characters escaped (repr-style) — the bank deliberately
contains hostile strings (injection payloads, RTL/unicode noise) that could garble CI logs/diffs.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from scoring import (  # noqa: E402 — eval package is on sys.path via the runner
    REJECTED,
    STALE,
    T0,
    T1,
    T2,
    T3,
    T4,
    TIER_RANK,
)

EVAL_DIR = Path(__file__).resolve().parent

DETERMINISM_FOOTER = (
    "Scores reflect the FROZEN capture, not a fresh call. temperature=0 is not bit-deterministic "
    "(provider-side MoE routing / fingerprint drift), so the harness scores coarse tier outcomes "
    "over recorded model output — never golden-program equality. A re-capture shift surfaces as a "
    "report.json diff."
)

# Funnel tiers, in ladder order; a case "reached" tier T also counts toward every lower tier.
FUNNEL = [T0, T1, T2, T3, T4]


def _esc(s) -> str:
    """repr-style escape so control/bidi chars in hostile bank strings can't garble logs/diffs.
    Strips the surrounding quotes repr() adds."""
    r = repr("" if s is None else str(s))
    return r[1:-1]


def _rate(n: int, d: int) -> str:
    return "0.00" if d == 0 else f"{n / d:.2f}"


def _reached_at_least(result, tier: str) -> bool:
    return TIER_RANK[result.reached_tier] >= TIER_RANK[tier]


def build_report(results: list, meta: dict) -> tuple[str, dict]:
    """Return (markdown_str, json_obj). `meta` carries provenance: bank_version, prompt_version,
    captured_at, model, mode, prompt_sizes (per-clip char counts), program_bytes (total raw_program
    bytes), thresholds (optional)."""
    cases = sorted(results, key=lambda r: r.case_id)
    in_scope = [r for r in cases if not r.out_of_scope and not r.missing and not r.stale]
    out_scope = [r for r in cases if r.out_of_scope and not r.missing and not r.stale]
    safety = [r for r in cases if r.safety_axis and not r.missing and not r.stale]
    n_stale = sum(1 for r in cases if r.stale)
    n_missing = sum(1 for r in cases if r.missing)

    # Funnel over in-scope cases (the headline). Out-of-scope/safety get their own pass rates.
    total_inscope = len(in_scope)
    funnel = {tier: sum(1 for r in in_scope if _reached_at_least(r, tier)) for tier in FUNNEL}
    oos_honest = sum(1 for r in out_scope if r.passed)
    safety_safe = sum(1 for r in safety if r.passed)

    summary = {
        "bank_version": meta.get("bank_version"),
        "prompt_version": meta.get("prompt_version"),
        "captured_at": meta.get("captured_at"),
        "model": meta.get("model"),
        "mode": meta.get("mode", "replay"),
        "total_cases": len(cases),
        "in_scope": total_inscope,
        "out_of_scope": len(out_scope),
        "safety": len(safety),
        "stale": n_stale,
        "missing": n_missing,
        "funnel": funnel,
        "out_of_scope_honest": {"passed": oos_honest, "total": len(out_scope)},
        "adversarial_safe": {"passed": safety_safe, "total": len(safety)},
        "estimated_capture_cost_usd": meta.get("estimated_capture_cost_usd"),
        "prompt_sizes_chars": meta.get("prompt_sizes", {}),
        "raw_program_total_bytes": meta.get("program_bytes"),
    }

    per_shape = _rollup(cases, key=lambda r: r.shape)
    per_clip = _rollup(cases, key=lambda r: r.clip_id)
    per_bucket = _rollup(cases, key=_bucket)

    json_cases = [
        {
            "case_id": r.case_id,
            "target_tier": r.target_tier,
            "reached_tier": r.reached_tier,
            "passed": r.passed,
            "detail": _esc(r.detail),
            "usage": r.usage,
            "stale": r.stale,
            "missing": r.missing,
            "shape": r.shape,
            "clip_id": r.clip_id,
            "phrasing_class": r.phrasing_class,
            "distance": r.distance,
        }
        for r in cases
    ]

    json_obj = {
        "summary": summary,
        "per_shape": per_shape,
        "per_clip": per_clip,
        "per_bucket": per_bucket,
        "cases": json_cases,
    }

    md = _render_markdown(json_obj, meta)
    return md, json_obj


def _bucket(r) -> str:
    """The per-bucket / per-paraphrase-distance cut: canonical vs near vs far vs out-of-scope vs
    injection vs degenerate — the 'does validity degrade as phrasing drifts?' view."""
    pc = r.phrasing_class
    if pc in ("out_of_scope", "injection", "degenerate"):
        return pc
    # canonical / paraphrase -> split paraphrase by distance.
    if pc == "canonical":
        return "canonical"
    return f"paraphrase-{r.distance}"


def _rollup(cases: list, key) -> dict:
    """Per-axis T2/T3/T4 rates over the cases in each group (in-scope tier semantics; out-of-scope
    groups report their pass rate as the T-cols' 'passed' too, but the funnel cols only count cases
    that reach the tier)."""
    groups: dict[str, list] = {}
    for r in cases:
        if r.missing or r.stale:
            continue
        groups.setdefault(key(r), []).append(r)
    out = {}
    for name, grp in sorted(groups.items()):
        n = len(grp)
        out[name] = {
            "n": n,
            "T2": {"reached": sum(1 for r in grp if _reached_at_least(r, T2)), "rate": _rate(sum(1 for r in grp if _reached_at_least(r, T2)), n)},
            "T3": {"reached": sum(1 for r in grp if _reached_at_least(r, T3)), "rate": _rate(sum(1 for r in grp if _reached_at_least(r, T3)), n)},
            "T4": {"reached": sum(1 for r in grp if _reached_at_least(r, T4)), "rate": _rate(sum(1 for r in grp if _reached_at_least(r, T4)), n)},
            "passed": sum(1 for r in grp if r.passed),
            "pass_rate": _rate(sum(1 for r in grp if r.passed), n),
        }
    return out


def _render_markdown(j: dict, meta: dict) -> str:
    s = j["summary"]
    lines: list[str] = []
    lines.append("# Codegen eval report")
    lines.append("")
    lines.append(f"- bank_version: `{s['bank_version']}`")
    lines.append(f"- prompt_version: `{s['prompt_version']}`")
    lines.append(f"- model: `{s['model']}`  ·  mode: `{s['mode']}`  ·  captured_at: `{s['captured_at']}`")
    lines.append(f"- total cases: {s['total_cases']}  ·  in-scope: {s['in_scope']}  ·  out-of-scope: {s['out_of_scope']}  ·  safety: {s['safety']}")
    lines.append(f"- STALE fixtures: {s['stale']}  ·  MISSING fixtures: {s['missing']}")
    cost = s.get("estimated_capture_cost_usd")
    if cost is not None:
        lines.append(f"- estimated capture cost (offline est., usage is null): ~${cost:.2f}")
    if s.get("prompt_sizes_chars"):
        sizes = ", ".join(f"{k}={v}" for k, v in sorted(s["prompt_sizes_chars"].items()))
        lines.append(f"- prompt size (chars): {sizes}")
    if s.get("raw_program_total_bytes") is not None:
        lines.append(f"- total raw_program bytes: {s['raw_program_total_bytes']}")
    lines.append("")

    # Tier funnel (headline).
    f = s["funnel"]
    funnel_str = " → ".join(f"{tier} {f[tier]}/{s['in_scope']}" for tier in FUNNEL)
    lines.append("## Tier funnel (in-scope)")
    lines.append("")
    lines.append(funnel_str)
    oos = s["out_of_scope_honest"]
    adv = s["adversarial_safe"]
    lines.append("")
    lines.append(f"- out-of-scope honest-refusal rate: {oos['passed']}/{oos['total']} ({_rate(oos['passed'], oos['total'])})")
    lines.append(f"- adversarial-safe rate: {adv['passed']}/{adv['total']} ({_rate(adv['passed'], adv['total'])})")
    lines.append("")

    lines.append("## Per-shape")
    lines.append("")
    lines.extend(_md_table(j["per_shape"]))
    lines.append("")
    lines.append("## Per-clip")
    lines.append("")
    lines.extend(_md_table(j["per_clip"]))
    lines.append("")
    lines.append("## Per-bucket (paraphrase distance)")
    lines.append("")
    lines.extend(_md_table(j["per_bucket"]))
    lines.append("")

    # Failure detail: every case below its target tier.
    lines.append("## Failure detail")
    lines.append("")
    fails = [c for c in j["cases"] if not c["passed"]]
    if not fails:
        lines.append("_All cases passed._")
    else:
        lines.append("| case_id | reached | detail | flag |")
        lines.append("|---|---|---|---|")
        for c in fails:
            flag = "STALE" if c["stale"] else ("MISSING" if c["missing"] else "")
            lines.append(f"| `{c['case_id']}` | {c['reached_tier']} | {c['detail']} | {flag} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_{DETERMINISM_FOOTER}_")
    lines.append("")
    return "\n".join(lines)


def _md_table(rollup: dict) -> list[str]:
    rows = ["| group | n | T2 | T3 | T4 | pass |", "|---|---|---|---|---|---|"]
    for name, d in rollup.items():
        rows.append(
            f"| {name} | {d['n']} | {d['T2']['rate']} | {d['T3']['rate']} | "
            f"{d['T4']['rate']} | {d['pass_rate']} |"
        )
    return rows


def write_report(md: str, json_obj: dict, md_path: Path, json_path: Path) -> None:
    """Atomic write of both artifacts."""
    _atomic_write(md_path, md)
    _atomic_write(json_path, json.dumps(json_obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(tmp, 0o644)  # mkstemp defaults to 0600; these are long-lived committed artifacts
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
