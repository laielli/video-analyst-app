#!/usr/bin/env python3
"""
Codegen eval harness CLI (D3): one entry point, two subcommands sharing one scoring path.

    # CAPTURE (live, billed, creds required) — the human runs this once after merge:
    python eval/run_eval.py capture [--only <case_id|shape|clip>] [--limit N] [--force] [--dry-run]

    # REPLAY (free, CI-safe, no creds, no network) — implementers + CI run this:
    python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json]
                                   [--shape S] [--clip C] [--fail-under <tier>=<rate>] [--no-gate]

Exit codes mirror codegen_probe.py: 0 = ok / threshold met, 1 = threshold missed / lint failed,
2 = setup problem (no creds on capture).

Path dance (NOT identical to codegen_probe.py — here HERE = api/eval, so insert api/ AND
api/scripts/ explicitly before importing validate_program/interpreter/canned/codegen; pytest's
`pythonpath = . scripts` applies only under pytest). The CI step and runbook invoke the bare
`python eval/run_eval.py replay` form from api/, so this header is load-bearing.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # api/eval
API_DIR = HERE.parent                            # api
for p in (str(API_DIR), str(API_DIR / "scripts"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import json  # noqa: E402

import bank  # noqa: E402
import canned  # noqa: E402
import fixtures as fx  # noqa: E402
import report as report_mod  # noqa: E402
import scoring  # noqa: E402

THRESHOLDS_PATH = HERE / "thresholds.json"
DEFAULT_REPORT_MD = HERE / "report.md"
DEFAULT_REPORT_JSON = HERE / "report.json"

# Per-invocation ceiling on ACTUAL live calls (capture only). Default = bank size band, so a
# fat-fingered bank can't bill thousands. --force does NOT bypass this; --limit is clamped to it.
MAX_CAPTURE_CALLS = 64

# Offline cost estimate (Resolved #1: usage is null; cost is an offline estimate). gpt-4o pricing
# ~ $2.50/M input, $10/M output; ~900 input + ~400 output tokens per case.
_EST_USD_PER_CALL = (900 * 2.50 + 400 * 10.0) / 1e6

# Provisioning hint (same posture as codegen_probe.py's exit-2 message).
OPENAI_HINT = """\
Azure OpenAI is not configured — set AZURE_OPENAI_* in api/.env (see .env.example).
Capture is the only billed step; replay (the CI gate) needs no creds. Without creds, run:

  python eval/run_eval.py replay

to score the committed seed fixtures. To capture real gpt-4o output, provision a gpt-4o
deployment (see scripts/codegen_probe.py for the az commands) and re-run capture."""


# ============================== REPLAY ==============================

def _load_clip_and_cache(clip_id: str, _cache_memo: dict):
    """Resolve a clip + load its cache (memoized so each clip's cache loads once)."""
    from interpreter import Cache

    if clip_id not in _cache_memo:
        clip = canned.clip_by_id(clip_id)
        _cache_memo[clip_id] = (clip, Cache.load(clip["cache"]) if clip else None)
    return _cache_memo[clip_id]


def replay(cases: list[dict], fixtures_dir: Path | None = None) -> list[scoring.CaseResult]:
    """Score every bank case against its recorded fixture (MISSING/STALE marked). No creds, no
    network. Pushes each fixture's untrusted raw_program through validation_errors BEFORE
    Interpreter.run (D7) by delegating to scoring.score_case."""
    results = []
    cache_memo: dict = {}
    for case in cases:
        clip, cache = _load_clip_and_cache(case["clip_id"], cache_memo)
        fixture = fx.load_fixture(case["case_id"], fixtures_dir)
        is_fresh = True
        if fixture is not None:
            pv = bank.prompt_version(clip)
            is_fresh = fx.is_fresh(fixture, pv)
        results.append(scoring.score_case(case, fixture, clip, cache, is_fresh=is_fresh))
    return results


def _build_meta(results: list[scoring.CaseResult], mode: str) -> dict:
    """Provenance + offline signals for the report (D5)."""
    bank_doc = bank.load_bank_doc()
    prompt_sizes = {}
    for clip_id in sorted({c.clip_id for c in results}):
        clip = canned.clip_by_id(clip_id)
        if clip is not None:
            import codegen

            prompt_sizes[clip_id] = len(codegen.build_system_message(clip))
    # Total raw_program bytes (offline output-size signal) — read off the loaded fixtures.
    total_bytes = 0
    for c in results:
        fixture = fx.load_fixture(c.case_id)
        if fixture is not None and fixture.raw_program is not None:
            total_bytes += len(json.dumps(fixture.raw_program).encode("utf-8"))
    n_calls = sum(1 for c in results if not c.missing)
    return {
        "bank_version": bank_doc.get("bank_version"),
        "prompt_version": bank_doc.get("prompt_version"),
        "captured_at": _capture_stamp(results),
        "model": _capture_model(results),
        "mode": mode,
        "prompt_sizes": prompt_sizes,
        "program_bytes": total_bytes,
        "estimated_capture_cost_usd": round(n_calls * _EST_USD_PER_CALL, 4),
    }


def _capture_stamp(results) -> str:
    for c in results:
        fixture = fx.load_fixture(c.case_id)
        if fixture is not None:
            return fixture.captured_at
    return ""


def _capture_model(results) -> str:
    for c in results:
        fixture = fx.load_fixture(c.case_id)
        if fixture is not None:
            return fixture.model
    return ""


def _rate(n: int, d: int) -> float:
    return 1.0 if d == 0 else n / d


def apply_gate(results: list[scoring.CaseResult], thresholds: dict) -> tuple[int, list[str]]:
    """Evaluate the D6 threshold gate. Returns (exit_code, messages).

    - Uniform all-stale (an honest prompt edit) -> advisory: exit 0, shout re-capture.
    - Partial staleness beyond max_stale -> FAIL (anti-gaming: stale counts as a failed gating
      outcome, never excluded from the denominator).
    - min_fixtures floor catches deletion; in-scope tier floors + out_of_scope_honest gate quality.
    """
    msgs: list[str] = []
    scored = [r for r in results if not r.missing]
    n_stale = sum(1 for r in results if r.stale)
    n_missing = sum(1 for r in results if r.missing)
    total = len(results)

    # min_fixtures floor (deletion guard).
    min_fixtures = thresholds.get("min_fixtures", 0)
    present = total - n_missing
    if present < min_fixtures:
        msgs.append(f"GATE FAIL: {present} fixtures present < min_fixtures {min_fixtures} "
                    f"({n_missing} MISSING)")
        return 1, msgs

    # Uniform all-stale: every present fixture is stale -> advisory downgrade.
    if n_stale > 0 and n_stale == present:
        msgs.append("ADVISORY: ALL fixtures are STALE (prompt edited) — re-capture needed. "
                    "Gate downgraded to advisory (exit 0).")
        return 0, msgs

    # Partial staleness beyond tolerance -> FAIL.
    max_stale = thresholds.get("max_stale", 0)
    if n_stale > max_stale:
        msgs.append(f"GATE FAIL: {n_stale} STALE fixtures > max_stale {max_stale} "
                    "(partial staleness is suspicious; re-capture or revert the prompt edit)")
        return 1, msgs

    # In-scope tier floors.
    in_scope = [r for r in scored if not r.out_of_scope and not r.stale]
    failed = False
    for tier_key, tier_const in (("T2_semantic_valid", scoring.T2),
                                 ("T3_grounded", scoring.T3),
                                 ("T4_answer_correct", scoring.T4)):
        floor = thresholds.get(tier_key)
        if floor is None:
            continue
        reached = sum(1 for r in in_scope if scoring.TIER_RANK[r.reached_tier] >= scoring.TIER_RANK[tier_const])
        rate = _rate(reached, len(in_scope))
        if rate < floor:
            msgs.append(f"GATE FAIL: {tier_key} rate {rate:.2f} < floor {floor:.2f} "
                        f"({reached}/{len(in_scope)})")
            failed = True

    # Out-of-scope honest-refusal floor.
    oos_floor = thresholds.get("out_of_scope_honest")
    if oos_floor is not None:
        oos = [r for r in scored if r.out_of_scope and not r.stale]
        passed = sum(1 for r in oos if r.passed)
        rate = _rate(passed, len(oos))
        if rate < oos_floor:
            msgs.append(f"GATE FAIL: out_of_scope_honest rate {rate:.2f} < floor {oos_floor:.2f} "
                        f"({passed}/{len(oos)})")
            failed = True

    if not failed:
        msgs.append("GATE PASS: all in-scope tier floors + out-of-scope honest-refusal floor met.")
    return (1 if failed else 0), msgs


def cmd_replay(args) -> int:
    cases = bank.load_bank()
    cases = bank.filter_cases(cases, clip=args.clip, shape=args.shape)
    results = replay(cases)

    meta = _build_meta(results, mode="replay")
    if args.fail_under:
        meta["thresholds"] = dict(args.fail_under)
    md, json_obj = report_mod.build_report(results, meta)

    md_path = Path(args.report)
    json_path = Path(args.json)
    report_mod.write_report(md, json_obj, md_path, json_path)

    # Thresholds: CLI --fail-under overrides file; otherwise the committed thresholds.json.
    thresholds = _load_thresholds()
    if args.fail_under:
        thresholds = {**thresholds, **dict(args.fail_under)}

    print(md)
    print(f"\n[report written: {md_path} + {json_path}]", file=sys.stderr)

    if args.no_gate:
        print("[--no-gate: scoring only, gate skipped]", file=sys.stderr)
        return 0

    code, msgs = apply_gate(results, thresholds)
    for m in msgs:
        print(m, file=sys.stderr)
    return code


def _load_thresholds() -> dict:
    raw = json.loads(THRESHOLDS_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ============================== CAPTURE ==============================

def cmd_capture(args) -> int:
    # Lazy import the Azure client; replay never reaches here so it never imports codegen-as-client.
    from ocr_probe import load_dotenv

    load_dotenv(API_DIR / ".env")
    import codegen  # after dotenv so enabled() sees the loaded vars

    if not codegen.enabled():
        print(OPENAI_HINT, file=sys.stderr)
        return 2  # never a silent no-op, never a billed call without creds

    cases = bank.load_bank()
    cases = bank.filter_cases(cases, clip=args.clip, shape=args.shape, only=args.only)

    # Determine which cases need a live call (resume by default).
    to_call = []
    for case in cases:
        clip = canned.clip_by_id(case["clip_id"])
        pv = bank.prompt_version(clip)
        existing = fx.load_fixture(case["case_id"])
        if not args.force and existing is not None and fx.is_fresh(existing, pv):
            continue  # resume-skip: fresh fixture already present
        to_call.append((case, clip, pv))

    # Clamp to the per-invocation ceiling; --limit caps further. --force never bypasses the ceiling.
    ceiling = MAX_CAPTURE_CALLS
    if args.limit is not None:
        ceiling = min(ceiling, args.limit)
    planned = to_call[:ceiling]

    if args.dry_run:
        est = len(planned) * _EST_USD_PER_CALL
        print(f"will make {len(planned)} live calls (~${est:.2f}) "
              f"[{len(to_call)} eligible, ceiling {ceiling}]", file=sys.stderr)
        return 0

    client = codegen.AzureCodegen.from_env()
    made = 0
    for case, clip, pv in planned:
        fixture = _capture_one(client, case, clip, pv)  # pragma: no cover — live network block
        fx.write_fixture(fixture)
        made += 1
    print(f"capture: wrote {made} fixtures ({len(to_call) - made} skipped beyond ceiling/limit)",
          file=sys.stderr)
    return 0


def _capture_one(client, case, clip, prompt_version):  # pragma: no cover — live network block
    """One billed gpt-4o call -> a fixture (usage:null per Resolved #1). On any generate() failure,
    record a fixed error tag (never raw exception text)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model = client._deployment if hasattr(client, "_deployment") else "gpt-4o"
    try:
        raw_program = client.generate(case["question"], clip=clip)
        error = None
    except Exception as e:  # noqa: BLE001 — map to a fixed tag (info-leak discipline)
        raw_program = None
        msg = str(e).lower()
        if "exceeds" in msg and "bytes" in msg:
            error = fx.ERR_OVERSIZED
        elif "missing 'program'" in msg or "json" in msg:
            error = fx.ERR_UNPARSEABLE
        else:
            error = fx.ERR_GENERATION_FAILED
    return fx.Fixture(
        case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
        prompt_version=prompt_version, captured_at=stamp, model=model,
        raw_program=raw_program, usage=None, error=error,
    )


# ============================== ARGPARSE ==============================

def _parse_fail_under(values) -> list[tuple[str, float]]:
    out = []
    for v in values or []:
        if "=" not in v:
            raise argparse.ArgumentTypeError(f"--fail-under expects <tier>=<rate>, got {v!r}")
        tier, rate = v.split("=", 1)
        out.append((tier, float(rate)))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Codegen eval harness — capture/replay")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="live, billed, creds-gated capture of gpt-4o output")
    cap.add_argument("--only", help="narrow to a case_id, shape, or clip")
    cap.add_argument("--clip", help="narrow to a clip_id")
    cap.add_argument("--shape", help="narrow to a shape")
    cap.add_argument("--limit", type=int, help="cap live calls this invocation (clamped to ceiling)")
    cap.add_argument("--force", action="store_true", help="re-capture even fresh fixtures (no resume-skip)")
    cap.add_argument("--dry-run", action="store_true", help="print planned call count + cost, make no calls")
    cap.set_defaults(func=cmd_capture)

    rep = sub.add_parser("replay", help="free, CI-safe replay + score over committed fixtures")
    rep.add_argument("--report", default=str(DEFAULT_REPORT_MD), help="markdown report path")
    rep.add_argument("--json", default=str(DEFAULT_REPORT_JSON), help="json report path")
    rep.add_argument("--shape", help="narrow to a shape")
    rep.add_argument("--clip", help="narrow to a clip_id")
    rep.add_argument("--fail-under", action="append", metavar="TIER=RATE",
                     help="override a threshold floor (repeatable)")
    rep.add_argument("--no-gate", action="store_true", help="score + report only, skip the gate")
    rep.set_defaults(func=cmd_replay)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "fail_under", None):
        args.fail_under = _parse_fail_under(args.fail_under)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
