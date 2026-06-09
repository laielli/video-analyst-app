#!/usr/bin/env python3
"""
Codegen eval harness CLI (D3) — one entry point, two subcommands sharing one scoring path.

    # CAPTURE (live, billed, creds required) — the human runs this once, after merge:
    python eval/run_eval.py capture [--only <case_id|shape|clip>] [--limit N] [--force] [--dry-run]

    # REPLAY (free, CI-safe, no creds, no network) — implementers + CI run this:
    python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json]
                                   [--shape S] [--clip C] [--fail-under <tier>=<rate>] [--no-gate]

Exit codes mirror codegen_probe.py: 0 = ok / thresholds met (or advisory uniform-stale bypass),
1 = threshold missed / lint failed, 2 = setup problem (no creds on capture).

PATH DANCE: HERE is api/eval, NOT scripts/ — pytest's `pythonpath = . scripts` applies only under
pytest, and the CI step + runbook invoke the bare `python eval/run_eval.py` form. So we insert both
api/ and api/scripts/ on sys.path BEFORE importing validate_program/interpreter/canned/codegen. A
bare invocation without these inserts raises `ModuleNotFoundError: No module named
'validate_program'`. Smoke-test `python eval/run_eval.py replay` from `api/` before raising a PR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # api/eval
API_DIR = HERE.parent                            # api/
for p in (str(API_DIR), str(API_DIR / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Allow `python eval/run_eval.py ...` (no package context) to still resolve the eval package
# imports below. When run as a module the package is already importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(API_DIR))

from eval import bank, fixtures, report, scoring  # noqa: E402
from eval.fixtures import Fixture  # noqa: E402

# Hard ceiling on actual live calls per capture invocation (cost guard, D3). Default = bank size,
# so a fat-fingered bank can't bill thousands. --force does NOT bypass this; --limit clamps to it.
MAX_CAPTURE_CALLS = 60

DEFAULT_REPORT_MD = HERE / "report.md"
DEFAULT_REPORT_JSON = HERE / "report.json"
THRESHOLDS_PATH = HERE / "thresholds.json"

# gpt-4o pricing for the offline cost estimate (USD per 1M tokens). Verify before a real run.
PRICE_IN_PER_M = 2.50
PRICE_OUT_PER_M = 10.00
EST_INPUT_TOKENS = 900   # SYSTEM_PROMPT + clip hint + question, rough
EST_OUTPUT_TOKENS = 400  # a real program, rough

OPENAI_HINT = """\
Azure OpenAI is not configured — set AZURE_OPENAI_* in api/.env (see .env.example).
Capture needs the same gpt-4o deployment codegen_probe.py uses. Without creds, replay still works
over the committed seed fixtures; only `capture` requires creds (it bills real tokens).
"""


# ---- shared helpers ----------------------------------------------------------------------
def _resolve_clip(clip_id: str):
    import canned

    clip = canned.clip_by_id(clip_id)
    if clip is None:
        raise SystemExit(f"unknown clip_id in bank: {clip_id}")
    return clip


def _est_cost(n_calls: int) -> str:
    per_call = (EST_INPUT_TOKENS * PRICE_IN_PER_M + EST_OUTPUT_TOKENS * PRICE_OUT_PER_M) / 1e6
    return f"~${n_calls * per_call:.2f}"


# ---- CAPTURE -----------------------------------------------------------------------------
def cmd_capture(args) -> int:
    """Live, billed, creds-gated capture. Lazy-imports the Azure client (replay never does)."""
    from ocr_probe import load_dotenv

    load_dotenv(API_DIR / ".env")
    import codegen  # after dotenv so enabled() sees the loaded vars

    if not codegen.enabled():
        print(OPENAI_HINT, file=sys.stderr)
        return 2

    cases = bank.filter_cases(bank.load_bank(), only=args.only)
    if not cases:
        print(f"no cases match --only {args.only!r}", file=sys.stderr)
        return 1

    # Determine which cases actually need a live call (resume-skip unless --force).
    pending = []
    for case in cases:
        clip = _resolve_clip(case["clip_id"])
        pv = bank.prompt_version(clip)
        existing = fixtures.load_fixture(case["case_id"])
        if existing is not None and fixtures.is_fresh(existing, pv) and not args.force:
            continue  # resume: a fresh fixture is skipped (free)
        pending.append((case, clip, pv))

    # --limit clamps this invocation; the ceiling is never exceeded by any flag combination.
    ceiling = min(MAX_CAPTURE_CALLS, args.limit) if args.limit is not None else MAX_CAPTURE_CALLS
    will_call = pending[:ceiling]

    if args.dry_run:
        print(f"will make {len(will_call)} live calls ({_est_cost(len(will_call))}); "
              f"{len(cases) - len(pending)} already-fresh fixtures skipped (resume).")
        return 0

    client = codegen.AzureCodegen.from_env()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    made = 0
    for case, clip, pv in will_call:
        fx = _capture_one(client, case, clip, pv, deployment)
        fixtures.write_fixture(fx)
        made += 1
        print(f"captured {case['case_id']} ({made}/{len(will_call)})")
    print(f"capture complete: {made} live call(s) ({_est_cost(made)}); "
          f"run `python eval/run_eval.py replay` to score.")
    return 0


def _capture_one(client, case: dict, clip: dict, pv: str, deployment: str) -> Fixture:
    """One billed codegen call -> a Fixture. Records usage:null (Resolved #1). Any generate()
    failure is recorded as a fixed error tag (no raw exception text — info-leak discipline)."""
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:  # pragma: no cover - the live network block (only path that bills; creds-gated)
        raw_program = client.generate(case["question"], clip=clip)
        error = None
    except ValueError as e:  # pragma: no cover - oversized/unparseable raise ValueError in generate
        raw_program, error = None, ("oversized" if "bytes" in str(e) else "unparseable")
    except Exception:  # pragma: no cover - any other generation failure
        raw_program, error = None, "generation-failed"
    return Fixture(
        case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
        prompt_version=pv, raw_program=raw_program, error=error,
        model=deployment, captured_at=captured_at, usage=None,
    )


# ---- REPLAY ------------------------------------------------------------------------------
def cmd_replay(args) -> int:
    """Free, deterministic, CI-safe replay. No creds, no network, no Azure client import."""
    from interpreter import Cache

    cases = bank.filter_cases(bank.load_bank(), clip=args.clip, shape=args.shape)
    bank_doc = bank.load_bank_doc()
    cases_by_id = {c["case_id"]: c for c in cases}

    # Cache the loaded interpreter Cache + prompt_version per clip (avoid reloading per case).
    clip_cache: dict[str, object] = {}
    clip_pv: dict[str, str] = {}
    prompt_sizes: dict[str, int] = {}
    import codegen  # NOT the Azure client — just build_system_message for the prompt-size signal

    results = []
    total_raw_bytes = 0
    for case in cases:
        clip = _resolve_clip(case["clip_id"])
        cid = case["clip_id"]
        if cid not in clip_cache:
            clip_cache[cid] = Cache.load(clip["cache"])
            clip_pv[cid] = bank.prompt_version(clip)
            prompt_sizes[cid] = len(codegen.build_system_message(clip))
        fx = fixtures.load_fixture(case["case_id"])
        result = scoring.score_case(case, fx, clip, clip_cache[cid])
        # Mark STALE if the fixture's prompt_version drifted from the current one.
        if fx is not None and not fixtures.is_fresh(fx, clip_pv[cid]):
            result.stale = True
            result.reached_tier = scoring.STALE
        if fx is not None and fx.raw_program is not None:
            total_raw_bytes += len(json.dumps(fx.raw_program).encode("utf-8"))
        results.append(result)

    meta = {
        "bank_version": bank_doc.get("bank_version"),
        "prompt_version": ",".join(sorted(set(clip_pv.values()))),
        "captured_at": _capture_stamp(cases),
        "model": _model_stamp(cases),
        "mode": "replay",
        "prompt_sizes": prompt_sizes,
        "total_raw_bytes": total_raw_bytes,
        "est_cost": _est_cost(len(cases)),
    }

    md, json_obj = report.build_report(results, cases_by_id, meta)
    md_path = Path(args.report) if args.report else DEFAULT_REPORT_MD
    json_path = Path(args.json) if args.json else DEFAULT_REPORT_JSON
    report.write_report(md, json_obj, md_path, json_path)
    print(md)

    if args.no_gate:
        return 0
    return apply_gate(results, cases_by_id, json_obj, _parse_fail_under(args.fail_under))


def _capture_stamp(cases: list[dict]) -> str:
    """First non-seed captured_at among fixtures, else 'seed' / 'none'."""
    stamps = set()
    for case in cases:
        fx = fixtures.load_fixture(case["case_id"])
        if fx is not None and fx.captured_at:
            stamps.add(fx.captured_at)
    if not stamps:
        return "none"
    return sorted(stamps)[0]


def _model_stamp(cases: list[dict]) -> str:
    models = set()
    for case in cases:
        fx = fixtures.load_fixture(case["case_id"])
        if fx is not None and fx.model:
            models.add(fx.model)
    return ",".join(sorted(models)) or "none"


# ---- the threshold gate (D6) -------------------------------------------------------------
def _parse_fail_under(pairs: list[str] | None) -> dict:
    """Parse `--fail-under T2_semantic_valid=1.0` repeats into {tier: rate}. CLI floors OVERRIDE
    the committed thresholds.json for the named tiers."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--fail-under expects <tier>=<rate>, got {pair!r}")
        tier, rate = pair.split("=", 1)
        out[tier.strip()] = float(rate)
    return out


def _load_thresholds() -> dict:
    return {k: v for k, v in json.loads(THRESHOLDS_PATH.read_text()).items() if not k.startswith("_")}


def apply_gate(results: list, cases_by_id: dict, json_obj: dict, cli_floors: dict) -> int:
    """Apply the calibrated threshold gate with the stale-fixture advisory bypass (D6). Returns the
    process exit code: 0 = met / advisory bypass, 1 = a floor missed / partial staleness / missing.

    STALE handling: uniform all-stale -> advisory (exit 0). Partial staleness -> stale fixtures
    count as FAILED gating outcomes beyond `max_stale` (the anti-gaming rule; never excluded)."""
    thresholds = _load_thresholds()
    thresholds.update(cli_floors)  # CLI floors override the file
    max_stale = int(thresholds.get("max_stale", 0))
    min_fixtures = int(thresholds.get("min_fixtures", 0))

    n_total = len(results)
    n_stale = sum(1 for r in results if r.stale)
    n_missing = sum(1 for r in results if r.missing)

    # Uniform all-stale: an honest prompt edit. Downgrade to advisory (the report already shouts
    # re-capture). An implementer/user editing SYSTEM_PROMPT must not red-CI before re-capturing.
    if n_stale and n_stale == n_total:
        print("\nGATE: ADVISORY — all fixtures stale (uniform prompt edit). Re-capture needed; "
              "exiting 0 per the D6 uniform-stale bypass.", file=sys.stderr)
        return 0

    failed = False

    # min_fixtures floor (catches deletion) + missing fixtures are a hard fail.
    n_present = sum(1 for r in results if not r.missing)
    if n_present < min_fixtures:
        print(f"GATE FAIL: {n_present} present fixtures < min_fixtures {min_fixtures}", file=sys.stderr)
        failed = True
    if n_missing:
        print(f"GATE FAIL: {n_missing} MISSING fixture(s) — capture them.", file=sys.stderr)
        failed = True

    # Partial staleness beyond max_stale is a FAIL (never excludes a case from the denominator).
    if 0 < n_stale <= max_stale:
        print(f"GATE: {n_stale} stale fixture(s) within max_stale {max_stale} (tolerated).",
              file=sys.stderr)
    elif n_stale > max_stale:
        print(f"GATE FAIL: {n_stale} stale fixture(s) > max_stale {max_stale} "
              f"(partial staleness — re-capture or revert).", file=sys.stderr)
        failed = True

    # Tier-rate floors. Each rate carries its denominator (`of`) so a floor with ZERO applicable
    # cases (e.g. a --shape slice with no out-of-scope cases) is skipped — a vacuous 0/0 rate must
    # not fail the gate.
    summary = json_obj["summary"]
    rate_sources = {
        "T2_semantic_valid": (float(summary["funnel"][scoring.T2_SEMANTIC]["rate"]),
                              summary["funnel"][scoring.T2_SEMANTIC]["of"]),
        "T3_executes_grounded": (float(summary["funnel"][scoring.T3_GROUNDED]["rate"]),
                                 summary["funnel"][scoring.T3_GROUNDED]["of"]),
        "T4_answer_correct": (float(summary["funnel"][scoring.T4_CORRECT]["rate"]),
                              summary["funnel"][scoring.T4_CORRECT]["of"]),
        "out_of_scope_honest": (float(summary["out_of_scope_honest"]["rate"]),
                                summary["out_of_scope_honest"]["of"]),
        "adversarial_safe": (float(summary["adversarial_safe"]["rate"]),
                             summary["adversarial_safe"]["of"]),
    }
    for key, floor in thresholds.items():
        if key in ("max_stale", "min_fixtures"):
            continue
        src = rate_sources.get(key)
        if src is None:
            continue  # unknown floor key — ignore (forward-compat)
        actual, denom = src
        if denom == 0:
            continue  # no applicable cases -> the floor is vacuous, skip it
        if actual < float(floor):
            print(f"GATE FAIL: {key} rate {actual:.2f} < floor {float(floor):.2f}", file=sys.stderr)
            failed = True

    if failed:
        return 1
    print("\nGATE PASS: all in-scope tier floors met; fixtures present and fresh.", file=sys.stderr)
    return 0


# ---- argparse ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Codegen eval harness — capture (live) / replay (CI-safe).")
    sub = ap.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture", help="live, billed, creds-gated capture of model output")
    cap.add_argument("--only", help="narrow to a case_id, shape, or clip_id")
    cap.add_argument("--limit", type=int, help="cap live calls this invocation (clamped to the ceiling)")
    cap.add_argument("--force", action="store_true", help="re-capture even fresh fixtures (resume off)")
    cap.add_argument("--dry-run", action="store_true", help="print the call count + cost, make no calls")
    cap.set_defaults(func=cmd_capture)

    rep = sub.add_parser("replay", help="free, deterministic, CI-safe scoring over committed fixtures")
    rep.add_argument("--report", help="markdown report path (default eval/report.md)")
    rep.add_argument("--json", help="json report path (default eval/report.json)")
    rep.add_argument("--shape", help="filter to one shape")
    rep.add_argument("--clip", help="filter to one clip_id")
    rep.add_argument("--fail-under", action="append", metavar="TIER=RATE",
                     help="floor a tier rate (repeatable); overrides thresholds.json")
    rep.add_argument("--no-gate", action="store_true", help="report only; never exit non-zero")
    rep.set_defaults(func=cmd_replay)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
