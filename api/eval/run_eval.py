#!/usr/bin/env python3
"""
Codegen eval harness CLI (D3) — capture (live, billed, creds-gated) + replay (free, CI-safe).

    # CAPTURE (the human runs this once, needs AZURE_OPENAI_* creds; BILLS real gpt-4o tokens):
    python eval/run_eval.py capture [--only X] [--limit N] [--force] [--dry-run]

    # REPLAY (implementers + CI; no creds, no network; deterministic over committed fixtures):
    python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json]
                                   [--shape S] [--clip C] [--fail-under T=R] [--no-gate]

Exit codes mirror codegen_probe.py: 0 = ok / threshold met, 1 = threshold missed / lint failed,
2 = setup problem (no creds on capture).

Path-dance note: unlike codegen_probe.py (whose HERE is scripts/), HERE is api/eval, so this file
must insert BOTH api/ and api/scripts/ on sys.path before importing validate_program/interpreter/
canned/codegen. A bare `python eval/run_eval.py replay` from api/ without these inserts raises
ModuleNotFoundError (pytest's pythonpath only applies under pytest).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
# Insert api/ and api/scripts/ so validate_program/interpreter/canned/codegen import bare.
for _p in (str(API_DIR), str(API_DIR / "scripts"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bank  # noqa: E402
import canned  # noqa: E402
import fixtures as fixtures_mod  # noqa: E402
import report as report_mod  # noqa: E402
import scoring  # noqa: E402

# A per-invocation hard ceiling on live calls — a fat-fingered bank can't bill thousands.
MAX_CAPTURE_CALLS = 100

DEFAULT_REPORT_MD = HERE / "report.md"
DEFAULT_REPORT_JSON = HERE / "report.json"
THRESHOLDS_PATH = HERE / "thresholds.json"

# gpt-4o offline cost estimate (no usage seam per Resolved #1): input ~900 tok, output ~400 tok.
_INPUT_USD_PER_M, _OUTPUT_USD_PER_M = 2.50, 10.0
_EST_INPUT_TOK, _EST_OUTPUT_TOK = 900, 400


def _est_cost(n_calls: int) -> str:
    usd = n_calls * (_EST_INPUT_TOK * _INPUT_USD_PER_M + _EST_OUTPUT_TOK * _OUTPUT_USD_PER_M) / 1e6
    return f"~${usd:.2f}"


# ---- CAPTURE ------------------------------------------------------------------------------

def _capture(args) -> int:
    # load_dotenv from the same shared probe helper, then the Azure-aware imports.
    sys.path.insert(0, str(API_DIR / "scripts"))
    from ocr_probe import load_dotenv

    load_dotenv(API_DIR / ".env")

    import codegen  # lazy: only the capture branch imports the Azure client surface

    if not codegen.enabled():
        # Same provisioning hint codegen_probe.py prints; never a silent no-op or a billed call.
        from codegen_probe import OPENAI_HINT

        print(OPENAI_HINT, file=sys.stderr)
        return 2

    cases = bank.filter_cases(bank.load_bank()["cases"], only=args.only)
    if not cases:
        print(f"no cases match --only {args.only!r}", file=sys.stderr)
        return 1

    # Determine which cases would actually be called (resume-skip unless --force).
    pending = []
    for case in cases:
        clip = canned.clip_by_id(case["clip_id"])
        pv = bank.prompt_version(clip)
        existing = fixtures_mod.load_fixture(case["case_id"])
        fresh = existing is not None and fixtures_mod.is_fresh(existing, pv)
        if fresh and not args.force:
            continue
        pending.append(case)

    # Apply --limit then clamp to the ceiling. --force never bypasses the ceiling.
    n_planned = len(pending)
    if args.limit is not None:
        n_planned = min(n_planned, args.limit)
    n_planned = min(n_planned, MAX_CAPTURE_CALLS)
    pending = pending[:n_planned]

    if args.dry_run:
        print(f"will make {len(pending)} live calls ({_est_cost(len(pending))}); "
              f"ceiling MAX_CAPTURE_CALLS={MAX_CAPTURE_CALLS}")
        return 0

    client = codegen.AzureCodegen.from_env()
    made = 0
    for case in pending:
        clip = canned.clip_by_id(case["clip_id"])
        pv = bank.prompt_version(clip)
        raw, error = _generate_one(client, case["question"], clip)
        fx = fixtures_mod.Fixture(
            case_id=case["case_id"],
            clip_id=case["clip_id"],
            question=case["question"],
            prompt_version=pv,
            raw_program=raw,
            captured_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            usage=None,
            error=error,
        )
        fixtures_mod.write_fixture(fx)
        made += 1
        steps_or_err = f"error={error}" if error else f"{len(raw)} steps"
        print(f"captured {case['case_id']} -> {steps_or_err}")
    print(f"capture complete: {made} live call(s), {len(cases) - made} skipped (fresh)")
    return 0


def _generate_one(client, question: str, clip: dict):
    """One codegen call (the only live, billed line is client.generate). Maps a generate() raise
    to a fixed error tag (no raw text leak)."""
    try:
        return client.generate(question, clip), None
    except ValueError as e:
        msg = str(e).lower()
        if "exceeds" in msg or "bytes" in msg:
            return None, fixtures_mod.ERROR_OVERSIZED
        if "missing 'program'" in msg or "json" in msg:
            return None, fixtures_mod.ERROR_UNPARSEABLE
        return None, fixtures_mod.ERROR_GENERATION_FAILED
    except Exception:  # noqa: BLE001
        return None, fixtures_mod.ERROR_GENERATION_FAILED


# ---- REPLAY -------------------------------------------------------------------------------

def replay(shape=None, clip=None, fixtures_dir=None) -> tuple[list, dict]:
    """Score every (filtered) bank case over its committed fixture. Pure: no creds, no network.
    Returns (results, meta). Used by the CLI and by tests."""
    from interpreter import Cache

    bank_obj = bank.load_bank()
    cases = bank.filter_cases(bank_obj["cases"], shape=shape, clip=clip)
    caches: dict[str, object] = {}
    results = []
    prompt_sizes = {}
    total_program_bytes = 0

    for case in cases:
        cl = canned.clip_by_id(case["clip_id"])
        if cl["id"] not in prompt_sizes:
            import codegen

            prompt_sizes[cl["id"]] = len(codegen.build_system_message(cl))
        cpath = cl["cache"]
        if cpath not in caches:
            caches[cpath] = Cache.load(cpath)
        pv = bank.prompt_version(cl)
        fx = fixtures_mod.load_fixture(case["case_id"], fixtures_dir)
        result = scoring.score_case(case, fx, cl, caches[cpath])
        # Mark STALE (overrides reached_tier for gating) when the fixture's prompt_version drifts.
        if fx is not None and not fixtures_mod.is_fresh(fx, pv):
            result.stale = True
            result.reached_tier = scoring.STALE
            result.passed = False
            result.detail = f"STALE: fixture prompt_version {fx.prompt_version} != current {pv}"
        total_program_bytes += result.raw_program_bytes
        results.append(result)

    n_fixtures = sum(1 for r in results if not r.missing)
    meta = {
        "bank_version": bank_obj["bank_version"],
        "prompt_version": bank.prompt_version(None),
        "captured_at": _capture_stamp(results),
        "model": _model_stamp(),
        "mode": "replay",
        "prompt_sizes": prompt_sizes,
        "total_program_bytes": total_program_bytes,
        "cost_estimate": _est_cost(len(cases)),
        "n_fixtures": n_fixtures,
    }
    return results, meta


def _capture_stamp(results) -> str:
    """Surface the capture timestamp from the first fixture (provenance only)."""
    for r in results:
        if not r.missing:
            fx = fixtures_mod.load_fixture(r.case_id)
            if fx and fx.captured_at:
                return fx.captured_at
    return "n/a"


def _model_stamp() -> str:
    cases = bank.load_bank()["cases"]
    for case in cases:
        fx = fixtures_mod.load_fixture(case["case_id"])
        if fx and fx.model:
            return fx.model
    return "n/a"


def _load_thresholds(overrides: list[str]) -> dict:
    thresholds = json.loads(THRESHOLDS_PATH.read_text())
    thresholds = {k: v for k, v in thresholds.items() if not k.startswith("_")}
    for spec in overrides or []:
        key, _, val = spec.partition("=")
        thresholds[key] = float(val)
    return thresholds


def evaluate_gate(results, meta, thresholds: dict) -> tuple[bool, list[str]]:
    """Apply the threshold gate (D6). Returns (passed, messages).

    Stale handling: uniform-all-stale downgrades to advisory (the only bypass); partial staleness
    FAILS beyond max_stale (anti-gaming) and never excludes a case from the denominator."""
    msgs = []
    in_scope = [r for r in results if r.expected_outcome == "grounded"]
    out_of_scope = [r for r in results if r.expected_outcome == "ungrounded" and r.shape != "adversarial"]
    adversarial = [r for r in results if r.shape == "adversarial"]
    n_fixtures = meta["n_fixtures"]

    stale_count = sum(1 for r in results if r.stale)
    non_missing = [r for r in results if not r.missing]
    if non_missing and stale_count == len(non_missing):
        msgs.append(f"ALL {stale_count} fixtures STALE (uniform prompt edit) — gate downgraded to "
                    f"ADVISORY. Re-capture: python eval/run_eval.py capture")
        return True, msgs

    failed = False
    # min_fixtures (catches deletion).
    min_fixtures = int(thresholds.get("min_fixtures", 0))
    if n_fixtures < min_fixtures:
        msgs.append(f"FAIL min_fixtures: {n_fixtures} < {min_fixtures} (fixtures missing)")
        failed = True
    # max_stale (partial staleness is suspicious; never excludes).
    max_stale = int(thresholds.get("max_stale", 0))
    if stale_count > max_stale:
        msgs.append(f"FAIL max_stale: {stale_count} stale > {max_stale} (partial staleness — re-capture all)")
        failed = True

    tier_rate = {tier: _tier_rate(in_scope, tier) for tier in scoring.TIER_ORDER}
    for key, floor in thresholds.items():
        if key in ("min_fixtures", "max_stale"):
            continue
        rate = _resolve_rate(key, tier_rate, out_of_scope, adversarial)
        if rate is None:
            continue
        status = "ok" if rate >= floor else "FAIL"
        msgs.append(f"{status} {key}: {rate:.2f} (floor {floor:.2f})")
        if rate < floor:
            failed = True
    return (not failed), msgs


def _tier_rate(in_scope, tier) -> float:
    if not in_scope:
        return 0.0
    reached = sum(1 for r in in_scope if report_mod._reached_at_least(r, tier))
    return reached / len(in_scope)


def _resolve_rate(key, tier_rate, out_of_scope, adversarial):
    if key == "out_of_scope_honest":
        return sum(1 for r in out_of_scope if r.passed) / len(out_of_scope) if out_of_scope else 1.0
    if key == "adversarial_safe":
        return sum(1 for r in adversarial if r.passed) / len(adversarial) if adversarial else 1.0
    return tier_rate.get(key)


def _replay(args) -> int:
    results, meta = replay(shape=args.shape, clip=args.clip)
    md, report_json = report_mod.build_report(results, meta)
    report_mod.write_report(md, report_json, args.report, args.json)
    print(f"wrote {args.report} + {args.json}")

    thresholds = _load_thresholds(args.fail_under)
    passed, msgs = evaluate_gate(results, meta, thresholds)
    for m in msgs:
        print(m)
    if args.no_gate:
        print("--no-gate: thresholds not enforced (advisory).")
        return 0
    return 0 if passed else 1


# ---- argparse -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Codegen eval harness — capture/replay")
    sub = ap.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture", help="live, billed, creds-gated — writes fixtures")
    cap.add_argument("--only", default=None, help="narrow to a case_id / shape / clip")
    cap.add_argument("--limit", type=int, default=None, help="cap live calls this invocation")
    cap.add_argument("--force", action="store_true", help="disable resume-skip (still capped)")
    cap.add_argument("--dry-run", action="store_true", help="print N calls / cost, make no calls")

    rep = sub.add_parser("replay", help="free, CI-safe — scores fixtures + writes report")
    rep.add_argument("--report", default=str(DEFAULT_REPORT_MD), help="markdown report path")
    rep.add_argument("--json", default=str(DEFAULT_REPORT_JSON), help="json report path")
    rep.add_argument("--shape", default=None, help="filter to a shape")
    rep.add_argument("--clip", default=None, help="filter to a clip")
    rep.add_argument("--fail-under", action="append", default=[], metavar="TIER=RATE",
                     help="override a threshold floor (repeatable)")
    rep.add_argument("--no-gate", action="store_true", help="report only, do not enforce thresholds")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "capture":
        return _capture(args)
    return _replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
