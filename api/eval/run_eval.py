#!/usr/bin/env python3
"""
Codegen eval harness CLI (D3): one entry point, two subcommands sharing one scoring path.

    # CAPTURE (live, billed, creds required) — the human runs this once after merge:
    python eval/run_eval.py capture [--only X] [--limit N] [--force] [--dry-run]

    # REPLAY (free, CI-safe, no creds, no network) — implementers + CI run this:
    python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json]
                                   [--shape S] [--clip C] [--fail-under T=R] [--no-gate]

Exit codes mirror codegen_probe.py: 0 = ok / threshold met, 1 = threshold missed / lint failed,
2 = setup problem (no creds on capture).

For a single-question gut-check use scripts/codegen_probe.py; this is the systematic N-question
measurement layer. Runbook + design notes: eval/README.md.

Path dance: HERE is api/eval, NOT scripts/, so we must insert BOTH api/ and api/scripts/ on
sys.path before importing validate_program/interpreter/canned/codegen — pytest's
`pythonpath = . scripts` applies only under pytest, but CI and the runbook invoke the bare
`python eval/run_eval.py ...` form. A bare invocation without these inserts raises
ModuleNotFoundError: No module named 'validate_program'.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
# api/ + api/scripts/ are needed for the cross-package imports (canned/interpreter/codegen/limits,
# validate_program/ocr_probe). pytest's `pythonpath = . scripts` already provides these under
# pytest; we insert them explicitly so the BARE `python eval/run_eval.py ...` form (used by CI and
# the runbook) also works — without them it raises ModuleNotFoundError: No module named
# 'validate_program' (pytest's pythonpath applies only under pytest). The __main__ block below
# bootstraps into the `eval` package so intra-package imports below stay RELATIVE — giving exactly
# ONE module identity (eval.fixtures, etc.), so a monkeypatch in a test patches the same object the
# runner uses (no dual `fixtures` vs `eval.fixtures` footgun).
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))

# BARE-SCRIPT bootstrap. When invoked as `python eval/run_eval.py ...` this module's __package__ is
# empty, so the relative imports below (`from . import bank`) would fail. Re-enter through the
# `eval` package (api/ is now on the path) and dispatch, then exit — so the relative imports run
# exactly once, under the package, giving a single module identity. (Smoke-tested:
# `python eval/run_eval.py replay` from api/.)
if __name__ == "__main__":
    from eval.run_eval import main as _packaged_main

    raise SystemExit(_packaged_main(sys.argv[1:]))

import canned  # noqa: E402
from interpreter import Cache  # noqa: E402

from . import bank  # noqa: E402
from . import fixtures as fx  # noqa: E402
from . import report as report_mod  # noqa: E402
from . import scoring  # noqa: E402

DEFAULT_REPORT_MD = HERE / "report.md"
DEFAULT_REPORT_JSON = HERE / "report.json"
THRESHOLDS_PATH = HERE / "thresholds.json"

# Hard ceiling on ACTUAL live calls per capture invocation (D3 cost control). No flag combination
# can exceed it; --force only disables resume-skip, forced calls still count against it.
MAX_CAPTURE_CALLS = 80  # > the ~53-case bank, leaves headroom for growth without unbounded billing

# Offline gpt-4o price estimate (per the runbook cost note; verify current pricing before a run).
_EST_INPUT_TOKENS = 900
_EST_OUTPUT_TOKENS = 400
_PRICE_IN_PER_M = 2.5
_PRICE_OUT_PER_M = 10.0


def _est_cost(n_calls: int) -> str:
    dollars = n_calls * (_EST_INPUT_TOKENS * _PRICE_IN_PER_M + _EST_OUTPUT_TOKENS * _PRICE_OUT_PER_M) / 1e6
    return f"~${dollars:.2f} ({n_calls} calls @ ~{_EST_INPUT_TOKENS}in/{_EST_OUTPUT_TOKENS}out tokens)"


OPENAI_HINT = """\
Azure OpenAI is not configured — set AZURE_OPENAI_* in api/.env (see .env.example).
Capture needs creds and bills real gpt-4o tokens; replay does not. Without creds:
  python eval/run_eval.py replay     # scores the committed seed fixtures, free + deterministic
The harness, its tests, and the CI gate are already green against the committed seed fixtures."""


# ---- capture ----------------------------------------------------------------------------

def cmd_capture(args) -> int:
    # load_dotenv before importing codegen so enabled() sees the loaded vars (mirrors the probe).
    from ocr_probe import load_dotenv

    load_dotenv(API_DIR / ".env")
    import codegen  # lazy: replay never imports the Azure client

    if not codegen.enabled():
        print(OPENAI_HINT, file=sys.stderr)
        return 2

    cases = bank.filter_cases(bank.load_cases(), only=args.only)
    if not cases:
        print(f"no cases match --only {args.only!r}", file=sys.stderr)
        return 1

    # Decide which cases will actually be called (resume-skip unless --force), then clamp to the
    # per-invocation ceiling and --limit. --force never bypasses the ceiling.
    to_call = []
    for case in cases:
        clip = canned.clip_by_id(case["clip_id"])
        pv = bank.prompt_version(clip)
        existing = fx.load_fixture(case["case_id"])
        if existing is not None and fx.is_fresh(existing, pv) and not args.force:
            continue  # resume: a fresh fixture for the current prompt is skipped
        to_call.append((case, clip, pv))

    ceiling = MAX_CAPTURE_CALLS
    if args.limit is not None:
        ceiling = min(ceiling, args.limit)
    if len(to_call) > ceiling:
        to_call = to_call[:ceiling]

    if args.dry_run:
        print(f"will make {len(to_call)} live calls ({_est_cost(len(to_call))}); --dry-run makes no calls")
        for case, _, _ in to_call:
            print(f"  - {case['case_id']}")
        return 0

    client = codegen.AzureCodegen.from_env()
    made = 0
    for case, clip, pv in to_call:
        if made >= MAX_CAPTURE_CALLS:  # belt-and-suspenders against the ceiling
            print(f"reached MAX_CAPTURE_CALLS={MAX_CAPTURE_CALLS}; stopping", file=sys.stderr)
            break
        raw, error = _capture_one(client, codegen, case["question"], clip)
        made += 1
        fixture = fx.Fixture(
            case_id=case["case_id"],
            clip_id=case["clip_id"],
            question=case["question"],
            prompt_version=pv,
            raw_program=raw,
            captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            model=client._deployment if hasattr(client, "_deployment") else "gpt-4o",
            usage=None,  # Resolved #1: no usage seam
            error=error,
        )
        fx.write_fixture(fixture)
        print(f"captured {case['case_id']} -> {'error=' + error if error else str(len(raw)) + ' steps'}")
    print(f"capture done: {made} live call(s)")
    return 0


def _capture_one(client, codegen, question, clip):  # pragma: no cover - live network only
    """The single billed call. Returns (raw_program, error_tag). Maps any generate() error to a
    fixed tag (no raw exception text — info-leak discipline). Covered by mocked CLI tests via the
    branch above; the live network path itself is pragma-excluded like precompute's Azure branch."""
    try:
        raw = client.generate(question, clip)
        return raw, None
    except ValueError as e:
        tag = "oversized" if "exceeds" in str(e) else "unparseable"
        return None, tag
    except Exception:  # noqa: BLE001
        return None, "generation-failed"


# ---- replay -----------------------------------------------------------------------------

def cmd_replay(args) -> int:
    cases = bank.filter_cases(bank.load_cases(), clip=args.clip, shape=args.shape)
    bank_obj = bank.load_bank()

    # Resolve clips/caches once (small, cached by path).
    clip_cache = {}
    clip_prompt_chars = {}
    results = []
    for case in cases:
        clip = canned.clip_by_id(case["clip_id"])
        pv = bank.prompt_version(clip)
        clip_prompt_chars.setdefault(case["clip_id"], len(_build_msg(clip)))
        if case["clip_id"] not in clip_cache:
            clip_cache[case["clip_id"]] = Cache.load(clip["cache"])
        fixture = fx.load_fixture(case["case_id"])
        result = scoring.score_case(case, fixture, clip, clip_cache[case["clip_id"]])
        # Mark staleness AFTER scoring (scoring shows what a stale fixture would score).
        if fixture is not None and not fx.is_fresh(fixture, pv):
            result.stale = True
        results.append(result)

    captured_at = next((fixture.captured_at for fixture in
                        (fx.load_fixture(c["case_id"]) for c in cases) if fixture), None)
    model = next((fixture.model for fixture in
                  (fx.load_fixture(c["case_id"]) for c in cases) if fixture), None)
    meta = {
        "bank_version": bank_obj.get("bank_version"),
        "prompt_version": bank.prompt_version(None),
        "captured_at": captured_at,
        "model": model,
        "mode": "replay",
        "clip_prompt_chars": clip_prompt_chars,
        "est_cost": _est_cost(len([r for r in results if not r.missing])),
    }
    md, json_obj = report_mod.build_report(results, meta)
    report_mod.write_report(md, json_obj, Path(args.report), Path(args.json))
    print(md)

    if args.no_gate:
        return 0
    return _apply_gate(results, json_obj, args)


def _build_msg(clip):
    import codegen  # local: replay-side prompt size only; never constructs an Azure client

    return codegen.build_system_message(clip)


def _load_thresholds(overrides: list[str] | None) -> dict:
    import json

    th = json.loads(THRESHOLDS_PATH.read_text())
    th = {k: v for k, v in th.items() if not k.startswith("_")}
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        th[key.strip()] = float(val)
    return th


def _apply_gate(results, json_obj, args) -> int:
    """The calibrated threshold gate (D6) with the stale-fixture advisory bypass. Returns exit code.
    Uniform all-stale -> advisory (exit 0). Partial staleness beyond max_stale -> FAIL. min_fixtures
    catches deletion. Any in-scope tier rate below its floor -> FAIL."""
    th = _load_thresholds(args.fail_under)
    summary = json_obj["summary"]
    total = summary["total_cases"]
    stale = summary["stale"]
    missing = summary["missing"]

    # Uniform all-stale -> advisory downgrade (an honest prompt edit before re-capture).
    if total > 0 and stale == total:
        print("\n[gate] ALL fixtures stale (uniform) — advisory downgrade, exit 0. Re-capture needed.")
        return 0

    failures = []
    min_fixtures = int(th.get("min_fixtures", 0))
    present = total - missing
    if present < min_fixtures:
        failures.append(f"min_fixtures: {present} present < floor {min_fixtures}")

    max_stale = int(th.get("max_stale", 0))
    if stale > max_stale:
        failures.append(f"partial staleness: {stale} stale > max_stale {max_stale} "
                        "(stale fixtures count as failures; re-capture or fix the doctored fixture)")

    # Tier-rate floors (in-scope) + safety axes.
    rate_keys = {
        "T2_semantic_valid": summary["T2_semantic_valid"],
        "T4_answer_correct": summary["T4_answer_correct"],
        "out_of_scope_honest": summary["out_of_scope_honest_rate"],
        "adversarial_safe": summary["adversarial_safe_rate"],
    }
    for key, floor in th.items():
        if key in rate_keys:
            actual = rate_keys[key]
            if actual < floor:
                failures.append(f"{key}: {actual:.2f} < floor {floor:.2f}")

    if failures:
        print("\n[gate] FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[gate] PASS — all in-scope tier floors met.")
    return 0


# ---- argparse ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Codegen eval harness — capture (live) / replay (CI-safe)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="live, billed, creds-gated — records fixtures")
    cap.add_argument("--only", help="narrow to a case_id, shape, or clip_id")
    cap.add_argument("--limit", type=int, help="cap live calls THIS invocation (clamped to the ceiling)")
    cap.add_argument("--force", action="store_true", help="disable resume-skip (still bounded by the ceiling)")
    cap.add_argument("--dry-run", action="store_true", help="print 'will make N live calls (~$X)'; make none")
    cap.set_defaults(func=cmd_capture)

    rep = sub.add_parser("replay", help="recorded fixtures, free, deterministic, CI-safe")
    rep.add_argument("--report", default=str(DEFAULT_REPORT_MD), help="markdown report path")
    rep.add_argument("--json", default=str(DEFAULT_REPORT_JSON), help="json report path")
    rep.add_argument("--shape", help="filter to one shape")
    rep.add_argument("--clip", help="filter to one clip")
    rep.add_argument("--fail-under", action="append", metavar="TIER=RATE",
                     help="override/add a threshold floor (repeatable)")
    rep.add_argument("--no-gate", action="store_true", help="write the report but never gate (exit 0)")
    rep.set_defaults(func=cmd_replay)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
