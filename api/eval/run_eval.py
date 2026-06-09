#!/usr/bin/env python3
"""Codegen eval harness CLI (D3) — capture (live, billed, creds-gated) + replay (free, CI-safe).

    # CAPTURE (live, billed, creds required) — the human runs this once after merge:
    python eval/run_eval.py capture [--only <case_id|shape|clip>] [--limit N] [--force] [--dry-run]

    # REPLAY (free, CI-safe, no creds, no network) — implementers + CI run this:
    python eval/run_eval.py replay [--report eval/report.md] [--json eval/report.json]
                                   [--shape S] [--clip C] [--fail-under <tier>=<rate>] [--no-gate]

Exit codes mirror codegen_probe.py: 0 = ok / threshold met, 1 = threshold missed / lint failed,
2 = setup problem (no creds on capture).

PATH DANCE (verified): HERE = api/eval (NOT scripts/ like codegen_probe), so we must insert BOTH
api/ and api/scripts/ before importing validate_program/interpreter/canned/codegen. A bare
`python eval/run_eval.py replay` from api/ without these raises
`ModuleNotFoundError: No module named 'validate_program'` (pytest's `pythonpath = . scripts` applies
only under pytest). The CI step + runbook both invoke the bare form, so this header is load-bearing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
# Insert BOTH api/ and api/scripts/ so validate_program (scripts/), interpreter/canned/codegen (api/)
# import under a bare CLI invocation. Idempotent so importing this module under pytest is harmless.
for _p in (str(API_DIR), str(API_DIR / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# eval-package imports work as `eval.<mod>` once api/ is on sys.path. Use absolute package imports so
# the same module runs as a script (api/ on path) and is importable under pytest.
from eval import bank as bank_mod  # noqa: E402
from eval import report as report_mod  # noqa: E402
from eval import scoring as scoring_mod  # noqa: E402
from eval.fixtures import Fixture, is_fresh, load_fixture, write_fixture  # noqa: E402

THRESHOLDS_PATH = HERE / "thresholds.json"
DEFAULT_REPORT_MD = HERE / "report.md"
DEFAULT_REPORT_JSON = HERE / "report.json"

# Capture cost controls (D3). MAX_CAPTURE_CALLS is the eval package's own per-invocation ceiling — a
# separate constant from server.MAX_CODEGEN_CALLS (capture bypasses server entirely, D7). No flag
# combination can exceed it in a single invocation; --force only disables resume-skip.
MAX_CAPTURE_CALLS = 80  # > bank size (~57), bounds a fat-fingered bank to <$1.

# Offline cost estimate (usage is null per Resolved #1): ~900 input + ~400 output tokens/case at
# gpt-4o pricing (~$2.50/M in, ~$10/M out). Verify current pricing before a real run.
_EST_INPUT_TOKENS = 900
_EST_OUTPUT_TOKENS = 400
_PRICE_IN_PER_M = 2.50
_PRICE_OUT_PER_M = 10.0


def _estimate_cost(n_calls: int) -> float:
    return n_calls * (_EST_INPUT_TOKENS * _PRICE_IN_PER_M + _EST_OUTPUT_TOKENS * _PRICE_OUT_PER_M) / 1e6


def _prompt_chars() -> dict:
    """len(build_system_message(clip)) per clip — the offline prompt-bloat signal."""
    import canned
    import codegen

    out = {}
    for cid, clip in canned.CLIPS.items():
        out[cid] = len(codegen.build_system_message(clip))
    return out


# ---- replay -------------------------------------------------------------------------------------

def run_replay(args) -> int:
    """Load bank + fixtures, score each case, build + write the report, apply the threshold gate."""
    import canned
    from interpreter import Cache, Interpreter

    bank = bank_mod.load_bank()
    cases = bank_mod.filter_cases(bank["cases"], clip=args.clip, shape=args.shape)

    # Current prompt_version per clip (for staleness). build_system_message(None) == SYSTEM_PROMPT.
    pv_by_clip = {cid: bank_mod.prompt_version(clip) for cid, clip in canned.CLIPS.items()}
    caches: dict[str, object] = {}

    results = []
    for case in cases:
        clip = canned.clip_by_id(case["clip_id"])
        fixture = load_fixture(case["case_id"])
        if fixture is None:
            results.append(_missing_result(case))
            continue
        stale = not is_fresh(fixture, pv_by_clip[case["clip_id"]])
        if case["clip_id"] not in caches:
            caches[case["clip_id"]] = Cache.load(clip["cache"])
        cache = caches[case["clip_id"]]
        res = scoring_mod.score_case(case, fixture, clip, cache,
                                     interpreter_run=Interpreter(cache).run)
        res.stale = stale
        results.append(res)

    # The first present fixture stamps the report's captured_at/model provenance (seeds carry
    # captured_at="...Z"/model="seed"; a real capture carries the live timestamp/deployment).
    stamp_fx = next((f for f in (load_fixture(c["case_id"]) for c in cases) if f is not None), None)
    meta = {
        "bank_version": bank.get("bank_version"),
        "prompt_version": pv_by_clip.get("single-goal"),
        "captured_at": stamp_fx.captured_at if stamp_fx else None,
        "model": stamp_fx.model if stamp_fx else None,
        "mode": "replay",
        "estimated_capture_cost_usd": _estimate_cost(len(results)),
        "prompt_chars": _prompt_chars(),
    }
    md, json_obj = report_mod.build_report(results, meta)

    report_md = Path(args.report) if args.report else DEFAULT_REPORT_MD
    report_json = Path(args.json) if args.json else DEFAULT_REPORT_JSON
    _atomic_write_text(report_md, md)
    _atomic_write_text(report_json, json.dumps(json_obj, indent=2, ensure_ascii=True) + "\n")
    print(md)

    if args.no_gate:
        return 0
    overrides = _parse_fail_under(args.fail_under)
    exit_code, gate_msg = apply_gate(results, json_obj, _load_thresholds(), overrides)
    print(gate_msg, file=sys.stderr)
    return exit_code


def _missing_result(case):
    return scoring_mod.CaseResult(
        case_id=case["case_id"], clip_id=case["clip_id"], shape=case["shape"],
        phrasing_class=case["phrasing_class"], distance=case["distance"],
        target_tier=scoring_mod._target_tier(case), reached_tier=scoring_mod.MISSING,
        passed=False, detail="no fixture (un-captured)", missing=True)


# ---- threshold gate (D6) ------------------------------------------------------------------------

def apply_gate(results, json_obj, thresholds: dict, overrides: dict) -> tuple[int, str]:
    """Return (exit_code, message). D6: a calibrated threshold gate with a uniform-stale advisory
    bypass; partial staleness FAILS (it never excludes a case from the denominator)."""
    floors = dict(thresholds)
    floors.update(overrides)

    total = len(results)
    stale = [r for r in results if r.stale]
    missing = [r for r in results if r.missing]

    # Uniform all-stale (an honest prompt edit) -> advisory downgrade (exit 0). Only bypass; never
    # weakens case-by-case. Requires at least one fixture present (all-missing is not all-stale).
    present = [r for r in results if not r.missing]
    if present and len(stale) == len(present):
        return 0, ("ADVISORY: all fixtures STALE (uniform prompt edit) — gate downgraded to "
                   "advisory. Re-capture to re-arm the gate.")

    msgs = []
    failed = False

    # min_fixtures floor (catches deletion).
    min_fixtures = floors.get("min_fixtures", 0)
    n_present = len(present)
    if n_present < min_fixtures:
        failed = True
        msgs.append(f"FAIL min_fixtures: {n_present} present < floor {min_fixtures}")

    # max_stale: partial staleness beyond tolerance FAILS (anti-gaming).
    max_stale = floors.get("max_stale", 0)
    if len(stale) > max_stale:
        failed = True
        msgs.append(f"FAIL max_stale: {len(stale)} stale > tolerance {max_stale} "
                    f"(partial staleness is a FAIL, not an exclusion)")

    # Missing fixtures are a fail (an un-captured case can't be silently passed).
    if missing:
        failed = True
        msgs.append(f"FAIL: {len(missing)} MISSING fixture(s) "
                    f"({', '.join(r.case_id for r in missing[:5])}...)")

    funnel = json_obj["summary"]["funnel"]
    den = funnel["total"]

    def reached_rate(tier_key):
        return 0.0 if den == 0 else funnel["counts"].get(tier_key, 0) / den

    if "T2_semantic_valid" in floors:
        rate = reached_rate(scoring_mod.T2_SEMANTIC)
        if rate < floors["T2_semantic_valid"]:
            failed = True
            msgs.append(f"FAIL T2_semantic_valid: {rate:.2f} < floor {floors['T2_semantic_valid']:.2f}")
    if "T4_answer_correct" in floors:
        rate = reached_rate(scoring_mod.T4_CORRECT)
        if rate < floors["T4_answer_correct"]:
            failed = True
            msgs.append(f"FAIL T4_answer_correct: {rate:.2f} < floor {floors['T4_answer_correct']:.2f}")
    if "out_of_scope_honest" in floors:
        oos = json_obj["summary"]["out_of_scope_honest"]
        if oos["rate"] < floors["out_of_scope_honest"]:
            failed = True
            msgs.append(f"FAIL out_of_scope_honest: {oos['rate']:.2f} < "
                        f"floor {floors['out_of_scope_honest']:.2f}")

    if failed:
        return 1, "GATE FAILED:\n  " + "\n  ".join(msgs)
    return 0, f"GATE PASSED ({total} cases scored; floors: {sorted(floors)})"


def _parse_fail_under(items) -> dict:
    """Parse repeated --fail-under TIER=RATE into {tier: float}."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--fail-under expects TIER=RATE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _load_thresholds() -> dict:
    raw = json.loads(THRESHOLDS_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; committed artifacts want world-readable.
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---- capture (live, billed, creds-gated) --------------------------------------------------------

OPENAI_HINT = """\
Azure OpenAI is not configured — set AZURE_OPENAI_* in api/.env (see .env.example).
Capture needs creds; replay does not. Provision a gpt-4o deployment (see scripts/codegen_probe.py
for the az commands). Without creds: run `python eval/run_eval.py replay` (free, CI-safe)."""


def run_capture(args) -> int:
    """Live, billed capture. Creds-gated (exit 2 if codegen.enabled() is False). Resume by default;
    --force disables resume-skip (but NOT the ceiling). Bounded by MAX_CAPTURE_CALLS + --limit."""
    # load_dotenv first so enabled() sees api/.env (mirrors codegen_probe).
    from ocr_probe import load_dotenv

    load_dotenv(API_DIR / ".env")

    import canned
    import codegen  # after dotenv so enabled() sees the loaded vars

    if not codegen.enabled():
        print(OPENAI_HINT, file=sys.stderr)
        return 2

    bank = bank_mod.load_bank()
    cases = bank_mod.filter_cases(bank["cases"], only=args.only)
    pv_by_clip = {cid: bank_mod.prompt_version(clip) for cid, clip in canned.CLIPS.items()}

    # Determine which cases would actually be called (resume-skip unless --force), then clamp.
    pending = []
    for case in cases:
        if not args.force:
            fx = load_fixture(case["case_id"])
            if fx is not None and is_fresh(fx, pv_by_clip[case["clip_id"]]):
                continue  # resume-skip: fresh fixture already present
        pending.append(case)

    ceiling = MAX_CAPTURE_CALLS
    limit = args.limit if args.limit is not None else ceiling
    n_calls = min(len(pending), limit, ceiling)
    to_call = pending[:n_calls]

    if args.dry_run:
        cost = _estimate_cost(n_calls)
        print(f"DRY RUN: will make {n_calls} live call(s) (~${cost:.2f}); "
              f"{len(cases) - len(pending)} already fresh (resume-skip).")
        return 0

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    client = codegen.AzureCodegen.from_env()  # pragma: no cover - constructed only with live creds
    made = 0
    for case in to_call:  # pragma: no cover - the live network loop (creds-gated, billed)
        if made >= ceiling:
            break
        clip = canned.clip_by_id(case["clip_id"])
        pv = pv_by_clip[case["clip_id"]]
        fixture = _capture_one(client, case, clip, pv, deployment)
        write_fixture(fixture)
        made += 1
        print(f"captured {case['case_id']} ({made}/{n_calls})")
    print(f"capture complete: {made} live call(s) made.")
    return 0


def _capture_one(client, case, clip, prompt_version, deployment) -> Fixture:  # pragma: no cover
    """One billed codegen call -> a Fixture. Maps any generate() error to a fixed tag (no text leak).
    Records usage:null (Resolved #1 — no production usage seam)."""
    from eval.fixtures import ERROR_GENERATION_FAILED, ERROR_OVERSIZED, ERROR_UNPARSEABLE

    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        raw_program = client.generate(case["question"], clip)
        return Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                       prompt_version=prompt_version, raw_program=raw_program, error=None,
                       usage=None, captured_at=captured_at, model=deployment)
    except ValueError as e:
        tag = ERROR_OVERSIZED if "exceeds" in str(e) else ERROR_UNPARSEABLE
        return Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                       prompt_version=prompt_version, raw_program=None, error=tag, usage=None,
                       captured_at=captured_at, model=deployment)
    except Exception:  # noqa: BLE001 — any API error -> fixed generation-failed tag (info-leak discipline)
        return Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                       prompt_version=prompt_version, raw_program=None, error=ERROR_GENERATION_FAILED,
                       usage=None, captured_at=captured_at, model=deployment)


# ---- argparse -----------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Codegen eval harness (capture/replay)")
    sub = ap.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture", help="live, billed, creds-gated capture of gpt-4o output")
    cap.add_argument("--only", default=None, help="narrow to a case_id, shape, or clip_id")
    cap.add_argument("--limit", type=int, default=None, help="cap live calls this invocation")
    cap.add_argument("--force", action="store_true", help="disable resume-skip (not the ceiling)")
    cap.add_argument("--dry-run", action="store_true", help="print cost estimate; make no calls")

    rep = sub.add_parser("replay", help="free, CI-safe scoring over recorded fixtures")
    rep.add_argument("--report", default=None, help="markdown report path")
    rep.add_argument("--json", default=None, help="json report path")
    rep.add_argument("--shape", default=None, help="filter to a shape")
    rep.add_argument("--clip", default=None, help="filter to a clip")
    rep.add_argument("--fail-under", action="append", default=[],
                     help="TIER=RATE floor override (repeatable)")
    rep.add_argument("--no-gate", action="store_true", help="score + report only; no threshold gate")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "capture":
        return run_capture(args)
    return run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
