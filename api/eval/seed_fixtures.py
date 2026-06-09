"""
Generate the committed SEED fixtures — the creds-free stand-in for a real capture (step 11).

The seeds make replay, its tests, and the CI gate green BEFORE any real capture; the human capture
overwrites them. They are GENERATED, not hand-authored (the same derived-not-hand-authored posture
as the committed bernabeu cache), and `test_seed_fixtures_regenerate_match` asserts regeneration
reproduces them — so a bank edit without re-seeding fails loudly.

In-scope seeds: the cell's matching pinned `examples/*_program.json`, retargeted by ONE precise
transform — copy the program steps verbatim, replace ONLY the final `answer` step's `question` with
the case's question text. Grounding depends on the cache + sample windows, not on the question
phrasing, so the retargeted seed still grounds to the cell's expected answer.

Out-of-scope / injection / degenerate seeds: canned templates inside this module — a valid-but-
non-grounding program (grounds:false -> honest no-ground -> PASS), a validator-rejected program
(non-whitelisted op -> REJECTED -> PASS for out-of-scope), and one error-tag fixture (generate
raised -> REJECTED -> PASS for out-of-scope). Every honest-refusal path is exercised.

Each seed's `prompt_version` is stamped with the CURRENT hash so the seeds are NOT stale and the
gate exercises the real ladder. `captured_at` is a fixed sentinel so regeneration is byte-stable.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bank import derive_ground_truth, load_bank, prompt_version
from .fixtures import FIXTURE_FORMAT, Fixture

EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent
EXAMPLES = API_DIR / "examples"

SEED_MODEL = "seed"  # provenance marker so a real capture's "gpt-4o" is visibly different
SEED_CAPTURED_AT = "seed"  # fixed sentinel -> byte-stable regeneration

# Map a (clip_id, shape) cell to its pinned program template (in-scope seeds).
CELL_PROGRAM = {
    ("single-goal", "first-goal"): "hero_program.json",
    ("single-goal", "presence"): "hero_presence_program.json",
    ("bernabeu-counter", "count"): "bernabeu-counter_count_program.json",
    ("bernabeu-counter", "scorer-number"): "bernabeu-counter_scorer-number_program.json",
    ("bernabeu-counter", "presence"): "bernabeu-counter_presence_program.json",
    # first-goal on bernabeu splits by expected outcome (present #7 vs absent #23) — handled below.
}


def _pinned_program(name: str) -> list[dict]:
    from validate_program import strip_comments

    raw = json.loads((EXAMPLES / name).read_text())
    return strip_comments(raw)["program"]


def _retarget(program: list[dict], question: str) -> list[dict]:
    """Copy steps verbatim, replacing ONLY the final answer step's `question` arg with `question`."""
    out = [json.loads(json.dumps(s)) for s in program]  # deep copy
    for s in reversed(out):
        if s.get("op") == "answer":
            s.setdefault("args", {})["question"] = question
            break
    return out


def _in_scope_program(case: dict) -> list[dict]:
    clip_id, shape = case["clip_id"], case["shape"]
    if shape == "first-goal" and clip_id == "bernabeu-counter":
        # present subject (#7, grounds Yes) vs absent subject (#23, grounds out honestly).
        name = ("bernabeu-counter_first-goal-23_program.json"
                if case["expect"]["outcome"] == "ungrounded"
                else "bernabeu-counter_first-goal-7_program.json")
    else:
        name = CELL_PROGRAM[(clip_id, shape)]
    return _retarget(_pinned_program(name), case["question"])


# ---- out-of-scope / injection / degenerate templates -------------------------------------
def _non_grounding_program(question: str) -> list[dict]:
    """Valid program that grounds out honestly (count over a window that misses every cached
    frame -> count 0 -> grounded:false). The honest-refusal seed for most out-of-scope cases."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 100, "end_ms": 200, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": question}},
    ]


def _rejected_program() -> list[dict]:
    """A program the validator REJECTS (non-whitelisted op). The clean-reject seed: for an
    out-of-scope/injection case a clean rejection is a PASS (the trust boundary held)."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 1}},
        {"id": "bomb", "op": "exec", "args": {"cmd": "rm -rf /"}},
        {"id": "result", "op": "answer", "args": {"from": "frames", "question": "x"}},
    ]


def _oom_program() -> list[dict]:
    """An OOM-window program the numeric-bounds gate REJECTS — the seed for the OOM-bomb injection."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2_000_000_000, "fps": 30}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "x"}},
    ]


def _out_of_scope_seed(case: dict) -> Fixture:
    """Pick an honest-refusal seed program for a non-grounding case. Most use the valid-but-non-
    grounding program; injection cases that name a non-whitelisted op / OOM window use a rejected
    program; one degenerate case uses an error-tag fixture (generate raised)."""
    cid = case["case_id"]
    pv = prompt_version(_clip(case["clip_id"]))
    base = dict(case_id=cid, clip_id=case["clip_id"], question=case["question"],
                prompt_version=pv, model=SEED_MODEL, captured_at=SEED_CAPTURED_AT,
                usage=None, fixture_format=FIXTURE_FORMAT)
    # Deterministic template choice by case_id so regeneration is stable AND every path is covered.
    if cid == "adv-injection-exec-op":
        return Fixture(raw_program=_rejected_program(), **base)
    if cid == "adv-injection-oom-window":
        return Fixture(raw_program=_oom_program(), **base)
    if cid == "degenerate-questionmark":
        # generate() raised on this case (unparseable / no program) -> error-tag fixture.
        return Fixture(raw_program=None, error="generation-failed", **base)
    return Fixture(raw_program=_non_grounding_program(case["question"]), **base)


def _clip(clip_id: str) -> dict:
    import canned

    return canned.clip_by_id(clip_id)


def build_seed_fixture(case: dict) -> Fixture:
    """The single seed fixture for one bank case (in-scope -> retargeted pinned program; otherwise
    -> an honest-refusal template)."""
    pv = prompt_version(_clip(case["clip_id"]))
    if case["expect"]["outcome"] == "grounded":
        return Fixture(
            case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
            prompt_version=pv, raw_program=_in_scope_program(case), error=None,
            model=SEED_MODEL, captured_at=SEED_CAPTURED_AT, usage=None,
            fixture_format=FIXTURE_FORMAT,
        )
    return _out_of_scope_seed(case)


def build_all_seeds(cases: list[dict] | None = None) -> list[Fixture]:
    """Build a seed Fixture for every bank case."""
    cases = cases if cases is not None else load_bank()
    return [build_seed_fixture(c) for c in cases]


def write_all_seeds(fixtures_dir: Path | None = None) -> int:
    """Generate + write every seed fixture to disk (the dev one-shot that produces the committed
    `fixtures/*.json`). Returns the count written."""
    from .fixtures import write_fixture

    seeds = build_all_seeds()
    for fx in seeds:
        write_fixture(fx, fixtures_dir)
    return len(seeds)


# Re-export so a derive sanity-check can run from the CLI/dev shell.
__all__ = ["build_seed_fixture", "build_all_seeds", "write_all_seeds", "derive_ground_truth"]


if __name__ == "__main__":  # pragma: no cover - dev one-shot; run as `python -m eval.seed_fixtures`
    import sys

    for _p in (str(API_DIR), str(API_DIR / "scripts")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    n = write_all_seeds()
    print(f"wrote {n} seed fixtures to {EVAL_DIR / 'fixtures'}")
