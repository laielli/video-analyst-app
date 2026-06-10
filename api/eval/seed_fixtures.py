"""
Generated seed fixtures — the creds-free stand-in for a real capture (step 11).

Every committed `eval/fixtures/<case_id>.json` is the output of `build_seed_fixture(case)`; the
human capture later OVERWRITES them with real gpt-4o output. Seeds are program TEMPLATES lifted
from the committed `examples/*_program.json` (the same derived-not-hand-authored posture as the
committed caches), so they exercise the real validate/execute/score path with realistic shapes.

In-scope groundable seeds: copy the matching pinned program's steps verbatim and replace ONLY the
final `answer` step's `question` arg with the case's question text. Grounding depends on the cache
+ sample windows, not on the question phrasing, so the retargeted seed still grounds to the cell's
expected answer (asserted by test_replay_over_committed_seeds_scores).

Out-of-scope / injection / degenerate seeds: a small set of canned templates inside this module —
a valid-but-honestly-ungrounded program (filter on an absent value), a validator-rejected program
(empty), and an `error`-tag fixture — so they score as PASS under the inverted ladder (D4).

This module is TESTED (test_seed_fixtures_regenerate_match), not dev-only dead code: regenerating
into a tmpdir must reproduce the committed fixtures modulo the prompt_version/captured_at stamps.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import canned
from validate_program import strip_comments

from .bank import load_cases, prompt_version
from .fixtures import Fixture, write_fixture

_HERE = Path(__file__).resolve().parent
API_DIR = _HERE.parent
EX = API_DIR / "examples"

# Fixed provenance stamp for committed seeds — deterministic so regeneration is byte-stable.
SEED_CAPTURED_AT = "1970-01-01T00:00:00Z"
SEED_MODEL = "seed"

# (clip_id, shape) -> the pinned grounded program to retarget for an in-scope case.
PINNED_BY_CELL = {
    ("single-goal", "first-goal"): EX / "hero_program.json",
    ("single-goal", "presence"): EX / "hero_presence_program.json",
    ("bernabeu-counter", "count"): EX / "bernabeu-counter_count_program.json",
    ("bernabeu-counter", "scorer-number"): EX / "bernabeu-counter_scorer-number_program.json",
    ("bernabeu-counter", "first-goal"): EX / "bernabeu-counter_first-goal-7_program.json",
    ("bernabeu-counter", "presence"): EX / "bernabeu-counter_presence_program.json",
}

# A valid program that grounds out HONESTLY (filter on a value absent from any cache) — used for
# out-of-scope / degenerate seeds so they PASS the inverted ladder via grounded:false.
HONEST_NEGATIVE_TEMPLATE = EX / "bernabeu-counter_first-goal-23_program.json"


def _load_program(path: Path) -> list[dict]:
    """Load a pinned examples/*_program.json, strip comments, unwrap ['program'] -> step list."""
    return strip_comments(json.loads(Path(path).read_text()))["program"]


def _retarget(program: list[dict], question: str) -> list[dict]:
    """Copy steps verbatim, replacing ONLY the final answer step's question arg."""
    prog = copy.deepcopy(program)
    for step in prog:
        if step["op"] == "answer":
            step["args"]["question"] = question
    return prog


# Clip-appropriate sample window for an honest-negative seed (must stay within the clip's
# duration so it validates, then filters on an absent jersey value -> grounded:false).
_NEGATIVE_WINDOW = {
    "single-goal": (3500, 5000, 8),
    "bernabeu-counter": (9000, 11000, 8),
}


def _honest_negative(question: str, clip_id: str) -> list[dict]:
    """A valid program that grounds out honestly (filter matches an absent jersey value -> empty
    subject -> grounded:false). The window is clamped to the clip's duration so it VALIDATES first,
    so the seed exercises the execute path (not just a reject)."""
    start, end, fps = _NEGATIVE_WINDOW[clip_id]
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": start, "end_ms": end, "fps": fps}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "absent", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "999"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "absent", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": question}},
    ]


def _rejected_program() -> list[dict]:
    """A structurally-invalid program (empty list) -> validator-rejected. PASS for out-of-scope."""
    return []


def build_seed_raw(case: dict):
    """Return (raw_program, error) for a case's seed fixture.

    Strategy by phrasing_class:
      canonical/paraphrase    -> retarget the cell's pinned grounded program (grounds to the answer).
      out_of_scope            -> the matching pinned absent-subject program if one exists for the
                                 cell, else an honest-negative program (grounded:false) or, for the
                                 'op the DSL can't express' flavor, a validator-rejected program.
      injection               -> a validator-rejected program (empty) so the trust boundary holds.
      degenerate              -> retarget the cell's pinned grounded program when the cell is
                                 groundable (the model would still ground a rambling but on-topic
                                 question), else an honest-negative program.
    """
    pc = case["phrasing_class"]
    cell = (case["clip_id"], case["shape"])

    if pc in ("canonical", "paraphrase"):
        return _retarget(_load_program(PINNED_BY_CELL[cell]), case["question"]), None

    if pc == "injection":
        # The trust boundary must reject the injected program; an empty program is structurally
        # invalid. Out-of-scope inversion makes a clean reject a PASS.
        return _rejected_program(), None

    if pc == "out_of_scope":
        # The #23 absent-subject cases retarget the pinned honest-negative program directly.
        if case["case_id"].startswith("bernabeu-first-goal-23"):
            return _retarget(_load_program(HONEST_NEGATIVE_TEMPLATE), case["question"]), None
        # 'Track ball speed', 'op the DSL can't express' -> the model would emit something the
        # validator rejects; model a clean reject. Others -> an honest-negative grounded:false.
        if case["case_id"] in ("oos-ball-speed", "oos-jersey-color", "oos-translate", "oos-offside", "oos-emotion"):
            return _rejected_program(), None
        return _honest_negative(case["question"], case["clip_id"]), None

    if pc == "degenerate":
        if case["case_id"] == "degen-rambling":
            # On-topic but rambling first-goal question -> grounds (retarget hero program).
            return _retarget(_load_program(PINNED_BY_CELL[(case["clip_id"], case["shape"])]), case["question"]), None
        if case["case_id"] == "degen-question-mark":
            # A bare "?" plausibly yields no usable output -> model the error/no-output path so a
            # seed exercises the REJECTED branch (out-of-scope inversion makes it a PASS).
            return None, "generation-failed"
        # Other degenerate cases -> honest no-ground.
        return _honest_negative(case["question"], case["clip_id"]), None

    raise ValueError(f"unhandled phrasing_class {pc!r} for {case['case_id']}")


def build_seed_fixture(case: dict) -> Fixture:
    """Build the Fixture for one case, stamped with the CURRENT prompt_version (so seeds are not
    stale and the gate exercises the real ladder)."""
    clip = canned.clip_by_id(case["clip_id"])
    raw, error = build_seed_raw(case)
    return Fixture(
        case_id=case["case_id"],
        clip_id=case["clip_id"],
        question=case["question"],
        prompt_version=prompt_version(clip),
        raw_program=raw,
        captured_at=SEED_CAPTURED_AT,
        model=SEED_MODEL,
        usage=None,
        error=error,
    )


def generate_all(fixtures_dir: Path | None = None) -> list[Path]:
    """Generate every seed fixture and write it. Returns the paths written."""
    written = []
    for case in load_cases():
        fx = build_seed_fixture(case)
        written.append(write_fixture(fx, fixtures_dir))
    return written


if __name__ == "__main__":
    # Bare script (python eval/seed_fixtures.py): bootstrap into the package so the relative imports
    # above resolve (api/ + api/scripts/ on path for the cross-package imports), then regenerate.
    import sys

    sys.path.insert(0, str(API_DIR))
    sys.path.insert(0, str(API_DIR / "scripts"))
    from eval.seed_fixtures import generate_all as _gen

    _paths = _gen()
    print(f"wrote {len(_paths)} seed fixtures to {_paths[0].parent if _paths else '(none)'}")
