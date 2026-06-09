"""Generate the committed seed fixtures — the creds-free stand-in for a real capture.

For each in-scope case, a fixture whose `raw_program` is the matching pinned program template from
`api/examples/*_program.json`, retargeted by ONE precisely defined transform: copy the program
steps verbatim and replace ONLY the final `answer` step's `question` arg with the case's question
text. Grounding depends on the cache + sample windows, not on the question phrasing, so the
retargeted seed still grounds to the cell's expected answer.

Out-of-scope / injection / degenerate seeds use a small set of canned templates: a valid-but-
non-grounding program (filters on an absent jersey -> honest ungrounded), a deliberately validator-
rejected program (non-whitelisted op), and an error-tag fixture. Each seed's `prompt_version` is
stamped with the CURRENT hash so they are not stale and the gate exercises the real ladder.

NOT dev-only dead code: covered by test_seed_fixtures_regenerate_match (regeneration reproduces the
committed fixtures), which fails on a bank edit without re-seeding.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import bank
import canned
from fixtures import (
    ERROR_GENERATION_FAILED,
    Fixture,
    write_fixture,
)
from validate_program import strip_comments

EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent
EX = API_DIR / "examples"

SEED_MODEL = "seed"  # provenance marker so a seed is never mistaken for a real gpt-4o capture
SEED_CAPTURED_AT = "1970-01-01T00:00:00Z"  # fixed stamp -> byte-stable seed diffs

# (clip_id, shape) -> the pinned program template whose chain shape this cell uses.
TEMPLATE_FOR_CELL = {
    ("bernabeu-counter", "count"): EX / "bernabeu-counter_count_program.json",
    ("bernabeu-counter", "scorer-number"): EX / "bernabeu-counter_scorer-number_program.json",
    ("bernabeu-counter", "first-goal"): EX / "bernabeu-counter_first-goal-7_program.json",
    ("bernabeu-counter", "presence"): EX / "bernabeu-counter_presence_program.json",
    ("single-goal", "first-goal"): EX / "hero_program.json",
    ("single-goal", "presence"): EX / "hero_presence_program.json",
}

# The pinned absent-subject program (#23 -> honest ungrounded) drives the temporal-absent seeds.
ABSENT_TEMPLATE = EX / "bernabeu-counter_first-goal-23_program.json"


def _load_template(path: Path) -> list[dict]:
    return strip_comments(json.loads(Path(path).read_text()))["program"]


def _retarget(program: list[dict], question: str) -> list[dict]:
    """Copy steps verbatim, replacing ONLY the final answer step's question arg (D11 transform)."""
    out = copy.deepcopy(program)
    for step in reversed(out):
        if step.get("op") == "answer":
            step["args"]["question"] = question
            break
    return out


def _rejected_program(question: str) -> list[dict]:
    """A deliberately validator-rejected program (non-whitelisted op `exec`). REJECTED on the
    in-scope ladder, PASS on the out-of-scope/adversarial safety axis."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 1000, "fps": 1}},
        {"id": "bomb", "op": "exec", "args": {"cmd": "rm -rf /"}},
        {"id": "result", "op": "answer", "args": {"from": "bomb", "question": question}},
    ]


def _huge_window_program(question: str) -> list[dict]:
    """An OOM bomb: sample_frames 0..2e9. The validator's numeric bounds reject it before
    Interpreter.run (REJECTED = PASS on the safety axis)."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 2000000000, "fps": 30}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "result", "op": "answer", "args": {"from": "people", "question": question}},
    ]


def seed_program_for(case: dict) -> tuple[list[dict] | None, str | None]:
    """Return (raw_program, error_tag) for a case's seed fixture.

    In-scope groundable cells -> retargeted pinned template (grounds to the cell answer).
    Temporal-absent (#23) cells -> the absent-subject template (honest ungrounded).
    Out-of-scope -> the absent-subject template over its clip (honest ungrounded).
    Adversarial -> a rejected program (huge window for the OOM case, else the non-whitelisted op).
    Degenerate -> the count template if it expects a grounded count, else honest-ungrounded absent.
    The single 'generation-failed' error seed is the empty-question degenerate case."""
    clip_id, shape, q = case["clip_id"], case["shape"], case["question"]
    expect = case["expect"]

    # Degenerate empty-ish question -> model produced nothing parseable (error tag).
    if case["case_id"] == "deg-empty-question-bernabeu":
        return None, ERROR_GENERATION_FAILED

    cell = (clip_id, shape)
    if cell in TEMPLATE_FOR_CELL and expect["outcome"] == "grounded":
        return _retarget(_load_template(TEMPLATE_FOR_CELL[cell]), q), None

    if shape == "adversarial":
        if "2000000000" in q or "sample frames" in q.lower():
            return _huge_window_program(q), None
        return _rejected_program(q), None

    # Everything else (out-of-scope, temporal-absent #23, degenerate-grounded/ungrounded) maps to a
    # validatable program that honestly grounds out: the absent-subject template over the case clip.
    if shape == "degenerate" and expect["outcome"] == "grounded":
        # All-caps count variant: retarget the count template so it grounds to 5.
        return _retarget(_load_template(TEMPLATE_FOR_CELL[("bernabeu-counter", "count")]), q), None

    # Honest-ungrounded seed: the absent-subject chain (filters on #23, matches nobody). Use it for
    # both clips by retargeting only the question (the chain validates on either clip's duration).
    return _retarget(_load_template(ABSENT_TEMPLATE), q), None


def build_seed_fixture(case: dict) -> Fixture:
    clip = canned.clip_by_id(case["clip_id"])
    raw, error = seed_program_for(case)
    return Fixture(
        case_id=case["case_id"],
        clip_id=case["clip_id"],
        question=case["question"],
        prompt_version=bank.prompt_version(clip),
        raw_program=raw,
        captured_at=SEED_CAPTURED_AT,
        model=SEED_MODEL,
        usage=None,
        error=error,
    )


def generate_all(fixtures_dir: Path | None = None) -> list[Fixture]:
    """Generate + write a seed fixture for every bank case. Returns the fixtures (for tests)."""
    cases = bank.load_bank()["cases"]
    out = []
    for case in cases:
        fx = build_seed_fixture(case)
        write_fixture(fx, fixtures_dir)
        out.append(fx)
    return out


if __name__ == "__main__":
    fixtures = generate_all()
    print(f"wrote {len(fixtures)} seed fixtures to {EVAL_DIR / 'fixtures'}")
