"""Generate the creds-free SEED fixtures (step 11) — the stand-in for a real gpt-4o capture.

These are committed under `eval/fixtures/*.json` and let replay + its tests + the CI gate go green
BEFORE any billed capture. The human capture later OVERWRITES them with real model output.

Honesty posture (same as precompute.py's fake-vision fixtures): seeds are program *templates* lifted
from the committed `examples/*_program.json`, so they exercise the real validate/execute/score path
with realistic shapes. The `git diff eval/report.json` after the human capture surfaces any in-scope
case the real model fails that the seed passed.

The precise transform for an in-scope groundable seed: copy the pinned program steps verbatim and
replace ONLY the final `answer` step's `question` arg with the case's question text (grounding
depends on the cache + sample windows, not on question phrasing, so the retargeted seed still grounds
to the cell's expected answer — asserted by test_replay_over_committed_seeds_scores).

Out-of-scope / injection / degenerate seeds come from canned templates here:
  - a valid-but-non-grounding program (filter on an absent value -> honest grounded:false) -> PASS,
  - a validator-rejected program (a non-whitelisted op) -> REJECTED -> PASS (out-of-scope),
  - an `error`-tag fixture (generation-failed) -> REJECTED -> PASS (out-of-scope).
"""
from __future__ import annotations

import json
from pathlib import Path

from .bank import load_cases, prompt_version
from .fixtures import ERROR_GENERATION_FAILED, FIXTURES_DIR, Fixture, write_fixture

EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent
EXAMPLES = API_DIR / "examples"

# Seeds carry a fixed captured_at/model so regeneration is byte-stable except prompt_version
# (the test compares modulo prompt_version/captured_at). We stamp the CURRENT prompt_version so the
# seeds are NOT stale and the gate exercises the real ladder.
SEED_CAPTURED_AT = "2026-06-09T00:00:00Z"
SEED_MODEL = "seed"


def _pinned_program(name: str) -> list[dict]:
    """Load a pinned examples/*_program.json: strip comments, unwrap ['program']."""
    from validate_program import strip_comments

    raw = json.loads((EXAMPLES / name).read_text())
    return strip_comments(raw)["program"]


def _presence_program(clip_id: str) -> list[dict]:
    """The derived presence program (D1 presence caveat): sample -> detect person -> answer from the
    detections kind. Window matches the clip's legible frame so it grounds to 'Yes'."""
    if clip_id == "single-goal":
        window = {"start_ms": 3500, "end_ms": 5000, "fps": 8}
    else:  # bernabeu-counter
        window = {"start_ms": 10000, "end_ms": 10001, "fps": 1}
    return [
        {"id": "frames", "op": "sample_frames", "args": window},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "result", "op": "answer", "args": {"from": "people", "question": "presence"}},
    ]


# Map (clip_id, shape) -> a factory for the pinned/derived program template.
def _grounded_template(case: dict) -> list[dict]:
    clip_id, shape = case["clip_id"], case["shape"]
    if shape == "first-goal":
        if case["expect"]["outcome"] == "ungrounded":
            # absent-subject (#23): the pinned ungrounded program (filters a value not in the cache).
            return _pinned_program("bernabeu-counter_first-goal-23_program.json")
        if clip_id == "single-goal":
            return _pinned_program("hero_program.json")
        return _pinned_program("bernabeu-counter_first-goal-7_program.json")
    if shape == "count":
        return _pinned_program("bernabeu-counter_count_program.json")
    if shape == "scorer-number":
        return _pinned_program("bernabeu-counter_scorer-number_program.json")
    if shape == "presence":
        return _presence_program(clip_id)
    raise ValueError(f"no grounded template for shape {shape!r}")


def _retarget(program: list[dict], question: str) -> list[dict]:
    """Copy the program and replace ONLY the final answer step's `question` arg with `question`."""
    prog = json.loads(json.dumps(program))  # deep copy
    for step in reversed(prog):
        if step.get("op") == "answer":
            step["args"]["question"] = question
            break
    return prog


def _non_grounding_program(clip_id: str, question: str) -> list[dict]:
    """A valid program that grounds out HONESTLY (filter on a jersey value absent from the cache ->
    grounded:false). The out-of-scope seed default: PASS via honest refusal."""
    if clip_id == "single-goal":
        window = {"start_ms": 3500, "end_ms": 5000, "fps": 8}
    else:
        window = {"start_ms": 9000, "end_ms": 11000, "fps": 8}
    return [
        {"id": "frames", "op": "sample_frames", "args": window},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "absent", "op": "filter",
         "args": {"items": "numbers", "where": {"field": "text", "equals": "99"}}},
        {"id": "ordered", "op": "temporal_order", "args": {"events": "absent", "by": "timestamp"}},
        {"id": "result", "op": "answer", "args": {"from": "ordered", "question": question}},
    ]


def _rejected_program(question: str) -> list[dict]:
    """A program the validator REJECTS (a non-whitelisted op). The injection seed for 'a bomb must be
    rejected, never executed': REJECTED -> PASS (out-of-scope)."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 1}},
        {"id": "danger", "op": "exec", "args": {"cmd": "rm -rf /"}},
        {"id": "result", "op": "answer", "args": {"from": "frames", "question": question}},
    ]


# Which injection/degenerate cases get the error-tag fixture vs the rejected-program vs non-grounding.
# Spread the three out-of-scope seed kinds so each is exercised + covered.
def _out_of_scope_fixture(case: dict, pv: str) -> Fixture:
    cid = case["case_id"]
    clip_id = case["clip_id"]
    q = case["question"]
    # An error-tag fixture for one degenerate case (covers the REJECTED/error path).
    if cid == "deg-empty-ish":
        return Fixture(case_id=cid, clip_id=clip_id, question=q, prompt_version=pv,
                       raw_program=None, error=ERROR_GENERATION_FAILED, usage=None,
                       captured_at=SEED_CAPTURED_AT, model=SEED_MODEL)
    # A validator-rejected program for the explicit injection cases (a bomb must be rejected).
    if case["phrasing_class"] == "injection":
        prog = _rejected_program(q)
    else:
        prog = _non_grounding_program(clip_id, q)
    return Fixture(case_id=cid, clip_id=clip_id, question=q, prompt_version=pv,
                   raw_program=prog, error=None, usage=None,
                   captured_at=SEED_CAPTURED_AT, model=SEED_MODEL)


def build_seed_fixture(case: dict) -> Fixture:
    """The seed Fixture for one bank case (deterministic; stamped with the current prompt_version)."""
    import canned

    clip = canned.clip_by_id(case["clip_id"])
    pv = prompt_version(clip)
    if case["phrasing_class"] in ("out_of_scope", "injection", "degenerate"):
        return _out_of_scope_fixture(case, pv)
    # In-scope (canonical/paraphrase, including the absent-subject #23 cases): retarget the pinned
    # program's answer question to the case text.
    template = _grounded_template(case)
    prog = _retarget(template, case["question"])
    return Fixture(case_id=case["case_id"], clip_id=case["clip_id"], question=case["question"],
                   prompt_version=pv, raw_program=prog, error=None, usage=None,
                   captured_at=SEED_CAPTURED_AT, model=SEED_MODEL)


def generate_seeds(cases=None, fixtures_dir: Path = FIXTURES_DIR) -> int:
    """Write a seed fixture for every bank case. Returns the count written. Used to regenerate the
    committed seeds (a bank edit without re-seeding fails test_seed_fixtures_regenerate_match)."""
    if cases is None:
        cases = load_cases()
    n = 0
    for case in cases:
        write_fixture(build_seed_fixture(case), fixtures_dir)
        n += 1
    return n


if __name__ == "__main__":  # pragma: no cover - dev one-shot to (re)write committed seeds
    import sys

    sys.path.insert(0, str(API_DIR))
    sys.path.insert(0, str(API_DIR / "scripts"))
    count = generate_seeds()
    print(f"wrote {count} seed fixtures to {FIXTURES_DIR}")
