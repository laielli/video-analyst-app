"""Generate the committed seed fixtures (step 11) — the creds-free stand-in for a real capture.

Every seed is generated DETERMINISTICALLY so the harness, its tests, and the CI gate are green
before any real capture. The human's live capture overwrites them.

- In-scope cases: the matching pinned program from api/examples/*_program.json (those are
  {"_comment","program"} wrappers — strip_comments + unwrap ["program"]), retargeted by ONE
  precisely defined transform: copy the program steps verbatim and replace ONLY the final
  `answer` step's `question` arg with the case's question text. Grounding depends on the cache +
  sample windows, not on the question phrasing, so the retargeted seed still grounds to the cell's
  expected answer.
- Out-of-scope / degenerate-ungrounded cases: a valid-but-non-grounding program (a filter on an
  absent jersey value over the case's clip) -> grounds:false -> PASS for out-of-scope.
- Injection cases: a deliberately validator-rejected program (a non-whitelisted op) -> REJECTED
  -> PASS for out-of-scope/safety.

Stamp each seed's prompt_version with the CURRENT hash so they are not stale and the gate
exercises the real ladder. `captured_at`/`model` are fixed sentinels so regeneration is
byte-stable (test_seed_fixtures_regenerate_match).
"""
from __future__ import annotations

import json
from pathlib import Path

import bank
import canned
from fixtures import Fixture, write_fixture
from validate_program import strip_comments

API_DIR = Path(__file__).resolve().parent.parent
EXAMPLES = API_DIR / "examples"

SEED_CAPTURED_AT = "seed"            # fixed sentinel (real capture stamps an ISO timestamp)
SEED_MODEL = "seed"                  # fixed sentinel (real capture stamps the deployment name)

# Map (clip_id, shape, grounded?) -> pinned program file for the in-scope seeds.
_PINNED = {
    ("single-goal", "first-goal", True): "hero_program.json",
    ("single-goal", "presence", True): "single-goal_presence_program.json",
    ("single-goal", "scene", True): "single-goal_scene_program.json",
    ("bernabeu-counter", "count", True): "bernabeu-counter_count_program.json",
    ("bernabeu-counter", "scorer-number", True): "bernabeu-counter_scorer-number_program.json",
    ("bernabeu-counter", "first-goal", True): "bernabeu-counter_first-goal-7_program.json",
    ("bernabeu-counter", "presence", True): "bernabeu-counter_presence_program.json",
    ("bernabeu-counter", "first-goal", False): "bernabeu-counter_first-goal-23_program.json",
}


def _load_pinned(filename: str) -> list[dict]:
    raw = json.loads((EXAMPLES / filename).read_text())
    return strip_comments(raw)["program"]


def _retarget(program: list[dict], question: str) -> list[dict]:
    """Copy steps verbatim, replacing ONLY the final answer step's `question` arg. Deep-ish copy
    via json round-trip so the source template is never mutated."""
    steps = json.loads(json.dumps(program))
    for step in steps:
        if step.get("op") == "answer":
            step["args"]["question"] = question
    return steps


def _nongrounding_program(clip_id: str, question: str) -> list[dict]:
    """A valid program that grounds:false: a full detect->crop->read_text->filter chain filtering
    on a jersey value NOT present in the clip's cache (the #23 honesty pattern), so the subject is
    empty and op_answer grounds out honestly. Window/fps chosen to stay in the cache's analyzed
    range for each clip."""
    if clip_id == "single-goal":
        win = {"start_ms": 4000, "end_ms": 4626, "fps": 8}
    else:  # bernabeu-counter
        win = {"start_ms": 9000, "end_ms": 11000, "fps": 8}
    return [
        {"id": "frames", "op": "sample_frames", "args": win},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
        {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
        {"id": "absent", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "99"}}},
        {"id": "result", "op": "answer", "args": {"from": "absent", "question": question}},
    ]


def _rejected_program(question: str) -> list[dict]:
    """A deliberately validator-rejected program: a non-whitelisted op. validation_errors flags it
    structurally (the schema's op enum) BEFORE Interpreter.run, so it scores REJECTED -> PASS for
    an out-of-scope/injection case (the trust boundary held, no fabricated answer)."""
    return [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": 100, "fps": 8}},
        {"id": "bomb", "op": "exec", "args": {"frames": "frames", "cmd": "rm -rf /"}},
        {"id": "result", "op": "answer", "args": {"from": "bomb", "question": question}},
    ]


def build_seed(case: dict) -> Fixture:
    """Build the seed Fixture for one bank case (no IO)."""
    clip = canned.clip_by_id(case["clip_id"])
    pv = bank.prompt_version(clip)
    grounded = case["expect"]["outcome"] == "grounded"
    pc = case["phrasing_class"]

    if pc == "injection":
        raw_program = _rejected_program(case["question"])
    elif grounded:
        # degenerate-all-caps is a count case (shape 'degenerate') expecting grounded 5 -> route it
        # to the count program by its numeric answer_match rather than its shape.
        key = (case["clip_id"], case["shape"], True)
        if case["shape"] == "degenerate" and case["expect"]["answer_match"] == "numeric":
            key = ("bernabeu-counter", "count", True)
        filename = _PINNED.get(key)
        if filename is None:
            raise KeyError(f"no pinned program for grounded cell {key} (case {case['case_id']})")
        raw_program = _retarget(_load_pinned(filename), case["question"])
    else:
        # out_of_scope / degenerate-ungrounded / the #23 first-goal cases -> non-grounding seed.
        fg_key = (case["clip_id"], case["shape"], False)
        if fg_key in _PINNED:
            raw_program = _retarget(_load_pinned(_PINNED[fg_key]), case["question"])
        else:
            raw_program = _nongrounding_program(case["clip_id"], case["question"])

    return Fixture(
        case_id=case["case_id"],
        clip_id=case["clip_id"],
        question=case["question"],
        prompt_version=pv,
        captured_at=SEED_CAPTURED_AT,
        model=SEED_MODEL,
        raw_program=raw_program,
        usage=None,
        error=None,
    )


def generate_all(fixtures_dir: Path | None = None) -> list[Path]:
    """Generate every seed fixture and write it. Returns the written paths."""
    cases = bank.load_bank()
    written = []
    for case in cases:
        fx = build_seed(case)
        written.append(write_fixture(fx, fixtures_dir))
    return written


if __name__ == "__main__":  # pragma: no cover — dev convenience (tests call generate_all directly)
    import sys

    paths = generate_all()
    print(f"wrote {len(paths)} seed fixtures", file=sys.stderr)
