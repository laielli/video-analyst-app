"""
Fixture IO for the eval harness (D2). A fixture is *untrusted recorded model output* — the raw
list `codegen.generate()` returns — committed one file per case at `eval/fixtures/<case_id>.json`,
the analogue of `examples/*_cache.json`. Replay re-pushes `raw_program` through the real
`validation_errors` + `Interpreter.run`, so a fixture can never smuggle an unvalidated program
into a "pass".

Pure IO: no Azure import, no scoring. Atomic writes (mkstemp + os.replace) mirror run_cache so a
crash mid-write never corrupts a fixture.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"

# Bump this kill-switch when the on-disk fixture shape changes (mirrors run_cache.CACHE_FORMAT).
FIXTURE_FORMAT = 1


@dataclass
class Fixture:
    """The D2 on-disk shape. `raw_program` is EXACTLY codegen.generate()'s return (untrusted),
    or None when generate() raised (then `error` carries a fixed tag). `usage` is reserved and
    always None in this entry (Resolved decision #1 — no production usage seam). `prompt_version`
    is the stamp replay compares against the current hash to detect staleness."""

    case_id: str
    clip_id: str
    question: str
    prompt_version: str
    raw_program: list | None
    captured_at: str | None = None
    model: str | None = None
    usage: None = None  # reserved; always null this entry (Resolved #1); never scored
    error: str | None = None
    fixture_format: int = FIXTURE_FORMAT

    def to_dict(self) -> dict:
        # Field order is chosen so a diff reads identity first (case/clip/question), then the
        # load-bearing `raw_program`, with provenance (captured_at/model) last to keep re-capture
        # restamp noise out of the lines a reviewer scans first (Deferred note: captured_at noise).
        return {
            "fixture_format": self.fixture_format,
            "case_id": self.case_id,
            "clip_id": self.clip_id,
            "question": self.question,
            "prompt_version": self.prompt_version,
            "raw_program": self.raw_program,
            "usage": self.usage,
            "error": self.error,
            "model": self.model,
            "captured_at": self.captured_at,
        }


def fixture_path(case_id: str, fixtures_dir: Path | None = None) -> Path:
    return (fixtures_dir or FIXTURES_DIR) / f"{case_id}.json"


def load_fixture(case_id: str, fixtures_dir: Path | None = None) -> Fixture | None:
    """Load a fixture by case_id, or None if it is absent (a MISSING outcome at replay)."""
    path = fixture_path(case_id, fixtures_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Fixture(
        case_id=data["case_id"],
        clip_id=data["clip_id"],
        question=data["question"],
        prompt_version=data["prompt_version"],
        raw_program=data.get("raw_program"),
        captured_at=data.get("captured_at"),
        model=data.get("model"),
        usage=data.get("usage"),
        error=data.get("error"),
        fixture_format=data.get("fixture_format", FIXTURE_FORMAT),
    )


def write_fixture(fixture: Fixture, fixtures_dir: Path | None = None) -> Path:
    """Write a fixture atomically (mkstemp + os.replace), pretty + trailing newline so the
    committed JSON diffs line-by-line. Returns the path written."""
    path = fixture_path(fixture.case_id, fixtures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixture.to_dict(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def is_fresh(fixture: Fixture, prompt_version: str) -> bool:
    """True iff the fixture was captured against the CURRENT prompt (its prompt_version matches).
    A False here marks the fixture STALE at replay (D6 stale policy)."""
    return fixture.prompt_version == prompt_version
