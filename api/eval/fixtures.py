"""Fixture IO for the codegen eval harness.

A fixture is *untrusted recorded model output* (the raw program `codegen.generate()` returned),
the analogue of `api/examples/*_cache.json`. One file per case, named by `case_id`, committed.
Replay re-pushes `raw_program` through the real validator + interpreter, so a fixture can never
smuggle an unvalidated program into a pass. Pure IO; no Azure import.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "fixtures"

# Bumped only on an on-disk-shape change (mirrors run_cache.CACHE_FORMAT discipline).
FIXTURE_FORMAT = 1

# Fixed error tags — never raw exception text (info-leak discipline, mirroring server._ungrounded).
ERROR_GENERATION_FAILED = "generation-failed"
ERROR_OVERSIZED = "oversized"
ERROR_UNPARSEABLE = "unparseable"


@dataclass
class Fixture:
    """The D2 on-disk shape. `raw_program` is the EXACT list codegen.generate() returned
    (untrusted); `usage` is reserved and always None in this entry (Resolved #1); `error` is a
    fixed tag (or None) when generate() raised."""

    case_id: str
    clip_id: str
    question: str
    prompt_version: str
    raw_program: list[dict] | None
    captured_at: str
    model: str
    usage: dict | None = None
    error: str | None = None
    fixture_format: int = FIXTURE_FORMAT

    def to_ordered_dict(self) -> dict:
        """Field order chosen so `raw_program` reads near the top of a diff (the load-bearing
        artifact), and the noisy provenance stamps (`captured_at`) read last."""
        return {
            "fixture_format": self.fixture_format,
            "case_id": self.case_id,
            "clip_id": self.clip_id,
            "question": self.question,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "raw_program": self.raw_program,
            "usage": self.usage,
            "error": self.error,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fixture":
        return cls(
            case_id=d["case_id"],
            clip_id=d["clip_id"],
            question=d["question"],
            prompt_version=d["prompt_version"],
            raw_program=d.get("raw_program"),
            captured_at=d.get("captured_at", ""),
            model=d.get("model", ""),
            usage=d.get("usage"),
            error=d.get("error"),
            fixture_format=d.get("fixture_format", FIXTURE_FORMAT),
        )


def fixture_path(case_id: str, fixtures_dir: Path | None = None) -> Path:
    return (fixtures_dir or FIXTURES_DIR) / f"{case_id}.json"


def load_fixture(case_id: str, fixtures_dir: Path | None = None) -> Fixture | None:
    """Load a fixture by case_id, or None if absent (the replay MISSING outcome)."""
    path = fixture_path(case_id, fixtures_dir)
    if not path.exists():
        return None
    return Fixture.from_dict(json.loads(path.read_text()))


def write_fixture(fixture: Fixture, fixtures_dir: Path | None = None) -> Path:
    """Atomically write a fixture (mkstemp + os.replace, mirroring run_cache's atomic write) so a
    crash mid-write never corrupts a fixture and a re-capture replaces in place."""
    target_dir = fixtures_dir or FIXTURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_path(fixture.case_id, target_dir)
    payload = json.dumps(fixture.to_ordered_dict(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(target_dir), prefix=f".{fixture.case_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def is_fresh(fixture: Fixture, prompt_version: str) -> bool:
    """True iff the fixture was captured against the current prompt (no STALE drift)."""
    return fixture.prompt_version == prompt_version


# asdict is re-exported for callers that want the full dataclass dict (tests).
__all__ = [
    "Fixture",
    "fixture_path",
    "load_fixture",
    "write_fixture",
    "is_fresh",
    "asdict",
    "FIXTURE_FORMAT",
    "ERROR_GENERATION_FAILED",
    "ERROR_OVERSIZED",
    "ERROR_UNPARSEABLE",
]
