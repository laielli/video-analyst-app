"""Fixture IO (D2): one file per case at `eval/fixtures/<case_id>.json`, committed.

A fixture is *untrusted recorded model output* — the raw list `codegen.generate()` returns, NOT the
validated program and NOT the run-doc — so replay must re-push it through the real
`validation_errors` + `Interpreter.run` gate (a fixture can never smuggle an unvalidated program into
a "pass"). Pure IO, no Azure import.

Field order is deliberate: the diff-noisy provenance fields (`captured_at`, `model`) sit AFTER
`raw_program` so a re-capture's diff reads the model output first (Deferred-notes cosmetic).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "fixtures"

# Bump only on an on-disk shape change. The nullable `usage` field stays in the format so a future
# additive `generate_with_usage` seam lands without a format bump (Resolved #1).
FIXTURE_FORMAT = 1

# Fixed error tags (info-leak discipline, mirroring server._ungrounded) — never raw exception text.
ERROR_GENERATION_FAILED = "generation-failed"
ERROR_OVERSIZED = "oversized"
ERROR_UNPARSEABLE = "unparseable"


@dataclass
class Fixture:
    """The D2 on-disk shape. `raw_program` is EXACTLY codegen.generate()'s return (untrusted), or
    None when generate() raised (then `error` carries a fixed tag). `usage` is reserved + always
    null in this entry (Resolved #1); never scored."""

    case_id: str
    clip_id: str
    question: str
    prompt_version: str
    raw_program: list | None
    error: str | None = None
    usage: dict | None = None
    captured_at: str | None = None
    model: str | None = None
    fixture_format: int = FIXTURE_FORMAT

    def to_dict(self) -> dict:
        """Serialize with raw_program ahead of the diff-noisy provenance fields."""
        return {
            "fixture_format": self.fixture_format,
            "case_id": self.case_id,
            "clip_id": self.clip_id,
            "question": self.question,
            "prompt_version": self.prompt_version,
            "raw_program": self.raw_program,
            "usage": self.usage,
            "error": self.error,
            "captured_at": self.captured_at,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fixture":
        return cls(
            case_id=d["case_id"],
            clip_id=d["clip_id"],
            question=d["question"],
            prompt_version=d["prompt_version"],
            raw_program=d.get("raw_program"),
            error=d.get("error"),
            usage=d.get("usage"),
            captured_at=d.get("captured_at"),
            model=d.get("model"),
            fixture_format=d.get("fixture_format", FIXTURE_FORMAT),
        )


def fixture_path(case_id: str, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    return fixtures_dir / f"{case_id}.json"


def load_fixture(case_id: str, fixtures_dir: Path = FIXTURES_DIR) -> Fixture | None:
    """Load a fixture by case_id, or None if absent (the replay MISSING outcome)."""
    p = fixture_path(case_id, fixtures_dir)
    if not p.exists():
        return None
    return Fixture.from_dict(json.loads(p.read_text()))


def write_fixture(fixture: Fixture, fixtures_dir: Path = FIXTURES_DIR) -> Path:
    """Atomic write (mkstemp + os.replace, mirroring run_cache) so a crash mid-write never corrupts a
    fixture. Returns the path written. Sorted keys are NOT used — the field order in to_dict() is the
    intended on-disk order (raw_program before provenance)."""
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    p = fixture_path(fixture.case_id, fixtures_dir)
    payload = json.dumps(fixture.to_dict(), indent=2, ensure_ascii=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(fixtures_dir), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; committed fixtures want world-readable.
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def is_fresh(fixture: Fixture, current_prompt_version: str) -> bool:
    """True iff the fixture was captured against the current prompt (prompt-version match). A
    mismatch marks the fixture STALE in the report and triggers D6's stale policy."""
    return fixture.prompt_version == current_prompt_version
