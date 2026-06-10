"""Fixture IO (D2). A fixture is one recorded gpt-4o codegen output per case, committed under
`fixtures/<case_id>.json`. It stores the UNTRUSTED raw model output (`raw_program`, the list
codegen.generate returns) — never the validated program, never the run-doc — so replay re-runs
the real trust boundary on every fixture. Atomic write mirrors run_cache's mkstemp + os.replace.
Pure IO; no Azure import.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "fixtures"

# Fixture on-disk schema version. Bump only on a breaking layout change (the nullable `usage`
# field reserved for a future additive seam means adding usage capture is NOT a bump — D2).
FIXTURE_FORMAT = 1

# Fixed error tags (never raw exception text — info-leak discipline, mirroring server._ungrounded).
ERR_GENERATION_FAILED = "generation-failed"
ERR_OVERSIZED = "oversized"
ERR_UNPARSEABLE = "unparseable"


@dataclass
class Fixture:
    """The D2 on-disk shape. `raw_program` is the parsed list codegen.generate returns (untrusted);
    on a generate() failure `raw_program` is None and `error` carries a fixed tag. `usage` is
    reserved and always None in this entry (Resolved #1)."""

    case_id: str
    clip_id: str
    question: str
    prompt_version: str
    captured_at: str
    model: str
    raw_program: list | None
    usage: dict | None = None
    error: str | None = None
    fixture_format: int = FIXTURE_FORMAT

    def to_dict(self) -> dict:
        # Field order favours diff legibility: raw_program reads before the provenance stamps so a
        # re-capture's program change shows first, even though captured_at restamps each run.
        return {
            "fixture_format": self.fixture_format,
            "case_id": self.case_id,
            "clip_id": self.clip_id,
            "question": self.question,
            "raw_program": self.raw_program,
            "error": self.error,
            "usage": self.usage,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fixture":
        return cls(
            case_id=d["case_id"],
            clip_id=d["clip_id"],
            question=d["question"],
            prompt_version=d["prompt_version"],
            captured_at=d.get("captured_at", ""),
            model=d.get("model", ""),
            raw_program=d.get("raw_program"),
            usage=d.get("usage"),
            error=d.get("error"),
            fixture_format=d.get("fixture_format", FIXTURE_FORMAT),
        )


def fixture_path(case_id: str, fixtures_dir: Path | None = None) -> Path:
    return (fixtures_dir or FIXTURES_DIR) / f"{case_id}.json"


def load_fixture(case_id: str, fixtures_dir: Path | None = None) -> Fixture | None:
    """Load a fixture by case_id; None if absent (a MISSING fixture is a distinct replay outcome)."""
    p = fixture_path(case_id, fixtures_dir)
    if not p.exists():
        return None
    return Fixture.from_dict(json.loads(p.read_text()))


def write_fixture(fixture: Fixture, fixtures_dir: Path | None = None) -> Path:
    """Atomic write (mkstemp + os.replace) so a crash mid-write never corrupts a fixture."""
    out_dir = fixtures_dir or FIXTURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = fixture_path(fixture.case_id, out_dir)
    body = json.dumps(fixture.to_dict(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), prefix=f".{fixture.case_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(tmp, 0o644)  # mkstemp defaults to 0600; these are long-lived committed artifacts
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def is_fresh(fixture: Fixture, prompt_version: str) -> bool:
    """True iff the fixture was captured under the current prompt version (no drift). A mismatch
    marks the fixture STALE in replay and triggers D6's stale policy."""
    return fixture.prompt_version == prompt_version


# Keep asdict importable for tests that want the dataclass form directly.
__all__ = [
    "Fixture",
    "FIXTURE_FORMAT",
    "ERR_GENERATION_FAILED",
    "ERR_OVERSIZED",
    "ERR_UNPARSEABLE",
    "fixture_path",
    "load_fixture",
    "write_fixture",
    "is_fresh",
    "asdict",
]
