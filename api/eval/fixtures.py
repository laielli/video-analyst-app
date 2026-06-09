"""
Fixture IO (D2): one recorded-model-output file per case at `fixtures/<case_id>.json`.

A fixture records the UNTRUSTED raw model output (`raw_program`, exactly what
`codegen.generate` returns) — NOT the validated program and NOT the run-doc. Replay re-pushes
`raw_program` through the real `validation_errors` + `Interpreter.run`, so a fixture can never
smuggle an unvalidated program into a "pass". Pure IO; no Azure import.

Field ordering puts `raw_program` near the top so a re-capture's diff reads the model-output
change first, ahead of the `captured_at` provenance restamp (the Deferred-notes cosmetic).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

FIXTURE_FORMAT = 1

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "fixtures"


@dataclass
class Fixture:
    """The D2 on-disk shape. `raw_program` is the parsed list `generate()` returns (None when
    `generate` raised and `error` carries a fixed tag). `usage` is reserved and always null in this
    entry (Resolved #1)."""
    case_id: str
    clip_id: str
    question: str
    prompt_version: str
    raw_program: list[dict] | None = None
    error: str | None = None
    captured_at: str | None = None
    model: str | None = None
    usage: dict | None = None
    fixture_format: int = FIXTURE_FORMAT

    def to_json_obj(self) -> dict:
        """Ordered dict for the on-disk file: identity + prompt_version, then raw_program (so the
        load-bearing diff reads first), then error, then provenance (model/captured_at/usage)."""
        return {
            "fixture_format": self.fixture_format,
            "case_id": self.case_id,
            "clip_id": self.clip_id,
            "question": self.question,
            "prompt_version": self.prompt_version,
            "raw_program": self.raw_program,
            "error": self.error,
            "model": self.model,
            "captured_at": self.captured_at,
            "usage": self.usage,
        }


def fixture_path(case_id: str, fixtures_dir: Path | None = None) -> Path:
    return (fixtures_dir or FIXTURES_DIR) / f"{case_id}.json"


def load_fixture(case_id: str, fixtures_dir: Path | None = None) -> Fixture | None:
    """Load a fixture by case_id, or None if absent (the MISSING outcome)."""
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
        error=data.get("error"),
        captured_at=data.get("captured_at"),
        model=data.get("model"),
        usage=data.get("usage"),
        fixture_format=data.get("fixture_format", FIXTURE_FORMAT),
    )


def write_fixture(fixture: Fixture, fixtures_dir: Path | None = None) -> Path:
    """Atomically write a fixture (mkstemp + os.replace, mirroring run_cache) so a crash mid-write
    never corrupts a fixture. Returns the path written."""
    path = fixture_path(fixture.case_id, fixtures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixture.to_json_obj(), indent=2, ensure_ascii=False) + "\n"
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
    """True iff the fixture was captured against the current prompt surface (D2 invalidation)."""
    return fixture.prompt_version == prompt_version
