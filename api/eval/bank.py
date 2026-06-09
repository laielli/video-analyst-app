"""Question-bank loader + the prompt-version stamp + ground-truth derivation.

Pure, creds-free, and Azure-free. NEVER imports `server` (module load would instantiate FastAPI +
FileRunStore state); the schema/maxLength drift guard lives in a *test* (which may import `server`).

The bank (`question_bank.json`) is the single self-describing source of truth: a flat list of typed
cases, each with a co-located `expect` block. Ground truth for the count/presence shapes is *derived*
from the pinned `examples/*_program.json` executed over the committed cache (never hand-typed), so the
answer key can't drift from the cache — `derive_ground_truth` is the dev one-shot that does it and a
test asserts the bank matches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent
BANK_PATH = EVAL_DIR / "question_bank.json"
BANK_SCHEMA_PATH = EVAL_DIR / "question_bank.schema.json"


def load_bank(path: Path | str = BANK_PATH) -> dict:
    """Read + schema-validate the bank. Raises on schema failure (fail-closed). Returns the full
    bank object ({bank_version, prompt_version, cases:[...]}). The schema enforces the case shape so
    a malformed case can never reach the scorer."""
    import jsonschema  # lazy: the only validation dep, same as validate_program

    bank = json.loads(Path(path).read_text())
    schema = json.loads(BANK_SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(bank)
    return bank


def load_cases(path: Path | str = BANK_PATH) -> list[dict]:
    """The bank's case list (schema-validated)."""
    return load_bank(path)["cases"]


def filter_cases(
    cases: list[dict],
    clip: str | None = None,
    shape: str | None = None,
    phrasing_class: str | None = None,
    only: str | None = None,
) -> list[dict]:
    """Filter the case list by clip / shape / phrasing_class, plus a single `only` token that matches
    a case_id, a shape, or a clip_id (the capture `--only` convenience; reused by replay's
    `--shape`/`--clip`)."""
    out = list(cases)
    if clip is not None:
        out = [c for c in out if c["clip_id"] == clip]
    if shape is not None:
        out = [c for c in out if c["shape"] == shape]
    if phrasing_class is not None:
        out = [c for c in out if c["phrasing_class"] == phrasing_class]
    if only is not None:
        out = [
            c
            for c in out
            if only in (c["case_id"], c["shape"], c["clip_id"])
        ]
    return out


def prompt_version(clip: dict | None) -> str:
    """The 16-hex prompt-version stamp = sha256(build_system_message(clip))[:16].

    `build_system_message(clip)` IS the full prompt surface the model sees (base SYSTEM_PROMPT rules
    AND the injected per-clip block) and it already *begins with* SYSTEM_PROMPT, so we hash it alone
    (concatenating SYSTEM_PROMPT in front would double-hash the base rules). With clip=None it equals
    SYSTEM_PROMPT exactly. This is the single source of the version stamp, imported by both capture
    and replay; any prompt edit auto-invalidates fixtures via the recorded mismatch."""
    import codegen  # imported here (not Azure-touching: build_system_message is a pure string fn)

    surface = codegen.build_system_message(clip)
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()[:16]


def derive_ground_truth(program_path: Path | str, cache_path: Path | str) -> str | None:
    """Load a pinned `examples/*_program.json`, run it over its cache, return `findings.answer`.

    Used as a dev one-shot to backfill the bank's count + presence answers so they can't drift from
    the cache (asserted by a test). The `examples/*_program.json` files are `{"_comment","program"}`
    wrappers, not bare lists — strip comments + unwrap `["program"]` before Interpreter.run (the same
    dance conftest's `hero_program` fixture does)."""
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    raw = json.loads(Path(program_path).read_text())
    program = strip_comments(raw)["program"]
    doc = Interpreter(Cache.load(str(cache_path))).run(program, query="ground-truth-derivation")
    return doc["findings"]["answer"]
