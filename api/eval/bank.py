"""
Question-bank loader + provenance + ground-truth derivation (D1, D4).

Pure library code: imports `codegen` (for the prompt-version stamp and build_system_message),
`validate_program`, `interpreter`, `canned`. It NEVER imports `server` — importing server
instantiates FastAPI + a FileRunStore at module load, which the eval library must not pull in
(the drift guard that the bank schema's maxLength equals server.MAX_QUERY_TEXT_LEN lives in the
*test*, which may import server freely).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import codegen
from validate_program import strip_comments

HERE = Path(__file__).resolve().parent
BANK_PATH = HERE / "question_bank.json"
SCHEMA_PATH = HERE / "question_bank.schema.json"


def load_bank(bank_path: Path | None = None) -> dict:
    """Read + schema-validate question_bank.json; return the full bank object (with `bank_version`,
    `prompt_version` provenance + the `cases` list). Raises ValueError on schema failure so a
    malformed case fails loud at load, not silently at scoring."""
    import jsonschema

    path = bank_path or BANK_PATH
    bank = json.loads(path.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(bank), key=lambda e: list(e.path))
    if errors:
        loc = "/".join(str(p) for p in errors[0].path) or "<root>"
        raise ValueError(f"question_bank.json fails schema at {loc}: {errors[0].message}")
    return bank


def load_cases(bank_path: Path | None = None) -> list[dict]:
    """The `cases` list from the (schema-valid) bank."""
    return load_bank(bank_path)["cases"]


def filter_cases(cases, clip=None, shape=None, phrasing_class=None, only=None) -> list[dict]:
    """Filter cases for the CLI. `only` matches a case_id, shape, or clip_id (so the capture
    `--only` flag and replay `--shape`/`--clip` share one filter)."""
    out = list(cases)
    if clip:
        out = [c for c in out if c["clip_id"] == clip]
    if shape:
        out = [c for c in out if c["shape"] == shape]
    if phrasing_class:
        out = [c for c in out if c["phrasing_class"] == phrasing_class]
    if only:
        out = [c for c in out if only in (c["case_id"], c["shape"], c["clip_id"])]
    return out


def prompt_version(clip: dict | None) -> str:
    """The 16-hex prompt-version stamp: sha256(build_system_message(clip))[:16]. build_system_message
    is the FULL prompt surface the model sees (base SYSTEM_PROMPT rules AND the per-clip injected
    block) and already BEGINS with SYSTEM_PROMPT, so we hash it alone (concatenating SYSTEM_PROMPT in
    front would double-hash the base rules). With clip=None it equals sha256(SYSTEM_PROMPT)[:16]. Any
    prompt edit auto-invalidates the stamp — the same invalidation discipline as run_cache.CACHE_FORMAT."""
    msg = codegen.build_system_message(clip)
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]


def derive_ground_truth(program_path: str | Path, cache_path: str | Path) -> str | None:
    """Load a pinned examples/*_program.json, run it over its committed cache, return findings.answer.
    Used as a dev one-shot to backfill the bank's count + presence answers so the answer key can't
    drift from the cache (asserted by test_count_ground_truth_derived_from_pinned). The example files
    are {"_comment","program"} wrappers, so strip_comments + unwrap ["program"] before Interpreter.run
    (the same dance conftest's hero_program fixture does)."""
    from interpreter import Cache, Interpreter

    raw = json.loads(Path(program_path).read_text())
    program = strip_comments(raw)["program"]
    doc = Interpreter(Cache.load(cache_path)).run(program)
    return doc["findings"].get("answer")
