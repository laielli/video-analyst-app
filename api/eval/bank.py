"""Question-bank loader + provenance helpers (D1/D4). Pure: no Azure import, NEVER imports
`server` (module load would instantiate FastAPI + FileRunStore state — the eval library must
stay HTTP/state-free; the `MAX_QUERY_TEXT_LEN` drift guard lives in the *test*, which may import
`server` freely).

Imports `codegen.build_system_message` (no network) for the prompt-version stamp, and
`validate_program` + `interpreter` + `canned` for ground-truth derivation. The version stamp is
the single source of truth shared by both capture and replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BANK_PATH = EVAL_DIR / "question_bank.json"
SCHEMA_PATH = EVAL_DIR / "question_bank.schema.json"


def prompt_version(clip: dict | None = None) -> str:
    """sha256(build_system_message(clip))[:16] — the version stamp. build_system_message already
    BEGINS with SYSTEM_PROMPT (with clip=None it equals SYSTEM_PROMPT exactly), so this hashes the
    full prompt surface the model sees (base rules + injected per-clip block) without
    double-hashing the base rules. Any prompt edit auto-invalidates the matching fixtures."""
    import codegen  # local import: keep bank importable without a configured Azure client

    msg = codegen.build_system_message(clip)
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]


def load_bank(path: Path | None = None) -> list[dict]:
    """Read + schema-validate the question bank; return the list of case dicts. Raises on a schema
    failure (a malformed case must not silently become a scored case)."""
    import jsonschema

    bank_path = path or BANK_PATH
    doc = json.loads(bank_path.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator(schema).validate(doc)
    return doc["cases"]


def load_bank_doc(path: Path | None = None) -> dict:
    """The whole bank doc (bank_version, prompt_version, cases) — for report provenance."""
    bank_path = path or BANK_PATH
    return json.loads(bank_path.read_text())


def filter_cases(
    cases: list[dict],
    clip: str | None = None,
    shape: str | None = None,
    phrasing_class: str | None = None,
    only: str | None = None,
) -> list[dict]:
    """Filter the bank for CLI selection (`--shape`, `--clip`, `--only <case_id|shape|clip>`).
    `only` matches a case_id exactly, OR a shape, OR a clip_id — the union, so `--only count`
    selects every count case and `--only bernabeu-count-canonical` selects the one case."""
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
            if c["case_id"] == only or c["shape"] == only or c["clip_id"] == only
        ]
    return out


def derive_ground_truth(program_path: Path | str, cache_path: Path | str) -> str | None:
    """Load a pinned examples/*_program.json, run it over its cache, return findings.answer.

    Used as a dev one-shot to backfill the bank's count (and presence) answers so they can't drift
    from the cache, and asserted by a test. The examples files are {"_comment", "program"}
    wrappers, not bare lists — strip_comments + unwrap ["program"] before Interpreter.run (the
    same dance conftest's hero_program fixture does)."""
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    raw = json.loads(Path(program_path).read_text())
    program = strip_comments(raw)["program"]
    doc = Interpreter(Cache.load(cache_path)).run(program, query="ground-truth derivation")
    return doc["findings"].get("answer")
