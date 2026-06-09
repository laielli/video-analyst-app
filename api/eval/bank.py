"""Question-bank loader, filtering, prompt-version stamp, and ground-truth derivation.

Pure data layer for the codegen eval harness. No Azure import, and — load-bearing — this module
NEVER imports `server` (importing server instantiates FastAPI + FileRunStore state). The bank's
ground truth is *derived* from the committed pinned programs executed over the committed caches,
never hand-typed, so the answer key cannot drift from the cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
API_DIR = EVAL_DIR.parent
BANK_PATH = EVAL_DIR / "question_bank.json"
BANK_SCHEMA_PATH = EVAL_DIR / "question_bank.schema.json"


def load_bank(path: Path | None = None) -> dict:
    """Read + schema-validate question_bank.json. Raises ValueError on a schema failure so a
    malformed case can never reach the scorer. Returns the full bank object (bank_version,
    prompt_version, cases)."""
    import jsonschema

    bank_path = path or BANK_PATH
    bank = json.loads(bank_path.read_text())
    schema = json.loads(BANK_SCHEMA_PATH.read_text())
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(bank),
        key=lambda e: list(e.path),
    )
    if errors:
        first = errors[0]
        loc = "/".join(str(p) for p in first.path) or "<root>"
        raise ValueError(f"question_bank.json fails schema at {loc}: {first.message}")
    seen: set[str] = set()
    for case in bank["cases"]:
        cid = case["case_id"]
        if cid in seen:
            raise ValueError(f"duplicate case_id in question_bank.json: {cid!r}")
        seen.add(cid)
    return bank


def load_cases(path: Path | None = None) -> list[dict]:
    """Just the `cases` list (schema-validated)."""
    return load_bank(path)["cases"]


def filter_cases(
    cases: list[dict],
    clip: str | None = None,
    shape: str | None = None,
    phrasing_class: str | None = None,
    only: str | None = None,
) -> list[dict]:
    """Filter cases by clip_id / shape / phrasing_class, plus a single `only` token that matches a
    case_id OR a shape OR a clip_id (the CLI's `--only` convenience)."""
    out = list(cases)
    if clip is not None:
        out = [c for c in out if c["clip_id"] == clip]
    if shape is not None:
        out = [c for c in out if c["shape"] == shape]
    if phrasing_class is not None:
        out = [c for c in out if c["phrasing_class"] == phrasing_class]
    if only is not None:
        out = [c for c in out if only in (c["case_id"], c["shape"], c["clip_id"])]
    return out


def prompt_version(clip: dict | None = None) -> str:
    """The fixture invalidation stamp: sha256(build_system_message(clip))[:16].

    `build_system_message(clip)` IS the full prompt surface the model sees (base SYSTEM_PROMPT
    rules AND the per-clip injected block) and it already begins with SYSTEM_PROMPT, so hashing it
    alone is correct (concatenating SYSTEM_PROMPT in front would double-hash the base rules). Any
    prompt edit — base rules or a clip hint — auto-invalidates every affected fixture."""
    import codegen

    msg = codegen.build_system_message(clip)
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]


def derive_ground_truth(program_path: str | Path, cache_path: str | Path) -> str | None:
    """Run a pinned `examples/*_program.json` over its cache and return `findings.answer`.

    Dev one-shot used while authoring the bank to backfill the count (and presence) answers so
    they can't drift from the cache; also asserted by a test. The examples files are
    `{"_comment", "program"}` wrappers, not bare lists — strip comments + unwrap ["program"]
    before Interpreter.run (the same dance the conftest hero_program fixture does)."""
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    raw = json.loads(Path(program_path).read_text())
    program = strip_comments(raw)["program"]
    doc = Interpreter(Cache.load(str(cache_path))).run(program)
    return doc["findings"]["answer"]
