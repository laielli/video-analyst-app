"""
Question-bank loader + the prompt-version stamp + ground-truth derivation (D1, D4).

Pure library code — no Azure import, and it NEVER imports `server` (importing server instantiates
FastAPI + a FileRunStore at module load; the eval library stays free of that state). The
server-drift guard for the bank schema's maxLength literal lives in the TEST, which may import
server freely.

`prompt_version(clip)` is the single source of the version stamp imported by both capture and
replay: `sha256(build_system_message(clip))[:16]`. `build_system_message` already BEGINS with
SYSTEM_PROMPT (and with clip=None equals it exactly), so this hashes the full surface the model
sees — base rules plus the injected per-clip block — without double-hashing the base rules.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BANK_PATH = EVAL_DIR / "question_bank.json"
BANK_SCHEMA_PATH = EVAL_DIR / "question_bank.schema.json"


def load_bank(path: Path | None = None) -> list[dict]:
    """Read + schema-validate the question bank, returning the typed case list. Raises ValueError
    on a schema failure so a malformed bank fails loudly (mirrors the repo's schema discipline)."""
    import jsonschema

    path = path or BANK_PATH
    doc = json.loads(path.read_text())
    schema = json.loads(BANK_SCHEMA_PATH.read_text())
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(doc),
        key=lambda e: list(e.path),
    )
    if errors:
        loc = "/".join(str(p) for p in errors[0].path) or "<root>"
        raise ValueError(f"question_bank.json fails its schema at {loc}: {errors[0].message}")
    return doc["cases"]


def load_bank_doc(path: Path | None = None) -> dict:
    """The full bank doc (bank_version/prompt_version provenance + cases). Schema-validated."""
    path = path or BANK_PATH
    load_bank(path)  # validate
    return json.loads(path.read_text())


def filter_cases(
    cases: list[dict],
    clip: str | None = None,
    shape: str | None = None,
    phrasing_class: str | None = None,
    only: str | None = None,
) -> list[dict]:
    """Filter the bank for the CLI. `only` matches a case_id, a shape, OR a clip_id (the capture
    `--only` convenience), so `--only count` and `--only bernabeu-count-canonical` both work."""
    out = cases
    if clip is not None:
        out = [c for c in out if c["clip_id"] == clip]
    if shape is not None:
        out = [c for c in out if c["shape"] == shape]
    if phrasing_class is not None:
        out = [c for c in out if c["phrasing_class"] == phrasing_class]
    if only is not None:
        out = [c for c in out if only in (c["case_id"], c["shape"], c["clip_id"])]
    return out


def prompt_version(clip: dict | None) -> str:
    """The 16-hex prompt-version stamp: sha256(build_system_message(clip))[:16]. The single source
    of the fixture version stamp (imported by both capture and replay). Any prompt edit — base
    rules or the injected per-clip block — auto-invalidates fixtures."""
    import codegen

    surface = codegen.build_system_message(clip)
    return hashlib.sha256(surface.encode("utf-8")).hexdigest()[:16]


def derive_ground_truth(program_path: str | Path, cache_path: str | Path) -> str | None:
    """Load a pinned `examples/*_program.json` (a {"_comment", "program"} wrapper), run it over its
    cache, and return `findings.answer`. The dev-time one-shot used to backfill the bank's count and
    presence answers so they can't drift from the cache (asserted by a test). Returns None when the
    program grounds out honestly."""
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    raw = json.loads(Path(program_path).read_text())
    program = strip_comments(raw)["program"]
    doc = Interpreter(Cache.load(str(cache_path))).run(program)
    return doc["findings"].get("answer")
