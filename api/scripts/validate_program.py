#!/usr/bin/env python3
"""
Validate a visual program against dsl.schema.json (structure) and the named-binding
rules the interpreter enforces (semantics). This is the "valid by construction" gate
from the dsl-json-schema-unification decision: the same schema that constrains Azure
OpenAI codegen is checked here, plus the binding/kind flow the JSON Schema can't express.

Usage:
    python validate_program.py                       # validates examples/hero_program.json
    python validate_program.py path/to/program.json

Exit 0 = valid, 1 = invalid (errors printed), 2 = setup problem (missing dep/file).

Authoring convention: keys beginning with "_" (e.g. "_comment") are treated as comments
and stripped before validation, so examples can carry inline notes while staying valid.
The Azure OpenAI strict-mode codegen never emits such keys.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE.parent / "schema" / "dsl.schema.json"
DEFAULT_PROGRAM = HERE.parent / "examples" / "hero_program.json"

# Output "kind" each op produces. filter/temporal_order pass their input collection
# kind through; we resolve those at walk time.
PRODUCES = {
    "sample_frames": "frames",
    "detect": "detections",
    "crop": "crops",
    "read_text": "texts",
    "count": "number",
    "answer": "answer",
}
# (binding arg field -> required input kind, or None for "any collection").
EXPECTS = {
    "detect": {"frames": "frames"},
    "crop": {"detections": "detections"},
    "read_text": {"crops": "crops"},
    "filter": {"items": None},
    "count": {"items": None},
    "temporal_order": {"events": None},
    "answer": {"from": None},
}
COLLECTION_KINDS = {"frames", "detections", "crops", "texts"}


def strip_comments(obj):
    """Recursively drop dict keys starting with '_' (authoring comments)."""
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_comments(v) for v in obj]
    return obj


def structural_errors(program_doc: dict, schema: dict) -> list[str]:
    """JSON Schema (structure) errors for a comment-stripped program doc. Empty = conforms."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(schema)
    out = []
    for e in sorted(validator.iter_errors(program_doc), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in e.path) or "<root>"
        out.append(f"at {loc}: {e.message}")
    return out


def validation_errors(program_doc: dict, schema: dict | None = None) -> list[str]:
    """The full valid-by-construction gate (structural then semantic) as one flat list — empty
    means valid. Shared by this CLI and the codegen fallback path (server._try_live). Comments
    are stripped here so callers can pass raw codegen/authoring output. Semantic checks are
    skipped when structure fails (malformed refs make them meaningless)."""
    if schema is None:
        schema = json.loads(SCHEMA_PATH.read_text())
    doc = strip_comments(program_doc)
    struct = structural_errors(doc, schema)
    if struct:
        return [f"structural: {m}" for m in struct]
    return [f"semantic: {m}" for m in semantic_errors(doc["program"])]


def semantic_errors(program: list[dict]) -> list[str]:
    """The checks the JSON Schema cannot express: ref existence/order, kind flow,
    unique ids, and answer-is-last."""
    errors: list[str] = []
    produced: dict[str, str] = {}  # id -> output kind, in declaration order

    for i, step in enumerate(program):
        sid = step.get("id")
        op = step.get("op")
        args = step.get("args", {})

        if not sid:
            errors.append(f"step[{i}] ({op}): missing id")
            continue
        if sid in produced:
            errors.append(f"step[{i}] ({op}): duplicate id '{sid}'")

        # Check each binding arg references an already-produced, kind-compatible step.
        for field, want in EXPECTS.get(op, {}).items():
            ref = args.get(field)
            if ref is None:
                continue  # structural validation already covers required args
            if ref not in produced:
                errors.append(
                    f"step[{i}] '{sid}' ({op}): arg '{field}' references "
                    f"'{ref}', which is not a prior step id"
                )
                continue
            got = produced[ref]
            if want is not None and got != want:
                errors.append(
                    f"step[{i}] '{sid}' ({op}): arg '{field}' expects kind "
                    f"'{want}' but '{ref}' produces '{got}'"
                )
            if want is None and got not in COLLECTION_KINDS:
                errors.append(
                    f"step[{i}] '{sid}' ({op}): arg '{field}' expects a collection "
                    f"but '{ref}' produces '{got}'"
                )

        # Resolve this step's output kind (filter/temporal_order pass input through).
        if op in PRODUCES:
            produced[sid] = PRODUCES[op]
        elif op == "filter":
            produced[sid] = produced.get(args.get("items"), "unknown")
        elif op == "temporal_order":
            produced[sid] = produced.get(args.get("events"), "unknown")
        else:
            produced[sid] = "unknown"

    if not program:
        errors.append("program is empty")
    elif program[-1].get("op") != "answer":
        errors.append("last step must be op 'answer' (the evidence-grounded verdict)")
    answers = [s for s in program if s.get("op") == "answer"]
    if len(answers) > 1:
        errors.append(f"exactly one 'answer' step allowed, found {len(answers)}")

    return errors


def main() -> int:
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema not installed. Run: pip install -r ../requirements.txt", file=sys.stderr)
        return 2

    prog_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PROGRAM
    if not SCHEMA_PATH.exists():
        print(f"ERROR: schema not found at {SCHEMA_PATH}", file=sys.stderr)
        return 2
    if not prog_path.exists():
        print(f"ERROR: program not found at {prog_path}", file=sys.stderr)
        return 2

    schema = json.loads(SCHEMA_PATH.read_text())
    raw = json.loads(prog_path.read_text())
    program_doc = strip_comments(raw)

    # 1) Structural validation against the DSL JSON Schema.
    struct = structural_errors(program_doc, schema)
    if struct:
        print(f"STRUCTURAL: FAIL ({len(struct)} error(s)) for {prog_path.name}")
        for m in struct:
            print(f"  - {m}")
        return 1
    print(f"STRUCTURAL: PASS — conforms to {SCHEMA_PATH.name}")

    # 2) Semantic named-binding validation (what the interpreter enforces).
    sem = semantic_errors(program_doc["program"])
    if sem:
        print(f"SEMANTIC: FAIL ({len(sem)} error(s))")
        for e in sem:
            print(f"  - {e}")
        return 1
    print("SEMANTIC: PASS — bindings resolve, kinds flow, answer is last")

    steps = program_doc["program"]
    chain = " -> ".join(s["id"] for s in steps)
    print(f"\nVALID: {prog_path.name} ({len(steps)} steps)\n  binding chain: {chain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
