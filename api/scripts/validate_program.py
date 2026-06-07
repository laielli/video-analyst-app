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
# `answer.from` uses the sentinel ANY_SYNTHESIZABLE: the generalized op_answer (Phase 0)
# synthesizes a verdict from a count (number), texts, detections/crops/frames, OR an ordered
# binding — so it accepts more than just collections. Don't pin it to one kind.
ANY_SYNTHESIZABLE = "<answerable>"
EXPECTS = {
    "detect": {"frames": "frames"},
    "crop": {"detections": "detections"},
    "read_text": {"crops": "crops"},
    "filter": {"items": None},
    "count": {"items": None},
    "temporal_order": {"events": None},
    "answer": {"from": ANY_SYNTHESIZABLE},
}
COLLECTION_KINDS = {"frames", "detections", "crops", "texts"}
# Kinds the generalized answer step can synthesize a verdict from (Phase 0). `number` is the
# count read-out; the collection kinds cover existence/text; `unknown` is the walk-time result
# of temporal_order/filter passthrough (ordered events), which answer also handles.
ANSWERABLE_KINDS = COLLECTION_KINDS | {"number", "ordered", "unknown"}


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


# Numeric arg ranges the JSON Schema documents only as prose (dsl.schema.json). The interpreter
# materializes frames via range(start, end+1, stride) (primitives.op_sample_frames), so an
# out-of-range window can OOM even for a structurally-valid program. This is the trust boundary:
# a validated free-text program MUST be bounded before Interpreter.run (plan §Risks CORRECTION).
FPS_MIN, FPS_MAX = 1, 30


def numeric_bounds_errors(program: list[dict], clip: dict | None = None) -> list[str]:
    """Bound sample_frames numeric args so a validated program can't OOM the interpreter:
    0 <= start < end <= clip.duration_ms (if a clip is given), and FPS_MIN <= fps <= FPS_MAX."""
    errors: list[str] = []
    duration = clip.get("duration_ms") if clip else None
    for i, step in enumerate(program):
        if step.get("op") != "sample_frames":
            continue
        a = step.get("args", {})
        sid = step.get("id")
        start, end, fps = a.get("start_ms"), a.get("end_ms"), a.get("fps")
        if not all(isinstance(v, int) for v in (start, end, fps)):
            continue  # structural validation already flags non-integers
        if start < 0:
            errors.append(f"step[{i}] '{sid}' (sample_frames): start_ms {start} < 0")
        if end <= start:
            errors.append(f"step[{i}] '{sid}' (sample_frames): end_ms {end} must be > start_ms {start}")
        if duration is not None and end > duration:
            errors.append(f"step[{i}] '{sid}' (sample_frames): end_ms {end} > clip duration_ms {duration}")
        if not (FPS_MIN <= fps <= FPS_MAX):
            errors.append(f"step[{i}] '{sid}' (sample_frames): fps {fps} outside [{FPS_MIN}, {FPS_MAX}]")
    return errors


def validation_errors(program_doc: dict, schema: dict | None = None, clip: dict | None = None) -> list[str]:
    """The full valid-by-construction gate (structural -> semantic -> numeric bounds) as one flat
    list — empty means valid. Shared by this CLI and the free-text/codegen path. Comments are
    stripped here so callers can pass raw codegen/authoring output. Semantic + numeric checks are
    skipped when structure fails (malformed refs/args make them meaningless). `clip` (optional)
    enables duration-bounding of sample_frames — the OOM trust boundary for free text."""
    if schema is None:
        schema = json.loads(SCHEMA_PATH.read_text())
    doc = strip_comments(program_doc)
    struct = structural_errors(doc, schema)
    if struct:
        return [f"structural: {m}" for m in struct]
    sem = [f"semantic: {m}" for m in semantic_errors(doc["program"])]
    if sem:
        return sem
    return [f"numeric: {m}" for m in numeric_bounds_errors(doc["program"], clip)]


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
            if want == ANY_SYNTHESIZABLE:
                if got not in ANSWERABLE_KINDS:
                    errors.append(
                        f"step[{i}] '{sid}' ({op}): arg '{field}' expects an answerable kind "
                        f"(collection / number / ordered) but '{ref}' produces '{got}'"
                    )
            elif want is not None and got != want:
                errors.append(
                    f"step[{i}] '{sid}' ({op}): arg '{field}' expects kind "
                    f"'{want}' but '{ref}' produces '{got}'"
                )
            elif want is None and got not in COLLECTION_KINDS:
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
