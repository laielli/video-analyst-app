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

# Trust-boundary limit constants live in the top-level `api/limits.py` (NOT here, NOT in
# interpreter/) so validate_program, the interpreter, and codegen all import the SAME numbers
# with zero cycle risk. `api/` is on sys.path for every entry point that reaches this module
# (server inserts it; the CLI/precompute run from api/). Imported lazily-tolerant: prefer the
# package-relative path, fall back to a bare top-level import for the `scripts/`-on-path callers.
try:  # pragma: no cover — exercised by whichever path the active entry point uses
    from limits import (  # noqa: E402
        MAX_DETECT_CLASSES,
        MAX_PROGRAM_BYTES,
        MAX_SAMPLED_FRAMES,
        MAX_STEPS,
        MAX_STR_ARG_LEN,
    )
except ImportError:  # pragma: no cover — api/ not yet on sys.path; add it and retry
    sys.path.insert(0, str(HERE.parent))
    from limits import (  # noqa: E402
        MAX_DETECT_CLASSES,
        MAX_PROGRAM_BYTES,
        MAX_SAMPLED_FRAMES,
        MAX_STEPS,
        MAX_STR_ARG_LEN,
    )

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
# (binding arg field -> required input kind: a literal kind, None for "any collection",
# or "any" for any synthesizable kind — used by `answer`, which op_answer now generalizes
# over collections + `number` + `ordered` (Phase 0). Without "any", a `count -> answer`
# program would not validate because `number` is not a collection.)
EXPECTS = {
    "detect": {"frames": "frames"},
    "crop": {"detections": "detections"},
    "read_text": {"crops": "crops"},
    "filter": {"items": None},
    "count": {"items": None},
    "temporal_order": {"events": None},
    "answer": {"from": "any"},
}
COLLECTION_KINDS = {"frames", "detections", "crops", "texts"}
# Kinds op_answer can synthesize a deterministic answer from (Phase 0). "unknown" is excluded
# so a malformed binding chain still surfaces, but the validator stays permissive for the four
# legitimate question shapes (temporal/count/text/existence).
ANSWERABLE_KINDS = COLLECTION_KINDS | {"number", "ordered"}

# Numeric arg-value bounds the JSON Schema documents as prose but cannot enforce. A *validated*
# program is the trust boundary, so the gate must bound these before Interpreter.run: an
# unbounded sample_frames window OOMs via range(start, end+1, stride) (primitives.py). fps is
# also bounded so the stride stays sane. clip.duration_ms is the per-clip upper bound.
FPS_MIN, FPS_MAX = 1, 30


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


def _oversize_error(program_doc: dict) -> list[str]:
    """A raw-size cap checked BEFORE any json/schema walk. The jsonschema pass is an 8-way
    `anyOf` per item — superlinear on hostile input — so a multi-megabyte blob must be rejected
    here, before `structural_errors` ever serializes or walks it. We measure the doc's serialized
    byte length (the same shape codegen would have emitted); an unserializable doc is also a
    reject (a validated program is always JSON-serializable). Returns a one-element error list
    when over the cap (or unserializable), else []."""
    try:
        size = len(json.dumps(program_doc).encode("utf-8"))
    except (TypeError, ValueError):
        return ["oversize: program is not serializable"]
    if size > MAX_PROGRAM_BYTES:
        return [f"oversize: program is {size} bytes; max is {MAX_PROGRAM_BYTES}"]
    return []


def validation_errors(program_doc: dict, schema: dict | None = None, clip: dict | None = None) -> list[str]:
    """The full valid-by-construction gate (size, then structural, then semantic) as one flat list
    — empty means valid. Shared by this CLI and the codegen fallback/free-text paths (server).
    Comments are stripped here so callers can pass raw codegen/authoring output. The raw-size cap
    runs FIRST, before json/schema walk, so the superlinear jsonschema pass never sees an unbounded
    blob (DoS guard). Semantic checks are skipped when structure fails (malformed refs make them
    meaningless). `clip` (with `duration_ms`) bounds numeric args so a *validated* program can't
    OOM the interpreter; the cumulative-frame cap bounds the OOM even with no clip."""
    oversize = _oversize_error(program_doc)
    if oversize:
        return oversize
    if schema is None:
        schema = json.loads(SCHEMA_PATH.read_text())
    doc = strip_comments(program_doc)
    struct = structural_errors(doc, schema)
    if struct:
        return [f"structural: {m}" for m in struct]
    return [f"semantic: {m}" for m in semantic_errors(doc["program"], clip=clip)]


def _sample_frames_bounds(i: int, sid: str, args: dict, clip: dict | None) -> list[str]:
    """Enforce sample_frames numeric bounds (trust boundary): 0 <= start < end <= duration_ms
    and 1 <= fps <= 30. The upper time bound uses clip.duration_ms when a clip is supplied; with
    no clip it still rejects start>=end and out-of-range fps (the OOM-relevant invariants)."""
    out: list[str] = []
    start, end, fps = args.get("start_ms"), args.get("end_ms"), args.get("fps")
    # structural validation guarantees these are ints when present; guard anyway.
    if not all(isinstance(v, int) for v in (start, end, fps)):
        return out
    if start < 0:
        out.append(f"step[{i}] '{sid}' (sample_frames): start_ms {start} < 0")
    if end <= start:
        out.append(f"step[{i}] '{sid}' (sample_frames): end_ms {end} must be > start_ms {start}")
    if clip is not None:
        dur = clip.get("duration_ms")
        if isinstance(dur, int) and end > dur:
            out.append(f"step[{i}] '{sid}' (sample_frames): end_ms {end} > clip duration_ms {dur}")
    if not (FPS_MIN <= fps <= FPS_MAX):
        out.append(f"step[{i}] '{sid}' (sample_frames): fps {fps} out of [{FPS_MIN}, {FPS_MAX}]")
    return out


def _sampled_frame_count(args: dict, clip: dict | None) -> int:
    """How many frames a `sample_frames` op WOULD materialize — byte-identical to op_sample_frames'
    `len(range(start, end+1, stride))` with `stride = max(1, round(1000/fps))`. This is O(1)
    arithmetic (Python's `len(range(...))` does not build the list), so the validator can sum the
    cumulative frame budget across all sample_frames ops without ever materializing one. The count
    is duration-INDEPENDENT (it bounds the OOM even when `clip=None`), so it closes the no-clip gap
    that `_sample_frames_bounds`' `end <= duration_ms` check can't. Returns 0 for malformed/missing
    args (structural validation surfaces those separately)."""
    start, end, fps = args.get("start_ms"), args.get("end_ms"), args.get("fps")
    if not all(isinstance(v, int) for v in (start, end, fps)):
        return 0
    if fps <= 0 or end < start:
        return 0  # out-of-domain args are rejected by _sample_frames_bounds; don't double-count
    stride = max(1, round(1000 / fps))
    return len(range(start, end + 1, stride))


def _arg_domain_errors(i: int, sid: str, op: str, args: dict) -> list[str]:
    """Per-op arg-domain checks the JSON Schema can't fully express under strict:false (the schema
    is advisory to codegen; THIS is the real gate). Bounds the abuse/OOM-relevant args:
    `detect.classes` (non-empty, length-capped, string elements ≤ MAX_STR_ARG_LEN) and the
    `filter.where` string args (length-capped). Semantically-harmless free strings (the `equals`
    VALUE semantics, `answer.question`) are left unconstrained beyond length — the default
    'defensive depth' that closes resource vectors without overfitting to today's question
    shapes. Structural validation already guarantees the rough types; these guard against the
    cases the schema's advisory limits don't stop under strict:false."""
    out: list[str] = []
    if op == "detect":
        cls = args.get("classes")
        if isinstance(cls, list):
            if not cls:
                out.append(f"step[{i}] '{sid}' (detect): classes is empty")
            if len(cls) > MAX_DETECT_CLASSES:
                out.append(
                    f"step[{i}] '{sid}' (detect): classes has {len(cls)} entries; "
                    f"max is {MAX_DETECT_CLASSES}"
                )
            for c in cls:
                if not isinstance(c, str):
                    out.append(f"step[{i}] '{sid}' (detect): classes element is not a string")
                elif len(c) > MAX_STR_ARG_LEN:
                    out.append(
                        f"step[{i}] '{sid}' (detect): a classes element is {len(c)} chars; "
                        f"max is {MAX_STR_ARG_LEN}"
                    )
    elif op == "filter":
        where = args.get("where")
        if isinstance(where, dict):
            for f in ("field", "equals"):
                v = where.get(f)
                if isinstance(v, str) and len(v) > MAX_STR_ARG_LEN:
                    out.append(
                        f"step[{i}] '{sid}' (filter): where.{f} is {len(v)} chars; "
                        f"max is {MAX_STR_ARG_LEN}"
                    )
    return out


def semantic_errors(program: list[dict], clip: dict | None = None) -> list[str]:
    """The checks the JSON Schema cannot fully enforce under strict:false: ref existence/order,
    kind flow, unique ids, answer-is-last, numeric arg bounds (trust boundary), AND the
    hardening caps — max program steps, max CUMULATIVE sampled frames across all sample_frames
    ops, and per-op arg-domain bounds (detect.classes, filter.where strings). The cumulative
    frame cap is the OOM-relevant invariant: it is duration-independent, so it rejects a frame
    bomb even when no clip is supplied (the gap `_sample_frames_bounds` can't close)."""
    errors: list[str] = []
    produced: dict[str, str] = {}  # id -> output kind, in declaration order
    total_frames = 0  # cumulative across all sample_frames ops (the OOM budget)

    # Whole-program step cap (the schema mirrors this as program.maxItems; re-checked here because
    # the schema is advisory under strict:false — the local validator is the real gate).
    if len(program) > MAX_STEPS:
        errors.append(f"program has {len(program)} steps; max is {MAX_STEPS}")

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
            if want == "any":
                # answer.from: accept any kind op_answer can synthesize (collections + number +
                # ordered). Reject only an unresolved/unknown binding chain.
                if got not in ANSWERABLE_KINDS:
                    errors.append(
                        f"step[{i}] '{sid}' ({op}): arg '{field}' expects an answerable kind "
                        f"(collection | number | ordered) but '{ref}' produces '{got}'"
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

        # Numeric arg bounds (trust boundary): an unbounded sample_frames window OOMs the
        # interpreter, so enforce 0 <= start < end <= duration_ms and 1 <= fps <= 30.
        if op == "sample_frames":
            errors.extend(_sample_frames_bounds(i, sid, args, clip))
            total_frames += _sampled_frame_count(args, clip)

        # Per-op arg-domain bounds (detect.classes, filter.where strings) — see _arg_domain_errors.
        errors.extend(_arg_domain_errors(i, sid, op, args))

        # Resolve this step's output kind (filter/temporal_order pass input through).
        if op in PRODUCES:
            produced[sid] = PRODUCES[op]
        elif op == "filter":
            produced[sid] = produced.get(args.get("items"), "unknown")
        elif op == "temporal_order":
            produced[sid] = produced.get(args.get("events"), "unknown")
        else:
            produced[sid] = "unknown"

    # Cumulative frame cap (the OOM budget): the SUM of would-be-materialized frames across all
    # sample_frames ops must stay under MAX_SAMPLED_FRAMES. Mirrors op_sample_frames' stride math
    # exactly (via _sampled_frame_count) so the validator and the runtime cap can never disagree.
    if total_frames > MAX_SAMPLED_FRAMES:
        errors.append(f"program samples {total_frames} frames; max is {MAX_SAMPLED_FRAMES}")

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
