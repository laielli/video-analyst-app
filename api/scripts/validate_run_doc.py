#!/usr/bin/env python3
"""
Validate a run-doc against run_doc.schema.json AND its embedded program against
dsl.schema.json — the backend<->UI contract check. Reuses the program validator so
the two schemas stay in lockstep (the run-doc embeds the program verbatim).

Usage:
    python validate_run_doc.py                      # validates examples/hero_run_doc.json
    python validate_run_doc.py path/to/run_doc.json

Exit 0 = valid, 1 = invalid, 2 = setup problem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from validate_program import strip_comments, semantic_errors, SCHEMA_PATH as DSL_SCHEMA_PATH  # noqa: E402

RUNDOC_SCHEMA_PATH = HERE.parent / "schema" / "run_doc.schema.json"
DEFAULT_RUNDOC = HERE.parent / "examples" / "hero_run_doc.json"


def main() -> int:
    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema not installed. Run: pip install -r ../requirements.txt", file=sys.stderr)
        return 2

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUNDOC
    for p in (RUNDOC_SCHEMA_PATH, DSL_SCHEMA_PATH, path):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    rundoc_schema = json.loads(RUNDOC_SCHEMA_PATH.read_text())
    dsl_schema = json.loads(DSL_SCHEMA_PATH.read_text())
    doc = strip_comments(json.loads(path.read_text()))

    # 1) Structural: run-doc against run_doc.schema.json
    errs = sorted(jsonschema.Draft202012Validator(rundoc_schema).iter_errors(doc), key=lambda e: list(e.path))
    if errs:
        print(f"RUN-DOC STRUCTURAL: FAIL ({len(errs)} error(s)) for {path.name}")
        for e in errs:
            loc = "/".join(str(p) for p in e.path) or "<root>"
            print(f"  - at {loc}: {e.message}")
        return 1
    print(f"RUN-DOC STRUCTURAL: PASS — conforms to {RUNDOC_SCHEMA_PATH.name}")

    # 2) Embedded program against the DSL schema (structure + named-binding semantics)
    prog_errs = sorted(jsonschema.Draft202012Validator(dsl_schema).iter_errors({"program": doc["program"]}),
                       key=lambda e: list(e.path))
    if prog_errs:
        print(f"EMBEDDED PROGRAM: FAIL ({len(prog_errs)} error(s))")
        for e in prog_errs:
            print(f"  - {e.message}")
        return 1
    sem = semantic_errors(doc["program"])
    if sem:
        print(f"EMBEDDED PROGRAM: FAIL ({len(sem)} semantic error(s))")
        for e in sem:
            print(f"  - {e}")
        return 1
    print("EMBEDDED PROGRAM: PASS — conforms to dsl.schema.json + bindings resolve")

    # 3) Cross-check: trace lines up with program, supporting_step is real
    cross: list[str] = []
    prog_ids = [s["id"] for s in doc["program"]]
    trace_ids = [t["id"] for t in doc["trace"]]
    if prog_ids != trace_ids:
        cross.append(f"trace ids {trace_ids} do not match program ids {prog_ids} (must be same order)")
    for t in doc["trace"]:
        if t["op"] != next((s["op"] for s in doc["program"] if s["id"] == t["id"]), t["op"]):
            cross.append(f"trace step '{t['id']}' op '{t['op']}' disagrees with program op")
    sup = doc["findings"].get("supporting_step")
    if sup and sup not in trace_ids:
        cross.append(f"findings.supporting_step '{sup}' is not a trace id")
    if cross:
        print(f"CROSS-CHECK: FAIL ({len(cross)} error(s))")
        for e in cross:
            print(f"  - {e}")
        return 1
    print("CROSS-CHECK: PASS — trace aligns with program, findings reference a real step")

    pinned = [t["id"] for t in doc["trace"] if t.get("source") == "pinned"]
    print(f"\nVALID: {path.name} — {len(doc['trace'])} steps, verdict: {doc['findings']['verdict']!r}")
    print(f"  provenance: {sum(1 for t in doc['trace'] if t['source']=='live')} live, "
          f"{sum(1 for t in doc['trace'] if t['source']=='cached')} cached, "
          f"{len(pinned)} pinned{f' ({pinned})' if pinned else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
