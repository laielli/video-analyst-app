#!/usr/bin/env python3
"""
Run a DSL program against a replay cache and emit a validated run-doc. Zero Azure calls.

    python run_program.py                                  # hero program + hero cache
    python run_program.py --program p.json --cache c.json --out run.json

Pipeline: validate program (dsl.schema + bindings) -> interpret over cache -> validate
the emitted run-doc (run_doc.schema). Exits non-zero if either validation fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
sys.path.insert(0, str(HERE))      # validate_program
sys.path.insert(0, str(API_DIR))   # interpreter package

from validate_program import strip_comments, semantic_errors, SCHEMA_PATH as DSL_SCHEMA  # noqa: E402
from interpreter import Cache, Interpreter  # noqa: E402

DEFAULT_PROGRAM = API_DIR / "examples" / "hero_program.json"
DEFAULT_CACHE = API_DIR / "examples" / "hero_cache.json"
RUNDOC_SCHEMA = API_DIR / "schema" / "run_doc.schema.json"

STATUS_MARK = {"done": "OK", "empty": "··", "error": "XX", "active": ">>", "pending": "  "}


def main() -> int:
    ap = argparse.ArgumentParser(description="Interpret a DSL program over a replay cache.")
    ap.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=None, help="write the run-doc JSON here")
    args = ap.parse_args()

    try:
        import jsonschema
    except ImportError:
        print("ERROR: jsonschema not installed. pip install -r ../requirements.txt", file=sys.stderr)
        return 2
    for p in (args.program, args.cache, DSL_SCHEMA, RUNDOC_SCHEMA):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    program = strip_comments(json.loads(args.program.read_text()))["program"]

    # 1) validate the input program
    dsl = json.loads(DSL_SCHEMA.read_text())
    errs = list(jsonschema.Draft202012Validator(dsl).iter_errors({"program": program}))
    sem = semantic_errors(program)
    if errs or sem:
        print(f"INVALID PROGRAM: {args.program.name}", file=sys.stderr)
        for e in errs:
            print(f"  - {e.message}", file=sys.stderr)
        for e in sem:
            print(f"  - {e}", file=sys.stderr)
        return 1

    # 2) interpret over the cache
    run_doc = Interpreter(Cache.load(args.cache)).run(program)

    # 3) report
    print(f"query:  {run_doc['query']}")
    print(f"clip:   {run_doc['clip']['id']} ({run_doc['clip']['width']}x{run_doc['clip']['height']})\n")
    for t in run_doc["trace"]:
        mark = STATUS_MARK.get(t["status"], "  ")
        n_ov = len(t.get("evidence", {}).get("overlays", []))
        print(f"  [{mark}] {t['id']:<8} {t['op']:<14} {t['source']:<7} -> {t.get('output_label',''):<22} "
              f"({n_ov} overlay{'s' if n_ov != 1 else ''})")
    f = run_doc["findings"]
    print(f"\nfindings: {f['verdict']}  (supporting: {f.get('supporting_step','-')})")

    # 4) validate the emitted run-doc against the contract
    rd_schema = json.loads(RUNDOC_SCHEMA.read_text())
    rd_errs = sorted(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc), key=lambda e: list(e.path))
    if rd_errs:
        print(f"\nRUN-DOC INVALID ({len(rd_errs)} error(s)):", file=sys.stderr)
        for e in rd_errs:
            loc = "/".join(str(p) for p in e.path) or "<root>"
            print(f"  - at {loc}: {e.message}", file=sys.stderr)
        return 1

    n = len(run_doc["trace"])
    prov = {k: sum(1 for t in run_doc["trace"] if t["source"] == k) for k in ("live", "cached", "pinned")}
    print(f"\nrun-doc valid (conforms to run_doc.schema.json) — {n} steps, "
          f"provenance: {prov['live']} live / {prov['cached']} cached / {prov['pinned']} pinned")
    if args.out:
        args.out.write_text(json.dumps(run_doc, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
