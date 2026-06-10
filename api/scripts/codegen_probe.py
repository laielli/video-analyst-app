#!/usr/bin/env python3
"""
Codegen probe — prove Azure OpenAI compiles the canned question into a VALID DSL program.

This is the server-independent gate for ViperGPT Layer 1 (program generation), the sibling of
vlm_probe.py for vision. It calls Azure OpenAI with the DSL schema as a strict structured-output
constraint, then runs the same validate_program gate the server uses.

For systematic measurement across a question bank (phrasing variants x shapes x clips, plus
adversarial/out-of-scope cases), with tiered scoring and a CI regression gate, see `api/eval/`
(the codegen eval harness). This probe stays the fast one-question gut-check.

    python scripts/codegen_probe.py                                  # default hero question
    python scripts/codegen_probe.py --question "Does #10 score the first goal?"
    python scripts/codegen_probe.py --run                            # also replay it over the hero cache

Exit 0 = a structurally + semantically valid program was generated (and, with --run, it executed
to a grounded answer). 1 = generated but invalid / didn't ground. 2 = setup problem (no creds).

Prereqs: AZURE_OPENAI_* in api/.env (see .env.example). With no creds the server still works —
it falls back to the pinned program (D-DR6); this probe just reports the missing setup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
sys.path.insert(0, str(HERE))      # validate_program, ocr_probe (shared load_dotenv)
sys.path.insert(0, str(API_DIR))   # codegen, interpreter

from ocr_probe import load_dotenv  # noqa: E402

DEFAULT_QUESTION = "Does #10 score the first goal?"
HERO_CACHE = API_DIR / "examples" / "hero_cache.json"

OPENAI_HINT = """\
Azure OpenAI is not configured — set AZURE_OPENAI_* in api/.env (see .env.example).
Provision a gpt-4o deployment (reuses the resource group from the Vision step):

  az cognitiveservices account create -n va-openai -g video-analyst-rg -l eastus \\
    --kind OpenAI --sku S0 --yes
  az cognitiveservices account deployment create -g video-analyst-rg -n va-openai \\
    --deployment-name gpt-4o --model-name gpt-4o --model-version 2024-11-20 \\
    --model-format OpenAI --sku-name GlobalStandard --sku-capacity 10

Without creds the app still runs: the server falls back to the pinned program (D-DR6)."""


def main() -> int:
    ap = argparse.ArgumentParser(description="Azure OpenAI question -> DSL program probe")
    ap.add_argument("--question", default=DEFAULT_QUESTION, help="the canned question to compile")
    ap.add_argument("--run", action="store_true", help="also replay the program over the hero cache")
    args = ap.parse_args()

    load_dotenv(API_DIR / ".env")

    import codegen  # after dotenv so enabled() sees the loaded vars
    from validate_program import validation_errors

    if not codegen.enabled():
        print(OPENAI_HINT, file=sys.stderr)
        return 2

    try:
        program = codegen.AzureCodegen.from_env().generate(args.question)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR calling Azure OpenAI codegen: {e}", file=sys.stderr)
        return 2

    chain = " -> ".join(s.get("id", "?") for s in program)
    print(f"# question: {args.question!r}")
    print(f"generated program ({len(program)} steps): {chain}\n")
    print(json.dumps({"program": program}, indent=2))

    errors = validation_errors({"program": program})
    if errors:
        print(f"\nCODEGEN FAIL — {len(errors)} validation error(s) (server would fall back to pinned):")
        for m in errors:
            print(f"  - {m}")
        return 1
    print("\nCODEGEN PASS — program is structurally + semantically valid (valid-by-construction).")

    if args.run:
        from interpreter import Cache, Interpreter

        doc = Interpreter(Cache.load(HERO_CACHE)).run(program)
        f = doc["findings"]
        grounded = not f.get("partial") and f.get("answer") not in (None, "", "Unknown")
        print(f"\nreplay over hero cache -> answer={f.get('answer')!r} verdict={f.get('verdict')!r}")
        if grounded:
            print("RUN PASS — live program executed to a grounded answer over the cache.")
            return 0
        print("RUN PARTIAL — program is valid but didn't ground over the cache (server would use pinned).")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
