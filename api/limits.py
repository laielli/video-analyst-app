"""
Trust-boundary limit constants — the single source of truth for the resource + semantic
guards on LLM-generated programs (the ViperGPT layer-2 trust boundary; see CLAUDE.md).

This is a NEW top-level module (not under scripts/) on purpose: it is imported by
`scripts/validate_program.py` (the validation gate), `interpreter/primitives.py` +
`interpreter/interpreter.py` (the runtime guard), and `codegen.py` (the codegen-output
byte cap). The interpreter package must NOT import from scripts/ — that path is only on
sys.path when server.py inserted it, while the precompute/CLI entry points do their own
path dance. A top-level limits.py is importable from all of them with zero cycle risk.

Where each constant is enforced (defense in depth — the validator is the PRIMARY gate
under strict:false; the runtime guard is the backstop for any path that bypasses it):

  MAX_PROGRAM_BYTES   codegen.generate (max_tokens + a byte check on the returned blob)
                      AND validation_errors (rejects an oversized doc BEFORE json/schema).
  MAX_STEPS           dsl.schema.json program.maxItems, semantic_errors, AND
                      Interpreter.run (top-of-loop guard).
  MAX_SAMPLED_FRAMES  semantic_errors (cumulative, sum across all sample_frames ops) AND
                      op_sample_frames (per-call pre-materialization cap — the load-bearing
                      check that stops the list(range(...)) OOM BEFORE it materializes).
  MAX_DETECT_CLASSES  dsl.schema.json detect.classes.maxItems AND semantic_errors.
  MAX_STR_ARG_LEN     dsl.schema.json maxLength on string args AND semantic_errors.

These mirror values in dsl.schema.json; test_program_limits.test_limit_constants_match_schema
asserts they never drift.
"""
from __future__ import annotations


class ProgramLimitExceeded(ValueError):
    """A program exceeded a resource limit at RUNTIME (the interpreter backstop). Subclasses
    ValueError so the existing `except Exception` reject paths (server.py) catch it and map it
    to a FIXED reason enum — the message must NEVER be interpolated into a client-visible doc.
    Lives here (not interpreter.py) so primitives.py and interpreter.py both raise the same
    type without an import cycle, and the constants travel with the exception."""


# Raw codegen output cap, checked BEFORE any json.loads / jsonschema walk. The schema pass is
# superlinear on hostile input (an 8-way anyOf per program item), so the validator must never
# see an unbounded blob; codegen.generate also caps the completion via max_tokens.
MAX_PROGRAM_BYTES = 65_536

# Whole-program step ceiling. The hero chain is 7 steps; the largest committed program is 7.
MAX_STEPS = 32

# Cumulative ceiling on frames a program may sample (SUM across every sample_frames op), and
# the per-call pre-materialization cap inside op_sample_frames. Real envelopes: hero window =
# 13 frames, bernabeu 9000-11000ms@fps8 = 17 frames, whole bernabeu clip (30086ms)@fps30 = 912
# frames. 2000 leaves headroom over the largest legitimate full-clip sweep while still rejecting
# the OOM bombs (end=2e9, fps30 => ~60M frames).
MAX_SAMPLED_FRAMES = 2_000

# detect.classes length ceiling. Every committed program uses ["person"] (length 1).
MAX_DETECT_CLASSES = 16

# Per-element string-arg length ceiling: detect.classes[] elements, filter.where.field/equals.
MAX_STR_ARG_LEN = 256
