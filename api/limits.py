"""
Trust-boundary limit constants — the single source of truth for the resource + arg-domain
caps that bound a generated program before (and during) execution.

WHY a top-level module
----------------------
The interpreter package (`interpreter/`) and `codegen.py` both need these constants, and so
does `scripts/validate_program.py`. `interpreter/` MUST NOT import from `scripts/` — that path
is only on `sys.path` when `server.py` inserts it; the precompute and CLI entry points do their
own path dance, so an `interpreter -> scripts` import would break those entry points. A
top-level `limits.py` (always importable from `api/`) is the cycle-free home all three share.

WHAT these bound
----------------
- `MAX_PROGRAM_BYTES`  — raw codegen output size, checked BEFORE any json.loads / jsonschema
  walk (the validator's 8-way `anyOf` pass is superlinear on hostile input, so it must never
  see an unbounded blob). Enforced in `codegen.generate` (plus `max_tokens` on the API call)
  AND at the top of `validate_program.validation_errors`.
- `MAX_STEPS`          — whole-program step count. Checked in `semantic_errors`, mirrored in
  the DSL schema (`program.maxItems`), and re-checked at the top of `Interpreter.run`.
- `MAX_SAMPLED_FRAMES` — SUM of frames across all `sample_frames` ops (the OOM vector). The
  validator sums the would-be `range(start, end+1, stride)` lengths with the SAME stride math
  as `op_sample_frames`; the interpreter caps each call's length BEFORE `list(range(...))`.
- `MAX_DETECT_CLASSES` — `detect.classes` array length.
- `MAX_STR_ARG_LEN`    — `detect.classes[]` elements, `filter.where.field`/`equals` strings.

Sizing rationale (real envelopes): hero window = 13 frames; bernabeu 9000-11000ms@fps8 = 17;
whole bernabeu clip (30086ms) @ fps30 = 912; @ fps8 = 241. The defaults sit comfortably above
the largest legitimate full-clip sweep while still rejecting the bombs (end=2e9 @ fps30 ~ 60M
frames). The hero chain is 7 steps; `detect.classes` is ["person"] in practice. The values are
one diff to tune and one import for tests; the schema's static mirrors are kept in sync by
`test_limit_constants_match_schema`.
"""
from __future__ import annotations

MAX_PROGRAM_BYTES = 65_536  # raw codegen output, checked BEFORE json/schema walk
MAX_STEPS = 32              # whole program; hero chain is 7
MAX_SAMPLED_FRAMES = 2000   # SUM of frames across all sample_frames ops
MAX_DETECT_CLASSES = 16     # detect.classes length
MAX_STR_ARG_LEN = 256       # filter.where.field/equals, classes[] elements


class ProgramLimitExceeded(ValueError):
    """A program exceeded a runtime resource cap (step count or per-call sampled-frame count).

    Lives here, beside the constants, so the interpreter (`Interpreter.run`) and the primitive
    (`op_sample_frames`) both raise the SAME type without an import cycle. Subclasses `ValueError`
    so the server's existing `except Exception` around `Interpreter.run` already catches it and
    maps it to the fixed `execution-error` reason — the exception text is NEVER interpolated into
    a client-visible doc (info-leak guard)."""

