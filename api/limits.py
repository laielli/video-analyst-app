"""
Trust-boundary resource + arg-domain limits for the generated-program executor.

This is a NEW top-level module (deliberately not under scripts/) so the same constants are
importable, with zero cycle risk and no sys.path dependency, by every layer that enforces
them:
  - validate_program.py  — the reject-before-execute gate (the real boundary under strict:false)
  - interpreter/primitives.py + interpreter/interpreter.py — the runtime guard (two checks)
  - codegen.py            — caps the raw LLM output before it ever reaches the validator

The interpreter package cannot safely import from scripts/ (that path is only on sys.path when
server.py inserted it; the precompute/CLI entry points do their own path dance), so a top-level
limits.py is the single source of truth.

The values are also mirrored, where statically expressible, into schema/dsl.schema.json so codegen
is steered at the source and the CLI/canned loaders inherit the static caps. That duplication is
guarded by test_limit_constants_match_schema, which asserts the schema's numbers equal these — so
they cannot silently drift. The local validator (semantic_errors) remains the AUTHORITATIVE gate;
the schema is advisory under the codegen call's strict:false.

Rationale for the chosen values (DESIGN-ROOM default from the finalized plan):
  - Real programs observed: hero = 7 steps / 13 frames; bernabeu first-goal = 7 steps / 17 frames;
    the largest legitimate full-clip sweep is the whole bernabeu clip (30086ms) @ fps30 ≈ 912
    frames. The defaults clear all of these with comfortable headroom while still rejecting the
    OOM bombs (e.g. end=2e9 @ fps30 ⇒ ~60M frames).
"""
from __future__ import annotations

# Raw codegen output, checked BEFORE any json.loads / jsonschema walk. The jsonschema pass is
# superlinear on hostile input (an 8-way anyOf per program item), so the validator must never
# see an unbounded blob; codegen.generate also caps the completion (max_tokens) and re-checks the
# returned content's byte length.
MAX_PROGRAM_BYTES = 65_536

# Whole-program step cap. The hero chain is 7 steps; 32 leaves wide headroom while bounding the
# tree-walk and the worst-case replay-cache lookups (steps x per-call frames).
MAX_STEPS = 32

# SUM of frames across ALL sample_frames ops in one program (the cumulative materialization the
# interpreter would build). 2000 sits above the largest legitimate full-clip sweep (~912 frames)
# and rejects the unbounded-window bombs.
MAX_SAMPLED_FRAMES = 2_000

# detect.classes length cap. Real programs use 1 ("person") or 2 classes.
MAX_DETECT_CLASSES = 16

# Length cap for free-string args (filter.where.field/equals, detect.classes[] elements). Closes
# the giant-string abuse vector independent of MAX_PROGRAM_BYTES; harmless free strings (e.g.
# answer.question) are length-bounded only, never whitelisted (the default defensive depth).
MAX_STR_ARG_LEN = 256


class ProgramLimitExceeded(ValueError):
    """Raised by the interpreter's runtime guard when a (somehow-unvalidated) program would
    exceed a resource limit: the pre-materialization frame cap inside op_sample_frames, or the
    MAX_STEPS check at the top of Interpreter.run. Subclasses ValueError so existing
    `except Exception` paths (server's free-text/canned handlers) catch it and map it to a fixed
    reason enum — the message text must NEVER be interpolated into a client-visible doc.

    Lives in limits.py (not interpreter.py) so primitives.py can import it without forming an
    interpreter <-> primitives import cycle.
    """

