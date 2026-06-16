"""
Trust-boundary resource + arg-domain limits for the generated-program executor.

The free-text path runs LLM-generated programs through the tree-walking interpreter, and
CLAUDE.md names the execution layer as the trust boundary: "generated code runs here, so
isolate and validate it." These constants are the single source of truth for the caps the
validator (scripts/validate_program.py), the runtime guards (interpreter/), and codegen.py
all enforce. They live in a TOP-LEVEL module (not scripts/) so the interpreter package can
import them with zero cycle risk and without depending on scripts/ being on sys.path — the
precompute and CLI entry points do their own path dance and must not break.

The matching static limits in schema/dsl.schema.json are kept in sync by
tests/test_program_limits.py::test_limit_constants_match_schema so they can never drift.

Sizing rationale (observed legitimate envelopes vs. the bombs they reject):
  - Real programs: hero = 7 steps / 13 frames; bernabeu first-goal = 7 steps / 17 frames.
  - Largest legitimate full-clip sweep: bernabeu (30086ms) @ fps30 ~= 912 frames.
  - The bombs: end=2e9 @ fps30 ~= 60M frames (the documented OOM via list(range(...))).
  - describe_scene: MAX_SCENE_CAPTIONS (16) caption-count cap; MAX_CAPTION_LEN (256) per-caption
    free-text cap (model-generated captions are truncated at the binding boundary).
The defaults sit comfortably above the legitimate full-clip sweep while rejecting the bombs.
"""
from __future__ import annotations

# Raw codegen output cap, checked BEFORE any json.loads / schema walk. The jsonschema pass is
# superlinear on hostile input (an 8-way anyOf per program item), so the validator itself is a
# DoS vector on an unbounded blob — it must never see one. Enforced in codegen.generate
# (max_tokens on the API call + a byte check on the returned content) AND at the top of
# validation_errors before any structural pass.
MAX_PROGRAM_BYTES = 65_536

# Whole-program step cap (the hero/bernabeu chains are 7 steps). Enforced statically in the
# schema (program.maxItems), in semantic_errors, and as a runtime guard at the top of
# Interpreter.run (the residual gate for any future caller that bypasses the validator).
MAX_STEPS = 32

# Cumulative cap on frames materialized across ALL sample_frames ops in a program. This is the
# OOM-relevant invariant: it is bounded by frame COUNT regardless of clip duration, so it closes
# the no-clip gap (validate_program previously skipped the duration bound when clip=None). Also
# enforced per-call (pre-materialization) inside op_sample_frames as the load-bearing runtime cap.
MAX_SAMPLED_FRAMES = 2000

# detect.classes length cap (a class list of thousands is abuse, not analysis).
MAX_DETECT_CLASSES = 16

# describe_scene caption-count cap: thousands of per-frame captions is abuse (mirrors
# MAX_DETECT_CLASSES). Caps the captions a single describe_scene binding may carry.
MAX_SCENE_CAPTIONS = 16

# Length cap for free-string args (detect.classes[] elements, filter.where.field/equals,
# answer.question; crop.region/temporal_order.by enums are already constrained structurally).
MAX_STR_ARG_LEN = 256

# Per-caption text-length cap: captions are model-generated free text; bound the STRING that
# enters a binding/answer/run-doc (run_doc.schema.json has no maxLength). Align with
# MAX_STR_ARG_LEN — the interpreter truncates each caption to this length at the boundary.
MAX_CAPTION_LEN = 256

# Static numeric bound on the sample_frames window endpoints (mirrored into the schema as
# start_ms/end_ms maximum). Far above any real clip (~30s) but finite, so a 2e9 end_ms is
# rejected structurally before the count math even runs. Kept in sync with the schema.
MAX_TIME_MS = 86_400_000  # 24h in ms — a generous absolute ceiling, never a real clip length


class ProgramLimitExceeded(ValueError):
    """A (pre-)validated program would exceed a resource cap at runtime. Subclasses ValueError
    so the interpreter's existing ``except Exception`` mapping at server.py turns it into the
    fixed ``execution-error`` reason (the message is NEVER interpolated into the client doc —
    info-leak guard). Raised by op_sample_frames (pre-materialization frame cap) and at the top
    of Interpreter.run (MAX_STEPS) — the two residual runtime guards for any caller that reaches
    the interpreter without going through validation_errors first."""


def sampled_frame_count(start_ms: int, end_ms: int, fps: int) -> int:
    """The number of frames op_sample_frames would materialize for this window, computed WITHOUT
    building the list (len(range(...)) is O(1)). MUST mirror op_sample_frames' stride math exactly
    (interpreter/primitives.py: ``stride = max(1, round(1000/fps)); range(start, end+1, stride)``)
    so the validator's cumulative count and the runtime cap agree byte-for-byte. Returns 0 for a
    degenerate (end < start) window; callers handle ordering separately."""
    stride = max(1, round(1000 / fps))
    return len(range(start_ms, end_ms + 1, stride))
