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
  - describe_scene: <= MAX_SCENE_CAPTIONS whole-image captions; each caption's model free-text
    is truncated to MAX_CAPTION_LEN before it enters a binding/answer/run-doc.
The defaults sit comfortably above the legitimate full-clip sweep while rejecting the bombs.
"""
from __future__ import annotations

import re

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
# MAX_DETECT_CLASSES). Mirrored into the schema (describe_scene.args.max_captions.maximum), kept
# in sync by test_limit_constants_match_schema.
MAX_SCENE_CAPTIONS = 16

# Length cap for free-string args (detect.classes[] elements, filter.where.field/equals,
# answer.question; crop.region/temporal_order.by enums are already constrained structurally).
MAX_STR_ARG_LEN = 256

# Invisible / bidirectional-control code points banned from EVERY free-string arg in a program
# (detect.classes[] elements, filter.where.field/equals, answer.question): C0+C1 controls,
# zero-width & directional marks (U+200B-U+200F), bidi embeddings/overrides (U+202A-U+202E),
# invisible operators + bidi isolates (U+2060-U+2069), and the BOM/ZWNBSP (U+FEFF). None of
# these can appear in a legitimate question about a clip, but all are classic smuggling /
# display-spoofing vectors — an answer.question carrying an RTL override would RENDER in the
# web UI differently than it executes, exactly the lie a glass-box demo cannot allow. Programs
# carrying them are rejected at the validator, so a noise-salted query grounds out honestly
# (noise-precedence) instead of being answered as if it were clean. Plane-14 Tags + Variation
# Selectors Supplement (U+E0000-U+E01EF) are the "ASCII smuggling" vector — invisible copies of
# the ASCII range — and are banned wholesale; the BMP variation selectors (U+FE00-U+FE0F) are
# deliberately NOT banned (legitimate emoji text, e.g. a soccer-ball emoji, carries U+FE0F).
BANNED_STR_CHARS = re.compile(
    "[\\x00-\\x1f\\x7f-\\x9f\\u200b-\\u200f\\u202a-\\u202e\\u2060-\\u2069\\ufeff"
    "\\U000E0000-\\U000E01EF]"
)


def has_banned_str_chars(s: str) -> bool:
    """True when a free-string program arg carries an invisible/bidi-control code point."""
    return bool(BANNED_STR_CHARS.search(s))

# Per-caption text-length cap: captions are model-generated free text; bound the STRING that
# enters a binding/answer/run-doc BEFORE it does (run_doc.schema.json has no maxLength on labels).
# Aligned with MAX_STR_ARG_LEN. Enforced in op_describe_scene (truncation at the boundary) and
# asserted by validate_cache_shape on committed caption slices.
MAX_CAPTION_LEN = 256

# Static numeric bound on the sample_frames window endpoints (mirrored into the schema as
# start_ms/end_ms maximum). Far above any real clip (~30s) but finite, so a 2e9 end_ms is
# rejected structurally before the count math even runs. Kept in sync with the schema.
MAX_TIME_MS = 86_400_000  # 24h in ms — a generous absolute ceiling, never a real clip length

# detect.on_pitch filter threshold (normalized 0-1 frame coords): a detection is kept only when
# its box BOTTOM edge (y+h) >= this value. Broadcast-framing heuristic: in these wide broadcast
# angles the pitch horizon sits roughly mid-frame, so a person actually standing/playing ON the
# pitch has their box bottom in the lower portion of the frame, while spectators/cameramen/staff
# in the stands (above the pitch horizon) have box bottoms well above it. Using the bottom edge
# (not the box center) also keeps players whose box is cropped at the top of the frame (tall/near
# camera). Verified against the committed hero cache (2026-07-19 real re-capture of
# single-goal.mov, 33 raw person detections across the 3500-5000ms window): every dropped box
# bottom is <= 0.533 (stands/cameraman region) and every kept bottom is >= 0.766 (on the pitch),
# so 0.55 sits cleanly inside the real gap between the two populations (23 of 33 kept).
ON_PITCH_MIN_BOTTOM_Y = 0.55


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
