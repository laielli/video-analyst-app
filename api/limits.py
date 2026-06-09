"""
Trust-boundary limit constants for the generated-program executor.

ONE top-level module so every layer that must agree on the envelope imports the SAME numbers
with zero cycle risk: `validate_program.py` (the reject-before-execute gate), `interpreter/`
(the runtime guard inside `op_sample_frames` + the `MAX_STEPS` check in `Interpreter.run`),
and `codegen.py` (the `max_tokens` + raw-byte cap on the LLM output). The interpreter package
deliberately does NOT import from `scripts/` — that path is only on `sys.path` when `server.py`
or a probe inserts it, while `api/` (where this lives) is always on the path wherever the
interpreter is loaded (server.py, precompute.py, pytest `pythonpath = . scripts`).

Why these values (see the plan's DESIGN-ROOM "limit values" knob — this impl took the draft
default, sized comfortably above every observed real program while still rejecting the bombs):
  - real programs top out at 7 steps and ~17 sampled frames (hero 3500-5000ms@8fps = 13 frames;
    bernabeu 9000-11000ms@8fps = 17 frames);
  - the largest *legitimate* full-clip sweep is the bernabeu clip (30086ms) @ fps30 ~= 912 frames;
  - the OOM bombs are `end=2e9, fps30` ~= 60M frames -- rejected by the cumulative cap and the
    pre-materialization cap alike.
So 2000 frames leaves >2x headroom over the largest legit sweep while rejecting the bombs, and
32 steps is >4x the 7-step hero chain. The string/class caps are anti-abuse hygiene (not OOM
vectors once MAX_PROGRAM_BYTES exists) sized to never clip a legitimate class list or value.

The schema (`dsl.schema.json`) re-encodes the statically-expressible subset of these; the
`test_limit_constants_match_schema` test asserts they never drift.
"""
from __future__ import annotations

# Raw codegen output (bytes), checked BEFORE any json.loads / schema walk. The jsonschema pass
# is an 8-way anyOf per item and superlinear on hostile input, so it must never see an
# unbounded blob. Enforced in TWO places: codegen.generate (max_tokens on the API call + a
# byte check on the returned content) AND at the top of validation_errors.
MAX_PROGRAM_BYTES = 65_536

# Whole-program step cap (hero chain is 7). Checked in semantic_errors (reject-before-execute),
# in the schema (program array maxItems), AND at the top of Interpreter.run (runtime guard for
# any future entry point that invokes run() without validating first).
MAX_STEPS = 32

# SUM of frames materialized across ALL sample_frames ops. The cumulative validator cap is
# duration-independent (it counts range() length), so it closes the no-clip OOM gap that the
# per-clip duration bound cannot. The per-call pre-materialization cap in op_sample_frames uses
# the same number so the validator and the runtime guard never disagree.
MAX_SAMPLED_FRAMES = 2000

# detect.classes length. A legitimate class list is a handful of nouns ("person", "ball").
MAX_DETECT_CLASSES = 16

# Max length of free-string args (detect.classes[] elements, filter.where.field/equals).
MAX_STR_ARG_LEN = 256
