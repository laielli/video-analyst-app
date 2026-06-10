"""
question -> DSL visual program, via Azure OpenAI structured output (gpt-4o).

This is ViperGPT Layer 1: the LLM compiles the canned question into the whitelisted JSON DSL
(schema/dsl.schema.json). The SAME schema is sent as a strict structured-output constraint
(valid-by-construction codegen) and re-checked locally by validate_program (the semantics
JSON Schema can't express). Execution stays replay-mode over the pinned cache, so a
live-generated program runs deterministically and free — only this one codegen call hits the
network.

Disabled cleanly when AZURE_OPENAI_* is unset (`enabled()` is False): callers fall back to the
query's pinned program (D-DR6). Same Azure OpenAI client pattern as vision/vlm_read.py.
SDK: openai>=1.40 (AzureOpenAI client).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from limits import MAX_PROGRAM_BYTES

API_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = API_DIR / "schema" / "dsl.schema.json"

# Cap the completion length so the model can never be steered into emitting an unbounded blob (a
# DoS vector for the downstream jsonschema walk). One token is >= 1 byte for ASCII JSON, so
# MAX_PROGRAM_BYTES // 3 keeps the worst-case raw size comfortably under MAX_PROGRAM_BYTES while
# leaving ample room for the largest legitimate program (the hero/bernabeu chains are ~1-2 KB).
# Clamped to gpt-4o's 16,384 completion-token ceiling: a larger value is a 400 invalid_request on
# every live call (the byte-cap below still bounds the worst case independently).
MODEL_MAX_COMPLETION_TOKENS = 16_384
MAX_PROGRAM_TOKENS = min(MAX_PROGRAM_BYTES // 3, MODEL_MAX_COMPLETION_TOKENS)

SYSTEM_PROMPT = """\
You are the program-generation layer of a ViperGPT-style video analyst. Compile the user's
question about a short soccer clip into a "visual program": an ordered list of named steps in
the provided JSON DSL. Each step has a unique `id`; later steps reference earlier step ids by
name (named bindings). Emit ONLY the program object {"program": [...]} — no prose.

The only ops allowed (with their output kind):
  sample_frames(start_ms, end_ms, fps)            -> frames
  detect(frames, classes[])                       -> detections   (objects per frame)
  crop(detections, region in [jersey, full, ball])-> crops
  read_text(crops)                                -> texts         (OCR per crop)
  filter(items, where{field, equals})             -> same kind as items
  count(items)                                     -> number
  temporal_order(events, by="timestamp")          -> ordered events
  answer(from, question)                          -> verdict       (MUST be the last step)

The final `answer` step summarizes whatever its `from` binding holds, honestly, to the question
asked — it is NOT limited to a first-scorer verdict:
- from an `ordered` (temporal_order) binding -> a boolean/temporal verdict ("did X score first?")
- from a `number` (count) binding             -> a count answer ("how many ...?")
- from a `texts` (read_text/filter) binding   -> a text readout ("what number is ...?")
- from a `detections`/`crops`/`frames` binding-> an existence answer ("is there a ...?")
Pick the chain whose final binding matches the QUESTION SHAPE: a counting question ends
count -> answer; a "what number" question ends read_text -> answer; a temporal/first question
ends temporal_order -> answer.

Counting ("how many X are visible?"):
- `count` returns the number of items in its input, SUMMED across EVERY sampled frame — so
  sampling N frames inflates the count by a factor of ~N (e.g. 5 people over 17 frames -> 85).
- So for a count, sample exactly ONE representative frame: a 1ms window at a single timestamp,
  at fps 1. Compute that timestamp as the MIDPOINT of the hint's event window — i.e.
  t = (window_start_ms + window_end_ms) // 2 — then sample_frames(start=t, end=t+1, fps=1),
  detect, count, answer. NEVER sample a multi-frame window for a count.

Rules:
- The load-bearing association is detect -> crop -> read_text: a detection box flows into a
  crop, then into OCR. To reason about a jersey number you MUST crop region "jersey", then
  read_text, then filter on the read text.
- To decide whether a given shirt number scores the first goal: sample frames over the clip's
  event window (see the Clip context below for the timing/event window and fps to use), detect
  "person", crop "jersey", read_text, filter text == the number in the question, temporal_order
  by "timestamp", then answer. Sample windows MUST stay within [0, duration_ms].
- Use only the whitelisted ops and arg shapes. Every binding must reference the id of an
  EARLIER step of a compatible kind. The final step must be `answer`, with the user's
  question passed through verbatim.
- Give each step a short, descriptive snake_case id that reads like a variable name for what
  it holds (e.g. frames, people, jerseys, numbers, tens, ordered, result) — NOT step1, step2.
  The program is shown to the user as the model's reasoning, so the ids should be legible.

Capability scope (answer honestly; do NOT fabricate):
- The ops above can ONLY observe: people on the pitch (person detection), jersey NUMBERS
  (jersey-region OCR), goal/first-scorer timing, and the presence of a numbered player. That
  is the whole observable surface.
- If a question asks for ANYTHING outside that listed surface — crowd/stadium size, emotion,
  weather, team formation, jersey COLOR, commentary, the referee, a coach, the final score,
  offside, ball speed, or any input that is just noise/punctuation with no answerable content —
  you CANNOT answer it. Do NOT invent a chain that grounds to an unrelated number or verdict.
- Instead emit a program that honestly grounds out: a normal detect -> crop "jersey" ->
  read_text -> filter chain whose filter tests a value the clip provably lacks (e.g.
  filter text == "99"), so the final binding is empty and the answer comes back grounded:false.
  This is the honest "I can't answer this" path — far better than a fabricated answer.
- Scope this ONLY to questions outside the listed capabilities. Do NOT ground out a normal
  in-scope question (count / jersey-number / first-goal / presence) just because the phrasing is
  unfamiliar — only when the thing ASKED FOR is not in the observable surface above."""


def enabled() -> bool:
    """True iff Azure OpenAI creds are configured. When False, callers use the pinned program."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY"))


def build_system_message(clip: dict | None = None) -> str:
    """The per-call system message: the stable SYSTEM_PROMPT rules + an injected, clip-specific
    'Clip context' block sourced from canned.CLIPS (Q2 -> Alt C). The hero timing now arrives as
    DATA via clip['hint'], not as a prompt constant — so the base prompt stays clip-agnostic and
    the next clip just supplies its own hint. With clip=None (the canned call site's back-compat
    path) the message is exactly SYSTEM_PROMPT with no clip block appended."""
    if not clip:
        return SYSTEM_PROMPT
    msg = SYSTEM_PROMPT + (
        f"\n\nClip context:\n- label: {clip['label']}\n"
        f"- duration_ms: {clip['duration_ms']}, sample within [0, duration_ms]\n"
    )
    if clip.get("hint"):  # per-clip timing/event window, as data (e.g. "goal ~3500-5000ms; ...")
        msg += f"- hint: {clip['hint']}\n"
    return msg


def _schema_for_openai() -> dict:
    """The DSL schema as a strict structured-output schema: strip the root keywords OpenAI
    rejects ($schema/$id/title); everything else is strict-compatible (per the schema's own
    note — object roots, additionalProperties:false, all-required, enums, $defs/$ref, anyOf)."""
    schema = json.loads(SCHEMA_PATH.read_text())
    for key in ("$schema", "$id", "title"):
        schema.pop(key, None)
    return schema


class AzureCodegen:
    """Thin wrapper over Azure OpenAI chat-completions with a strict json_schema response
    format. from_env() or explicit creds. generate() returns a program (list of steps);
    validity is the caller's gate (validate_program.validation_errors)."""

    def __init__(self, endpoint: str, key: str, deployment: str, api_version: str = "2024-10-21"):
        from openai import AzureOpenAI  # lazy: keep import optional until actually used

        if not (endpoint and key and deployment):
            raise ValueError("Azure OpenAI endpoint, key, and deployment are required.")
        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
        self._deployment = deployment

    @classmethod
    def from_env(cls) -> "AzureCodegen":
        return cls(
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            key=os.environ.get("AZURE_OPENAI_KEY", ""),
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    def generate(self, question: str, clip: dict | None = None) -> list[dict]:
        """Compile `question` into a program (list of DSL steps). When `clip` is given, its
        metadata (label, duration_ms, hint) is injected into the system message so the program is
        compiled against the ACTUAL clip (Q2 -> Alt C). Raises on API error or unparseable output;
        the caller validates structure/semantics (and, for free text, grounds out honestly on any
        miss — there is no pinned fallback for a novel question)."""
        resp = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            # Bound the output so a hostile/confused prompt can't make the model emit a giant blob
            # that DoS-es the downstream jsonschema walk (trust boundary; mirrors MAX_PROGRAM_BYTES).
            max_tokens=MAX_PROGRAM_TOKENS,
            response_format={
                "type": "json_schema",
                # strict=False on purpose. Our program is a discriminated union (8 op objects in
                # an `anyOf`, all sharing id/op/args keyed by the `op` enum); OpenAI's STRICT
                # structured-output subset rejects that shape ("Invalid response_format provided").
                # strict=False still feeds the full schema as strong guidance, and correctness is
                # enforced locally anyway — validate_program re-checks every program and D-DR6 falls
                # back to the pinned program on any miss. So the schema stays the single source of truth.
                "json_schema": {"name": "visual_program", "strict": False, "schema": _schema_for_openai()},
            },
            messages=[
                {"role": "system", "content": build_system_message(clip)},
                {"role": "user", "content": question},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        # Raw-size cap BEFORE json.loads (trust boundary): max_tokens bounds the API output, but
        # belt-and-suspenders against a misconfigured deployment or a multibyte blow-up. Reject the
        # raw bytes here so the unbounded string never reaches the json parser / validator. The
        # caller maps any generate() error to the fixed `codegen-error` reason (no text leak).
        if len(content.encode("utf-8")) > MAX_PROGRAM_BYTES:
            raise ValueError(f"codegen output exceeds {MAX_PROGRAM_BYTES} bytes")
        doc = json.loads(content)
        program = doc.get("program")
        if not isinstance(program, list):
            raise ValueError("codegen output missing 'program' array")
        return program
