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

API_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = API_DIR / "schema" / "dsl.schema.json"

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

The `answer` step summarizes whatever binding it consumes, HONESTLY. Choose the chain whose
final binding fits the question — do NOT force a temporal_order when the question is a count
or a read-out:
- ordered binding (temporal_order) -> a temporal/boolean verdict ("did #N score first?").
- number binding (count)           -> a count answer ("how many players are visible?").
- texts binding (read_text)        -> a text read-out ("what number is the scorer wearing?").
- detections/crops binding         -> an existence answer ("is there a referee?").

Rules:
- The load-bearing association is detect -> crop -> read_text: a detection box flows into a
  crop, then into OCR. To reason about a jersey number you MUST crop region "jersey", then
  read_text, then filter on the read text.
- To decide whether a given shirt number scores the first goal: sample frames around the goal
  window (see the Clip context block for THIS clip's timing/event window), detect "person",
  crop "jersey", read_text, filter text == the number in the question, temporal_order by
  "timestamp", then answer. To count people: sample frames, detect "person", count, answer.
- Use only the whitelisted ops and arg shapes. Every binding must reference the id of an
  EARLIER step of a compatible kind. The final step must be `answer`, with the user's
  question passed through verbatim.
- Give each step a short, descriptive snake_case id that reads like a variable name for what
  it holds (e.g. frames, people, jerseys, numbers, tens, ordered, result) — NOT step1, step2.
  The program is shown to the user as the model's reasoning, so the ids should be legible."""


def build_system_message(clip: dict | None = None) -> str:
    """The stable rules prompt + an injected, per-clip Clip-context block (Q2 -> Alt C). The
    hero timing is NOT a SYSTEM_PROMPT constant — it arrives as clip DATA (label, duration_ms,
    hint from canned.CLIPS), so the prompt stays clip-agnostic while the demo keeps grounding.
    clip=None -> base prompt only (back-compat with the canned path call site)."""
    if not clip:
        return SYSTEM_PROMPT
    block = (
        "\n\nClip context (compile against THIS clip):\n"
        f"- label: {clip.get('label', '')}\n"
        f"- duration_ms: {clip.get('duration_ms')}; sample windows must stay within "
        "[0, duration_ms]\n"
    )
    if clip.get("hint"):
        block += f"- hint: {clip['hint']}\n"
    return SYSTEM_PROMPT + block


def enabled() -> bool:
    """True iff Azure OpenAI creds are configured. When False, callers use the pinned program."""
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY"))


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
        """Compile `question` into a program (list of DSL steps). When `clip` is supplied, the
        per-clip metadata (label, duration_ms, timing/event hint from canned.CLIPS) is injected
        into the system message so the question compiles against the ACTUAL clip (Q2 -> Alt C);
        clip=None keeps the canned-path behavior. Raises on API error or unparseable output; the
        caller validates structure/semantics. For free text there is NO pinned fallback — a miss
        grounds out honestly (Q1 -> A)."""
        sys_msg = build_system_message(clip)
        resp = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
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
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": question},
            ],
        )
        doc = json.loads(resp.choices[0].message.content or "{}")
        program = doc.get("program")
        if not isinstance(program, list):
            raise ValueError("codegen output missing 'program' array")
        return program
