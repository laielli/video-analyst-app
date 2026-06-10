"""
Phase 1 (Q2 -> Alt C): the codegen prompt is clip-aware. Per-clip metadata (label, duration_ms,
and the timing/event-window `hint` sourced from canned.CLIPS) is injected into the system message,
and the hardcoded hero `start_ms 3500` constant is removed from the base SYSTEM_PROMPT (the timing
now arrives as DATA, not a prompt literal).

The Azure OpenAI client is mocked: AzureCodegen.generate is called and the captured system
message is inspected. Mock-seam style from conftest.py.
"""
from __future__ import annotations

import json

import codegen
from canned import CLIPS


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _CaptureClient:
    """Stands in for the AzureOpenAI client; records the messages of the last create() call."""
    def __init__(self):
        self.last_messages = None
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.last_messages = kwargs["messages"]
        # return a minimal valid program so generate() doesn't raise.
        return _FakeResp(json.dumps({"program": [
            {"id": "result", "op": "answer", "args": {"from": "x", "question": "q"}}
        ]}))


def _codegen_with_capture():
    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)  # bypass __init__ (no real client)
    client = _CaptureClient()
    cg._client = client
    cg._deployment = "gpt-4o"
    return cg, client


# --------------------------------------------------------------------------------------
# test_codegen_prompt_injects_clip_metadata
# --------------------------------------------------------------------------------------

def test_codegen_prompt_injects_clip_metadata():
    cg, client = _codegen_with_capture()
    clip = CLIPS["single-goal"]
    cg.generate("how many players?", clip=clip)

    sys_msg = next(m["content"] for m in client.last_messages if m["role"] == "system")
    # the clip label, duration, and the per-clip hint are all injected.
    assert clip["label"] in sys_msg
    assert str(clip["duration_ms"]) in sys_msg
    assert clip["hint"] in sys_msg
    assert "Clip context:\n- label:" in sys_msg  # the injected block header, not the prose mention

    # the base SYSTEM_PROMPT no longer hardcodes the hero timing literal — it survives as clip data.
    assert "3500" not in codegen.SYSTEM_PROMPT
    assert "start_ms 3500" not in codegen.SYSTEM_PROMPT


# --------------------------------------------------------------------------------------
# test_codegen_prompt_omits_clip_block_when_clip_none — back-compat with the canned call site
# --------------------------------------------------------------------------------------

def test_codegen_prompt_omits_clip_block_when_clip_none():
    cg, client = _codegen_with_capture()
    cg.generate("Does #10 score the first goal?", clip=None)

    sys_msg = next(m["content"] for m in client.last_messages if m["role"] == "system")
    assert sys_msg == codegen.SYSTEM_PROMPT
    # the INJECTED clip block (header + label line) is absent — its presence is what marks injection.
    assert "Clip context:\n- label:" not in sys_msg


# --------------------------------------------------------------------------------------
# build_system_message is the pure seam the injection rides on
# --------------------------------------------------------------------------------------

def test_build_system_message_pure_helper():
    base = codegen.build_system_message(None)
    assert base == codegen.SYSTEM_PROMPT
    withclip = codegen.build_system_message(CLIPS["single-goal"])
    assert withclip.startswith(codegen.SYSTEM_PROMPT)
    assert "hint:" in withclip and CLIPS["single-goal"]["hint"] in withclip


# --------------------------------------------------------------------------------------
# Phase 1 count-shape grounding rule + Phase 2 capability-scope / honest-grounding rule
# --------------------------------------------------------------------------------------

def test_codegen_prompt_has_counting_rule():
    """The single-frame counting rule is present: it names the multi-frame summation failure and
    prescribes a deterministic single-frame (fps 1) sample. It must NOT hardcode a clip timestamp
    literal — the timestamp is derived from the injected `hint` (mirrors the `3500`-literal guard
    above)."""
    sp = codegen.SYSTEM_PROMPT.lower()
    # prescribes the single-representative-frame pattern at fps 1
    assert "single" in sp and "fps 1" in sp
    assert "one representative frame" in sp or "exactly one" in sp
    # names the count summation failure mode so the rule is self-explaining
    assert "count" in sp and "sum" in sp
    # deterministic timestamp choice (midpoint of the hint's event window), not "a representative t"
    assert "midpoint" in sp
    # no hardcoded clip timestamp literal — the hint supplies it (same negative-literal pattern as
    # the de-hardcoded hero timing).
    assert "3500" not in codegen.SYSTEM_PROMPT
    assert "4250" not in codegen.SYSTEM_PROMPT


def test_codegen_prompt_has_capability_scope_rule():
    """The capability-scope rule lists the observable surface and instructs honest grounding-out
    for out-of-scope questions rather than a fabricated grounded answer."""
    sp = codegen.SYSTEM_PROMPT.lower()
    # names the observable surface (person/presence/count + jersey number + first-scorer timing)
    assert "person" in sp and "jersey" in sp
    # names at least a few of the out-of-scope attributes the report flagged
    for attr in ("color", "emotion", "formation", "crowd"):
        assert attr in sp, f"capability rule should name '{attr}' as out of scope"
    # instructs honest grounding-out (the grounded:false discriminator), not fabrication
    assert "ground" in sp
    assert "fabricate" in sp or "invent" in sp


def test_codegen_prompt_refusal_rule_is_scoped():
    """Negative-wording guard on the over-refusal regression: the refusal rule must scope by
    CAPABILITY ('outside the listed capabilities' or equivalent), never by blanket uncertainty.
    A 'when unsure / when in doubt, refuse' rule would push borderline in-scope paraphrases to
    ground out, silently regressing presence/scorer-number — invisible to seeds, only surfaced at
    the human's re-capture. This is the only agent-side, creds-free guard on that risk."""
    sp = codegen.SYSTEM_PROMPT.lower()
    assert "outside the listed capabilities" in sp
    assert "when unsure" not in sp
    assert "when in doubt" not in sp
    assert "if unsure" not in sp
