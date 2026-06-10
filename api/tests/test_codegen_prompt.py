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
# Count-shape grounding rule (Phase 1): single-frame sampling for counting questions.
# --------------------------------------------------------------------------------------

def test_codegen_prompt_has_counting_rule():
    """The prompt names the count-window failure (count sums across all sampled frames) and
    prescribes the single-representative-frame pattern (fps 1, a 1ms window). It must steer the
    sampling without hardcoding a clip-specific timestamp literal — the t comes from the injected
    hint, mirroring the existing 'no 3500 literal' guard."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    # the failure mode is named explicitly so the model understands *why* one frame.
    assert "count" in low and ("sums" in low or "summed" in low or "multiplies" in low)
    # the single-frame remedy is prescribed concretely.
    assert "one" in low or "single" in low
    assert "fps=1" in low or "fps 1" in low or "fps1" in low
    # deterministic timestamp choice, not "a representative frame" hand-waving.
    assert "midpoint" in low or "event window" in low
    # clip-agnostic: no hardcoded clip timestamp literals in the counting rule (the t is data).
    assert "10000" not in p and "10001" not in p


# --------------------------------------------------------------------------------------
# Capability-scope / honest-grounding rule (Phase 2).
# --------------------------------------------------------------------------------------

def test_codegen_prompt_has_capability_scope_rule():
    """The prompt enumerates the observable capability surface and instructs honest grounding-out
    for questions outside it, rather than fabricating an unrelated grounded answer."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    # the observable surface is listed.
    assert "jersey" in low and "number" in low and "presence" in low
    # at least some of the canonical out-of-scope attributes are named so the model can map novel Qs.
    assert "color" in low or "colour" in low
    assert "emotion" in low or "formation" in low or "weather" in low
    # the honest-grounding instruction is present (ground out / cannot answer, don't fabricate).
    assert "ground out" in low or "grounds out" in low or "not grounded" in low
    assert "cannot" in low or "fabricate" in low


def test_codegen_prompt_refusal_rule_is_scoped():
    """Negative-wording guard on the over-refusal regression risk: the refusal must be scoped by
    CAPABILITY ('outside the listed capabilities' / 'outside ... surface'), never triggered by
    blanket uncertainty. A 'when unsure / when in doubt, refuse' rule would push borderline
    in-scope paraphrases to ground out and regress presence/scorer cells, invisible to seeds."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    # scoped to capability, not uncertainty.
    assert "outside" in low and ("capabilit" in low or "surface" in low)
    # explicit anti-uncertainty triggers must be absent.
    assert "when unsure" not in low
    assert "when in doubt" not in low
    assert "if unsure" not in low
