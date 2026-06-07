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
