"""
Phase 1: the codegen prompt is clip-aware. Per-clip metadata (label, duration_ms, and the
timing/event-window `hint` sourced from canned.CLIPS — Q2 -> Alt C) is injected into the system
message, and the hero timing constant is NO LONGER hardcoded in SYSTEM_PROMPT (it survives as
clip data). The AzureOpenAI client is mocked so no network call is made.
"""
from __future__ import annotations

import codegen
import canned


class _FakeChatClient:
    """Captures the messages handed to chat.completions.create and returns a canned program."""

    def __init__(self):
        self.captured_messages = None
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, model, temperature, response_format, messages):
        self.captured_messages = messages

        class _Msg:
            content = '{"program": []}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _codegen_with_fake_client():
    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)  # bypass __init__ (no openai SDK / creds)
    cg._client = _FakeChatClient()
    cg._deployment = "gpt-4o"
    return cg


def test_codegen_prompt_injects_clip_metadata():
    cg = _codegen_with_fake_client()
    clip = canned.CLIPS["single-goal"]
    cg.generate("How many players are visible?", clip=clip)

    sys_msg = next(m["content"] for m in cg._client.captured_messages if m["role"] == "system")
    # Clip metadata reached the prompt: label, duration_ms, and the per-clip hint (Q2 -> Alt C).
    assert clip["label"] in sys_msg
    assert str(clip["duration_ms"]) in sys_msg
    assert clip["hint"] in sys_msg

    # The base SYSTEM_PROMPT constant no longer hardcodes the hero timing literal — it arrives
    # as clip DATA now, not a prompt constant.
    assert "start_ms 3500" not in codegen.SYSTEM_PROMPT
    assert "3500" not in codegen.SYSTEM_PROMPT


def test_codegen_prompt_omits_clip_block_when_clip_none():
    cg = _codegen_with_fake_client()
    cg.generate("Does #10 score the first goal?", clip=None)
    sys_msg = next(m["content"] for m in cg._client.captured_messages if m["role"] == "system")
    # Back-compat with the canned path call site (server.build_run_doc): the injected Clip-context
    # block (with its concrete clip data) is absent — sys_msg is exactly the base prompt.
    assert sys_msg == codegen.SYSTEM_PROMPT
    assert "Clip context (compile against THIS clip)" not in sys_msg


def test_build_system_message_is_clip_agnostic_by_default():
    # build_system_message(None) is exactly the stable rules prompt.
    assert codegen.build_system_message() == codegen.SYSTEM_PROMPT
    msg = codegen.build_system_message(canned.CLIPS["single-goal"])
    assert msg.startswith(codegen.SYSTEM_PROMPT)
    # The injected block (not the prose reference) carries the concrete clip data.
    assert "Clip context (compile against THIS clip)" in msg
