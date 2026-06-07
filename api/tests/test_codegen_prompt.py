"""
Phase 1: clip metadata (label, duration_ms, and the per-clip timing/event-window `hint`, Q2 ->
Alt C) is injected into the codegen system message, and the base SYSTEM_PROMPT no longer hardcodes
the hero's start_ms literal — the timing now arrives as clip DATA. The AzureOpenAI client is mocked
so no creds / network are needed (mock-seam style from conftest.py).
"""
from __future__ import annotations

import codegen
from canned import CLIPS


class _FakeChatCompletions:
    """Captures the messages handed to chat.completions.create and returns a canned program."""

    def __init__(self, sink):
        self._sink = sink

    def create(self, **kwargs):
        self._sink["messages"] = kwargs["messages"]

        class _Msg:
            content = '{"program": []}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, sink):
        self.chat = type("C", (), {"completions": _FakeChatCompletions(sink)})()


def _codegen_with_capture():
    sink: dict = {}
    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)  # bypass __init__ (no real client)
    cg._client = _FakeClient(sink)
    cg._deployment = "gpt-4o"
    return cg, sink


def _system_message(sink: dict) -> str:
    return next(m["content"] for m in sink["messages"] if m["role"] == "system")


def test_codegen_prompt_injects_clip_metadata():
    cg, sink = _codegen_with_capture()
    clip = CLIPS["single-goal"]
    cg.generate("how many players are visible?", clip=clip)
    sys_msg = _system_message(sink)

    # clip label, duration, and the per-clip hint (Q2 -> Alt C) are all injected.
    assert clip["label"] in sys_msg
    assert str(clip["duration_ms"]) in sys_msg
    assert clip["hint"] in sys_msg

    # and the base prompt no longer hardcodes the hero timing literal — it survives as clip DATA.
    assert "3500" not in codegen.SYSTEM_PROMPT
    assert "start_ms 3500" not in codegen.SYSTEM_PROMPT


def test_codegen_prompt_omits_clip_block_when_clip_none():
    cg, sink = _codegen_with_capture()
    cg.generate("Does #10 score the first goal?", clip=None)
    sys_msg = _system_message(sink)
    # back-compat: no clip -> base SYSTEM_PROMPT verbatim, no "Clip context" block.
    assert "Clip context" not in sys_msg
    assert sys_msg == codegen.SYSTEM_PROMPT


def test_codegen_prompt_question_passed_through():
    cg, sink = _codegen_with_capture()
    cg.generate("is there a referee?", clip=CLIPS["single-goal"])
    user_msg = next(m["content"] for m in sink["messages"] if m["role"] == "user")
    assert user_msg == "is there a referee?"
