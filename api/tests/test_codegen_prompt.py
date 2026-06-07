"""
Phase 1: clip-aware codegen prompt. The hero timing/event-window is no longer a SYSTEM_PROMPT
constant — it lives in canned.CLIPS as a per-clip `hint` and is injected as DATA (Q2 -> Alt C).
These tests assert the injection (with the AzureOpenAI client mocked) and that the base prompt
no longer hardcodes the timing literal. Mock-seam style from conftest.py.
"""
from __future__ import annotations

import codegen
import canned


class _CapturingClient:
    """Captures the messages handed to chat.completions.create and returns a canned program."""
    def __init__(self):
        self.captured = None
        program = {"program": [
            {"id": "frames", "op": "sample_frames", "args": {"start_ms": 3500, "end_ms": 5000, "fps": 8}},
            {"id": "result", "op": "answer", "args": {"from": "frames", "question": "q"}},
        ]}
        import json

        class _Msg:
            content = json.dumps(program)

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(_self, **kwargs):
                self.captured = kwargs
                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _codegen_with_mock_client():
    """An AzureCodegen instance whose _client is the capturing mock (no real Azure)."""
    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    client = _CapturingClient()
    cg._client = client
    cg._deployment = "gpt-4o"
    return cg, client


def test_codegen_prompt_injects_clip_metadata():
    cg, client = _codegen_with_mock_client()
    clip = canned.CLIPS["single-goal"]
    cg.generate("Does #10 score the first goal?", clip=clip)

    sys_msg = client.captured["messages"][0]["content"]
    # the injected Clip context block carries label, duration_ms, and the per-clip hint.
    assert "\n\nClip context:\n" in sys_msg  # the injected block heading (not the rules prose)
    assert clip["label"] in sys_msg
    assert str(clip["duration_ms"]) in sys_msg
    assert clip["hint"] in sys_msg
    # the timing now arrives via clip DATA, not a prompt constant.
    assert "start_ms 3500" not in codegen.SYSTEM_PROMPT
    assert "3500" in sys_msg  # injected via the hint
    # the user message is the question verbatim.
    assert client.captured["messages"][1]["content"] == "Does #10 score the first goal?"


def test_codegen_prompt_omits_clip_block_when_clip_none():
    cg, client = _codegen_with_mock_client()
    cg.generate("how many players?", clip=None)
    sys_msg = client.captured["messages"][0]["content"]
    assert "\n\nClip context:\n" not in sys_msg  # no injected block
    # back-compat: the base SYSTEM_PROMPT is used as-is for the canned (no-clip) call site.
    assert sys_msg == codegen.SYSTEM_PROMPT


def test_system_prompt_no_longer_hardcodes_hero_timing():
    # The hero timing constant must NOT survive in the prompt literal (Q2 -> Alt C).
    assert "start_ms 3500" not in codegen.SYSTEM_PROMPT
    assert "end_ms 5000" not in codegen.SYSTEM_PROMPT
    # but it IS preserved as clip data.
    assert "3500" in canned.CLIPS["single-goal"]["hint"]


def test_system_message_describes_generalized_answer():
    # The op catalog must tell the LLM answer summarizes count/text/temporal/existence honestly.
    msg = codegen.AzureCodegen._system_message(None)
    low = msg.lower()
    assert "count" in low and "temporal" in low
    assert "existence" in low or "presence" in low
