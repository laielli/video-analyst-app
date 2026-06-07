"""
Phase 1: the codegen system prompt is split into a stable rules section + an injected, per-clip
context block sourced from canned.CLIPS (Q2 → Alt C). These tests assert:
  - generate(question, clip=CLIPS['single-goal']) injects the clip label, duration, and the
    per-clip `hint` into the system message,
  - the base SYSTEM_PROMPT constant no longer hardcodes the hero `start_ms 3500` literal (the
    timing now arrives as clip DATA),
  - generate(question, clip=None) falls back to the bare prompt (back-compat with the canned path).

The AzureOpenAI client is mocked so no network call is made; we capture the messages it received.
"""
from __future__ import annotations

import codegen
import canned


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeCompletions:
    """Records the kwargs of the last create() call so the test can inspect the system message."""
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type("R", (), {"choices": [_FakeChoice('{"program": []}')]})()


class _FakeClient:
    def __init__(self):
        self.chat = type("C", (), {"completions": _FakeCompletions()})()


def _codegen_with_fake_client():
    """An AzureCodegen whose Azure client is the recording fake (bypasses __init__ creds check)."""
    cg = codegen.AzureCodegen.__new__(codegen.AzureCodegen)
    cg._client = _FakeClient()
    cg._deployment = "gpt-4o"
    return cg


def _system_message_of(cg) -> str:
    msgs = cg._client.chat.completions.last_kwargs["messages"]
    return next(m["content"] for m in msgs if m["role"] == "system")


def test_codegen_prompt_injects_clip_metadata():
    clip = canned.CLIPS["single-goal"]
    cg = _codegen_with_fake_client()
    cg.generate("How many players are visible?", clip=clip)
    sys_msg = _system_message_of(cg)

    # clip label, duration, and the per-clip hint (Q2 → Alt C) are all injected.
    assert clip["label"] in sys_msg
    assert str(clip["duration_ms"]) in sys_msg
    assert clip["hint"] in sys_msg
    assert "Clip context" in sys_msg

    # The base prompt no longer hardcodes the hero timing constant — it survives as clip data.
    assert "start_ms 3500" not in codegen.SYSTEM_PROMPT


def test_codegen_prompt_omits_clip_block_when_clip_none():
    cg = _codegen_with_fake_client()
    cg.generate("anything", clip=None)
    sys_msg = _system_message_of(cg)
    # No clip -> bare SYSTEM_PROMPT, no "Clip context" block (back-compat with the canned path).
    assert sys_msg == codegen.SYSTEM_PROMPT
    assert "Clip context" not in sys_msg


def test_codegen_prompt_describes_generalized_answer_shapes():
    """The op catalog now tells the LLM `answer` summarizes count/text/temporal/existence honestly
    (Phase 0), not only the first-scorer recipe."""
    p = codegen.SYSTEM_PROMPT
    assert "count readout" in p
    assert "text readout" in p
    assert "existence" in p or "presence" in p


def test_system_message_helper_is_pure():
    """_system_message(None) is the bare prompt; with a clip it appends, never mutating the const."""
    base = codegen.SYSTEM_PROMPT
    assert codegen.AzureCodegen._system_message(None) == base
    withclip = codegen.AzureCodegen._system_message(canned.CLIPS["single-goal"])
    assert withclip.startswith(base) and len(withclip) > len(base)
    assert codegen.SYSTEM_PROMPT == base  # unchanged
