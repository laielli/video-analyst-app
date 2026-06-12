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
# Phase 1 — count-shape grounding rule (single representative frame)
# --------------------------------------------------------------------------------------

def test_codegen_prompt_has_counting_rule():
    """The prompt must steer counting questions to a SINGLE representative frame (the count
    over-counts when summed across a multi-frame window). It must prescribe the single-frame
    pattern (1ms window, fps 1) WITHOUT hardcoding a clip timestamp literal — the timestamp comes
    from the injected hint (mirrors the existing `assert "3500" not in SYSTEM_PROMPT` guard)."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    assert "count" in low
    # the single-frame prescription is present (sample ONE frame, 1ms window, fps 1).
    assert "one representative frame" in low or "single representative frame" in low
    assert "fps 1" in low or "fps=1" in low
    # deterministic timestamp choice (the midpoint of the hint window), not a free guess.
    assert "midpoint" in low
    # clip-agnostic: no hardcoded bernabeu count timestamp literal leaked into the base prompt.
    assert "10000" not in p, "the counting rule must not hardcode a clip timestamp literal"


# --------------------------------------------------------------------------------------
# Phase 2 — capability-scope / honest-grounding rule
# --------------------------------------------------------------------------------------

def test_codegen_prompt_has_capability_scope_rule():
    """The prompt must enumerate the observable capability surface and instruct an honest
    ground-out (not a fabricated chain) for questions outside it."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    # the observable surface is named.
    assert "presence" in low
    assert "jersey" in low and "number" in low
    # at least some out-of-scope attributes are named so the rule is concrete.
    assert "color" in low  # jersey COLOR is the canonical out-of-scope attribute
    assert "formation" in low or "emotion" in low
    # the honest-grounding instruction: ground out / ungrounded rather than fabricate.
    assert "ground out" in low or "grounds out" in low or "ungrounded" in low
    assert "cannot" in low or "not answerable" in low or "unanswerable" in low


def test_codegen_prompt_refusal_rule_is_scoped():
    """Negative-wording guard for the over-refusal regression risk: the refusal rule must scope by
    CAPABILITY ('outside the listed capabilities' / 'outside that surface'), never by blanket
    uncertainty ('when unsure' / 'when in doubt'). An over-broad rule would push borderline in-scope
    paraphrases to ground out, regressing presence/scorer-number — invisible until the live
    re-capture. This is the only creds-free guard on that risk."""
    low = codegen.SYSTEM_PROMPT.lower()
    assert "when unsure" not in low
    assert "when in doubt" not in low
    assert "if unsure" not in low
    # the rule IS scoped to capability, positively.
    assert "outside the listed capabilities" in low or "outside that surface" in low


# --------------------------------------------------------------------------------------
# Post-recapture regression fixes — analyzed-window law + digit-identify shape
# --------------------------------------------------------------------------------------

def test_codegen_prompt_has_analyzed_window_rule():
    """The 2026-06-11 re-capture showed chains sampling outside the precomputed cache (whole-clip
    midpoint 15043 on bernabeu; fps-1 grids missing single-goal's lone 4625 frame) ground out
    empty — presence fell 11/11 -> 1/11 and count grounded out instead of overcounting. The prompt
    must (a) state the analyzed-window law (analysis exists ONLY in the hint's window at the
    hint's fps; copy both verbatim) and (b) NOT offer the whole-clip-midpoint fallback that
    invited off-window sampling."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    assert "analyzed" in low
    assert "verbatim" in low
    assert "grounds out" in low or "ground out" in low
    # the landmine: a whole-clip-midpoint fallback for counts must never come back.
    assert "duration_ms / 2" not in p
    assert "midpoint of the whole clip" not in low


def test_codegen_prompt_digit_identify_is_what_number_shape():
    """Scorer-number paraphrases like "Can you identify the digit ..." regressed to yes/no chains
    (filter(text == '7') -> temporal_order -> 'Yes') at the 2026-06-11 re-capture. The shape rule
    must classify identify/name-the-digits phrasings as "what number" questions ending
    read_text -> answer, and confine filter-on-a-number to questions that themselves name it."""
    low = codegen.SYSTEM_PROMPT.lower()
    assert "identify" in low
    assert "what number" in low
    assert "read_text -> answer" in low
