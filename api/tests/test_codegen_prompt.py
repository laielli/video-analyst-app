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
import re
from pathlib import Path

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


# --------------------------------------------------------------------------------------
# 2026-06-12 re-capture fixes — noise precedence, midpoint exclusivity, hint-number filter
# --------------------------------------------------------------------------------------

def test_codegen_prompt_noise_precedence_rule():
    """The 2026-06-12 capture showed adv-injection-huge-window and adv-unicode-rtl-noise getting
    grounded '5' answers: the (now working) count chain claimed noise/injection inputs the refusal
    rule should own. The ground-out rules must take explicit PRECEDENCE over the chain recipes,
    covering instruction-embedding and control/RTL/zero-width corruption."""
    low = codegen.SYSTEM_PROMPT.lower()
    assert "precedence" in low
    assert "zero-width" in low and "rtl" in low
    assert "embeds instructions" in low


def test_codegen_prompt_midpoint_pattern_is_count_exclusive():
    """The 2026-06-12 capture showed single-goal-presence-far-3 using the single-frame midpoint
    count pattern for a presence question and grounding out empty. The midpoint pattern must be
    scoped to counting questions only; presence/yes-no chains use the hint's full window + fps."""
    low = codegen.SYSTEM_PROMPT.lower()
    assert "exclusively for counting" in low


def test_codegen_prompt_forbids_hint_number_filter():
    """The 2026-06-12 capture showed three scorer-number paraphrases answering 'Yes' via
    filter(text == '7') with the 7 sourced from the HINT, ending temporal_order -> answer. The
    shape rule must forbid hint-sourced number filters and temporal_order endings for
    "what number" questions."""
    low = codegen.SYSTEM_PROMPT.lower()
    assert "never filter on a number taken from the clip context hint" in low
    assert "never end temporal_order -> answer" in low


# --------------------------------------------------------------------------------------
# Few-shot iteration (#3) — worked examples carry the failing-shape behavior. The 66-case
# capture under the #50 prompt showed prose rules alone do not steer the 3 remaining shapes,
# so each gets a worked program; these guards pin the examples' shape invariants creds-free.
# --------------------------------------------------------------------------------------

def _example(shape: str) -> dict:
    return next(e for e in codegen.FEW_SHOT_EXAMPLES if e["shape"] == shape)


def _assert_uses_hint_window_verbatim(ex: dict):
    """The example's sample step must copy the EXAMPLE_CLIP hint's window + fps verbatim — the
    analyzed-window behavior the examples are supposed to demonstrate."""
    win = next(s for s in ex["program"] if s["op"] == "sample_frames")["args"]
    hint = codegen.EXAMPLE_CLIP["hint"]
    assert f"{win['start_ms']}-{win['end_ms']}ms" in hint
    assert f"fps {win['fps']}" in hint


def test_few_shot_examples_validate_against_dsl():
    """Every few-shot program must pass the REAL validator (the same gate generated programs
    face) against the fictional example clip — an example that wouldn't validate would teach the
    model an illegal shape."""
    from validate_program import validation_errors

    for ex in codegen.FEW_SHOT_EXAMPLES:
        errs = validation_errors({"program": ex["program"]}, clip=codegen.EXAMPLE_CLIP)
        assert errs == [], f"few-shot example {ex['shape']!r} fails the real validator: {errs}"


def test_few_shot_examples_rendered_into_prompt():
    """The examples are data; the prompt is their render. Every question and step must appear in
    SYSTEM_PROMPT (compact JSON), and the examples block sits at the very END so the injected
    Clip context lands directly under the 'substitute the real Clip context's window and fps'
    pointer (recency: the real hint is the last thing the model reads)."""
    for ex in codegen.FEW_SHOT_EXAMPLES:
        assert json.dumps(ex["question"]) in codegen.SYSTEM_PROMPT
        for step in ex["program"]:
            assert json.dumps(step, separators=(",", ":")) in codegen.SYSTEM_PROMPT
    assert codegen.SYSTEM_PROMPT.index("Worked examples") > codegen.SYSTEM_PROMPT.index("Rules:")
    # nothing may be appended after the last example — the clip block must land right under it.
    assert codegen.SYSTEM_PROMPT.endswith(codegen._render_example(codegen.FEW_SHOT_EXAMPLES[-1]))


def test_few_shot_scorer_number_example_shape():
    """scorer-number (3/6 paraphrases ended temporal_order -> 'Yes' at the 2026-06-12 capture):
    the worked program must end read_text -> answer with NO filter and NO temporal_order, over
    the hint window verbatim."""
    ex = _example("scorer-number")
    ops = [s["op"] for s in ex["program"]]
    assert ops[-2:] == ["read_text", "answer"]
    assert "temporal_order" not in ops and "filter" not in ops
    _assert_uses_hint_window_verbatim(ex)
    # the example must face the REAL failure stimulus: a hint that names a (decoy) scorer number
    # — the failing rows lifted bernabeu's 'scorer wears #7' into filter(text == '7'). A hint
    # with no number would demonstrate restraint against nothing.
    assert re.search(r"scorer wears #\d", codegen.EXAMPLE_CLIP["hint"])


def test_few_shot_presence_example_shape():
    """presence (far-3 used the 1ms count-midpoint pattern and grounded out empty): the worked
    program is detect -> answer over the hint's full multi-frame window, never the 1ms midpoint
    and never a count."""
    ex = _example("presence")
    assert [s["op"] for s in ex["program"]] == ["sample_frames", "detect", "answer"]
    win = next(s for s in ex["program"] if s["op"] == "sample_frames")["args"]
    assert win["end_ms"] - win["start_ms"] > 1, "presence must not use the 1ms midpoint pattern"
    _assert_uses_hint_window_verbatim(ex)


def test_few_shot_ground_out_example_shape():
    """injection/noise (adv-injection-huge-window + adv-unicode-rtl-noise leaked grounded '5'
    via the count chain): the example question embeds sampling instructions + count bait, and
    the worked program is the absent-jersey ground-out chain — never count."""
    ex = _example("ground-out")
    ops = [s["op"] for s in ex["program"]]
    assert "count" not in ops
    assert ops[-2:] == ["filter", "answer"]
    filt = next(s for s in ex["program"] if s["op"] == "filter")
    assert filt["args"]["where"] == {"field": "text", "equals": "99"}
    q = ex["question"].lower()
    assert "how many" in q, "the example must carry count bait"
    assert "fps" in q and "ms" in q, "the example must embed sampling instructions"


# --------------------------------------------------------------------------------------
# describe_scene (new primitive) — scene-description shape rule + worked example + scope guard
# --------------------------------------------------------------------------------------

def test_few_shot_scene_example_shape():
    """The new scene-description example must end describe_scene -> answer, with NO detect/count/
    temporal_order, copying the EXAMPLE_CLIP window+fps verbatim (the analyzed-window behavior)."""
    ex = _example("scene-description")
    ops = [s["op"] for s in ex["program"]]
    assert ops == ["sample_frames", "describe_scene", "answer"]
    assert "detect" not in ops and "count" not in ops and "temporal_order" not in ops
    _assert_uses_hint_window_verbatim(ex)


def test_codegen_prompt_has_scene_rule():
    """The prompt must name describe_scene and route 'what is happening' to the scene chain."""
    p = codegen.SYSTEM_PROMPT
    low = p.lower()
    assert "describe_scene" in low
    assert "what is happening" in low
    assert "describe_scene -> answer" in low


def test_codegen_prompt_scene_rule_stays_scoped():
    """Over-refusal regression guard: widening 'what is happening' into scope must NOT pull
    formation/emotion/jersey-COLOR/weather in-scope — they stay listed as NOT answerable."""
    low = codegen.SYSTEM_PROMPT.lower()
    # the out-of-scope attributes remain named in the NOT-answerable list.
    for oos in ("formation", "emotion", "color", "weather"):
        assert oos in low, f"{oos} must stay named (out-of-scope) after the scene widening"
    # the scene widening did not weaken the capability-scoped refusal rule.
    assert "outside the listed capabilities" in low
    assert "is not answerable" in low or "not answerable" in low


def test_few_shot_examples_teach_structure_not_the_test_set():
    """Anti-contamination: no bank question may appear verbatim in the prompt (the eval would
    measure memorization, not generalization), and no real clip timing literal may leak into the
    base prompt (the examples compile against a FICTIONAL clip)."""
    bank_doc = json.loads((Path(codegen.API_DIR) / "eval" / "question_bank.json").read_text())
    for case in bank_doc["cases"]:
        q = case["question"]
        if len(q) >= 15:  # skip degenerate stubs like '?' that match anything
            assert q not in codegen.SYSTEM_PROMPT, f"bank question leaked into prompt: {q!r}"
            # examples render questions via json.dumps — also scan the escaped form, or a
            # quoted/non-ASCII bank question could leak invisibly to the raw scan above.
            assert json.dumps(q) not in codegen.SYSTEM_PROMPT, f"bank question leaked (escaped): {q!r}"
    for clip in CLIPS.values():
        literals = set(re.findall(r"\d{4,}", clip["hint"])) | {str(clip["duration_ms"])}
        for lit in literals:
            assert lit not in codegen.SYSTEM_PROMPT, f"real clip literal leaked into prompt: {lit}"


def test_prompt_size_budget():
    """The few-shot iteration TRIMMED the rulebook (6,751 -> ~4.3k chars) while the whole prompt
    (rules + examples) stayed under the pre-iteration 6,751-char base. Budget-guard the trim so
    rule-accretion can't creep back: shrink a rule or convert it to a worked example instead of
    appending prose."""
    assert len(codegen._RULES) <= 4500, "rulebook crept past its budget — trim or convert to an example"
    assert len(codegen.SYSTEM_PROMPT) <= 6700, "prompt crept past its budget (pre-iteration base: 6,751)"
