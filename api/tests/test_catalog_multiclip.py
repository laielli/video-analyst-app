"""
Multi-clip / multi-query catalog suite (plan: "prove repeatability beyond the single hero path").

Proves the registries + interpreter + precompute pipeline generalize to a SECOND clip (best-goal)
and query shapes the hero chain never hits: count / filter-on-a-non-hero-value / out-of-bounds /
read_text readout. Every assertion replays deterministically with ZERO live Azure calls.

Two patterns are mirrored:
  - test_generalized_answer.py — Cache.load + Interpreter().run over a COMMITTED cache.
  - test_precompute.py        — mocked pipeline (FakeAzureVision + injected providers), zero
                                 real Azure/ffmpeg.

The best-goal cache is DERIVED (Alt C) from a checked-in detect fixture, so the grounding proof
is non-circular: the fixture sets the detect boxes, the manifest sets the pin/event, and the
committed cache is reproduced byte-for-byte by run_precompute over that fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import canned
import precompute
from precompute import OcrLadder, run_precompute, resolve_program_and_out, validate_cache_shape
from interpreter import Cache, Interpreter
from validate_program import strip_comments, validation_errors
from conftest import FakeAzureVision, CallSpyVLM

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

NEW_CLIP = "best-goal"
NEW_CACHE = API_DIR / "examples" / "best-goal_cache.json"
FAKE_DETECT = FIXTURES / "fake_detect_best-goal.json"


# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------

def _new_clip_objects():
    """The checked-in detect fixture as real Detection/Box dataclasses, for FakeAzureVision(objects=)."""
    from vision.azure_vision import Detection, Box
    raw = json.loads(FAKE_DETECT.read_text())["objects"]
    return [
        Detection(cls=o["class"], confidence=o["confidence"],
                  box=Box(o["box"]["x"], o["box"]["y"], o["box"]["w"], o["box"]["h"]))
        for o in raw
    ]


@pytest.fixture
def new_manifest():
    """The best-goal manifest (comment-stripped), source neutralized so no .mov is needed."""
    raw = json.loads((REPO_ROOT / "clips" / NEW_CLIP / "manifest.json").read_text())
    m = precompute._strip_comments(raw)
    m["_dir"] = REPO_ROOT / "clips" / NEW_CLIP
    return m


@pytest.fixture
def new_frame_provider():
    def provider(ts):
        return b"\x89PNG\r\n\x1a\n" + ts.to_bytes(4, "big")
    return provider


@pytest.fixture
def new_crop_provider():
    def provider(ts, box):
        return b"\x89PNG-crop" + ts.to_bytes(4, "big")
    return provider


def _query(qid: str) -> dict:
    q = canned.by_id(qid)
    assert q is not None, f"query '{qid}' not registered in canned.QUERIES"
    return q


def _replay(qid: str) -> dict:
    """Replay a query's pinned program over the COMMITTED best-goal cache -> findings."""
    q = _query(qid)
    cache = Cache.load(q["cache"])
    program = strip_comments(json.loads(Path(q["program"]).read_text()))["program"]
    return Interpreter(cache).run(program, query=q["text"])


# --------------------------------------------------------------------------------------
# registry shape
# --------------------------------------------------------------------------------------

def test_catalog_enumerates_new_clip_and_queries():
    assert len(canned.CLIPS) >= 2, "expected at least 2 clips after the multi-clip catalog"
    assert len(canned.QUERIES) >= 4, "expected at least 4 queries after the multi-clip catalog"
    assert NEW_CLIP in canned.CLIPS
    # every query's clip resolves in CLIPS.
    for q in canned.QUERIES:
        assert q["clip"] in canned.CLIPS, f"query '{q['id']}' references unknown clip '{q['clip']}'"


def test_new_clip_carries_full_field_set():
    """Required-field invariant: label + duration_ms (bracket-read in codegen.build_system_message)
    and cache (bracket-read in server.build_run_doc) must be present or a request KeyErrors."""
    clip = canned.CLIPS[NEW_CLIP]
    for k in ("id", "label", "width", "height", "duration_ms", "cache", "hint"):
        assert k in clip, f"best-goal clip missing required field '{k}'"
    assert clip["duration_ms"] is not None  # validate_cache_shape rejects null fps/duration


def test_every_query_program_and_cache_path_exists():
    for q in canned.QUERIES:
        assert Path(q["program"]).exists(), f"{q['id']}: program path missing {q['program']}"
        assert Path(q["cache"]).exists(), f"{q['id']}: cache path missing {q['cache']}"


def test_every_pinned_program_validates():
    for q in canned.QUERIES:
        clip = canned.clip_by_id(q["clip"])
        raw = json.loads(Path(q["program"]).read_text())
        errs = validation_errors(raw, clip=clip)
        assert errs == [], f"{q['id']}: program failed validation: {errs}"


def test_validate_program_enforces_bounds_with_clip():
    """The CLI validate_program.py calls semantic_errors with NO clip, so it can't enforce
    end_ms <= duration_ms. This asserts the bound IS enforced when a clip is passed (the path the
    server/free-text trust boundary uses): an out-of-range window is rejected against best-goal."""
    clip = canned.clip_by_id(NEW_CLIP)
    over = {"program": [
        {"id": "frames", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": clip["duration_ms"] + 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]}
    errs = validation_errors(over, clip=clip)
    assert any("duration_ms" in e for e in errs), errs
    # the same program WITHIN bounds validates.
    over["program"][0]["args"]["end_ms"] = clip["duration_ms"]
    assert validation_errors(over, clip=clip) == []


# --------------------------------------------------------------------------------------
# query-shape replays over the COMMITTED (fixture-derived) cache
# --------------------------------------------------------------------------------------

def test_count_query_grounds_to_a_number():
    f = _replay("best-goal-count-players")["findings"]
    assert f["grounded"] is True
    # the fixture carries 5 person dets per frame; the count program samples ts 9000 -> 5.
    assert f["answer"] == "5"
    # honesty: NEVER the hero first-scorer verdict.
    assert "#10 scored the first goal" not in (f["verdict"] or "")


def test_filter_nonhero_value_query_grounds_to_its_subject():
    f = _replay("best-goal-seven-first-goal")["findings"]
    assert f["grounded"] is True
    # the verdict names the query's subject #7, derived from the filter — NOT #10.
    assert f["verdict"] == "Yes — #7 scored the first goal"
    assert "#10" not in (f["verdict"] or "")


def test_out_of_bounds_query_grounds_out_honestly():
    f = _replay("best-goal-ninetynine-first-goal")["findings"]
    assert f["grounded"] is False
    assert f["verdict"] is None
    assert f["answer"] is None
    assert f["reason"] == "no-grounded-answer"


def test_read_text_readout_query_grounds():
    f = _replay("best-goal-scorer-number")["findings"]
    assert f["grounded"] is True
    # the bound cache carries read_text[p_seven].text == '7'; the readout reads it back.
    assert "7" in (f["answer"] or "")
    assert "#10 scored the first goal" not in (f["verdict"] or "")


def test_new_clip_cache_replays_its_grounding_query():
    """The clip's grounding query (filter #7) replays over the COMMITTED cache to its grounded
    verdict — the multi-clip analogue of the hero replay test."""
    doc = _replay("best-goal-seven-first-goal")
    answer = next(t for t in doc["trace"] if t["op"] == "answer")
    assert answer["output_label"] == "Yes"
    tens = next(t for t in doc["trace"] if t["op"] == "filter")
    assert tens["status"] == "done"
    cache = Cache.load(NEW_CACHE)
    scorer = cache.goal_events()[0]["scorer_det"]
    assert scorer == "p_seven"
    rt = cache.read_text_for(scorer)
    assert rt["text"] == "7" and rt["source"] == "pinned"
    # run-doc conforms to the schema.
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    assert not list(jsonschema.Draft202012Validator(rd_schema).iter_errors(doc))


# --------------------------------------------------------------------------------------
# precompute: derive the committed cache from the checked-in fixture (Alt C), zero live Azure
# --------------------------------------------------------------------------------------

def test_new_clip_precompute_derives_cache_from_fixture(tmp_path, new_manifest, new_frame_provider, new_crop_provider):
    """Run run_precompute on the best-goal manifest with a per-clip FakeAzureVision(objects=...)
    loaded from the checked-in fixture; assert exit 0, the produced cache passes
    validate_cache_shape, AND equals the committed best-goal_cache.json byte-for-byte (proving the
    artifact is reproducible, not hand-authored). FakeAzureVision must only ever see read=False."""
    out = tmp_path / "best-goal_cache.json"
    program_path, _registered_out = resolve_program_and_out(NEW_CLIP, "best-goal-seven-first-goal")
    vision = FakeAzureVision(objects=_new_clip_objects())
    rep = run_precompute(
        new_manifest, out=out, program_path=program_path,
        vision=vision, vlm_ladder=OcrLadder(),
        frame_provider=new_frame_provider, jersey_crop_provider=new_crop_provider,
        emit_frames=False,
    )
    assert rep["exit"] == 0
    assert rep["vlm_calls"] == 0  # the #7 jersey is pinned -> no VLM call

    produced = json.loads(out.read_text())
    assert validate_cache_shape(produced) == []

    committed = json.loads(NEW_CACHE.read_text())
    assert produced == committed, "derived cache drifted from the committed best-goal_cache.json"

    # The committed file is byte-identical too (canonical writer is deterministic).
    assert out.read_text() == NEW_CACHE.read_text()

    # Azure Read was never a rung: FakeAzureVision.analyze asserts `not read` on every call, so a
    # successful run already proves read=False suite-wide. analyze was called once per sampled frame.
    assert vision.analyze_calls == len(rep["detect_frames"]) == 41


def test_new_clip_precompute_makes_zero_vlm_calls(tmp_path, new_manifest, new_frame_provider, new_crop_provider):
    """A spy VLM that would say '99' is never consulted, because the scorer jersey is pinned."""
    spy = CallSpyVLM(digits="99")
    ladder = OcrLadder()
    ladder._vlm = spy
    ladder._vlm_attempted = True
    out = tmp_path / "cache.json"
    program_path, _ = resolve_program_and_out(NEW_CLIP, "best-goal-seven-first-goal")
    run_precompute(
        new_manifest, out=out, program_path=program_path,
        vision=FakeAzureVision(objects=_new_clip_objects()), vlm_ladder=ladder,
        frame_provider=new_frame_provider, jersey_crop_provider=new_crop_provider, emit_frames=False,
    )
    assert spy.calls == 0
    cache = Cache.load(out)
    assert cache.read_text_for("p_seven")["text"] == "7"


# --------------------------------------------------------------------------------------
# precompute resolver wiring (Phase 4) — the path --dry-run does NOT exercise
# --------------------------------------------------------------------------------------

def test_new_clip_resolver_threads_program_and_out():
    """resolve_program_and_out maps --clip/--query to the right pinned program + cache path."""
    # explicit query wins.
    prog, out = resolve_program_and_out(NEW_CLIP, "best-goal-seven-first-goal")
    assert prog == Path(canned.by_id("best-goal-seven-first-goal")["program"])
    assert out == Path(canned.by_id("best-goal-seven-first-goal")["cache"])

    # clip alone resolves the FIRST registered query for that clip (the count query).
    prog2, out2 = resolve_program_and_out(NEW_CLIP, None)
    first_for_clip = next(q for q in canned.QUERIES if q["clip"] == NEW_CLIP)
    assert prog2 == Path(first_for_clip["program"])

    # no clip, no query -> hero defaults (back-compat).
    prog3, out3 = resolve_program_and_out(None, None)
    assert prog3.name == "hero_program.json"
    assert out3.name == "hero_cache.json"


def test_resolver_unknown_query_is_setup_error():
    from precompute import SetupError
    with pytest.raises(SetupError):
        resolve_program_and_out(NEW_CLIP, "no-such-query")


def test_resolver_threaded_run_self_validates_grounding(tmp_path, new_manifest, new_frame_provider, new_crop_provider):
    """Drive the resolver NON-dry with mocked seams so the self_validate_replay path (which
    --dry-run skips) actually runs: the resolved grounding program must self-validate to a grounded
    answer, and expect_grounded follows the manifest's `events` presence."""
    assert new_manifest.get("events"), "best-goal manifest must declare a goal event"
    program_path, _ = resolve_program_and_out(NEW_CLIP, "best-goal-seven-first-goal")
    out = tmp_path / "cache.json"
    rep = run_precompute(
        new_manifest, out=out, program_path=program_path,
        vision=FakeAzureVision(objects=_new_clip_objects()), vlm_ladder=OcrLadder(),
        frame_provider=new_frame_provider, jersey_crop_provider=new_crop_provider, emit_frames=False,
    )
    # run_precompute raises GroundingError if self_validate fails; reaching exit 0 proves it ran + grounded.
    assert rep["exit"] == 0
    assert rep["verdict"] == "Yes — #7 scored the first goal"


def test_generalized_self_validate_accepts_count_grounding_program(tmp_path, new_manifest, new_frame_provider, new_crop_provider):
    """The generalized self_validate_replay (Phase 4) must accept a NON-temporal grounding program.
    The count program (count -> answer, no temporal_order step) self-validates because the manifest
    declares an event (expect_grounded) and the count grounds to a number."""
    program_path, _ = resolve_program_and_out(NEW_CLIP, "best-goal-count-players")
    assert program_path.name == "best-goal_count_program.json"
    out = tmp_path / "cache.json"
    rep = run_precompute(
        new_manifest, out=out, program_path=program_path,
        vision=FakeAzureVision(objects=_new_clip_objects()), vlm_ladder=OcrLadder(),
        frame_provider=new_frame_provider, jersey_crop_provider=new_crop_provider, emit_frames=False,
    )
    assert rep["exit"] == 0  # no temporal_order step, yet self-validation passed


# --------------------------------------------------------------------------------------
# codegen repeatability across clips (mocked free-text path — no live Azure)
# --------------------------------------------------------------------------------------

def test_codegen_prompt_injects_new_clip_hint():
    """Canned queries are clip-blind (server.py _live_run_doc calls generate(text) with no clip),
    so codegen-repeatability across clips is proven on the FREE-TEXT path: build_system_message
    injects the NEW clip's hint/label/duration into the prompt, distinct from the hero's. No live
    Azure call is needed — this asserts the prompt-injection seam directly."""
    import codegen
    hero_msg = codegen.build_system_message(canned.CLIPS["single-goal"])
    new_msg = codegen.build_system_message(canned.CLIPS[NEW_CLIP])
    assert new_msg != hero_msg
    assert canned.CLIPS[NEW_CLIP]["hint"] in new_msg
    assert canned.CLIPS[NEW_CLIP]["label"] in new_msg
    assert "30086" in new_msg  # best-goal duration_ms, distinct from the hero's 5067
    assert canned.CLIPS["single-goal"]["hint"] not in new_msg
