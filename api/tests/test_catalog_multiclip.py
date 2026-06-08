"""
Multi-clip / multi-query catalog proof. This module proves the system is REPEATABLE beyond the
single hero path: a SECOND clip (`best-goal`) with its own manifest, fixture-derived replay cache,
pinned programs, and codegen `hint`, plus FOUR new queries that stress op-paths the hero chain
never hits — count, filter-on-a-non-hero-value, an honest out-of-bounds ground-out, and a
read_text readout.

Two patterns, mirrored from the existing suite:
  - registry + interpreter replay over the COMMITTED cache (test_generalized_answer.py):
    Cache.load + Interpreter().run, asserting each query's expected grounded-or-ungrounded findings.
  - mocked precompute derive (test_precompute.py + conftest.py seams): run_precompute over the new
    clip's manifest with a CONFIGURABLE FakeAzureVision(objects=fixture) + injected providers, with
    ZERO live Azure calls (and Azure Read never a rung), asserting the derive reproduces the
    committed cache byte-for-byte (Alt-C, non-circular).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import canned
import precompute
from precompute import (
    OcrLadder, run_precompute, validate_cache_shape, resolve_program_and_out,
)
from interpreter import Cache, Interpreter
from validate_program import strip_comments, validation_errors
from conftest import FakeAzureVision, CallSpyVLM

API_DIR = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
NEW_CLIP = "best-goal"
NEW_CACHE = API_DIR / "examples" / "best-goal_cache.json"
NEW_FIXTURE = FIXTURES / "fake_detect_best-goal.json"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _program_steps(query_id: str) -> list[dict]:
    """The comment-stripped pinned program for a canned query id."""
    entry = canned.by_id(query_id)
    assert entry is not None, f"query '{query_id}' not in canned.QUERIES"
    return strip_comments(json.loads(Path(entry["program"]).read_text()))["program"]


def _run_query_over_committed_cache(query_id: str) -> dict:
    """Replay a query's pinned program over the new clip's COMMITTED cache -> run-doc."""
    entry = canned.by_id(query_id)
    cache = Cache.load(entry["cache"])
    return Interpreter(cache).run(_program_steps(query_id), query=entry["text"])


def _new_clip_objects():
    """Configurable FakeAzureVision objects for the new clip, loaded from the checked-in fixture
    (Alt-C). The DEFAULT FakeAzureVision returns HERO_OBJECTS — this is the configurable variant."""
    from vision.azure_vision import Detection, Box
    data = json.loads(NEW_FIXTURE.read_text())
    return [
        Detection(cls=o["class"], confidence=o["confidence"],
                  box=Box(o["box"]["x"], o["box"]["y"], o["box"]["w"], o["box"]["h"]))
        for o in data["objects"]
    ]


@pytest.fixture
def new_manifest():
    """The new clip's manifest (comment-stripped), `_dir` wired so `source` resolves."""
    cdir = API_DIR.parent / "clips" / NEW_CLIP
    raw = json.loads((cdir / "manifest.json").read_text())
    m = precompute._strip_comments(raw)
    m["_dir"] = cdir
    return m


# --------------------------------------------------------------------------------------
# registry-shape assertions
# --------------------------------------------------------------------------------------

def test_catalog_enumerates_new_clip_and_queries():
    assert len(canned.CLIPS) >= 2, "expected >= 2 clips after adding the second clip"
    assert len(canned.QUERIES) >= 4, "expected >= 4 queries after adding the new queries"
    assert NEW_CLIP in canned.CLIPS, f"new clip '{NEW_CLIP}' missing from CLIPS"
    # every query's clip resolves in CLIPS.
    for q in canned.QUERIES:
        assert q["clip"] in canned.CLIPS, f"query '{q['id']}' references unknown clip '{q['clip']}'"


def test_new_clip_carries_required_fields():
    """label + duration_ms are bracket-accessed in build_system_message; cache in build_run_doc.
    A missing one is a KeyError at request time, so carry the full single-goal field set."""
    clip = canned.CLIPS[NEW_CLIP]
    for k in ("id", "label", "width", "height", "duration_ms", "cache", "hint"):
        assert k in clip and clip[k] not in (None, ""), f"clip['{k}'] missing/empty"
    # the committed cache exists and its clip block carries a non-null fps (validate_cache_shape gate).
    cache = json.loads(Path(clip["cache"]).read_text())
    assert cache["clip"]["fps"] is not None


def test_every_query_program_and_cache_path_exists():
    for q in canned.QUERIES:
        assert Path(q["program"]).exists(), f"program path missing for '{q['id']}': {q['program']}"
        assert Path(q["cache"]).exists(), f"cache path missing for '{q['id']}': {q['cache']}"


def test_every_pinned_program_validates():
    """Each query's program validates structurally + semantically, with numeric bounds enforced
    against its OWN clip (validation_errors with clip=clip, which the CLI cannot do)."""
    for q in canned.QUERIES:
        program = strip_comments(json.loads(Path(q["program"]).read_text()))
        clip = canned.clip_by_id(q["clip"])
        errs = validation_errors(program, clip=clip)
        assert errs == [], f"query '{q['id']}' program failed validation: {errs}"


def test_out_of_bounds_window_is_rejected_with_clip_bounds():
    """The validate_program CLI does NOT bounds-check (no clip), so prove the clip-aware gate
    rejects an end_ms past the new clip's duration_ms — the trust-boundary invariant."""
    clip = canned.clip_by_id(NEW_CLIP)
    over = clip["duration_ms"] + 5000
    bad = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 0, "end_ms": over, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]}
    errs = validation_errors(bad, clip=clip)
    assert any("end_ms" in e and "duration_ms" in e for e in errs), errs


# --------------------------------------------------------------------------------------
# per-query replay assertions (over the COMMITTED, fixture-derived cache)
# --------------------------------------------------------------------------------------

def test_count_query_grounds_to_a_number():
    doc = _run_query_over_committed_cache("best-count-players")
    f = doc["findings"]
    assert f["grounded"] is True
    # the fixture has 5 person dets per frame; the single-frame count window yields exactly 5.
    assert f["answer"] == "5"
    # NEVER the hero verdict.
    assert "#10 scored the first goal" not in (f["verdict"] or "")


def test_filter_nonhero_value_query_grounds_to_its_subject():
    doc = _run_query_over_committed_cache("best-seven-first-goal")
    f = doc["findings"]
    assert f["grounded"] is True
    # subject is DERIVED as #7 from the filter, NOT a literal #10.
    assert f["verdict"] == "Yes — #7 scored the first goal"
    assert "#10" not in (f["verdict"] or "")


def test_out_of_bounds_query_grounds_out_honestly():
    doc = _run_query_over_committed_cache("best-ninetynine-out-of-bounds")
    f = doc["findings"]
    # a real first goal exists, but #99 matches nobody -> honest ground-out, never a fake "No".
    assert f["grounded"] is False
    assert f["verdict"] is None
    assert f["answer"] is None
    assert f["reason"] == "no-grounded-answer"


def test_read_text_readout_query_grounds():
    doc = _run_query_over_committed_cache("best-scorer-number")
    f = doc["findings"]
    assert f["grounded"] is True
    # the committed cache pins p_seven's jersey OCR to "7"; the readout reads it back.
    assert "7" in f["answer"]
    assert "#10 scored the first goal" not in (f["verdict"] or "")


def test_new_clip_cache_replays_its_grounding_query():
    """The clip's grounding (positive-claim) query replays over the COMMITTED cache to a grounded
    Yes — the end-to-end repeatability proof for the second clip."""
    doc = _run_query_over_committed_cache("best-seven-first-goal")
    # run-doc validates against the schema.
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(doc))
    assert not errs, errs
    # the scorer survives filter text==7 and reads "7".
    cache = Cache.load(NEW_CACHE)
    scorer = cache.goal_events()[0]["scorer_det"]
    assert scorer == "p_seven"
    assert cache.read_text_for(scorer)["text"] == "7"
    assert doc["findings"]["grounded"] is True


# --------------------------------------------------------------------------------------
# mocked precompute derive (Alt-C) — zero live Azure, reproduces the committed cache
# --------------------------------------------------------------------------------------

def test_new_clip_precompute_derives_cache_from_fixture(new_manifest, fake_frame_provider, fake_crop_provider, tmp_path):
    """Run run_precompute over the new clip's manifest with a CONFIGURABLE FakeAzureVision loaded
    from the checked-in detect fixture + injected providers. Assert exit 0, a shape-valid cache,
    that it EQUALS the committed best-goal_cache.json (non-circular proof), and that Azure saw
    read=False only (zero live Azure / Read calls)."""
    out = tmp_path / "best-goal_cache.json"
    vision = FakeAzureVision(objects=_new_clip_objects())
    rep = run_precompute(
        new_manifest, out=out,
        vision=vision,
        vlm_ladder=OcrLadder(),
        frame_provider=fake_frame_provider,
        jersey_crop_provider=fake_crop_provider,
        program_path=API_DIR / "examples" / "best-goal_seven_program.json",
        emit_frames=False,
    )
    assert rep["exit"] == 0
    assert rep["vlm_calls"] == 0  # the pin wins -> zero VLM calls

    produced = json.loads(out.read_text())
    assert validate_cache_shape(produced) == []
    # Alt-C: the committed artifact IS this derive's output.
    committed = json.loads(NEW_CACHE.read_text())
    assert produced == committed, "derived cache diverged from the committed best-goal_cache.json"

    # Azure Read was never a rung (FakeAzureVision asserts read=False internally); detection-only.
    assert vision.analyze_calls >= 1
    assert CallSpyVLM.total_calls == 0


def test_new_clip_resolver_threads_program_and_out(tmp_path):
    """The Phase-4 resolver maps (--clip, --query) -> (program_path, out) from canned.QUERIES so a
    NON-hero clip self-validates its OWN pinned program and writes its OWN cache path. Drive the
    resolver directly (the --dry-run command returns before self_validate_replay, so it doesn't
    exercise this)."""
    # explicit --query -> that query's program + cache.
    prog, out = resolve_program_and_out(NEW_CLIP, "best-seven-first-goal")
    assert prog == Path(canned.by_id("best-seven-first-goal")["program"])
    assert out == Path(canned.by_id("best-seven-first-goal")["cache"])

    # --clip only -> the clip's GROUNDING (temporal_order) query is picked.
    prog2, out2 = resolve_program_and_out(NEW_CLIP, None)
    assert "temporal_order" in {s["op"] for s in strip_comments(json.loads(prog2.read_text()))["program"]}
    assert out2 == Path(canned.by_id("best-seven-first-goal")["cache"])

    # an explicit --out override wins for the path; the program still resolves per the query.
    override = tmp_path / "elsewhere.json"
    prog3, out3 = resolve_program_and_out(NEW_CLIP, "best-count-players", override)
    assert out3 == override
    assert prog3 == Path(canned.by_id("best-count-players")["program"])

    # hero defaults unchanged (no args).
    prog4, out4 = resolve_program_and_out(None, None)
    assert prog4 == API_DIR / "examples" / "hero_program.json"
    assert out4 == precompute.DEFAULT_OUT


def test_new_clip_precompute_self_validates_grounding_program(new_manifest, fake_frame_provider, fake_crop_provider, tmp_path):
    """Non-dry mocked run threading the grounding program: the generalized self_validate_replay
    must accept it (an `answer` step + grounded findings). Writes to tmp_path (not the committed
    artifact) so this test doesn't regenerate the cache."""
    out = tmp_path / "cache.json"
    rep = run_precompute(
        new_manifest, out=out,
        vision=FakeAzureVision(objects=_new_clip_objects()),
        vlm_ladder=OcrLadder(),
        frame_provider=fake_frame_provider,
        jersey_crop_provider=fake_crop_provider,
        program_path=API_DIR / "examples" / "best-goal_seven_program.json",
        emit_frames=False,
    )
    # self_validate_replay ran inside run_precompute (it raises on failure); reaching here + the
    # grounded verdict proves the generic validator accepted a clip whose claim grounds.
    assert rep["exit"] == 0
    assert rep["verdict"] == "Yes — #7 scored the first goal"


def test_self_validate_replay_accepts_count_program(new_manifest, fake_frame_provider, fake_crop_provider, tmp_path):
    """The generalized self_validate_replay accepts a NON-temporal program: a count->answer chain
    over the new clip's cache validates without a temporal_order step (the old hero-shaped guard
    would have raised 'program lacks temporal_order/answer step'). The count window grounds to a
    number, so with the clip's positive claim it still asserts findings.grounded is True."""
    out = tmp_path / "cache.json"
    # First derive the cache (so self_validate has detect to replay against), then re-run
    # self_validate_replay directly with the count program.
    run_precompute(
        new_manifest, out=out,
        vision=FakeAzureVision(objects=_new_clip_objects()),
        vlm_ladder=OcrLadder(),
        frame_provider=fake_frame_provider,
        jersey_crop_provider=fake_crop_provider,
        program_path=API_DIR / "examples" / "best-goal_seven_program.json",
        emit_frames=False,
    )
    cache = json.loads(out.read_text())
    run_doc = precompute.self_validate_replay(
        cache,
        program_path=API_DIR / "examples" / "best-goal_count_program.json",
        expect_grounded=True,
    )
    assert run_doc["findings"]["grounded"] is True
    assert run_doc["findings"]["answer"] == "5"
    # there is no temporal_order step in this program.
    assert all(t["op"] != "temporal_order" for t in run_doc["trace"])


# --------------------------------------------------------------------------------------
# codegen repeatability (mocked free-text path) — the new clip's hint reaches the prompt
# --------------------------------------------------------------------------------------

def test_build_system_message_injects_new_clip_hint():
    """Codegen-repeatability across clips, proven WITHOUT live Azure: build_system_message(new_clip)
    injects the new clip's label + duration + per-clip hint into the codegen prompt (Q2 -> Alt C),
    so a free-text question is compiled against the SECOND clip's context, not the hero's."""
    import codegen
    clip = canned.CLIPS[NEW_CLIP]
    msg = codegen.build_system_message(clip)
    assert clip["hint"] in msg
    assert clip["label"] in msg
    assert str(clip["duration_ms"]) in msg
    # the hero hint must NOT leak into the new clip's prompt.
    assert canned.CLIPS["single-goal"]["hint"] not in msg


def test_free_text_path_compiles_new_clip_program_against_its_hint(mock_codegen):
    """End-to-end (mocked-codegen) free-text repeatability: build_free_text_run_doc for the new
    clip drives codegen with clip=best-goal, validates the returned program, and replays it over
    the new clip's cache to a grounded answer — zero live Azure (FakeCodegen records the clip)."""
    import server
    from conftest import FakeCodegen
    program = strip_comments(
        json.loads((API_DIR / "examples" / "best-goal_count_program.json").read_text())
    )["program"]
    mock_codegen(program=program)

    doc = server.build_free_text_run_doc("How many players are visible?", NEW_CLIP)
    # codegen was asked to compile against the NEW clip (its hint/label reach the prompt builder).
    assert FakeCodegen.last_clip is not None
    assert FakeCodegen.last_clip["id"] == NEW_CLIP
    assert doc["program_source"] == "live"
    assert doc["findings"]["grounded"] is True
    assert doc["findings"]["answer"] == "5"
