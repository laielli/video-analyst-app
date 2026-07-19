"""
Multi-clip / multi-query catalog proof. Exercises the registries + the precompute pipeline +
the interpreter against a SECOND clip (bernabeu-counter) and FOUR query shapes the hero chain
never hits: count, filter-on-a-non-hero-value, an out-of-bounds honest-ungrounded question, and
a read_text readout. Zero live Azure calls: the new clip's cache is DERIVED (Alt C) by running
run_precompute over a checked-in detect fixture under a configurable FakeAzureVision, then the
replay tests ground each query off that derived cache.

Pattern mirror: test_generalized_answer.py (Cache.load + Interpreter().run over a committed
cache) + test_precompute.py (mocked pipeline via FakeAzureVision + injected providers).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import precompute
from precompute import (
    OcrLadder, run_precompute, resolve_clip_query, validate_cache_shape,
)
from interpreter import Cache, Interpreter
from validate_program import strip_comments, validation_errors
from vision.azure_vision import AnalysisResult, Detection, Box
import canned

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
EX = API_DIR / "examples"

NEW_CLIP = "bernabeu-counter"
NEW_CACHE = EX / "bernabeu-counter_cache.json"
GROUNDING_QUERY = "bernabeu-7-first-goal"  # the temporal grounding program self-validated by precompute


# --------------------------------------------------------------------------------------
# helpers — the Alt-C fixture-derived seam (FakeAzureVision(objects=...) from the fixture)
# --------------------------------------------------------------------------------------

def _bernabeu_objects() -> list[Detection]:
    """Reuse the real Detection dataclass so the fake is structurally identical to production,
    loaded from the checked-in detect fixture that drives the Alt-C derive."""
    fix = json.loads((FIXTURES / "fake_detect_bernabeu.json").read_text())
    return [Detection(cls=o["class"], confidence=o["confidence"], box=Box(**o["box"]))
            for o in fix["objects"]]


class _ConfigurableVision:
    """A FakeAzureVision-shaped stand-in seeded with the bernabeu fixture objects (the conftest
    default returns HERO_OBJECTS for every manifest; Alt C needs the per-clip objects= variant).
    analyze() returns the same objects for every frame and records read= so the zero-Azure-Read
    invariant is assertable suite-wide."""

    def __init__(self, objects):
        self.objects = list(objects)
        self.analyze_calls = 0
        self.read_flags: list[bool] = []

    def analyze(self, image_bytes, detect=True, read=False):
        assert not read, "Azure Read must NOT be a rung in the OCR ladder (Resolved #2)"
        self.read_flags.append(read)
        self.analyze_calls += 1
        return AnalysisResult(
            width=2636, height=1474,
            objects=list(self.objects) if detect else [],
            lines=[],
        )


@pytest.fixture
def bernabeu_manifest():
    """The committed bernabeu-counter manifest (comment-stripped), `source` left intact but never
    touched (providers are injected, so no .mov is needed)."""
    raw = json.loads((REPO_ROOT / "clips" / NEW_CLIP / "manifest.json").read_text())
    m = precompute._strip_comments(raw)
    m["_dir"] = REPO_ROOT / "clips" / NEW_CLIP
    return m


@pytest.fixture
def fake_frame_provider():
    def provider(ts):
        return b"\x89PNG\r\n\x1a\n" + ts.to_bytes(4, "big")
    return provider


@pytest.fixture
def fake_crop_provider():
    def provider(ts, box):
        return b"\x89PNG-crop" + ts.to_bytes(4, "big")
    return provider


def _derive_cache(manifest, out, frame_provider, crop_provider, *, program_path, vision=None):
    """Run the pipeline with mocked Azure + injected providers, writing to `out`. Returns
    (report, vision) so tests can assert read=False was the only flag the fake ever saw."""
    vision = vision or _ConfigurableVision(_bernabeu_objects())
    rep = run_precompute(
        manifest, out=out, vision=vision, vlm_ladder=OcrLadder(),
        frame_provider=frame_provider, jersey_crop_provider=crop_provider,
        program_path=program_path, emit_frames=False,
    )
    return rep, vision


def _program(query_id: str) -> list[dict]:
    entry = canned.by_id(query_id)
    return strip_comments(json.loads(Path(entry["program"]).read_text()))["program"]


def _replay(query_id: str, cache: Cache) -> dict:
    entry = canned.by_id(query_id)
    return Interpreter(cache).run(_program(query_id), query=entry["text"])


# --------------------------------------------------------------------------------------
# registry shape
# --------------------------------------------------------------------------------------

def test_catalog_enumerates_new_clip_and_queries():
    assert len(canned.CLIPS) >= 2
    assert len(canned.QUERIES) >= 4
    assert NEW_CLIP in canned.CLIPS
    # every query binds to a real clip.
    for q in canned.QUERIES:
        assert q["clip"] in canned.CLIPS, f"query {q['id']} binds to unknown clip {q['clip']}"


def test_new_clip_carries_required_fields():
    # label + duration_ms are bracket-accessed in build_system_message; cache in build_run_doc.
    # A missing one is a KeyError at request time — carry the full single-goal field set.
    clip = canned.CLIPS[NEW_CLIP]
    for k in ("id", "label", "width", "height", "duration_ms", "cache", "hint"):
        assert k in clip, f"new clip missing required field {k!r}"
    # cache's clip block must carry non-null fps or validate_cache_shape rejects it.
    cache_data = json.loads(Path(clip["cache"]).read_text())
    assert cache_data["clip"]["fps"] is not None


def test_every_query_program_and_cache_path_exists():
    for q in canned.QUERIES:
        assert Path(q["program"]).exists(), f"missing program for {q['id']}: {q['program']}"
        assert Path(q["cache"]).exists(), f"missing cache for {q['id']}: {q['cache']}"


def test_every_pinned_program_validates():
    for q in canned.QUERIES:
        program = strip_comments(json.loads(Path(q["program"]).read_text()))["program"]
        clip = canned.clip_by_id(q["clip"])
        errs = validation_errors({"program": program}, clip=clip)
        assert errs == [], f"{q['id']} program has validation errors: {errs}"


def test_pinned_program_bounds_enforced_with_clip():
    # validate_program's CLI skips numeric bounds (no clip); the registry path passes clip=clip so
    # end_ms <= duration_ms IS enforced. Prove an out-of-range window is rejected for the new clip.
    clip = canned.clip_by_id(NEW_CLIP)
    bad = {"program": [
        {"id": "frames", "op": "sample_frames",
         "args": {"start_ms": 0, "end_ms": clip["duration_ms"] + 5000, "fps": 8}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
        {"id": "n", "op": "count", "args": {"items": "people"}},
        {"id": "result", "op": "answer", "args": {"from": "n", "question": "how many?"}},
    ]}
    errs = validation_errors(bad, clip=clip)
    assert any("end_ms" in e and "duration_ms" in e for e in errs), errs


# --------------------------------------------------------------------------------------
# query-shape replay over the COMMITTED (derived) cache
# --------------------------------------------------------------------------------------

def test_count_query_grounds_to_a_number():
    cache = Cache.load(NEW_CACHE)
    doc = _replay("bernabeu-count-players", cache)
    f = doc["findings"]
    assert f["grounded"] is True
    # the fixture carries 5 person detections; the single-frame count window samples exactly one
    # frame, so the count is 5 — NOT multiplied across the window, and NOT the hero verdict.
    assert f["answer"] == "5"
    assert "#10 scored the first goal" not in (f["verdict"] or "")


def test_filter_nonhero_value_query_grounds_to_its_subject():
    cache = Cache.load(NEW_CACHE)
    doc = _replay("bernabeu-7-first-goal", cache)
    f = doc["findings"]
    assert f["grounded"] is True
    # the verdict names the query's subject #7, derived from the filter — NEVER the hero #10.
    assert "#7" in f["verdict"]
    assert "#10" not in f["verdict"]
    assert f["verdict"] == "Yes — #7 scored the first goal"


def test_out_of_bounds_query_grounds_out_honestly():
    cache = Cache.load(NEW_CACHE)
    doc = _replay("bernabeu-23-first-goal", cache)
    f = doc["findings"]
    # #23 is not in the cache -> empty subject -> honest ungrounded, not a fabricated 'No'.
    assert f["grounded"] is False
    assert f["verdict"] is None
    assert f["answer"] is None
    assert f["reason"] == "no-grounded-answer"


def test_read_text_readout_query_grounds():
    cache = Cache.load(NEW_CACHE)
    # the bound cache MUST carry a read_text[p_vini] entry whose text is the queried number.
    assert cache.read_text_for("p_vini")["text"] == "7"
    doc = _replay("bernabeu-scorer-number", cache)
    f = doc["findings"]
    assert f["grounded"] is True
    assert "7" in f["answer"]
    assert "#10 scored the first goal" not in (f["verdict"] or "")


def test_new_clip_cache_replays_its_grounding_query():
    # the committed cache replays the clip's grounding (temporal) query to the grounded chain.
    cache = Cache.load(NEW_CACHE)
    doc = _replay(GROUNDING_QUERY, cache)
    f = doc["findings"]
    assert f["grounded"] is True
    goals = cache.goal_events()
    scorer = goals[0]["scorer_det"]
    assert scorer == "p_vini"
    rt = cache.read_text_for(scorer)
    assert rt["text"] == "7" and rt["source"] == "pinned"
    # run-doc validates against the schema.
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(doc))
    assert not errs, errs


# --------------------------------------------------------------------------------------
# Alt-C: the committed cache is DERIVED from the fixture (non-circular proof)
# --------------------------------------------------------------------------------------

def test_new_clip_precompute_derives_cache_from_fixture(
    tmp_path, bernabeu_manifest, fake_frame_provider, fake_crop_provider
):
    out = tmp_path / "derived.json"
    program_path = Path(canned.by_id(GROUNDING_QUERY)["program"])
    rep, vision = _derive_cache(
        bernabeu_manifest, out, fake_frame_provider, fake_crop_provider, program_path=program_path
    )
    assert rep["exit"] == 0
    derived = json.loads(out.read_text())
    assert validate_cache_shape(derived) == []
    # the derived cache EQUALS the committed artifact (deterministic from the fixture).
    committed = json.loads(NEW_CACHE.read_text())
    assert derived == committed, "derived cache drifted from the committed bernabeu-counter_cache.json"
    # FakeAzureVision only ever saw read=False (Azure Read is not a rung).
    assert vision.read_flags and all(r is False for r in vision.read_flags)
    # the pin won with ZERO VLM calls (p_vini is pinned).
    assert rep["vlm_calls"] == 0


def test_new_clip_precompute_self_validates_grounded(
    tmp_path, bernabeu_manifest, fake_frame_provider, fake_crop_provider
):
    # the generalized self_validate_replay must pass for the new clip's temporal grounding program
    # (manifest declares goal events -> expect_grounded=True -> findings.grounded asserted True).
    out = tmp_path / "derived.json"
    program_path = Path(canned.by_id(GROUNDING_QUERY)["program"])
    rep, _ = _derive_cache(
        bernabeu_manifest, out, fake_frame_provider, fake_crop_provider, program_path=program_path
    )
    assert rep["exit"] == 0
    assert rep["verdict"] == "Yes — #7 scored the first goal"


# --------------------------------------------------------------------------------------
# Phase-4 resolver: --clip/--query -> (program_path, out)
# --------------------------------------------------------------------------------------

def test_resolver_threads_program_and_out_for_new_clip():
    # the resolver maps the new clip's grounding query to its pinned program + committed cache out.
    program_path, out = resolve_clip_query(NEW_CLIP, GROUNDING_QUERY)
    assert program_path == Path(canned.by_id(GROUNDING_QUERY)["program"])
    assert out == Path(canned.by_id(GROUNDING_QUERY)["cache"])
    # --clip alone (no --query match) falls back to the first query for that clip.
    program_path2, out2 = resolve_clip_query(NEW_CLIP, "not-a-real-query")
    first_for_clip = next(q for q in canned.QUERIES if q["clip"] == NEW_CLIP)
    assert program_path2 == Path(first_for_clip["program"])
    assert out2 == Path(first_for_clip["cache"])


def test_resolver_is_hero_backcompat():
    # no clip/query -> hero defaults (back-compat: no-arg precompute is unchanged).
    program_path, out = resolve_clip_query(None, None)
    assert program_path == EX / "hero_program.json"
    assert out == precompute.DEFAULT_OUT
    # the default --query (hero) also resolves to the hero program + hero cache.
    hp, ho = resolve_clip_query("single-goal", "hero-10-first-goal")
    assert hp == Path(canned.by_id("hero-10-first-goal")["program"])
    assert ho == Path(canned.by_id("hero-10-first-goal")["cache"])


def test_new_clip_resolver_self_validates_via_run_precompute(
    tmp_path, bernabeu_manifest, fake_frame_provider, fake_crop_provider
):
    # Drive the resolver NON-dry (the dry-run path returns before self_validate_replay, so the
    # resolver's self-validation is NOT exercised by --dry-run). Resolve the program_path the same
    # way main() does, write to tmp_path, assert it self-validates (grounded) without regenerating
    # the committed artifact, and that expect_grounded follows the manifest's events presence.
    program_path, _ = resolve_clip_query(NEW_CLIP, GROUNDING_QUERY)
    out = tmp_path / "resolver.json"
    rep, _ = _derive_cache(
        bernabeu_manifest, out, fake_frame_provider, fake_crop_provider, program_path=program_path
    )
    assert rep["exit"] == 0
    assert bool(bernabeu_manifest.get("events")) is True  # manifest declares events -> grounded required
    assert out.exists()


# --------------------------------------------------------------------------------------
# generalized self_validate_replay: count/read_text programs are valid targets now
# --------------------------------------------------------------------------------------

def test_self_validate_replay_accepts_non_temporal_grounding_program():
    # The prior gate REQUIRED a temporal_order step; the generalized version requires only an
    # `answer` step and asserts findings.grounded. A count->answer program over the derived cache
    # is now a valid self-validation target (proves the de-hero-shaping).
    cache = json.loads(NEW_CACHE.read_text())
    count_program = Path(canned.by_id("bernabeu-count-players")["program"])
    run_doc = precompute.self_validate_replay(cache, program_path=count_program, expect_grounded=True)
    assert run_doc["findings"]["grounded"] is True
    assert run_doc["findings"]["answer"] == "5"


def test_self_validate_replay_raises_on_missing_answer_step():
    cache = json.loads(NEW_CACHE.read_text())
    # a program with no answer step (write a temp program file) must raise.
    import tempfile
    prog = {"program": [
        {"id": "frames", "op": "sample_frames", "args": {"start_ms": 10000, "end_ms": 10001, "fps": 1}},
        {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
    ]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(prog, f)
        ppath = Path(f.name)
    with pytest.raises(precompute.GroundingError, match="answer step"):
        precompute.self_validate_replay(cache, program_path=ppath, expect_grounded=False)


# --------------------------------------------------------------------------------------
# CLI dry-run sanity for the new clip
# --------------------------------------------------------------------------------------

def test_new_clip_dry_run_via_cli(tmp_path):
    # the new clip's dry-run samples its window, exits 0, writes nothing (no Azure, no .mov).
    out = tmp_path / "cache.json"
    rc = precompute.main(["--clip", NEW_CLIP, "--query", GROUNDING_QUERY, "--dry-run", "--out", str(out)])
    assert rc == 0
    assert not out.exists()


# --------------------------------------------------------------------------------------
# codegen-repeatability across clips (mocked free-text-path prompt injection)
# --------------------------------------------------------------------------------------

def test_codegen_prompt_injects_new_clip_hint():
    # The canned path is clip-blind (server.py:85 calls generate(text) with no clip), so
    # codegen-repeatability across clips is proven on the FREE-TEXT path: build_system_message
    # injects the NEW clip's hint/label/duration (no live Azure needed).
    import codegen
    clip = canned.CLIPS[NEW_CLIP]
    msg = codegen.build_system_message(clip)
    assert clip["label"] in msg
    assert str(clip["duration_ms"]) in msg
    assert clip["hint"] in msg
    # the new clip's hint is DISTINCT from the hero's (a genuinely different window/event).
    assert clip["hint"] != canned.CLIPS["single-goal"]["hint"]
    # 2026-07-19: window widened 9000-11000 -> 9000-14800ms (real goal is ~14250ms; the prior
    # 9000-11000/10500 window was derived from a fake fixture, not the source video).
    assert "9000-14800ms" in msg


def test_free_text_path_compiles_new_clip_program_against_its_hint(mock_codegen):
    """End-to-end (mocked-codegen) free-text repeatability: build_free_text_run_doc for the new
    clip drives codegen with clip=bernabeu-counter, validates the returned program, and replays it
    over the new clip's cache to a grounded answer — zero live Azure (FakeCodegen records the clip).

    Cherry-picked from PR #23 (parallel-impl #2): deeper than the build_system_message-only check
    above — it exercises the FULL free-text compile+validate+replay path against the second clip,
    not just prompt construction."""
    import server
    from conftest import FakeCodegen
    program = strip_comments(
        json.loads((EX / "bernabeu-counter_count_program.json").read_text())
    )["program"]
    mock_codegen(program=program)

    doc = server.build_free_text_run_doc("How many players are visible?", NEW_CLIP)
    # codegen was asked to compile against the NEW clip (its hint/label reach the prompt builder).
    assert FakeCodegen.last_clip is not None
    assert FakeCodegen.last_clip["id"] == NEW_CLIP
    assert doc["program_source"] == "live"
    assert doc["findings"]["grounded"] is True
    assert doc["findings"]["answer"] == "5"
