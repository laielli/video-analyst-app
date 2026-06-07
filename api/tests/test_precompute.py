"""
First pytest suite for api/. Covers the clip -> cache precompute pipeline end-to-end with
Azure + ffmpeg fully mocked (no creds, no .mov, no ffmpeg). The named tests map 1:1 to the
plan's test list.

Two mock seams (see conftest.py): FakeAzureVision/CallSpyVLM via from_env, and injected
frame/crop providers returning canned bytes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import precompute
from precompute import (
    OcrLadder, GroundingError, SetupError,
    mint_det_ids, run_precompute, sampled_timestamps,
    canonical_cache, validate_cache_shape,
)
from interpreter import Cache, Interpreter
from conftest import FakeAzureVision, CallSpyVLM

API_DIR = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_minting_from_fixture_filters_classes_and_orders():
    """det_id minting over a canned-detect JSON fixture: `ball` filtered out, p0 = highest conf."""
    data = json.loads((FIXTURES / "fake_detect_4625.json").read_text())
    persons = [
        {"cls": o["class"], "confidence": o["confidence"], "box": o["box"]}
        for o in data["objects"] if o["class"].lower() == "person"
    ]
    minted = mint_det_ids(persons)
    assert [d["det_id"] for d in minted] == ["p0", "p1", "p2", "p3", "p4"]
    assert next(d for d in minted if d["det_id"] == "p0")["confidence"] == 0.753
    assert all(d["cls"] == "person" for d in minted)  # ball excluded upstream


def _run(manifest, frame_provider, crop_provider, *, vlm=None, require_ocr=False, out, emit_frames=False):
    """Helper: run the pipeline with mocked Azure + injected providers."""
    ladder = OcrLadder(require_ocr=require_ocr)
    if vlm is not None:
        # pin the ladder's VLM directly (still goes through from_env-less path).
        ladder._vlm = vlm
        ladder._vlm_attempted = True
    return run_precompute(
        manifest,
        out=out,
        vision=FakeAzureVision(),
        vlm_ladder=ladder,
        frame_provider=frame_provider,
        jersey_crop_provider=crop_provider,
        require_ocr=require_ocr,
        emit_frames=emit_frames,
    )


# --------------------------------------------------------------------------------------
# test_cache_loads_with_expected_accessors
# --------------------------------------------------------------------------------------

def test_cache_loads_with_expected_accessors(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    out = tmp_path / "cache.json"
    rep = _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    assert rep["exit"] == 0

    cache = Cache.load(out)
    # .clip
    assert cache.clip["id"] == "single-goal"
    assert cache.clip["width"] == 1872 and cache.clip["height"] == 1042
    assert cache.clip["duration_ms"] == 5067 and cache.clip["fps"] == 60
    # .analyzed_frames() — every sampled ts carries detections (default: detect on all)
    frames = cache.analyzed_frames()
    assert 4625 in frames
    assert frames == sorted(frames)
    # .detections_at
    dets = cache.detections_at(4625)
    assert any(d["det_id"] == "p_messi" for d in dets)
    # .read_text_for / .goal_events
    assert cache.read_text_for("p_messi")["text"] == "10"
    goals = cache.goal_events()
    assert goals and goals[0]["scorer_det"] == "p_messi"

    # structural invariants
    data = json.loads(out.read_text())
    all_ids = set()
    for ts, frame_dets in data["detect"].items():
        ids = [d["det_id"] for d in frame_dets]
        assert len(ids) == len(set(ids)), f"det_ids not unique at {ts}"
        all_ids.update(ids)
    for did in data["read_text"]:
        assert did in all_ids, f"read_text key {did} is not a real det_id"
    for ev in data["events"]:
        assert ev["scorer_det"] in all_ids, "event scorer_det not in detect map"


# --------------------------------------------------------------------------------------
# test_hero_cache_replays_to_grounded_answer
# --------------------------------------------------------------------------------------

def test_hero_cache_replays_to_grounded_answer(tmp_path, manifest, fake_frame_provider, fake_crop_provider, hero_program_path):
    out = tmp_path / "cache.json"
    _run(manifest, fake_frame_provider, fake_crop_provider, out=out)

    from validate_program import strip_comments
    program = strip_comments(json.loads(hero_program_path.read_text()))["program"]
    cache = Cache.load(out)
    run_doc = Interpreter(cache).run(program)

    # assert the GROUNDED chain, not the hardcoded verdict string.
    ordered = next(t for t in run_doc["trace"] if t["op"] == "temporal_order")
    assert ordered["output_label"].startswith("goal @ 4000ms")
    goals = cache.goal_events()
    scorer = goals[0]["scorer_det"]
    assert scorer == "p_messi"
    # scorer survives op_filter text==10
    numbers = next(t for t in run_doc["trace"] if t["op"] == "read_text")
    assert numbers["output_label"] == "10"
    tens = next(t for t in run_doc["trace"] if t["op"] == "filter")
    assert tens["status"] == "done"
    rt = cache.read_text_for(scorer)
    assert rt["text"] == "10" and rt["source"] == "pinned"
    # subject_is_first_scorer is True (via temporal_order binding semantics)
    answer = next(t for t in run_doc["trace"] if t["op"] == "answer")
    assert answer["output_label"] == "Yes"
    assert run_doc["findings"]["verdict"] == "Yes — #10 scored the first goal"

    # run-doc validates against the schema.
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc))
    assert not errs, errs


def test_degraded_sibling_carries_partial_signal(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """A degraded clip (no pin, no VLM digits, no goal event) -> no read_text -> filter empty ->
    temporal_order has no goal to order -> the answer step honestly grounds out (Phase 0 / Q1->A:
    grounded:false), instead of the old fabricated 'No — #10 didn't score' verdict (there was no
    goal to deny). Proves the honest-empty path is now legible, not a confident lie."""
    out = tmp_path / "cache.json"
    m = json.loads(json.dumps({k: v for k, v in manifest.items() if k != "_dir"}))
    m["_dir"] = manifest["_dir"]
    m["pins"] = []           # no pin
    m["events"] = []         # no goal -> nothing to temporally order -> ungrounded
    m["sampling"]["read_all_jerseys"] = True  # attempt VLM reads on every det
    rep = run_precompute(
        m, out=out, vision=FakeAzureVision(),
        vlm_ladder=OcrLadder(), frame_provider=fake_frame_provider,
        jersey_crop_provider=fake_crop_provider, emit_frames=False,
    )
    assert rep["exit"] == 0
    cache = Cache.load(out)
    assert cache.read_text_for("p0") is None       # no read written (honest-empty)
    assert cache.goal_events() == []
    from validate_program import strip_comments
    program = strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))["program"]
    run_doc = Interpreter(cache).run(program)
    # No goal -> no first scorer -> honest ungrounded, NOT a fabricated #10 verdict.
    assert run_doc["findings"]["grounded"] is False
    assert run_doc["findings"]["verdict"] is None
    numbers = next(t for t in run_doc["trace"] if t["op"] == "read_text")
    assert numbers["status"] == "empty"


# --------------------------------------------------------------------------------------
# test_detect_keys_align_with_program_sampling  (DEPENDENCY invariant)
# --------------------------------------------------------------------------------------

def test_detect_keys_align_with_program_sampling(tmp_path, manifest, fake_frame_provider, fake_crop_provider, hero_program_path):
    out = tmp_path / "cache.json"
    _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    cache = Cache.load(out)

    # the verdict depends on the scorer det at the goal frame; the program samples on the same
    # stride op_sample_frames uses. The real invariant: every sampled ts the verdict relies on
    # has a detect key (not a literal superset).
    from validate_program import strip_comments
    program = strip_comments(json.loads(hero_program_path.read_text()))["program"]
    sf = next(s for s in program if s["op"] == "sample_frames")
    start, end, fps = sf["args"]["start_ms"], sf["args"]["end_ms"], sf["args"]["fps"]
    stride = max(1, round(1000 / fps))
    program_ts = set(range(start, end + 1, stride))

    goals = cache.goal_events()
    scorer = goals[0]["scorer_det"]
    # the scorer det must be reachable: it appears in detect at some sampled ts.
    scorer_frames = [ts for ts in cache.analyzed_frames()
                     if any(d["det_id"] == scorer for d in cache.detections_at(ts))]
    assert scorer_frames, "scorer det not present in any detect frame"
    assert all(ts in program_ts for ts in cache.analyzed_frames()), \
        "detect keys must be a subset of the program's sampled ts (stride alignment)"
    assert 4625 in program_ts and 4625 in cache.analyzed_frames()


# --------------------------------------------------------------------------------------
# test_det_id_minting_is_deterministic
# --------------------------------------------------------------------------------------

def test_det_id_minting_is_deterministic():
    fixed = [
        {"cls": "person", "confidence": 0.598, "box": {"x": 0.2452, "y": 0.5739, "w": 0.2618, "h": 0.2553}},
        {"cls": "person", "confidence": 0.753, "box": {"x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276}},
        {"cls": "person", "confidence": 0.629, "box": {"x": 0.7959, "y": 0.2649, "w": 0.1277, "h": 0.2534}},
    ]
    a = mint_det_ids(fixed)
    b = mint_det_ids(list(reversed(fixed)))  # order-independent
    assert [d["det_id"] for d in a] == ["p0", "p1", "p2"]
    assert {d["det_id"]: d["confidence"] for d in a} == {d["det_id"]: d["confidence"] for d in b}
    # p0 = highest confidence
    p0 = next(d for d in a if d["det_id"] == "p0")
    assert p0["confidence"] == 0.753

    # a pin renames the nearest-box det to p_messi.
    detect = {"4625": mint_det_ids(fixed)}
    pin = {"match": {"frame_ts_ms": 4625, "nearest_box": {"x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276}},
           "det_id": "p_messi"}
    resolved = precompute.apply_pin_rename(detect, pin)
    assert resolved == "p_messi"
    assert any(d["det_id"] == "p_messi" for d in detect["4625"])
    # the renamed det is the one that was p0 (highest conf, matching box).
    messi = next(d for d in detect["4625"] if d["det_id"] == "p_messi")
    assert messi["confidence"] == 0.753


# --------------------------------------------------------------------------------------
# test_ocr_ladder_pin_wins
# --------------------------------------------------------------------------------------

def test_ocr_ladder_pin_wins(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    out = tmp_path / "cache.json"
    vlm = CallSpyVLM(digits="99")  # would say 99 if ever called
    rep = _run(manifest, fake_frame_provider, fake_crop_provider, vlm=vlm, out=out)
    cache = Cache.load(out)
    rt = cache.read_text_for("p_messi")
    assert rt["text"] == "10" and rt["source"] == "pinned"
    # VLM never called for the pinned det.
    assert vlm.calls == 0
    assert rep["vlm_calls"] == 0


# --------------------------------------------------------------------------------------
# test_ocr_ladder_vlm_then_skip
# --------------------------------------------------------------------------------------

def test_ocr_ladder_vlm_then_skip(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """No pin: a VLM that raises 429 -> det SKIPPED, no read_text entry, Azure Read never called."""
    out = tmp_path / "cache.json"
    m = {k: v for k, v in manifest.items() if k != "_dir"}
    m = json.loads(json.dumps(m)); m["_dir"] = manifest["_dir"]
    m["pins"] = []  # no pin so the VLM rung is reached
    m["events"] = []  # isolate the OCR ladder (no scorer to ground)
    m["sampling"]["read_all_jerseys"] = True  # every det is an OCR target -> VLM rung reached

    class Boom(Exception):
        pass
    vlm = CallSpyVLM(raise_exc=Boom("429 rate limit"))

    ladder = OcrLadder()
    ladder._vlm = vlm
    ladder._vlm_attempted = True
    rep = run_precompute(
        m, out=out, vision=FakeAzureVision(), vlm_ladder=ladder,
        frame_provider=fake_frame_provider, jersey_crop_provider=fake_crop_provider, emit_frames=False,
    )
    cache = Cache.load(out)
    # VLM was attempted (raised) for at least one det, and NO read_text entry was written.
    assert vlm.calls >= 1
    assert json.loads(out.read_text())["read_text"] == {}
    # Azure Read is never a rung — FakeAzureVision.analyze is only ever called with read=False
    # (we asserted detection-only by the fact no lines ever appear). No Azure Read entry exists.


# --------------------------------------------------------------------------------------
# test_vlm_tpm_gated_degrades_gracefully
# --------------------------------------------------------------------------------------

def test_vlm_tpm_gated_degrades_gracefully(tmp_path, manifest, fake_frame_provider, fake_crop_provider, monkeypatch):
    """A fake AzureVLM.from_env that raises -> pipeline completes, valid cache, warns, exit 0
    (no --require-ocr). Hero unaffected because p_messi is pinned."""
    import vision.vlm_read as vlm_mod

    def boom_from_env():
        raise ValueError("AZURE_OPENAI_* missing")
    monkeypatch.setattr(vlm_mod.AzureVLM, "from_env", staticmethod(boom_from_env))

    out = tmp_path / "cache.json"
    # Opt into reading every jersey so the (patched, raising) VLM from_env is actually reached;
    # the p_messi pin still wins, so the hero verdict is unaffected.
    m = json.loads(json.dumps({k: v for k, v in manifest.items() if k != "_dir"}))
    m["_dir"] = manifest["_dir"]
    m["sampling"]["read_all_jerseys"] = True
    ladder = OcrLadder()  # will call the patched from_env lazily -> warn
    rep = run_precompute(
        m, out=out, vision=FakeAzureVision(), vlm_ladder=ladder,
        frame_provider=fake_frame_provider, jersey_crop_provider=fake_crop_provider, emit_frames=False,
    )
    assert rep["exit"] == 0
    assert any("unavailable" in w for w in rep["warnings"])
    cache = Cache.load(out)
    assert cache.read_text_for("p_messi")["text"] == "10"  # pin still wins


def test_require_ocr_hard_fails_when_read_skipped(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """--require-ocr + a non-pinned det with no VLM -> GroundingError (exit 1)."""
    out = tmp_path / "cache.json"
    m = {k: v for k, v in manifest.items() if k != "_dir"}
    m = json.loads(json.dumps(m)); m["_dir"] = manifest["_dir"]
    m["pins"] = []
    m["events"] = []
    m["sampling"]["read_all_jerseys"] = True
    ladder = OcrLadder(require_ocr=True)
    ladder._vlm = None
    ladder._vlm_attempted = True  # no VLM available
    with pytest.raises(GroundingError):
        run_precompute(
            m, out=out, vision=FakeAzureVision(), vlm_ladder=ladder,
            frame_provider=fake_frame_provider, jersey_crop_provider=fake_crop_provider,
            require_ocr=True, emit_frames=False,
        )


# --------------------------------------------------------------------------------------
# test_idempotent_rerun_is_noop
# --------------------------------------------------------------------------------------

def test_idempotent_rerun_is_noop(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    out = tmp_path / "cache.json"
    _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    first = out.read_bytes()
    _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    second = out.read_bytes()
    assert first == second, "re-run produced a different cache (not idempotent)"


# --------------------------------------------------------------------------------------
# test_dry_run_writes_nothing
# --------------------------------------------------------------------------------------

def test_dry_run_writes_nothing(tmp_path, manifest):
    out = tmp_path / "cache.json"
    rep = run_precompute(manifest, out=out, dry_run=True)
    assert rep["exit"] == 0
    assert rep["dry_run"] is True
    assert not out.exists(), "dry-run must not write the cache"
    # zero Azure calls: no FakeAzureVision was constructed, no VLM call.
    assert FakeAzureVision.instances == []
    assert CallSpyVLM.total_calls == 0
    assert 4625 in rep["timestamps"]


def test_dry_run_via_cli(tmp_path, monkeypatch):
    """End-to-end CLI dry-run exits 0 and writes nothing."""
    out = tmp_path / "cache.json"
    rc = precompute.main(["--clip", "single-goal", "--dry-run", "--out", str(out)])
    assert rc == 0
    assert not out.exists()


# --------------------------------------------------------------------------------------
# test_box_coords_normalized
# --------------------------------------------------------------------------------------

def test_box_coords_normalized(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    out = tmp_path / "cache.json"
    _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    data = json.loads(out.read_text())
    for dets in data["detect"].values():
        for d in dets:
            for k, v in d["box"].items():
                assert 0 <= v <= 1, f"box.{k}={v} out of [0,1]"
                # <= 4 decimals
                assert round(v, 4) == v, f"box.{k}={v} not rounded to 4dp"
    for ev in data["events"]:
        for k, v in ev["box"].items():
            assert 0 <= v <= 1 and round(v, 4) == v


# --------------------------------------------------------------------------------------
# test_stills_emitted_for_evidence_frames
# --------------------------------------------------------------------------------------

def test_stills_emitted_for_evidence_frames(tmp_path, manifest, monkeypatch):
    """One still per unique evidence ts, with the EXACT filenames goal-4000.jpg / scorer-4625.jpg.
    ffmpeg is stubbed (subprocess.run records the dest); the .mov is faked as existing."""
    import precompute as pc
    frames_dir = tmp_path / "frames"
    fake_video = tmp_path / "single-goal.mov"
    fake_video.write_bytes(b"\x00")  # exists()

    calls = []

    def fake_run(cmd, **kw):
        # last arg is the dest path; create the file so the pipeline "writes" it.
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8\xff")  # JPEG magic
        calls.append(dest.name)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(pc.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pc.subprocess, "run", fake_run)

    written = pc.emit_stills(manifest, video=fake_video, frames_dir=frames_dir)
    assert set(written) == {"goal-4000.jpg", "scorer-4625.jpg"}
    assert (frames_dir / "goal-4000.jpg").exists()
    assert (frames_dir / "scorer-4625.jpg").exists()
    # exactly one ffmpeg call per unique evidence ts (2).
    assert len(calls) == 2


# --------------------------------------------------------------------------------------
# manifest-absence + setup errors
# --------------------------------------------------------------------------------------

def test_missing_manifest_errors_exit_2():
    rc = precompute.main(["--clip", "does-not-exist", "--dry-run"])
    assert rc == 2


def test_no_clip_or_manifest_errors_exit_2():
    rc = precompute.main([])
    assert rc == 2


# --------------------------------------------------------------------------------------
# canonical / shape validation unit checks
# --------------------------------------------------------------------------------------

def test_validate_cache_shape_catches_orphan_scorer():
    clip = {"id": "x", "width": 10, "height": 10, "duration_ms": 100, "fps": 30}
    detect = {"100": [{"det_id": "p0", "cls": "person", "confidence": 0.5, "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}]}
    cache = canonical_cache(clip, detect, {}, [{"ts_ms": 100, "type": "goal", "scorer_det": "ghost", "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}])
    errs = validate_cache_shape(cache)
    assert any("scorer_det" in e for e in errs)


def test_validate_cache_shape_catches_orphan_read_text():
    clip = {"id": "x", "width": 10, "height": 10, "duration_ms": 100, "fps": 30}
    detect = {"100": [{"det_id": "p0", "cls": "person", "confidence": 0.5, "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}]}
    cache = canonical_cache(clip, detect, {"ghost": {"text": "10", "confidence": None, "source": "pinned"}}, [])
    errs = validate_cache_shape(cache)
    assert any("read_text key 'ghost'" in e for e in errs)
