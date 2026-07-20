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
    full_person_crop_box, crop_jersey_box,
    emit_stills, emit_clip_video, emit_frames_manifest, build_frames_manifest_entry,
)
from interpreter import Cache, Interpreter
from conftest import FakeAzureVision, CallSpyVLM
from vision.azure_vision import AnalysisResult, Detection, Box

API_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = API_DIR.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_bernabeu_manifest() -> dict:
    """The committed bernabeu-counter manifest (comment-stripped), loaded directly (this file
    doesn't share conftest.py fixtures with test_catalog_multiclip.py, which owns its own
    `bernabeu_manifest` fixture)."""
    raw = json.loads((REPO_ROOT / "clips" / "bernabeu-counter" / "manifest.json").read_text())
    m = precompute._strip_comments(raw)
    m["_dir"] = REPO_ROOT / "clips" / "bernabeu-counter"
    return m


class _BernabeuVision:
    """A FakeAzureVision-shaped stand-in that returns HAND-BUILT detections (not the shared
    fake_detect_bernabeu.json fixture, which test_catalog_multiclip.py owns) so this file's
    bernabeu-manifest tests are fully self-contained. The scorer box matches the manifest pin's
    nearest_box; a couple of decoys keep det_id minting honest."""

    def __init__(self):
        self.objects = [
            Detection(cls="person", confidence=0.58, box=Box(0.62, 0.55, 0.09, 0.30)),
            Detection(cls="person", confidence=0.81, box=Box(0.35, 0.30, 0.06, 0.33)),  # p_vini
            Detection(cls="person", confidence=0.64, box=Box(0.06, 0.27, 0.19, 0.34)),
            Detection(cls="ball",   confidence=0.35, box=Box(0.5, 0.5, 0.03, 0.03)),
        ]
        self.analyze_calls = 0

    def analyze(self, image_bytes, detect=True, read=False, caption=False):
        assert not read, "Azure Read must NOT be a rung in the OCR ladder (Resolved #2)"
        self.analyze_calls += 1
        return AnalysisResult(width=2636, height=1474,
                              objects=list(self.objects) if detect else [], lines=[], captions=[])


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


_DEFAULT_VLM = object()  # sentinel: "use the default CallSpyVLM", distinct from an explicit None


def _run(manifest, frame_provider, crop_provider, *, vlm=_DEFAULT_VLM, require_ocr=False, out,
         emit_frames=False):
    """Helper: run the pipeline with mocked Azure + injected providers.

    `vlm` defaults to a CallSpyVLM reading '10' — the single-goal pin's disclosed expectation —
    so the now-mandatory live_vlm pin verification (Task 1) passes silently for tests that don't
    care about OCR. Pass an explicit CallSpyVLM (wrong digits / raising) to test the mismatch
    path, or `vlm=None` to simulate "no VLM available at all" (pin hard-fails). Either way the
    ladder's VLM is pre-attached (`_vlm_attempted = True`) so it NEVER falls through to a real
    AzureVLM.from_env() — this suite makes zero real Azure calls even though api/.env carries
    real creds for the live demo.
    """
    ladder = OcrLadder(require_ocr=require_ocr)
    resolved_vlm = CallSpyVLM(digits="10") if vlm is _DEFAULT_VLM else vlm
    ladder._vlm = resolved_vlm
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
    assert rt["text"] == "10" and rt["source"] == "live"  # verified live_vlm read, never hardcoded
    # subject_is_first_scorer is True (via temporal_order binding semantics)
    answer = next(t for t in run_doc["trace"] if t["op"] == "answer")
    assert answer["output_label"] == "Yes"
    assert run_doc["findings"]["verdict"] == "Yes — #10 scored the goal"

    # run-doc validates against the schema.
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc))
    assert not errs, errs


def test_degraded_sibling_carries_partial_signal(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """A degraded clip (no pin, no VLM digits, no goal event) -> no read_text -> filter empty ->
    temporal_order finds no goal -> the answer grounds out HONESTLY (grounded=false) rather than
    emitting a fabricated 'No' verdict. Proves the honest-empty path under Phase 0 (Q1 -> A): the
    old behavior synthesized a confident 'No' from no evidence; now it is an explicit ungrounded
    state with answer/verdict null."""
    out = tmp_path / "cache.json"
    m = json.loads(json.dumps({k: v for k, v in manifest.items() if k != "_dir"}))
    m["_dir"] = manifest["_dir"]
    m["pins"] = []           # no pin
    m["events"] = []         # no goal -> nothing to ground against
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
    # Honest degradation: no fabricated verdict, an explicit ungrounded discriminator.
    assert run_doc["findings"]["grounded"] is False
    assert run_doc["findings"]["verdict"] is None
    assert run_doc["findings"]["answer"] is None
    assert run_doc["findings"]["reason"] == "no-grounded-answer"
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
# test_ocr_ladder_pin_verifies_live_read  (Task 1: pins disclose, never hardcode)
# --------------------------------------------------------------------------------------

def test_ocr_ladder_pin_verifies_live_read(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """The pin's `read` is {"mode":"live_vlm","expected_text":"10"} — a disclosed expectation,
    NOT hardcoded text. A real (mocked) gpt-4o read agreeing with it produces source=live with
    the FIXED disclosure note, and the VLM IS called exactly once (unlike the old hardcoded pin,
    which never called it)."""
    out = tmp_path / "cache.json"
    vlm = CallSpyVLM(digits="10")  # the honest gpt-4o read this pin expects
    rep = _run(manifest, fake_frame_provider, fake_crop_provider, vlm=vlm, out=out)
    cache = Cache.load(out)
    rt = cache.read_text_for("p_messi")
    assert rt["text"] == "10" and rt["source"] == "live"
    assert rt["note"] == (
        "gpt-4o read of the full player crop at precompute; expectation disclosed and verified"
    )
    assert "confidence" in rt and rt["confidence"] is None
    # the pin now MANDATES exactly one real (mocked) VLM call to verify the disclosure.
    assert vlm.calls == 1
    assert rep["vlm_calls"] == 1


def test_ocr_ladder_pin_mismatch_hard_fails(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """A gpt-4o read that DISAGREES with the pin's expected_text is a hard GroundingError — never
    a silent skip and never a fallback to the (retired) hardcoded text. This is the honest
    human-verification gate: a wrong live read must block the run, not degrade it."""
    out = tmp_path / "cache.json"
    vlm = CallSpyVLM(digits="77")  # disagrees with the pin's expected_text ('10')
    with pytest.raises(GroundingError, match="expected gpt-4o to read '10'"):
        _run(manifest, fake_frame_provider, fake_crop_provider, vlm=vlm, out=out)
    assert not out.exists(), "a failed grounding gate must not write a cache"


def test_ocr_ladder_pin_requires_vlm_when_unavailable(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """A pin is a mandatory verification gate, not a degradable OCR rung: with NO VLM available
    at all, the pin hard-fails (GroundingError) rather than skipping or falling back to a
    hardcoded read."""
    out = tmp_path / "cache.json"
    with pytest.raises(GroundingError, match="no VLM is available"):
        _run(manifest, fake_frame_provider, fake_crop_provider, vlm=None, out=out)
    assert not out.exists()


def test_ocr_ladder_pin_rejects_non_live_vlm_mode(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    """A pin `read` with any mode other than 'live_vlm' (e.g. a residual hardcoded-text shape)
    is rejected outright — hardcoded pinned text is retired, not merely deprioritized."""
    import copy
    m = copy.deepcopy({k: v for k, v in manifest.items() if k != "_dir"})
    m["_dir"] = manifest["_dir"]
    m["pins"][0]["read"] = {"text": "10", "confidence": None, "source": "pinned"}
    out = tmp_path / "cache.json"
    with pytest.raises(GroundingError, match="unsupported mode"):
        _run(m, fake_frame_provider, fake_crop_provider, vlm=CallSpyVLM(digits="10"), out=out)


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
# test_non_pinned_reads_degrade_gracefully_when_vlm_unavailable
# --------------------------------------------------------------------------------------

def test_non_pinned_reads_degrade_gracefully_when_vlm_unavailable(
    tmp_path, manifest, fake_frame_provider, fake_crop_provider, monkeypatch,
):
    """A fake AzureVLM.from_env that raises -> a NON-pinned read (sampling.read_all_jerseys)
    degrades gracefully: pipeline completes, valid cache, warns, exit 0 (no --require-ocr). Pins
    are NOT exercised here (pins=[]) because — since Task 1 — a pin is a mandatory verification
    gate that hard-fails without a VLM (see test_ocr_ladder_pin_requires_vlm_when_unavailable),
    not a degradable rung; only non-pinned OCR targets degrade this way."""
    import vision.vlm_read as vlm_mod

    def boom_from_env():
        raise ValueError("AZURE_OPENAI_* missing")
    monkeypatch.setattr(vlm_mod.AzureVLM, "from_env", staticmethod(boom_from_env))

    out = tmp_path / "cache.json"
    m = json.loads(json.dumps({k: v for k, v in manifest.items() if k != "_dir"}))
    m["_dir"] = manifest["_dir"]
    m["pins"] = []            # isolate the non-pinned degrade path from the pin verification gate
    m["events"] = []          # no scorer to ground (irrelevant to this test)
    m["sampling"]["read_all_jerseys"] = True
    ladder = OcrLadder()  # will call the patched from_env lazily -> warn
    rep = run_precompute(
        m, out=out, vision=FakeAzureVision(), vlm_ladder=ladder,
        frame_provider=fake_frame_provider, jersey_crop_provider=fake_crop_provider, emit_frames=False,
    )
    assert rep["exit"] == 0
    assert any("unavailable" in w for w in rep["warnings"])
    cache = Cache.load(out)
    assert json.loads(out.read_text())["read_text"] == {}  # honest-empty, no fabricated read


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
# test_stills_emitted_for_every_sampled_frame  (Task 4: replaces the evidence_frames subset)
# --------------------------------------------------------------------------------------

def test_stills_emitted_for_every_sampled_frame(tmp_path, manifest, monkeypatch):
    """One still per SAMPLED ts (not a curated evidence_frames subset, which is dropped), named
    f<ts>.jpg under frames_dir/<clip_id>/. ffmpeg is stubbed (subprocess.run records the cmd +
    dest); the .mov is faked as existing."""
    import precompute as pc
    frames_dir = tmp_path / "frames"
    fake_video = tmp_path / "single-goal.mov"
    fake_video.write_bytes(b"\x00")  # exists()

    calls = []

    def fake_run(cmd, **kw):
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8\xff")  # JPEG magic
        calls.append(cmd)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(pc.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pc.subprocess, "run", fake_run)

    expected_ts = pc.sampled_timestamps(manifest)
    written = pc.emit_stills(manifest, video=fake_video, frames_dir=frames_dir)

    assert set(written) == {f"single-goal/f{ts}.jpg" for ts in expected_ts}
    assert len(calls) == len(expected_ts), "exactly one ffmpeg call per sampled ts"
    for ts in expected_ts:
        assert (frames_dir / "single-goal" / f"f{ts}.jpg").exists()
    # scale filter + jpeg quality are part of the constructed cmd.
    assert any("scale=960:-2" in " ".join(c) for c in calls)
    assert all("-q:v" in c and "3" in c for c in calls)


def test_emit_stills_skips_when_ffmpeg_absent(tmp_path, manifest, monkeypatch):
    import precompute as pc
    monkeypatch.setattr(pc.shutil, "which", lambda _: None)
    written = pc.emit_stills(manifest, video=tmp_path / "x.mov", frames_dir=tmp_path / "frames")
    assert written == []
    assert not (tmp_path / "frames").exists()


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


# --------------------------------------------------------------------------------------
# describe_scene precompute (Phase 4) — captions slice gated by sampling.caption_frames
# --------------------------------------------------------------------------------------

def test_precompute_emits_captions_slice(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    # opting into sampling.caption_frames yields a `captions` map keyed by sampled ts, each entry
    # carrying text/confidence/box; the produced cache passes validate_cache_shape.
    manifest = dict(manifest)
    manifest["sampling"] = {**manifest["sampling"], "caption_frames": True}
    out = tmp_path / "cache.json"
    rep = _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    assert rep["exit"] == 0
    data = json.loads(out.read_text())
    assert "captions" in data and data["captions"], "captions slice must be emitted when opted in"
    for ts, caps in data["captions"].items():
        assert str(ts).isdigit()
        for c in caps:
            assert c["text"] and isinstance(c["confidence"], (int, float))
            for k in ("x", "y", "w", "h"):
                assert 0 <= c["box"][k] <= 1
    assert validate_cache_shape(data) == []
    # the cache's captions_at accessor resolves the slice.
    cache = Cache.load(out)
    sampled = sampled_timestamps(manifest)
    assert cache.captions_at(sampled[0]), "captions_at must return the cached caption"


def test_precompute_no_captions_when_not_opted_in(tmp_path, manifest, fake_frame_provider, fake_crop_provider):
    # sampling.caption_frames off (the committed single-goal manifest opts IN — Task 3 — so this
    # test explicitly opts back OUT to prove run_describe_scene's own default-off behavior) yields
    # NO captions key.
    manifest = dict(manifest)
    manifest["sampling"] = {**manifest["sampling"], "caption_frames": False}
    out = tmp_path / "cache.json"
    rep = _run(manifest, fake_frame_provider, fake_crop_provider, out=out)
    assert rep["exit"] == 0
    data = json.loads(out.read_text())
    assert "captions" not in data


def test_validate_cache_shape_catches_overlong_caption():
    # a caption longer than MAX_CAPTION_LEN is rejected by validate_cache_shape (security guard).
    from limits import MAX_CAPTION_LEN
    clip = {"id": "x", "width": 10, "height": 10, "duration_ms": 100, "fps": 30}
    detect = {"100": [{"det_id": "p0", "cls": "person", "confidence": 0.5, "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}]}
    captions = {"100": [{"text": "z" * (MAX_CAPTION_LEN + 1), "confidence": 0.5, "box": {"x": 0, "y": 0, "w": 1, "h": 1}}]}
    cache = canonical_cache(clip, detect, {}, [], captions=captions)
    errs = validate_cache_shape(cache)
    assert any("text is" in e and "max is" in e for e in errs)


def test_run_describe_scene_off_by_default(manifest, fake_frame_provider):
    # run_describe_scene returns {} (no Azure CAPTION call) unless the manifest opts in — proven
    # against a manifest copy with caption_frames explicitly off (the committed single-goal
    # manifest opts IN per Task 3, so the raw fixture no longer demonstrates the off-by-default
    # case on its own).
    manifest = dict(manifest)
    manifest["sampling"] = {**manifest["sampling"], "caption_frames": False}
    ts = sampled_timestamps(manifest)
    assert precompute.run_describe_scene(manifest, ts, frame_provider=fake_frame_provider,
                                         vision=FakeAzureVision()) == {}


# --------------------------------------------------------------------------------------
# full-person VLM crop math (Task 2) — kills the brittle crop_jersey_box heuristic for reads
# --------------------------------------------------------------------------------------

def test_full_person_crop_box_adds_margin():
    box = {"x": 0.30, "y": 0.30, "w": 0.10, "h": 0.40}
    crop = full_person_crop_box(box, margin=0.05)
    assert crop["x"] == pytest.approx(0.30 - 0.05 * 0.10, abs=1e-9)
    assert crop["y"] == pytest.approx(0.30 - 0.05 * 0.40, abs=1e-9)
    # width/height each grow by 2*margin (margin on BOTH sides).
    assert crop["w"] == pytest.approx(0.10 * 1.10, abs=1e-9)
    assert crop["h"] == pytest.approx(0.40 * 1.10, abs=1e-9)
    # it is NOT the tight jersey-box crop (which shrinks to ~56%/22% of the person box).
    jersey = crop_jersey_box(box)
    assert crop["w"] > jersey["w"] and crop["h"] > jersey["h"]


def test_full_person_crop_box_clamps_to_unit_box():
    # a box already touching every edge: margin must clamp, not go negative / past 1.
    edge_box = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    crop = full_person_crop_box(edge_box)
    assert crop == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}

    # a box near one edge only: that edge clamps, the far edge still gets its margin.
    near_edge = {"x": 0.0, "y": 0.4, "w": 0.02, "h": 0.3}
    crop = full_person_crop_box(near_edge, margin=0.05)
    assert crop["x"] == 0.0  # clamped (0 - margin*w < 0)
    assert crop["x"] + crop["w"] <= 1.0
    assert crop["y"] + crop["h"] <= 1.0


def test_pin_read_uses_full_person_crop_not_jersey_box(tmp_path, manifest, fake_frame_provider):
    """The crop handed to the VLM crop provider for a pinned det is the FULL-PERSON + margin box
    (Task 2), never the tight jersey-box heuristic."""
    captured = {}

    def spy_crop_provider(ts, box):
        captured[ts] = box
        return b"\x89PNG-crop"

    out = tmp_path / "cache.json"
    _run(manifest, fake_frame_provider, spy_crop_provider, vlm=CallSpyVLM(digits="10"), out=out)

    got_box = captured[4625]
    p_messi_box = {"x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276}
    assert got_box == full_person_crop_box(p_messi_box)
    assert got_box != crop_jersey_box(p_messi_box)


# --------------------------------------------------------------------------------------
# clip video emission (Task 5)
# --------------------------------------------------------------------------------------

def test_emit_clip_video_builds_expected_ffmpeg_cmd(tmp_path, manifest, monkeypatch):
    import precompute as pc
    clips_dir = tmp_path / "clips"
    fake_video = tmp_path / "single-goal.mov"
    fake_video.write_bytes(b"\x00")

    calls = []

    def fake_run(cmd, **kw):
        Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(cmd[-1]).write_bytes(b"\x00")
        calls.append(cmd)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(pc.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pc.subprocess, "run", fake_run)

    m = dict(manifest)
    m["video"] = {"cut_start_ms": 0, "cut_end_ms": 5067}
    written = pc.emit_clip_video(m, video=fake_video, clips_dir=clips_dir)

    assert written == "single-goal.mp4"
    assert (clips_dir / "single-goal.mp4").exists()
    assert len(calls) == 1
    cmd = calls[0]
    joined = " ".join(cmd)
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "0.0"
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "5.067"
    assert "scale=-2:720" in joined
    assert "libx264" in cmd and "yuv420p" in cmd
    assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "23"
    assert "-an" in cmd
    assert "+faststart" in cmd


def test_emit_clip_video_skips_without_manifest_video_block(tmp_path, manifest, monkeypatch):
    import precompute as pc
    monkeypatch.setattr(pc.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    m = dict(manifest)
    m.pop("video", None)
    fake_video = tmp_path / "single-goal.mov"
    fake_video.write_bytes(b"\x00")
    assert pc.emit_clip_video(m, video=fake_video, clips_dir=tmp_path / "clips") is None


def test_emit_clip_video_skips_when_ffmpeg_absent(tmp_path, manifest, monkeypatch):
    import precompute as pc
    monkeypatch.setattr(pc.shutil, "which", lambda _: None)
    m = dict(manifest)
    m["video"] = {"cut_start_ms": 0, "cut_end_ms": 5067}
    assert pc.emit_clip_video(m, video=tmp_path / "x.mov", clips_dir=tmp_path / "clips") is None


# --------------------------------------------------------------------------------------
# frames-manifest emission (Task 6) — shape + read-modify-write merge
# --------------------------------------------------------------------------------------

def test_build_frames_manifest_entry_shape():
    m = {
        "clip": {"id": "single-goal"},
        "video": {"cut_start_ms": 0, "cut_end_ms": 5067},
        "poster_ts": 4625,
    }
    entry = build_frames_manifest_entry(m, stills_ts=[4625, 3500, 4000], width=1872, height=1042)
    assert entry == {
        "width": 1872,
        "height": 1042,
        "stills": [3500, 4000, 4625],   # sorted
        "still_path": "/frames/single-goal/f{ts}.jpg",
        "poster_ts": 4625,
        "video": {"src": "/clips/single-goal.mp4", "offset_ms": 0, "duration_ms": 5067},
    }
    # deterministic key order (the exact shape the mission specifies).
    assert list(entry.keys()) == ["width", "height", "stills", "still_path", "poster_ts", "video"]


def test_build_frames_manifest_entry_omits_video_when_manifest_has_none():
    m = {"clip": {"id": "x"}, "poster_ts": 100}
    entry = build_frames_manifest_entry(m, stills_ts=[100], width=10, height=10)
    assert "video" not in entry


def test_emit_frames_manifest_merges_and_preserves_other_clips(tmp_path, manifest):
    path = tmp_path / "frames-manifest.json"
    path.write_text(json.dumps({"other-clip": {"width": 1, "height": 1, "stills": [1],
                                                "still_path": "/frames/other-clip/f{ts}.jpg",
                                                "poster_ts": 1}}) + "\n")
    clip = {"id": "single-goal", "width": 1872, "height": 1042, "duration_ms": 5067, "fps": 60}
    merged = emit_frames_manifest(manifest, clip=clip, stills_ts=[4000, 4625], path=path)

    assert set(merged) == {"other-clip", "single-goal"}
    assert merged["other-clip"] == {"width": 1, "height": 1, "stills": [1],
                                    "still_path": "/frames/other-clip/f{ts}.jpg", "poster_ts": 1}
    sg = merged["single-goal"]
    assert sg["stills"] == [4000, 4625]
    assert sg["poster_ts"] == 4625
    assert sg["video"] == {"src": "/clips/single-goal.mp4", "offset_ms": 0, "duration_ms": 5067}

    # written to disk, atomically, deterministically (sorted clip-id key order).
    on_disk = json.loads(path.read_text())
    assert list(on_disk.keys()) == sorted(on_disk.keys())
    assert on_disk == merged


def test_emit_frames_manifest_dry_run_does_not_touch_disk(tmp_path, manifest):
    path = tmp_path / "frames-manifest.json"
    clip = {"id": "single-goal", "width": 1872, "height": 1042, "duration_ms": 5067, "fps": 60}
    result = emit_frames_manifest(manifest, clip=clip, stills_ts=[4625], path=path, dry_run=True)
    assert result == {}
    assert not path.exists()


def test_emit_frames_manifest_starts_fresh_when_no_stub_exists(tmp_path, manifest):
    path = tmp_path / "does-not-exist" / "frames-manifest.json"
    clip = {"id": "single-goal", "width": 1872, "height": 1042, "duration_ms": 5067, "fps": 60}
    merged = emit_frames_manifest(manifest, clip=clip, stills_ts=[4625], path=path)
    assert set(merged) == {"single-goal"}
    assert path.exists()


# --------------------------------------------------------------------------------------
# bernabeu-counter manifest (Task 3) — exact re-verified values, self-contained (no shared
# fixtures/other test files' registries): loaded directly from clips/bernabeu-counter/manifest.json.
# --------------------------------------------------------------------------------------

def test_bernabeu_manifest_carries_the_reverified_ground_truth():
    m = _load_bernabeu_manifest()
    assert m["sampling"] == {"start_ms": 9000, "end_ms": 14800, "fps": 8}
    assert m["video"] == {"cut_start_ms": 7870, "cut_end_ms": 14820}
    assert m["poster_ts"] == 10000
    assert "evidence_frames" not in m  # superseded by all-sampled-frames stills (Task 4)

    pin = m["pins"][0]
    assert pin["match"] == {
        "frame_ts_ms": 10000,
        "nearest_box": {"x": 0.35, "y": 0.30, "w": 0.06, "h": 0.33},
    }
    assert pin["read"]["mode"] == "live_vlm"
    assert pin["read"]["expected_text"] == "7"
    assert "text" not in pin["read"], "no hardcoded jersey text may enter the pin's read spec"

    ev = m["events"][0]
    assert ev["ts_ms"] == 14250  # the REAL goal moment; NOT the fake-fixture-derived 10500
    assert ev["box"] == {"x": 0.06, "y": 0.27, "w": 0.20, "h": 0.20}
    assert ev["scorer_pin"] == "p_vini"

    # the goal + scorer frame are both on the sampled stride (no silent grounding gap).
    ts = sampled_timestamps(m)
    assert 14250 in ts and 10000 in ts


def test_single_goal_manifest_carries_task3_updates():
    m = json.loads((REPO_ROOT / "clips" / "single-goal" / "manifest.json").read_text())
    assert m["sampling"]["caption_frames"] is True
    assert m["video"] == {"cut_start_ms": 0, "cut_end_ms": 5067}
    assert m["poster_ts"] == 4625
    assert "evidence_frames" not in m
    pin = m["pins"][0]
    assert pin["read"]["mode"] == "live_vlm"
    assert pin["read"]["expected_text"] == "10"
    assert "text" not in pin["read"], "no hardcoded jersey text may enter the pin's read spec"


def _write_bernabeu_local_program(tmp_path) -> Path:
    """A locally-authored program (window 9000-14800, filter text==7) — NOT the shared
    examples/bernabeu-counter_first-goal-7_program.json, which another agent owns/edits
    concurrently — so these tests are hermetic against that file's in-flight state."""
    program = {
        "program": [
            {"id": "frames", "op": "sample_frames", "args": {"start_ms": 9000, "end_ms": 14800, "fps": 8}},
            {"id": "people", "op": "detect", "args": {"frames": "frames", "classes": ["person"]}},
            {"id": "jerseys", "op": "crop", "args": {"detections": "people", "region": "jersey"}},
            {"id": "numbers", "op": "read_text", "args": {"crops": "jerseys"}},
            {"id": "sevens", "op": "filter", "args": {"items": "numbers", "where": {"field": "text", "equals": "7"}}},
            {"id": "ordered", "op": "temporal_order", "args": {"events": "sevens", "by": "timestamp"}},
            {"id": "result", "op": "answer", "args": {"from": "ordered", "question": "Does #7 score the goal?"}},
        ]
    }
    path = tmp_path / "bernabeu_program.json"
    path.write_text(json.dumps(program))
    return path


def test_bernabeu_precompute_pin_and_event_ground_via_live_vlm(tmp_path):
    """End-to-end (mocked): the bernabeu manifest's pin resolves p_vini at the re-verified 10000ms
    box, requires (and gets) a live gpt-4o read of '7', and the REAL goal ts (14250, from the
    manifest — not a hardcoded literal) is a reachable, minted det_id-bearing frame. Self-validates
    against a LOCAL program (see _write_bernabeu_local_program) so this test doesn't depend on
    another agent's concurrent edits to the shared example program."""
    manifest = _load_bernabeu_manifest()
    out = tmp_path / "cache.json"
    program_path = _write_bernabeu_local_program(tmp_path)

    def frame_provider(ts):
        return b"\x89PNG" + ts.to_bytes(4, "big")

    def crop_provider(ts, box):
        return b"\x89PNG-crop" + ts.to_bytes(4, "big")

    ladder = OcrLadder()
    ladder._vlm = CallSpyVLM(digits="7")
    ladder._vlm_attempted = True

    rep = run_precompute(
        manifest, out=out, vision=_BernabeuVision(), vlm_ladder=ladder,
        frame_provider=frame_provider, jersey_crop_provider=crop_provider,
        program_path=program_path, emit_frames=False,
    )
    assert rep["exit"] == 0
    cache = Cache.load(out)
    assert cache.read_text_for("p_vini") == {
        "text": "7", "confidence": None, "source": "live",
        "note": "gpt-4o read of the full player crop at precompute; expectation disclosed and verified",
    }
    goals = cache.goal_events()
    assert goals and goals[0]["ts_ms"] == 14250 and goals[0]["scorer_det"] == "p_vini"
    assert 14250 in cache.analyzed_frames()


def test_bernabeu_precompute_replays_grounded_with_local_program(tmp_path):
    """Self-validation replay path (mission requirement): mint a cache from the manifest's OWN
    event ts (14250 — not the retired fake-fixture 10500) and replay a program windowed
    9000-14800 filtering on '7'; the produced cache must ground to Yes. Uses a LOCALLY authored
    program (not the shared examples/bernabeu-counter_first-goal-7_program.json, which another
    agent owns/edits concurrently) so this test is hermetic."""
    manifest = _load_bernabeu_manifest()
    out = tmp_path / "cache.json"
    program_path = _write_bernabeu_local_program(tmp_path)

    ladder = OcrLadder()
    ladder._vlm = CallSpyVLM(digits="7")
    ladder._vlm_attempted = True

    def frame_provider(ts):
        return b"\x89PNG" + ts.to_bytes(4, "big")

    def crop_provider(ts, box):
        return b"\x89PNG-crop" + ts.to_bytes(4, "big")

    rep = run_precompute(
        manifest, out=out, vision=_BernabeuVision(), vlm_ladder=ladder,
        frame_provider=frame_provider, jersey_crop_provider=crop_provider,
        program_path=program_path, emit_frames=False,
    )
    assert rep["exit"] == 0
    assert rep["verdict"] == "Yes — #7 scored the goal"
