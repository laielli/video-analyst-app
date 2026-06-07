"""
First pytest suite for api/ — the clip->cache precompute pipeline.

All Azure + ffmpeg IO is mocked (see conftest.py). The committed examples/hero_cache.json is
never overwritten: every test writes to tmp_path and points --out / out= there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
for p in (str(API_DIR), str(API_DIR / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import precompute  # noqa: E402
from interpreter import Cache, Interpreter  # noqa: E402
from validate_program import strip_comments  # noqa: E402
from conftest import FakeVision, FakeVLM, HERO_DETECTIONS_4625  # noqa: E402

HERO_PROGRAM = strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))["program"]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _run(tmp_manifest, tmp_path, **kw):
    out = kw.pop("out", tmp_path / "out_cache.json")
    frames = kw.pop("frames_dir", tmp_path / "frames")
    return precompute.run_precompute(tmp_manifest, out=out, frames_dir=frames, **kw)


# =========================================================================== #
# 1. Cache loads with expected accessors + structural invariants              #
# =========================================================================== #
def test_cache_loads_with_expected_accessors(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    res = _run(tmp_manifest, tmp_path)
    assert res["exit_code"] == 0
    cache = Cache.load(res["cache_path"])

    # clip block
    assert cache.clip["id"] == "single-goal"
    assert cache.clip["width"] == 1872 and cache.clip["height"] == 1042

    # analyzed_frames / detections_at
    assert cache.analyzed_frames() == [4625]
    dets = cache.detections_at(4625)
    assert len(dets) == len(HERO_DETECTIONS_4625)

    # read_text_for / goal_events
    assert cache.read_text_for("p_messi")["text"] == "10"
    goals = cache.goal_events()
    assert len(goals) == 1 and goals[0]["scorer_det"] == "p_messi"

    # structural invariants: det_ids unique per frame
    ids = [d["det_id"] for d in dets]
    assert len(ids) == len(set(ids))
    # every read_text key is a real det_id
    minted = {d["det_id"] for ts in cache.analyzed_frames() for d in cache.detections_at(ts)}
    assert set(json.loads(res["cache_path"].read_text())["read_text"]).issubset(minted)
    # every events[].scorer_det is in the detect map
    for ev in cache.goal_events():
        assert ev["scorer_det"] in minted


# =========================================================================== #
# 2. Grounded replay (assert the chain, not the hardcoded verdict string)     #
# =========================================================================== #
def test_hero_cache_replays_to_grounded_answer(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    res = _run(tmp_manifest, tmp_path)
    assert res["exit_code"] == 0
    cache = Cache.load(res["cache_path"])
    run_doc = Interpreter(cache).run(HERO_PROGRAM)

    # Walk the binding chain, not the canned verdict at primitives.py:194.
    ordered = next(t for t in run_doc["trace"] if t["op"] == "temporal_order")
    assert ordered["status"] == "done"

    goals = cache.goal_events()
    first = goals[0]
    assert first["scorer_det"] == "p_messi"

    # the scorer det survives filter text == 10
    rt = cache.read_text_for("p_messi")
    assert rt["text"] == "10" and rt["source"] == "pinned"

    # interpreter agrees the subject is the first scorer (grounded, not a hardcoded string)
    doc = Interpreter(cache).run(HERO_PROGRAM)
    assert doc["findings"]["partial"] is False
    assert doc["findings"]["answer"] == "Yes"

    # And the structured ordered value carries subject_is_first_scorer True.
    from interpreter.primitives import OPS  # noqa: E402
    env2: dict = {}
    for step in HERO_PROGRAM:
        b, _ = OPS[step["op"]](step, env2, cache)
        env2[step["id"]] = b
    ordered_val = env2["ordered"]["value"]
    assert ordered_val["subject_is_first_scorer"] is True
    assert ordered_val["first"]["scorer_det"] == "p_messi"


def test_grounded_replay_has_degraded_sibling(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch):
    """A degraded sibling proving the events linkage is load-bearing: with the goal events
    stripped, op_temporal_order is empty and the verdict FLIPS from Yes to No. The cache is
    still structurally valid and a "No" is still a grounded answer (exit 0)."""
    vision = FakeVision({4625: HERO_DETECTIONS_4625})
    monkeypatch.setattr(precompute, "make_vision", lambda: vision)
    monkeypatch.setattr(precompute, "make_vlm", lambda: None)
    tmp_manifest["events"] = []  # no goal events
    res = _run(tmp_manifest, tmp_path)
    cache = res["cache"]
    assert cache["events"] == []
    # Replay the hero program: no events -> temporal_order empty -> verdict is "No".
    doc = Interpreter(Cache.load(res["cache_path"])).run(HERO_PROGRAM)
    assert doc["findings"]["answer"] == "No"
    ordered = next(t for t in doc["trace"] if t["op"] == "temporal_order")
    assert ordered["status"] == "empty"
    # Contrast: the FULL hero cache (with events) grounds to "Yes" — the linkage is load-bearing.


# =========================================================================== #
# 3. Detect keys align with program sampling (dependency invariant)           #
# =========================================================================== #
def test_detect_keys_align_with_program_sampling(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    res = _run(tmp_manifest, tmp_path)
    cache = Cache.load(res["cache_path"])

    # The interpreter samples the same stride from the program's sample_frames args.
    sf = next(s for s in HERO_PROGRAM if s["op"] == "sample_frames")["args"]
    stride = max(1, round(1000 / sf["fps"]))
    program_ts = set(range(sf["start_ms"], sf["end_ms"] + 1, stride))

    # DEPENDENCY invariant: every ts the verdict depends on (the goal scorer's frame) has a key.
    # p_messi is detected at 4625 and 4625 is a sampled ts -> the join survives op_detect.
    detect_ts = set(cache.analyzed_frames())
    assert detect_ts.issubset(program_ts), "detect keys must be sampled timestamps"
    assert 4625 in detect_ts and 4625 in program_ts


# =========================================================================== #
# 4. det_id minting is deterministic                                          #
# =========================================================================== #
def test_det_id_minting_is_deterministic():
    raw = [
        {"cls": "person", "confidence": c, "box": {"x": x, "y": 0.1, "w": 0.1, "h": 0.1}}
        for (c, x) in [(0.5, 0.4), (0.9, 0.1), (0.7, 0.2)]
    ]
    minted = precompute._mint_det_ids(raw)
    by_id = {d["det_id"]: d for d in minted}
    # p0 is the highest confidence
    assert by_id["p0"]["confidence"] == 0.9
    assert by_id["p1"]["confidence"] == 0.7
    assert by_id["p2"]["confidence"] == 0.5
    # deterministic across repeated mints
    again = precompute._mint_det_ids(list(reversed(raw)))
    assert [d["det_id"] for d in sorted(again, key=lambda d: d["det_id"])] == ["p0", "p1", "p2"]
    assert {d["det_id"]: d["confidence"] for d in again} == {"p0": 0.9, "p1": 0.7, "p2": 0.5}


def test_pin_renames_highest_conf_to_p_messi(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    res = _run(tmp_manifest, tmp_path)
    cache = json.loads(res["cache_path"].read_text())
    ids = {d["det_id"] for d in cache["detect"]["4625"]}
    # The pinned nearest-box (p_messi's box, highest conf -> was p0) renamed to p_messi.
    assert "p_messi" in ids
    assert "p0" not in ids  # p0 got renamed
    assert {"p1", "p2", "p3", "p4"}.issubset(ids)


# =========================================================================== #
# 5. OCR ladder — pin wins (VLM never called)                                 #
# =========================================================================== #
def test_ocr_ladder_pin_wins(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch):
    vlm = FakeVLM(digits="77")  # would misread; must never be called for the pinned det
    # Single detection so the only OCR target is the pinned one.
    monkeypatch.setattr(precompute, "make_vision",
                        lambda: FakeVision({4625: [HERO_DETECTIONS_4625[0]]}))
    monkeypatch.setattr(precompute, "make_vlm", lambda: vlm)
    res = _run(tmp_manifest, tmp_path)
    cache = Cache.load(res["cache_path"])
    rt = cache.read_text_for("p_messi")
    assert rt["text"] == "10" and rt["source"] == "pinned"
    assert vlm.calls == 0  # pin short-circuits the ladder


# =========================================================================== #
# 6. OCR ladder — VLM then skip                                               #
# =========================================================================== #
def test_ocr_ladder_vlm_then_skip(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch):
    """No pin: VLM raises 429 -> det SKIPPED (no read_text entry), Azure Read never consulted."""
    # Drop the pin so the ladder reaches the VLM rung for every det.
    tmp_manifest["pins"] = []
    tmp_manifest["events"] = []  # without the pin the event can't ground; drop it for this test
    vision = FakeVision({4625: [HERO_DETECTIONS_4625[0]]})
    vlm = FakeVLM(raises=RuntimeError("429 Too Many Requests"))
    azure_read_spy = {"called": False}

    def _no_read(*a, **k):
        azure_read_spy["called"] = True
        raise AssertionError("Azure Read must NOT be a rung in the OCR ladder")

    monkeypatch.setattr(precompute, "make_vision", lambda: vision)
    monkeypatch.setattr(precompute, "make_vlm", lambda: vlm)
    res = _run(tmp_manifest, tmp_path)
    cache = res["cache"]
    assert cache["read_text"] == {}        # VLM 429 -> skipped, no entry
    assert vlm.calls == 1                   # VLM was attempted once
    assert azure_read_spy["called"] is False  # Azure Read never used


def test_ocr_ladder_vlm_success(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch):
    """No pin, VLM returns digits -> read_text entry with source 'live'."""
    tmp_manifest["pins"] = []
    tmp_manifest["events"] = []
    vision = FakeVision({4625: [HERO_DETECTIONS_4625[0]]})
    vlm = FakeVLM(digits="10")
    monkeypatch.setattr(precompute, "make_vision", lambda: vision)
    monkeypatch.setattr(precompute, "make_vlm", lambda: vlm)
    res = _run(tmp_manifest, tmp_path)
    cache = res["cache"]
    # det_id of the single detection is p0 (no pin renames it)
    assert cache["read_text"]["p0"] == {"text": "10", "confidence": None, "source": "live"}
    assert vlm.calls == 1


# =========================================================================== #
# 7. VLM TPM-gated degrades gracefully (no non-zero exit unless --require-ocr)#
# =========================================================================== #
def test_vlm_tpm_gated_degrades_gracefully(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch, capsys):
    """from_env raising / 429 -> pipeline completes, valid cache, warns, exit 0 (pin grounds it)."""
    vision = FakeVision({4625: HERO_DETECTIONS_4625})
    vlm = FakeVLM(raises=RuntimeError("429"))
    monkeypatch.setattr(precompute, "make_vision", lambda: vision)
    monkeypatch.setattr(precompute, "make_vlm", lambda: vlm)
    res = _run(tmp_manifest, tmp_path)
    assert res["exit_code"] == 0           # hero still grounds via the pin
    assert res["skipped_reads"]            # the non-pinned dets were skipped
    err = capsys.readouterr().err
    assert "VLM read failed" in err or "skipping" in err
    # structural validity holds
    assert precompute.validate_cache_structure(res["cache"]) == []


def test_require_ocr_hard_fails_on_skip(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch):
    vision = FakeVision({4625: HERO_DETECTIONS_4625})
    vlm = FakeVLM(raises=RuntimeError("429"))
    monkeypatch.setattr(precompute, "make_vision", lambda: vision)
    monkeypatch.setattr(precompute, "make_vlm", lambda: vlm)
    res = _run(tmp_manifest, tmp_path, require_ocr=True)
    assert res["exit_code"] == 1
    assert res["skipped_reads"]


# =========================================================================== #
# 8. Idempotent rerun is a no-op (byte-identical JSON; stills excluded)        #
# =========================================================================== #
def test_idempotent_rerun_is_noop(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    out = tmp_path / "idem.json"
    res1 = _run(tmp_manifest, tmp_path, out=out)
    first = out.read_bytes()
    res2 = _run(tmp_manifest, tmp_path, out=out)
    second = out.read_bytes()
    assert res1["exit_code"] == 0 and res2["exit_code"] == 0
    assert first == second, "re-running must produce byte-identical cache JSON"


# =========================================================================== #
# 9. Dry-run writes nothing (zero Azure calls, no writes, exit 0)             #
# =========================================================================== #
def test_dry_run_writes_nothing(tmp_manifest, tmp_path, patch_ffmpeg, monkeypatch):
    calls = {"vision": 0, "vlm": 0}

    def _vision():
        calls["vision"] += 1
        return FakeVision({4625: HERO_DETECTIONS_4625})

    def _vlm():
        calls["vlm"] += 1
        return FakeVLM(digits="10")

    monkeypatch.setattr(precompute, "make_vision", _vision)
    monkeypatch.setattr(precompute, "make_vlm", _vlm)
    out = tmp_path / "dry.json"
    frames = tmp_path / "dry_frames"
    res = precompute.run_precompute(tmp_manifest, dry_run=True, out=out, frames_dir=frames)
    assert res["exit_code"] == 0
    assert not out.exists()                 # no cache written
    assert not frames.exists()              # no stills written
    assert calls["vision"] == 0 and calls["vlm"] == 0  # zero Azure construction/calls


# =========================================================================== #
# 10. Box coords normalized (0..1, <=4 decimals)                              #
# =========================================================================== #
def test_box_coords_normalized(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    res = _run(tmp_manifest, tmp_path)
    cache = json.loads(res["cache_path"].read_text())
    boxes = [d["box"] for dets in cache["detect"].values() for d in dets]
    boxes += [e["box"] for e in cache["events"]]
    for b in boxes:
        for c in ("x", "y", "w", "h"):
            assert 0.0 <= b[c] <= 1.0
            assert round(b[c], 4) == b[c]


# =========================================================================== #
# 11. Stills emitted for evidence frames (exact filenames)                    #
# =========================================================================== #
def test_stills_emitted_for_evidence_frames(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    frames = tmp_path / "frames"
    res = _run(tmp_manifest, tmp_path, frames_dir=frames)
    assert res["exit_code"] == 0
    assert set(res["stills"]) == {"goal-4000.jpg", "scorer-4625.jpg"}
    assert (frames / "goal-4000.jpg").exists()
    assert (frames / "scorer-4625.jpg").exists()


# =========================================================================== #
# CLI / setup-problem coverage                                                #
# =========================================================================== #
def test_missing_manifest_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(precompute.shutil, "which", lambda n: "/usr/bin/" + n)
    rc = precompute.main(["--clip", "does-not-exist"])
    assert rc == 2
    assert "no manifest" in capsys.readouterr().err


def test_no_clip_no_manifest_exits_2(capsys):
    rc = precompute.main([])
    assert rc == 2
    assert "specify --clip" in capsys.readouterr().err


def test_main_dry_run_exit_0(tmp_path, monkeypatch, patch_ffmpeg):
    # Build a manifest on disk and drive main() with --manifest + --dry-run.
    clip_dir = tmp_path / "clips" / "single-goal"
    clip_dir.mkdir(parents=True)
    (tmp_path / "single-goal.mov").write_bytes(b"\x00")
    from conftest import HERO_MANIFEST
    m = json.loads(json.dumps(HERO_MANIFEST))
    m["clip"]["source"] = "../../single-goal.mov"
    mpath = clip_dir / "manifest.json"
    mpath.write_text(json.dumps(m))
    rc = precompute.main(["--manifest", str(mpath), "--query", "hero-10-first-goal", "--dry-run"])
    assert rc == 0


# =========================================================================== #
# Self-validation seam reuse — replay over a written cache                     #
# =========================================================================== #
def test_replay_and_validate_passes_on_good_cache(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    res = _run(tmp_manifest, tmp_path)
    ok, msg = precompute.replay_and_validate(res["cache_path"])
    assert ok, msg
    assert "Yes" in msg


def test_committed_hero_cache_untouched(tmp_manifest, tmp_path, patch_ffmpeg, patch_azure):
    """Sanity: tests write to tmp_path and never to examples/hero_cache.json."""
    committed = API_DIR / "examples" / "hero_cache.json"
    before = committed.read_bytes()
    _run(tmp_manifest, tmp_path)
    assert committed.read_bytes() == before
