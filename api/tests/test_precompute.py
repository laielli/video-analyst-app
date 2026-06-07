"""
First pytest suite for api/ — the precompute pipeline (api/scripts/precompute.py).

Azure + ffmpeg are mocked end to end (conftest fakes); nothing here calls Azure or touches the
gitignored single-goal.mov. Tests assert the GROUNDED verdict chain (not the hardcoded verdict
string in primitives.py) and the cache invariants from docs/cache-schema.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))

import precompute as pc  # noqa: E402
from interpreter import Cache, Interpreter  # noqa: E402


# --------------------------------------------------------------------------- helpers


def _run(manifest, manifest_path, tmp_out, *, vision, vlm, extract, **kw):
    return pc.run_precompute(
        manifest,
        manifest_path,
        out=tmp_out,
        program_path=API_DIR / "examples" / "hero_program.json",
        vision_factory=lambda: vision,
        vlm_factory=lambda: vlm,
        extract=extract,
        **kw,
    )


def _full_cache(manifest, manifest_path, tmp_path, *, vision, vlm, extract, emit_frames=False, **kw):
    out = tmp_path / "cache.json"
    _run(manifest, manifest_path, out, vision=vision, vlm=vlm, extract=extract,
         emit_frames=emit_frames, **kw)
    return out, json.loads(out.read_text())


# --------------------------------------------------------------------------- named tests


def test_cache_loads_with_expected_accessors(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract):
    out, data = _full_cache(manifest, manifest_path, tmp_path,
                            vision=fake_vision, vlm=fake_vlm_none, extract=fake_extract)
    cache = Cache.load(out)

    # .clip
    assert cache.clip["id"] == "single-goal"
    assert cache.clip["width"] == 1872 and cache.clip["height"] == 1042

    # .analyzed_frames() — sampled-frame ts with detections; 4625 present
    frames = cache.analyzed_frames()
    assert frames == sorted(frames)
    assert 4625 in frames

    # .detections_at — det_ids unique per frame
    dets = cache.detections_at(4625)
    ids = [d["det_id"] for d in dets]
    assert len(ids) == len(set(ids)), "det_ids must be unique within a frame"
    assert "p_messi" in ids

    # .read_text_for — every read_text key is a real det_id
    all_ids = {d["det_id"] for ts in frames for d in cache.detections_at(ts)}
    for det_id in data["read_text"]:
        assert det_id in all_ids, f"read_text key {det_id} is not a real det_id"
    assert cache.read_text_for("p_messi")["text"] == "10"

    # .goal_events — every scorer_det is in the detect map
    for ev in cache.goal_events():
        assert ev["scorer_det"] in all_ids


def test_hero_cache_replays_to_grounded_answer(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract, hero_program_path):
    from validate_program import strip_comments

    out, data = _full_cache(manifest, manifest_path, tmp_path,
                            vision=fake_vision, vlm=fake_vlm_none, extract=fake_extract)
    cache = Cache.load(out)
    program = strip_comments(json.loads(hero_program_path.read_text()))["program"]
    run_doc = Interpreter(cache).run(program)

    # The GROUNDED chain, not the verdict literal: first goal's scorer survives filter text==10,
    # carries read_text "10" (pinned), and is flagged the first scorer.
    ordered = next(t for t in run_doc["trace"] if t["op"] == "temporal_order")
    assert ordered["status"] == "done"

    goals = cache.goal_events()
    scorer = goals[0]["scorer_det"]
    assert scorer == "p_messi", "first goal's scorer must be the pinned scorer id"
    rt = cache.read_text_for(scorer)
    assert rt is not None and rt["text"] == "10" and rt["source"] == "pinned"

    # subject_is_first_scorer via re-running temporal_order semantics: the scorer survives the
    # text==10 filter (it has read_text 10) AND is the first goal scorer.
    tens = [t for t in run_doc["trace"] if t["op"] == "filter"][0]
    assert tens["status"] == "done", "filter text==10 must keep the scorer"

    assert run_doc["findings"]["answer"] == "Yes"
    assert run_doc["findings"]["verdict"] == "Yes — #10 scored the first goal"

    # run-doc validates against the contract
    import jsonschema
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc))
    assert not errs, errs

    # Degraded sibling: drop the pin's read_text -> verdict must flip / partial signal appears.
    degraded = json.loads(out.read_text())
    degraded["read_text"] = {}
    deg_path = tmp_path / "degraded.json"
    deg_path.write_text(json.dumps(degraded))
    deg_doc = Interpreter(Cache.load(deg_path)).run(program)
    assert deg_doc["findings"]["answer"] == "No", "no #10 read -> filter empty -> verdict flips"


def test_detect_keys_align_with_program_sampling(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract, hero_program_path):
    from validate_program import strip_comments
    from interpreter.primitives import op_sample_frames

    out, data = _full_cache(manifest, manifest_path, tmp_path,
                            vision=fake_vision, vlm=fake_vlm_none, extract=fake_extract)
    cache = Cache.load(out)
    program = strip_comments(json.loads(hero_program_path.read_text()))["program"]
    sample_step = next(s for s in program if s["op"] == "sample_frames")
    binding, _ = op_sample_frames(sample_step, {}, cache)
    program_ts = [f["frame_ts_ms"] for f in binding["items"]]

    # DEPENDENCY invariant: every sampled ts the verdict depends on (here, all of them, since we
    # detect on every sampled frame) has a detect key; the scorer frame is present.
    detect_keys = {int(k) for k in data["detect"]}
    assert 4625 in detect_keys
    assert set(program_ts) <= detect_keys, "every program-sampled ts must have a detect key"


def test_det_id_minting_is_deterministic():
    raw = [
        {"cls": "person", "confidence": 0.50, "box": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}},
        {"cls": "person", "confidence": 0.90, "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}},
        {"cls": "person", "confidence": 0.70, "box": {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}},
    ]
    minted = pc.mint_det_ids(raw)
    by_id = {m["det_id"]: m for m in minted}
    assert by_id["p0"]["confidence"] == 0.90, "p0 = highest confidence"
    assert by_id["p1"]["confidence"] == 0.70
    assert by_id["p2"]["confidence"] == 0.50
    # deterministic: re-mint -> same ids
    again = pc.mint_det_ids(list(reversed(raw)))
    assert [m["det_id"] for m in again] == [m["det_id"] for m in minted]
    assert {m["det_id"]: m["confidence"] for m in again} == {m["det_id"]: m["confidence"] for m in minted}

    # pin renames the matched det to p_messi by nearest box
    detect = {4625: pc.mint_det_ids(raw)}
    pins = [{"det_id": "p_messi", "text": "10",
             "match": {"frame_ts_ms": 4625, "nearest_box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}}]
    pc.apply_pins(detect, pins)
    ids_after = {d["det_id"] for d in detect[4625]}
    assert "p_messi" in ids_after
    messi = next(d for d in detect[4625] if d["det_id"] == "p_messi")
    assert messi["confidence"] == 0.90, "pin must land on the nearest box (the 0.90 det)"


def test_ocr_ladder_pin_wins(manifest, manifest_path, tmp_path, fake_vision, fake_extract):
    # A VLM that would return digits if called — but the pin must win and the VLM is NEVER called
    # for the pinned det. (It MAY be called for other non-pinned dets; we assert pin entry source.)
    from conftest import FakeVLM
    spy = FakeVLM(digits="99")
    out, data = _full_cache(manifest, manifest_path, tmp_path,
                            vision=fake_vision, vlm=spy, extract=fake_extract)
    assert data["read_text"]["p_messi"]["text"] == "10"
    assert data["read_text"]["p_messi"]["source"] == "pinned"
    # The pinned det never goes through the VLM; any 99s would be on OTHER dets only.
    assert "99" != data["read_text"]["p_messi"]["text"]


def test_ocr_ladder_vlm_then_skip(manifest, manifest_path, tmp_path, fake_vision, fake_extract):
    # No pin scenario: a VLM raising 429 -> that det is SKIPPED (no read_text entry), and Azure
    # Read is never requested (read=False on every analyze call).
    from conftest import FakeVLM
    no_pin = json.loads(manifest_path.read_text())
    no_pin["pins"] = []
    no_pin["events"] = []  # scorer_pin would dangle without the pin
    vlm = FakeVLM(raise_exc=RuntimeError("429"))
    out = tmp_path / "c.json"
    _run(no_pin, manifest_path, out, vision=fake_vision, vlm=vlm, extract=fake_extract, emit_frames=False)
    data = json.loads(out.read_text())
    assert data["read_text"] == {}, "VLM 429 with no pin -> no read_text entries"
    assert vlm.calls > 0, "VLM was attempted"
    assert fake_vision.read_requested == 0, "Azure Read must never be requested"


def test_vlm_tpm_gated_degrades_gracefully(manifest, manifest_path, tmp_path, fake_vision, fake_extract):
    # from_env-style factory raising -> pipeline still completes with a valid cache, warns, no
    # non-zero exit (unless --require-ocr). Here vlm_factory returns None (the no-VLM rung).
    out = tmp_path / "c.json"
    report = _run(manifest, manifest_path, out, vision=fake_vision, vlm=None,
                  extract=fake_extract, emit_frames=False)
    assert report["wrote_cache"] is True
    assert report["run_doc_valid"] is True
    data = json.loads(out.read_text())
    # pin still present (pin rung does not need the VLM)
    assert data["read_text"]["p_messi"]["text"] == "10"

    # --require-ocr with a non-pinned det and no VLM -> hard fail (exit 1)
    with pytest.raises(pc.PrecomputeError) as ei:
        _run(manifest, manifest_path, tmp_path / "c2.json", vision=fake_vision, vlm=None,
             extract=fake_extract, emit_frames=False, require_ocr=True)
    assert ei.value.exit_code == 1


def test_idempotent_rerun_is_noop(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract):
    out = tmp_path / "c.json"
    _run(manifest, manifest_path, out, vision=fake_vision, vlm=fake_vlm_none, extract=fake_extract, emit_frames=False)
    first = out.read_bytes()
    _run(manifest, manifest_path, out, vision=fake_vision, vlm=fake_vlm_none, extract=fake_extract, emit_frames=False)
    second = out.read_bytes()
    assert first == second, "re-running the pipeline must be byte-identical (idempotent, JSON only)"


def test_dry_run_writes_nothing(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_digits, fake_extract):
    out = tmp_path / "should-not-exist.json"
    report = _run(manifest, manifest_path, out, vision=fake_vision, vlm=fake_vlm_digits,
                  extract=fake_extract, dry_run=True)
    assert report["wrote_cache"] is False
    assert not out.exists(), "dry-run must not write the cache"
    # zero Azure calls
    assert fake_vision.analyze_calls == 0
    assert fake_vlm_digits.calls == 0
    # zero extract calls (dry-run only reports the sampling plan)
    assert fake_extract.calls == []


def test_box_coords_normalized(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract):
    out, data = _full_cache(manifest, manifest_path, tmp_path,
                            vision=fake_vision, vlm=fake_vlm_none, extract=fake_extract)
    for ts, dets in data["detect"].items():
        for d in dets:
            for c in ("x", "y", "w", "h"):
                v = d["box"][c]
                assert 0.0 <= v <= 1.0, f"box.{c}={v} out of [0,1]"
                # <= 4 decimals
                assert round(v, 4) == v, f"box.{c}={v} has > 4 decimals"


def test_stills_emitted_for_evidence_frames(manifest, manifest_path, tmp_path, fake_vision, fake_vlm_none, fake_extract, monkeypatch):
    # Patch emit_stills' ffmpeg/file work to a fake that just records filenames, so we assert the
    # ts->filename contract without invoking ffmpeg or the .mov.
    written = {}

    def fake_emit(video, evidence_frames, *, extract=None):
        out_paths = []
        for ef in evidence_frames:
            written[ef["filename"]] = ef["ts_ms"]
            out_paths.append(pc.FRAMES_DIR / ef["filename"])
        return out_paths

    monkeypatch.setattr(pc, "emit_stills", fake_emit)
    out = tmp_path / "c.json"
    report = _run(manifest, manifest_path, out, vision=fake_vision, vlm=fake_vlm_none,
                  extract=fake_extract, emit_frames=True)
    assert report["wrote_stills"] is True
    # exact filenames the unchanged EvidencePanel.frameSrc expects
    assert written == {"goal-4000.jpg": 4000, "scorer-4625.jpg": 4625}
