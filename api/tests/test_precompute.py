"""
First pytest suite for api/ — the clip->cache precompute pipeline + the cache contract it must
satisfy. Azure and ffmpeg are mocked (conftest.py): no creds, no ffmpeg, no gitignored .mov.

These tests do NOT overwrite the committed examples/hero_cache.json — every run writes to a
tmp_path. The committed file stays untouched (it was hand-verified; we can't regenerate it live
in this env).
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

import precompute  # noqa: E402
from interpreter import Cache, Interpreter  # noqa: E402
from conftest import FakeVision, FakeVLM, HERO_FRAME_4625  # noqa: E402


def run_hero(tmp_path, hero_manifest, hero_manifest_path, *, vision=None, vlm=None,
             require_ocr=False, regenerated_at=None):
    """Run the pipeline over the hero manifest with mocked Azure, writing to tmp_path."""
    out = tmp_path / "cache.json"
    vision = vision if vision is not None else FakeVision()
    vlm = vlm if vlm is not None else FakeVLM("none")
    result = precompute.run_precompute(
        hero_manifest, hero_manifest_path,
        dry_run=False, require_ocr=require_ocr, out=out,
        azure_vision_factory=lambda: vision,
        azure_vlm_factory=lambda: vlm,
        regenerated_at=regenerated_at,
        log=lambda *a, **k: None,
    )
    return out, result, vision, vlm


# --------------------------------------------------------------------------- cache contract

def test_cache_loads_with_expected_accessors(tmp_path, hero_manifest, hero_manifest_path,
                                             fake_extract, no_stills):
    out, result, _, _ = run_hero(tmp_path, hero_manifest, hero_manifest_path)
    cache = Cache.load(out)

    assert cache.clip["id"] == "single-goal"
    assert cache.clip["width"] == 1872 and cache.clip["height"] == 1042

    frames = cache.analyzed_frames()
    assert 4625 in frames

    dets = cache.detections_at(4625)
    det_ids = [d["det_id"] for d in dets]
    # structural invariant: det_ids unique per frame
    assert len(det_ids) == len(set(det_ids))
    assert "p_messi" in det_ids

    # every read_text key is a real det_id
    all_ids = {d["det_id"] for ts in frames for d in cache.detections_at(ts)}
    for det_id in cache._read:  # the read_text map
        assert det_id in all_ids

    # every events[].scorer_det is in the detect map
    for ev in cache.goal_events():
        assert ev["scorer_det"] in all_ids

    assert cache.read_text_for("p_messi")["text"] == "10"
    goals = cache.goal_events()
    assert goals and goals[0]["scorer_det"] == "p_messi"


def test_hero_cache_replays_to_grounded_answer(tmp_path, hero_manifest, hero_manifest_path,
                                               fake_extract, no_stills):
    out, result, _, _ = run_hero(tmp_path, hero_manifest, hero_manifest_path)
    cache = Cache.load(out)
    program = json.loads((API_DIR / "examples" / "hero_program.json").read_text())["program"]
    run_doc = Interpreter(cache).run(program)

    # Assert the GROUNDED chain, not the hardcoded verdict string.
    ordered = next(b for b in run_doc["trace"] if b["op"] == "temporal_order")
    answer = next(b for b in run_doc["trace"] if b["op"] == "answer")

    # ordered.first.scorer_det == pinned scorer id
    env_ordered = _binding_value(program, cache, "temporal_order")
    assert env_ordered["first"]["scorer_det"] == "p_messi"
    assert env_ordered["subject_is_first_scorer"] is True
    # the scorer det survives filter text==10 (it is in subject_dets)
    assert "p_messi" in env_ordered["subject_dets"]
    # read_text[scorer].text == "10" with expected source
    rt = cache.read_text_for("p_messi")
    assert rt["text"] == "10" and rt["source"] == "pinned"

    # run-doc validates
    _assert_run_doc_valid(run_doc)
    assert run_doc["findings"]["verdict"] == "Yes — #10 scored the first goal"

    # degraded sibling: drop the pin's read so findings carry partial signal (no #10 -> No)
    no_pin_manifest = json.loads(json.dumps(hero_manifest))
    no_pin_manifest["pins"] = []
    no_pin_manifest["events"] = []  # no events -> no grounded scorer to assert
    out2 = tmp_path / "cache_degraded.json"
    precompute.run_precompute(
        no_pin_manifest, hero_manifest_path, dry_run=False, require_ocr=False, out=out2,
        azure_vision_factory=lambda: FakeVision(),
        azure_vlm_factory=lambda: FakeVLM("none"),  # no VLM digits -> no read_text
        log=lambda *a, **k: None,
    )
    run_doc2 = Interpreter(Cache.load(out2)).run(program)
    _assert_run_doc_valid(run_doc2)
    assert run_doc2["findings"]["verdict"].startswith("No")


def _assert_run_doc_valid(run_doc):
    import jsonschema
    schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(run_doc))
    assert not errs, "; ".join(e.message for e in errs[:5])


def _binding_value(program, cache, op):
    """Re-run the program collecting the env binding value for a given op (test helper)."""
    from interpreter.primitives import OPS
    env = {}
    for step in program:
        binding, _ = OPS[step["op"]](step, env, cache)
        env[step["id"]] = binding
        if step["op"] == op:
            return binding["value"]
    raise AssertionError(f"op {op} not found")


# --------------------------------------------------------------------------- sampling alignment

def test_detect_keys_align_with_program_sampling(tmp_path, hero_manifest, hero_manifest_path,
                                                 fake_extract, no_stills):
    out, _, _, _ = run_hero(tmp_path, hero_manifest, hero_manifest_path)
    cache = Cache.load(out)
    program = json.loads((API_DIR / "examples" / "hero_program.json").read_text())["program"]
    sf = next(s for s in program if s["op"] == "sample_frames")["args"]

    from interpreter.primitives import OPS
    binding, _ = OPS["sample_frames"]({"id": "f", "op": "sample_frames", "args": sf}, {}, cache)
    sampled_ts = {it["frame_ts_ms"] for it in binding["items"]}

    # DEPENDENCY invariant (not literal superset): every det key the verdict depends on is a
    # sampled ts, and the scorer's detect frame is among the sampled timestamps.
    scorer_ts = [int(ts) for ts in cache._detect
                 if any(d["det_id"] == "p_messi" for d in cache.detections_at(int(ts)))]
    assert scorer_ts, "scorer must appear in at least one detect frame"
    for ts in scorer_ts:
        assert ts in sampled_ts, f"detect frame {ts} not on the program's sampling stride"


# --------------------------------------------------------------------------- det_id minting

def test_det_id_minting_is_deterministic():
    raw = [
        {"cls": "person", "confidence": 0.5, "box": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
        {"cls": "person", "confidence": 0.9, "box": {"x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}},
        {"cls": "person", "confidence": 0.7, "box": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}},
    ]
    minted = precompute.mint_det_ids(raw)
    again = precompute.mint_det_ids(raw)
    ids = [d["det_id"] for d in minted]
    # p0 is highest confidence
    assert minted[0]["det_id"] == "p0" and minted[0]["confidence"] == 0.9
    assert ids == ["p0", "p1", "p2"]
    assert [d["det_id"] for d in again] == ids  # deterministic

    # pin renames to p_messi by nearest box
    detect = {"4625": precompute.mint_det_ids(HERO_FRAME_4625)}
    manifest = {"pins": [{"det_id": "p_messi", "text": "10",
                          "match": {"frame_ts_ms": 4625, "nearest_box": HERO_FRAME_4625[0]["box"]}}]}
    resolved = precompute.apply_pins(detect, manifest, warn=lambda *_: None)
    assert resolved == {"p_messi": "p_messi"}
    assert any(d["det_id"] == "p_messi" for d in detect["4625"])
    # the renamed det carries the conf-0.753 box (the original p0)
    messi = next(d for d in detect["4625"] if d["det_id"] == "p_messi")
    assert messi["confidence"] == 0.753


# --------------------------------------------------------------------------- OCR ladder

def test_ocr_ladder_pin_wins(tmp_path, hero_manifest, hero_manifest_path, fake_extract, no_stills):
    vlm = FakeVLM("digits")  # would return 10 for everyone, but pin must win for p_messi
    out, _, _, vlm = run_hero(tmp_path, hero_manifest, hero_manifest_path, vlm=vlm)
    cache = Cache.load(out)
    rt = cache.read_text_for("p_messi")
    assert rt["text"] == "10" and rt["source"] == "pinned"
    # VLM was never called FOR the pinned det (it is excluded from VLM targets). With a 'digits'
    # VLM the other 4 dets do get reads, but p_messi's value came from the pin, not the VLM.
    assert rt["source"] == "pinned"


def test_ocr_ladder_pin_wins_zero_vlm_when_only_pin(tmp_path, hero_manifest, hero_manifest_path,
                                                    fake_extract, no_stills):
    # Reduce detect to ONLY the pinned person so the VLM has no other targets -> zero VLM calls.
    vision = FakeVision(per_ts={4625: [HERO_FRAME_4625[0]]})
    vlm = FakeVLM("digits")
    out, _, _, vlm = run_hero(tmp_path, hero_manifest, hero_manifest_path, vision=vision, vlm=vlm)
    cache = Cache.load(out)
    assert cache.read_text_for("p_messi")["source"] == "pinned"
    assert vlm.calls == 0  # pin-only -> VLM never invoked


def test_ocr_ladder_vlm_then_skip(tmp_path, hero_manifest, hero_manifest_path, fake_extract, no_stills):
    # No pins; VLM raises 429 -> every det SKIPPED, no read_text entries, Azure Read never called.
    no_pin = json.loads(json.dumps(hero_manifest))
    no_pin["pins"] = []
    no_pin["events"] = []
    vlm = FakeVLM("none", raises=RuntimeError("429 Too Many Requests"))
    out = tmp_path / "cache.json"
    precompute.run_precompute(
        no_pin, hero_manifest_path, dry_run=False, require_ocr=False, out=out,
        azure_vision_factory=lambda: FakeVision(),
        azure_vlm_factory=lambda: vlm,
        log=lambda *a, **k: None,
    )
    cache = Cache.load(out)
    assert cache._read == {}  # no read_text entries written


def test_vlm_tpm_gated_degrades_gracefully(tmp_path, hero_manifest, hero_manifest_path,
                                           fake_extract, no_stills):
    # from_env (the factory) itself raises (missing creds / TPM) -> pipeline completes, valid cache,
    # warns, no non-zero exit. Hero is pinned so the verdict still grounds.
    def boom():
        raise ValueError("Azure OpenAI endpoint, key, and deployment are required.")

    out = tmp_path / "cache.json"
    result = precompute.run_precompute(
        hero_manifest, hero_manifest_path, dry_run=False, require_ocr=False, out=out,
        azure_vision_factory=lambda: FakeVision(),
        azure_vlm_factory=boom,
        log=lambda *a, **k: None,
    )
    assert result["ok"] is True and result["written"] is True
    assert any("VLM unavailable" in w for w in result["warnings"])
    cache = Cache.load(out)
    assert cache.read_text_for("p_messi")["source"] == "pinned"  # hero unaffected


def test_require_ocr_hard_fails_on_skip(tmp_path, hero_manifest, hero_manifest_path,
                                        fake_extract, no_stills):
    no_pin = json.loads(json.dumps(hero_manifest))
    no_pin["pins"] = []
    no_pin["events"] = []
    out = tmp_path / "cache.json"
    with pytest.raises(precompute.GroundingError):
        precompute.run_precompute(
            no_pin, hero_manifest_path, dry_run=False, require_ocr=True, out=out,
            azure_vision_factory=lambda: FakeVision(),
            azure_vlm_factory=lambda: FakeVLM("none"),  # returns no digits -> skip
            log=lambda *a, **k: None,
        )


# --------------------------------------------------------------------------- idempotency

def test_idempotent_rerun_is_noop(tmp_path, hero_manifest, hero_manifest_path, fake_extract, no_stills):
    out1, _, _, _ = run_hero(tmp_path, hero_manifest, hero_manifest_path)
    first = out1.read_bytes()
    # second run, fresh fakes, same inputs -> byte-identical JSON (no regenerated_at)
    out2, _, _, _ = run_hero(tmp_path / "second", hero_manifest, hero_manifest_path)
    # write second over a fresh path then compare bytes
    second = out2.read_bytes()
    assert first == second


def test_regenerated_at_excluded_from_idempotency(tmp_path, hero_manifest, hero_manifest_path,
                                                  fake_extract, no_stills):
    # CLI default omits regenerated_at; passing it is opt-in and is the ONLY differing field.
    out_a, _, _, _ = run_hero(tmp_path / "a", hero_manifest, hero_manifest_path, regenerated_at=None)
    out_b, _, _, _ = run_hero(tmp_path / "b", hero_manifest, hero_manifest_path,
                              regenerated_at="2026-06-06T00:00:00+00:00")
    a = json.loads(out_a.read_text())
    b = json.loads(out_b.read_text())
    assert "regenerated_at" not in a
    assert b.pop("regenerated_at") == "2026-06-06T00:00:00+00:00"
    assert a == b  # identical apart from the timestamp


# --------------------------------------------------------------------------- dry run

def test_dry_run_writes_nothing(tmp_path, hero_manifest, hero_manifest_path, fake_extract):
    out = tmp_path / "cache.json"
    vision = FakeVision()
    vlm = FakeVLM("digits")
    result = precompute.run_precompute(
        hero_manifest, hero_manifest_path, dry_run=True, require_ocr=False, out=out,
        azure_vision_factory=lambda: vision,
        azure_vlm_factory=lambda: vlm,
        log=lambda *a, **k: None,
    )
    assert result["dry_run"] is True and result["written"] is False
    assert not out.exists()              # no cache written
    assert vision.calls == 0             # zero Azure detect calls
    assert vlm.calls == 0               # zero Azure VLM calls
    assert len(fake_extract.calls) == 0  # zero ffmpeg extractions


# --------------------------------------------------------------------------- box coords

def test_box_coords_normalized(tmp_path, hero_manifest, hero_manifest_path, fake_extract, no_stills):
    out, _, _, _ = run_hero(tmp_path, hero_manifest, hero_manifest_path)
    cache = Cache.load(out)
    for ts in cache.analyzed_frames():
        for d in cache.detections_at(ts):
            b = d["box"]
            for k in ("x", "y", "w", "h"):
                assert 0 <= b[k] <= 1, f"{k}={b[k]} out of [0,1]"
                assert round(b[k], 4) == b[k], f"{k}={b[k]} has >4 decimals"
    for ev in cache.goal_events():
        if "box" in ev:
            for k in ("x", "y", "w", "h"):
                assert 0 <= ev["box"][k] <= 1
                assert round(ev["box"][k], 4) == ev["box"][k]


# --------------------------------------------------------------------------- stills

def test_stills_emitted_for_evidence_frames(tmp_path, hero_manifest, hero_manifest_path,
                                            fake_extract, monkeypatch):
    # Patch the JPEG writer to record (frame ts, dest) without ffmpeg, asserting one still per
    # unique evidence ts with the exact committed filenames.
    written: list[tuple[int, str]] = []

    def fake_jpeg(video, ts_s, dest, *, warn, q=2):
        written.append((int(round(ts_s * 1000)), Path(dest).name))
        return True

    monkeypatch.setattr(precompute, "extract_frame_to_jpeg", fake_jpeg)
    out, result, _, _ = run_hero(tmp_path, hero_manifest, hero_manifest_path)

    names = {n for _, n in written}
    assert names == {"goal-4000.jpg", "scorer-4625.jpg"}
    name_for_ts = {ts: n for ts, n in written}
    assert name_for_ts[4000] == "goal-4000.jpg"
    assert name_for_ts[4625] == "scorer-4625.jpg"
    # one still per unique evidence ts
    assert len(written) == len({ts for ts, _ in written})


# --------------------------------------------------------------------------- CLI / setup errors

def test_missing_manifest_exits_2(tmp_path, monkeypatch, capsys):
    # --clip with no manifest -> exit 2, never synthesizes defaults.
    rc = precompute.main(["--clip", "does-not-exist", "--out", str(tmp_path / "c.json")])
    assert rc == 2
    assert "manifest not found" in capsys.readouterr().err


def test_validate_cache_structure_catches_bad_scorer():
    bad = {
        "clip": {"id": "x", "width": 10, "height": 10, "duration_ms": 1, "fps": 60},
        "detect": {"100": [{"det_id": "p0", "cls": "person", "confidence": 0.5,
                            "box": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]},
        "read_text": {},
        "events": [{"ts_ms": 100, "type": "goal", "scorer_det": "ghost",
                    "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}}],
    }
    errs = precompute.validate_cache_structure(bad)
    assert any("scorer_det 'ghost'" in e for e in errs)
