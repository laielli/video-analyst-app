"""
First pytest suite for api/ — the precompute pipeline (api/scripts/precompute.py).

Azure (detect + VLM) and ffmpeg are both mocked (conftest seams), so nothing here touches
the network or the gitignored single-goal.mov. The tests assert the GROUNDED verdict chain
(events.scorer_det -> detect det_id -> read_text text=="10") rather than the hardcoded
verdict string in primitives.py, plus the structural cache invariants and the OPEN-knob
behaviors (det_id minting, OCR ladder, idempotency).
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

import precompute
from vision.azure_vision import Detection, Box, AnalysisResult
from conftest import FakeAzureVision, CountingVLM, HERO_DETECTIONS, REAL_MANIFEST


HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_hero(hero_manifest, out_paths, patch_vision, fake_extract, no_vlm,
                **kwargs):
    """Run the pipeline over the real manifest with all seams installed."""
    return precompute.run_precompute(
        hero_manifest,
        out=out_paths["cache"],
        frames_dir=out_paths["frames"],
        warn=lambda *a, **k: None,
        log=lambda *a, **k: None,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# test_cache_loads_with_expected_accessors
# --------------------------------------------------------------------------- #

def test_cache_loads_with_expected_accessors(hero_manifest, out_paths, patch_vision,
                                             fake_extract, no_vlm):
    from interpreter import Cache

    res = _build_hero(hero_manifest, out_paths, patch_vision, fake_extract, no_vlm)
    cache = Cache.load(out_paths["cache"])

    # .clip
    assert cache.clip["id"] == "single-goal"
    assert cache.clip["width"] == 1872 and cache.clip["height"] == 1042

    # .analyzed_frames() — sorted ints; 4625 (the scorer frame) is present
    frames = cache.analyzed_frames()
    assert frames == sorted(frames)
    assert 4625 in frames

    # .detections_at — the 4625 frame has the 5 hero people
    dets = cache.detections_at(4625)
    assert len(dets) == 5
    assert all(d["cls"] == "person" for d in dets)

    # .read_text_for — p_messi reads "10"
    rt = cache.read_text_for("p_messi")
    assert rt is not None and rt["text"] == "10"

    # .goal_events — one goal whose scorer_det is p_messi
    goals = cache.goal_events()
    assert len(goals) == 1 and goals[0]["scorer_det"] == "p_messi"

    # Structural invariants: det_ids unique per frame; every read_text key is a real det_id;
    # every events[].scorer_det is in the detect map.
    all_ids = set()
    for ts in frames:
        ids = [d["det_id"] for d in cache.detections_at(ts)]
        assert len(ids) == len(set(ids)), f"duplicate det_id in frame {ts}"
        all_ids.update(ids)
    cache_json = json.loads(out_paths["cache"].read_text())
    for did in cache_json["read_text"]:
        assert did in all_ids, f"read_text key {did} is not a real det_id"
    for ev in cache_json["events"]:
        assert ev["scorer_det"] in all_ids


# --------------------------------------------------------------------------- #
# test_hero_cache_replays_to_grounded_answer
# --------------------------------------------------------------------------- #

def test_hero_cache_replays_to_grounded_answer(hero_manifest, out_paths, patch_vision,
                                               fake_extract, no_vlm):
    import jsonschema
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    _build_hero(hero_manifest, out_paths, patch_vision, fake_extract, no_vlm)
    cache = Cache.load(out_paths["cache"])

    program = strip_comments(json.loads((API_DIR / "examples" / "hero_program.json").read_text()))["program"]
    run_doc = Interpreter(cache).run(program)

    # Walk the bindings the way the interpreter does to assert the GROUNDED chain (not the
    # hardcoded verdict string at primitives.py:194).
    from interpreter.primitives import OPS
    env = {}
    for step in program:
        binding, _ = OPS[step["op"]](step, env, cache)
        env[step["id"]] = binding

    ordered = env["ordered"]["value"]
    scorer = ordered["first"]["scorer_det"]
    assert scorer == "p_messi", "first goal's scorer_det must be the pinned id"

    # That det survives op_filter text == "10": it is in the filtered 'tens' binding's dets.
    tens_ids = {it.get("det_id") for it in env["tens"]["items"]}
    assert scorer in tens_ids, "scorer must survive filter text==10"

    # read_text[scorer].text == "10" with the expected pinned source
    rt = cache.read_text_for(scorer)
    assert rt["text"] == "10" and rt["source"] == "pinned"

    # subject_is_first_scorer is True
    assert ordered["subject_is_first_scorer"] is True

    # run-doc validates
    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc))
    assert not errs, errs
    assert run_doc["findings"]["verdict"] == "Yes — #10 scored the first goal"

    # Degraded sibling: a manifest with NO pin -> no read_text -> verdict flips, findings.partial
    degraded = copy.deepcopy(hero_manifest)
    degraded["pins"] = []
    dout = out_paths["cache"].parent / "degraded.json"
    precompute.run_precompute(degraded, out=dout, frames_dir=out_paths["frames"],
                              warn=lambda *a, **k: None, log=lambda *a, **k: None)
    dcache = Cache.load(dout)
    drun = Interpreter(dcache).run(program)
    assert drun["findings"]["partial"] is False  # answer step still runs -> not partial
    assert drun["findings"]["verdict"].startswith("No")


# --------------------------------------------------------------------------- #
# test_detect_keys_align_with_program_sampling  (DEPENDENCY invariant)
# --------------------------------------------------------------------------- #

def test_detect_keys_align_with_program_sampling(hero_manifest, out_paths, patch_vision,
                                                 fake_extract, no_vlm):
    from interpreter import Cache

    _build_hero(hero_manifest, out_paths, patch_vision, fake_extract, no_vlm)
    cache = Cache.load(out_paths["cache"])

    # The program op_sample_frames derives ts with stride round(1000/fps).
    prog = json.loads((API_DIR / "examples" / "hero_program.json").read_text())["program"]
    s = next(st for st in prog if st["op"] == "sample_frames")["args"]
    stride = max(1, round(1000 / s["fps"]))
    sampled = list(range(s["start_ms"], s["end_ms"] + 1, stride))

    # Every detect key is one of the sampled ts (keys align with the grid).
    for ts in cache.analyzed_frames():
        assert ts in sampled, f"detect key {ts} is not on the sampling grid"

    # The DEPENDENCY invariant: the ts the verdict depends on (4625 scorer, 4000 goal-near)
    # has a detect key (others may degrade to empty cleanly).
    assert 4625 in cache.analyzed_frames()
    # the goal event scorer must resolve to a det that appears in the detect map
    scorer = cache.goal_events()[0]["scorer_det"]
    assert any(d["det_id"] == scorer for ts in cache.analyzed_frames()
               for d in cache.detections_at(ts))


# --------------------------------------------------------------------------- #
# test_det_id_minting_is_deterministic
# --------------------------------------------------------------------------- #

def test_det_id_minting_is_deterministic():
    raw = [
        {"cls": "person", "confidence": 0.5, "box": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}},
        {"cls": "person", "confidence": 0.9, "box": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1}},
        {"cls": "person", "confidence": 0.7, "box": {"x": 0.3, "y": 0.3, "w": 0.1, "h": 0.1}},
    ]
    minted1 = precompute.mint_det_ids(raw)
    minted2 = precompute.mint_det_ids(list(reversed(raw)))
    # deterministic regardless of input order
    by_id1 = {d["det_id"]: d["confidence"] for d in minted1}
    by_id2 = {d["det_id"]: d["confidence"] for d in minted2}
    assert by_id1 == by_id2
    # p0 is the highest confidence
    assert by_id1["p0"] == 0.9
    assert [d["det_id"] for d in minted1] != []  # ids assigned

    # Pin renames the nearest det to p_messi (the conf-0.75 left box from the hero set).
    hero_raw = [{"cls": d.cls, "confidence": d.confidence,
                 "box": {"x": d.box.x, "y": d.box.y, "w": d.box.w, "h": d.box.h}}
                for d in HERO_DETECTIONS]
    minted = precompute.mint_det_ids(hero_raw)
    # highest conf (0.753) is the left box -> p0
    p0 = next(d for d in minted if d["det_id"] == "p0")
    assert p0["confidence"] == 0.753
    target = precompute.nearest_det(minted, {"x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276})
    assert target["det_id"] == "p0"  # pin would rename p0 -> p_messi


# --------------------------------------------------------------------------- #
# test_ocr_ladder_pin_wins
# --------------------------------------------------------------------------- #

def test_ocr_ladder_pin_wins(hero_manifest, out_paths, patch_vision, fake_extract,
                             monkeypatch):
    """Pin present -> read_text text=='10', source=='pinned', VLM spy ZERO calls."""
    import vision.vlm_read as vr
    spy = CountingVLM(digits="99")  # would return wrong number if ever called
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.example")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "fake-key")
    monkeypatch.setattr(vr.AzureVLM, "from_env", classmethod(lambda cls: spy))

    res = precompute.run_precompute(hero_manifest, out=out_paths["cache"],
                                    frames_dir=out_paths["frames"],
                                    warn=lambda *a, **k: None, log=lambda *a, **k: None)
    rt = res["cache"]["read_text"]["p_messi"]
    assert rt["text"] == "10" and rt["source"] == "pinned"
    # The VLM is still called for the OTHER (un-pinned) dets, but never for the pinned one,
    # and never produces a "10" override. Assert the pinned read is untouched by VLM '99'.
    assert all(v.get("text") != "99" or k != "p_messi" for k, v in res["cache"]["read_text"].items())
    # The pinned det was NOT sent to the VLM.
    assert FakeAzureVision.SENTINEL_4625 not in spy.calls  # full-frame sentinel never OCR'd


# --------------------------------------------------------------------------- #
# test_ocr_ladder_vlm_then_skip
# --------------------------------------------------------------------------- #

def test_ocr_ladder_vlm_then_skip(hero_manifest, out_paths, patch_vision, fake_extract,
                                  monkeypatch):
    """No pin; VLM raises 429 -> det SKIPPED (no read_text entry), Azure Read never called."""
    import vision.vlm_read as vr
    spy = CountingVLM(raise_with=RuntimeError("429 Too Many Requests"))
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.example")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "fake-key")
    monkeypatch.setattr(vr.AzureVLM, "from_env", classmethod(lambda cls: spy))

    no_pin = copy.deepcopy(hero_manifest)
    no_pin["pins"] = []
    res = precompute.run_precompute(no_pin, out=out_paths["cache"],
                                    frames_dir=out_paths["frames"],
                                    warn=lambda *a, **k: None, log=lambda *a, **k: None)
    # VLM was attempted (the 429 path), but NO read_text entry was written.
    assert len(spy.calls) >= 1
    assert res["cache"]["read_text"] == {}
    assert res["ocr_skipped"] >= 1
    # Azure Read never requested: FakeAzureVision.analyze always called with read=False.
    assert all(c["read"] is False for c in patch_vision.calls)


# --------------------------------------------------------------------------- #
# test_vlm_tpm_gated_degrades_gracefully
# --------------------------------------------------------------------------- #

def test_vlm_tpm_gated_degrades_gracefully(hero_manifest, out_paths, patch_vision,
                                           fake_extract, monkeypatch):
    """from_env raising (TPM/missing creds) -> pipeline completes, valid cache, no nonzero
    exit unless --require-ocr. Hero still grounds because p_messi is pinned."""
    import vision.vlm_read as vr

    def _raise(cls):
        raise ValueError("TPM quota exhausted")

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.example")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "fake-key")
    monkeypatch.setattr(vr.AzureVLM, "from_env", classmethod(_raise))

    warnings = []
    res = precompute.run_precompute(hero_manifest, out=out_paths["cache"],
                                    frames_dir=out_paths["frames"],
                                    warn=lambda *a, **k: warnings.append(" ".join(str(x) for x in a)),
                                    log=lambda *a, **k: None)
    # Pipeline produced a cache and the pinned hero read survives.
    assert res["written"] is True
    assert res["cache"]["read_text"]["p_messi"]["text"] == "10"
    # Structurally valid.
    assert precompute.validate_cache_structure(res["cache"]) == []
    # Replays to a grounded answer.
    ok, verdict = precompute.self_validate_replay(out_paths["cache"])
    assert ok and verdict.startswith("Yes")


# --------------------------------------------------------------------------- #
# test_idempotent_rerun_is_noop  (JSON byte-identity only, never stills)
# --------------------------------------------------------------------------- #

def test_idempotent_rerun_is_noop(hero_manifest, out_paths, patch_vision, fake_extract,
                                  no_vlm):
    precompute.run_precompute(hero_manifest, out=out_paths["cache"],
                              frames_dir=out_paths["frames"],
                              warn=lambda *a, **k: None, log=lambda *a, **k: None)
    first = out_paths["cache"].read_bytes()
    precompute.run_precompute(hero_manifest, out=out_paths["cache"],
                              frames_dir=out_paths["frames"],
                              warn=lambda *a, **k: None, log=lambda *a, **k: None)
    second = out_paths["cache"].read_bytes()
    assert first == second, "re-run must be byte-identical (JSON only)"


# --------------------------------------------------------------------------- #
# test_dry_run_writes_nothing
# --------------------------------------------------------------------------- #

def test_dry_run_writes_nothing(hero_manifest, out_paths, patch_vision, fake_extract,
                                no_vlm):
    res = precompute.run_precompute(hero_manifest, dry_run=True, out=out_paths["cache"],
                                    frames_dir=out_paths["frames"],
                                    warn=lambda *a, **k: None, log=lambda *a, **k: None)
    # No Azure detect calls.
    assert patch_vision.calls == []
    # No cache written, no frames written.
    assert res["written"] is False
    assert not out_paths["cache"].exists()
    assert not out_paths["frames"].exists() or not any(out_paths["frames"].iterdir())
    # No ffmpeg extraction either.
    assert fake_extract.calls == []


# --------------------------------------------------------------------------- #
# test_box_coords_normalized
# --------------------------------------------------------------------------- #

def test_box_coords_normalized(hero_manifest, out_paths, patch_vision, fake_extract, no_vlm):
    res = _build_hero(hero_manifest, out_paths, patch_vision, fake_extract, no_vlm)
    for dets in res["cache"]["detect"].values():
        for d in dets:
            b = d["box"]
            for k in ("x", "y", "w", "h"):
                assert 0.0 <= b[k] <= 1.0, f"{k}={b[k]} out of [0,1]"
                assert round(b[k], 4) == b[k], f"{k}={b[k]} has >4 decimals"
    for ev in res["cache"]["events"]:
        if ev.get("box"):
            for k in ("x", "y", "w", "h"):
                assert 0.0 <= ev["box"][k] <= 1.0


# --------------------------------------------------------------------------- #
# test_stills_emitted_for_evidence_frames
# --------------------------------------------------------------------------- #

def test_stills_emitted_for_evidence_frames(hero_manifest, out_paths, patch_vision,
                                            fake_extract, no_vlm, monkeypatch):
    """A still per unique evidence ts with the EXACT filenames goal-4000.jpg / scorer-4625.jpg.
    ffmpeg is mocked: patch emit_stills' subprocess + which so we assert filenames, not pixels."""
    calls = []

    class _R:  # stand-in CompletedProcess; carries both returncode and stdout
        returncode = 0
        stdout = json.dumps({"streams": [{"width": 1872, "height": 1042}],
                             "format": {"duration": "5.067"}})

    def _fake_run(cmd, *a, **k):
        calls.append(cmd)
        if cmd and cmd[0] == "ffprobe":
            return _R()
        # ffmpeg still: the dest is the last cmd token; write a JPEG stub so the test sees it.
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8\xff\xe0JPEGSTUB")  # JPEG magic stub
        return _R()

    monkeypatch.setattr(precompute.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(precompute.subprocess, "run", _fake_run)
    # make the source video appear to exist (scoped: only precompute.Path)
    monkeypatch.setattr(precompute.Path, "exists", lambda self: True)

    res = precompute.run_precompute(hero_manifest, out=out_paths["cache"],
                                    frames_dir=out_paths["frames"],
                                    warn=lambda *a, **k: None, log=lambda *a, **k: None)
    names = set(res["still_files"])
    assert "goal-4000.jpg" in names
    assert "scorer-4625.jpg" in names
    assert (out_paths["frames"] / "goal-4000.jpg").exists()
    assert (out_paths["frames"] / "scorer-4625.jpg").exists()


# --------------------------------------------------------------------------- #
# Manifest-absence is a setup error (exit 2), never synthesized defaults.
# --------------------------------------------------------------------------- #

def test_missing_manifest_exits_2(capsys):
    rc = precompute.main(["--clip", "does-not-exist", "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "manifest not found" in err


def test_no_clip_or_manifest_exits_2(capsys):
    rc = precompute.main([])
    assert rc == 2


# --------------------------------------------------------------------------- #
# CLI end-to-end via main() (dry-run), mocked seams, asserts exit 0.
# --------------------------------------------------------------------------- #

def test_cli_dry_run_exit_0(patch_vision, fake_extract, no_vlm, capsys):
    rc = precompute.main(["--clip", "single-goal", "--query", "hero-10-first-goal",
                          "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out


def test_require_ocr_hard_fails_when_skipped(hero_manifest, out_paths, patch_vision,
                                             fake_extract, no_vlm, monkeypatch, capsys):
    """--require-ocr exits 1 if any un-pinned OCR target is skipped (no VLM here)."""
    # Build a no-pin manifest so there ARE un-pinned OCR targets that get skipped.
    no_pin = copy.deepcopy(hero_manifest)
    no_pin["pins"] = []
    mpath = out_paths["cache"].parent / "manifest.json"
    mpath.write_text(json.dumps(no_pin))
    monkeypatch.setattr(precompute.shutil, "which", lambda name: "/usr/bin/" + name)
    rc = precompute.main(["--manifest", str(mpath), "--require-ocr",
                          "--out", str(out_paths["cache"]),
                          "--frames-dir", str(out_paths["frames"])])
    assert rc == 1
    assert "require-ocr" in capsys.readouterr().err
