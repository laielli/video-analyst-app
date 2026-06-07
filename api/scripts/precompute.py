#!/usr/bin/env python3
"""
Automated clip -> replay-cache precompute pipeline (ViperGPT Layer 2/3 ingest).

Turns a raw clip + a per-clip manifest into the exact two artifacts the demo replays:
  (a) frame stills under web/public/frames/ (the EvidencePanel renders these), and
  (b) a per-clip replay cache JSON in the shape api/interpreter/cache.py loads
      (`clip`, `detect` keyed by sampled-frame ms, `read_text` keyed by det_id, `events`).

The pipeline is the orchestrator: it reuses the ffmpeg frame extraction from ocr_probe.py,
calls Azure AI Vision for `detect`, mints stable det_ids (Azure returns none), resolves
manifest pins by nearest-box to mint semantic ids (p_messi), runs the OCR ladder
(pin -> gpt-4o VLM -> skip; Azure Read is deliberately NOT a rung), derives the `events`
block from the pin resolution, and self-validates by replaying the hero program over the
freshly written cache and re-checking it against run_doc.schema.json.

It degrades gracefully when Azure creds are absent or TPM-gated: detect/OCR failures warn
and skip (honest-empty) rather than aborting, unless --require-ocr is set.

    python scripts/precompute.py --clip single-goal --query hero-10-first-goal
    python scripts/precompute.py --manifest ../clips/single-goal/manifest.json --dry-run
    python scripts/precompute.py --clip single-goal --query hero-10-first-goal --require-ocr

Exit 0 = cache written + self-validated · 1 = produced but validation/grounding failed
(or --require-ocr and a read was skipped) · 2 = setup problem (no ffmpeg / no manifest /
no creds when required).

OPEN-KNOB choices (justified in the PR body):
  - frame sampling : detect on EVERY sampled ts in the window (manifest sampling.detect_on
    = "all"); cheap (~13 frames @ 8fps) and guarantees stride alignment with op_sample_frames.
  - det_id scheme  : p{N} by descending confidence per frame (p0 = highest); manifest pins
    RENAME the nearest-box det to a semantic id (p_messi). Reproduces the hero cache.
  - JPEG stills    : ffmpeg-direct (-q:v, .jpg destination) — avoids the Pillow dependency.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent

# scripts-importing-scripts path hack (same pattern as codegen_probe.py / run_program.py):
# extract_frame + load_dotenv live in ocr_probe.py; the interpreter/vision packages live in api/.
sys.path.insert(0, str(HERE))      # ocr_probe (extract_frame, load_dotenv)
sys.path.insert(0, str(API_DIR))   # interpreter, vision, validate_program

from ocr_probe import extract_frame, load_dotenv  # noqa: E402

DEFAULT_FRAMES_DIR = REPO_ROOT / "web" / "public" / "frames"
DEFAULT_CACHE_OUT = API_DIR / "examples" / "hero_cache.json"
DSL_SCHEMA = API_DIR / "schema" / "dsl.schema.json"
RUNDOC_SCHEMA = API_DIR / "schema" / "run_doc.schema.json"
DEFAULT_PROGRAM = API_DIR / "examples" / "hero_program.json"


# --------------------------------------------------------------------------- #
# Manifest loading + clip resolution
# --------------------------------------------------------------------------- #

def _strip_comments(obj):
    """Drop dict keys starting with '_' (authoring comments), recursively."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def manifest_path_for_clip(clip_id: str) -> Path:
    return REPO_ROOT / "clips" / clip_id / "manifest.json"


def load_manifest(path: Path) -> dict:
    """Load + comment-strip a manifest. Raises FileNotFoundError if absent (caller -> exit 2)."""
    if not path.exists():
        raise FileNotFoundError(path)
    return _strip_comments(json.loads(path.read_text()))


def resolve_source(manifest: dict) -> Path:
    """The clip -> .mov resolution lives IN the manifest (so canned.py stays untouched).
    Relative paths are anchored at the repo root."""
    src = manifest["clip"]["source"]
    p = Path(src)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


# --------------------------------------------------------------------------- #
# Sampling — same stride op_sample_frames uses, so detect keys align
# --------------------------------------------------------------------------- #

def sampled_timestamps(sampling: dict) -> list[int]:
    """The sampled-frame ms for this clip, derived EXACTLY as op_sample_frames does
    (primitives.py:28-29): stride = max(1, round(1000/fps)); range(start, end+1, stride)."""
    start, end, fps = sampling["start_ms"], sampling["end_ms"], sampling["fps"]
    stride = max(1, round(1000 / fps))
    return list(range(start, end + 1, stride))


def detect_timestamps(manifest: dict) -> list[int]:
    """Which sampled ts to actually run detect on. detect_on='all' (default) => every
    sampled ts; 'evidence' => only evidence + event ts (sparse). Always a subset of the
    sampled grid so keys align."""
    sampling = manifest["sampling"]
    all_ts = sampled_timestamps(sampling)
    mode = sampling.get("detect_on", "all")
    if mode == "all":
        return all_ts
    # sparse: evidence frames + event frames, snapped to the nearest sampled ts.
    wanted = {ef["ts_ms"] for ef in manifest.get("evidence_frames", [])}
    wanted |= {e["ts_ms"] for e in manifest.get("events", [])}
    grid = set(all_ts)
    out = sorted(t for t in all_ts if t in wanted or t in grid and t in wanted)
    # snap any non-grid wanted ts to the nearest grid ts
    for t in wanted:
        if t not in grid and all_ts:
            out.append(min(all_ts, key=lambda g: abs(g - t)))
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# det_id minting + pin resolution
# --------------------------------------------------------------------------- #

def mint_det_ids(detections: list[dict]) -> list[dict]:
    """Assign p{N} by descending confidence (p0 = highest). Stable, deterministic.
    Ties broken by box position (x, then y) so the ordering is total. Returns NEW dicts."""
    ordered = sorted(
        detections,
        key=lambda d: (-(d["confidence"] if d["confidence"] is not None else -1.0),
                       d["box"]["x"], d["box"]["y"]),
    )
    out = []
    for i, d in enumerate(ordered):
        out.append({**d, "det_id": f"p{i}"})
    return out


def _box_center(b: dict) -> tuple[float, float]:
    return (b["x"] + b["w"] / 2.0, b["y"] + b["h"] / 2.0)


def _center_dist2(a: dict, b: dict) -> float:
    (ax, ay), (bx, by) = _box_center(a), _box_center(b)
    return (ax - bx) ** 2 + (ay - by) ** 2


def nearest_det(detections: list[dict], target_box: dict) -> dict | None:
    """The detection whose box center is nearest the target (manifest pin) box."""
    if not detections:
        return None
    return min(detections, key=lambda d: _center_dist2(d["box"], target_box))


# --------------------------------------------------------------------------- #
# Box math — normalized <-> source pixels for the OCR crop
# --------------------------------------------------------------------------- #

def jersey_subbox(person_box: dict) -> dict:
    """The jersey sub-region, matching op_crop's math (primitives.py:78-79)."""
    b = person_box
    return {
        "x": b["x"] + 0.22 * b["w"], "y": b["y"] + 0.10 * b["h"],
        "w": 0.56 * b["w"], "h": 0.22 * b["h"],
    }


def norm_box_to_px(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Normalized 0-1 box -> SOURCE pixel (x,y,w,h) for extract_frame's --crop arg."""
    x = max(0, round(box["x"] * width))
    y = max(0, round(box["y"] * height))
    w = max(1, round(box["w"] * width))
    h = max(1, round(box["h"] * height))
    return x, y, w, h


def round_box(box: dict, p: int = 4) -> dict:
    return {k: round(box[k], p) for k in ("x", "y", "w", "h")}


# --------------------------------------------------------------------------- #
# Frame extraction guards (extract_frame sys.exit(2)s on error — guard it)
# --------------------------------------------------------------------------- #

def safe_extract(video: Path, ts_ms: int, crop: str | None = None, scale: float = 1.0,
                 *, warn=print) -> bytes | None:
    """extract_frame sys.exit(2)s on a bad frame; wrap so one bad frame doesn't abort the
    run. Returns the frame bytes, or None on failure (caller degrades that frame)."""
    try:
        return extract_frame(video, ts_ms / 1000.0, crop=crop, scale=scale)
    except SystemExit:
        warn(f"WARN: frame extraction failed at {ts_ms}ms (crop={crop}); skipping.")
        return None
    except (subprocess.CalledProcessError, OSError) as e:
        warn(f"WARN: frame extraction error at {ts_ms}ms: {e}; skipping.")
        return None


# --------------------------------------------------------------------------- #
# ffprobe — clip dims/duration (manifest is the fallback / source of truth)
# --------------------------------------------------------------------------- #

def probe_clip(video: Path) -> dict | None:
    """Best-effort ffprobe for {width,height,duration_ms}. None if ffprobe absent/fails.
    Registry/manifest values stay authoritative; this only fills gaps."""
    if not shutil.which("ffprobe") or not video.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "json", str(video)],
            check=True, capture_output=True, text=True,
        ).stdout
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        dur = data.get("format", {}).get("duration")
        res = {}
        if stream.get("width"):
            res["width"] = int(stream["width"])
        if stream.get("height"):
            res["height"] = int(stream["height"])
        if dur:
            res["duration_ms"] = int(round(float(dur) * 1000))
        return res or None
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Vision adapters — constructed via from_env so tests can monkeypatch that seam
# --------------------------------------------------------------------------- #

class _SkipOCR(Exception):
    """Raised internally to signal a graceful OCR skip (no read_text entry written)."""


def _make_vision(warn=print):
    """AzureVision.from_env() or None if creds/SDK are unavailable."""
    try:
        from vision.azure_vision import AzureVision
        if not os.environ.get("AZURE_VISION_ENDPOINT") or not os.environ.get("AZURE_VISION_KEY"):
            warn("WARN: AZURE_VISION_* not set; detect will be empty (honest-empty).")
            return None
        return AzureVision.from_env()
    except Exception as e:  # noqa: BLE001 — surface any SDK/auth problem, keep going
        warn(f"WARN: could not construct AzureVision: {e}; detect will be empty.")
        return None


def _make_vlm(warn=print):
    """AzureVLM.from_env() or None if creds/SDK are unavailable."""
    try:
        from vision.vlm_read import AzureVLM
        if not (os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY")):
            return None
        return AzureVLM.from_env()
    except Exception:  # noqa: BLE001 — VLM is optional; the ladder falls to skip
        return None


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #

def run_precompute(
    manifest: dict,
    *,
    dry_run: bool = False,
    require_ocr: bool = False,
    out: Path | None = None,
    frames_dir: Path | None = None,
    warn=print,
    log=print,
) -> dict:
    """Build the replay cache (and stills, unless dry_run) for one clip from its manifest.

    Returns a dict: {cache, summary, written: bool, ocr_skipped: int, still_files: [...]}.
    Raises _SkipOCR-derived RuntimeError only via require_ocr enforcement at the CLI.
    """
    out = Path(out) if out else DEFAULT_CACHE_OUT
    frames_dir = Path(frames_dir) if frames_dir else DEFAULT_FRAMES_DIR

    clip = dict(manifest["clip"])
    video = resolve_source(manifest)

    # 1) clip block — registry/manifest authoritative, ffprobe only fills gaps.
    probed = probe_clip(video)
    clip_block = {
        "id": clip["id"],
        "width": clip.get("width"),
        "height": clip.get("height"),
        "duration_ms": clip.get("duration_ms"),
        "fps": clip.get("fps"),
    }
    if probed:
        for k, v in probed.items():
            if clip_block.get(k) is None:
                clip_block[k] = v
    for k in ("width", "height", "duration_ms"):
        if clip_block.get(k) is None:
            raise ValueError(f"clip.{k} missing from manifest and ffprobe could not supply it")
    width, height = clip_block["width"], clip_block["height"]

    vision = None if dry_run else _make_vision(warn=warn)
    classes = [c.lower() for c in manifest.get("detect_classes", ["person"])]
    det_ts = detect_timestamps(manifest)

    # 2) detect on each chosen ts -> mint det_ids
    detect_map: dict[str, list[dict]] = {}
    per_frame_summary: list[tuple[int, int]] = []
    for ts in det_ts:
        frame = None if dry_run else safe_extract(video, ts, warn=warn)
        raw_dets: list[dict] = []
        if frame is not None and vision is not None:
            try:
                result = vision.analyze(frame, detect=True, read=False)
                for d in result.objects:
                    if d.cls.lower() not in classes:
                        continue
                    b = d.box.rounded()
                    raw_dets.append({
                        "cls": d.cls,
                        "confidence": round(d.confidence, 4) if d.confidence is not None else None,
                        "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
                    })
            except Exception as e:  # noqa: BLE001 — one bad detect frame degrades to empty
                warn(f"WARN: detect failed at {ts}ms: {e}; frame degrades to empty.")
        minted = mint_det_ids(raw_dets) if raw_dets else []
        if minted:
            detect_map[str(ts)] = minted
        per_frame_summary.append((ts, len(minted)))

    # 3) resolve pins (nearest-box) AFTER detect so det_ids are stable, then OCR ladder.
    #    THE VERDICT LINCHPIN: pin resolution mints semantic ids, writes read_text under the
    #    SAME id, and feeds the events scorer_det. One ordered step.
    read_text_map: dict[str, dict] = {}
    ocr_skipped = 0
    # det_id remap: {(ts_key, old_det_id) -> new_det_id} applied to the detect map.
    pin_resolution: dict[str, str] = {}  # pin index marker -> resolved det_id

    vlm = None if dry_run else _make_vlm(warn=warn)
    ocr_cfg = manifest.get("ocr", {})
    ocr_scale = float(ocr_cfg.get("scale", 3.0))

    for pidx, pin in enumerate(manifest.get("pins", [])):
        match = pin["match"]
        ts = match["frame_ts_ms"]
        ts_key = str(ts)
        frame_dets = detect_map.get(ts_key, [])
        target = nearest_det(frame_dets, match["nearest_box"])
        if target is None:
            warn(f"WARN: pin '{pin['det_id']}' found no detection at {ts}ms to rename; "
                 "pin not applied (det may have degraded to empty).")
            continue
        new_id = pin["det_id"]
        old_id = target["det_id"]
        # RENAME the matched det in the detect map to the semantic id.
        for d in frame_dets:
            if d["det_id"] == old_id:
                d["det_id"] = new_id
        pin_resolution[f"pin{pidx}"] = new_id
        # Write read_text under the SAME renamed id (source: pinned, verbatim).
        read_text_map[new_id] = {
            "text": pin["text"],
            "confidence": pin.get("confidence"),
            "source": pin.get("source", "pinned"),
        }
        if pin.get("note"):
            read_text_map[new_id]["note"] = pin["note"]
        log(f"pin: {ts}ms nearest det '{old_id}' -> '{new_id}' = {pin['text']!r} "
            f"(source={read_text_map[new_id]['source']})")

    # 3b) VLM OCR ladder for any OCR-target det NOT already pinned. Default OCR targets:
    #     the same dets the program crops to jersey (all detected persons). The ladder:
    #     pin (already handled) -> gpt-4o VLM -> skip. Azure Read is NOT a rung.
    if not dry_run and vlm is not None:
        for ts_key, dets in detect_map.items():
            for d in dets:
                if d["det_id"] in read_text_map:
                    continue  # pinned already
                sub = jersey_subbox(d["box"])
                px = norm_box_to_px(sub, width, height)
                crop_arg = f"{px[0]},{px[1]},{px[2]},{px[3]}"
                frame = safe_extract(video, int(ts_key), crop=crop_arg, scale=ocr_scale, warn=warn)
                if frame is None:
                    ocr_skipped += 1
                    continue
                try:
                    read = vlm.read_jersey_number(frame)
                    if read.digits:
                        read_text_map[d["det_id"]] = {
                            "text": read.digits, "confidence": None, "source": "live",
                        }
                        log(f"vlm: {ts_key}ms {d['det_id']} -> {read.digits!r} (source=live)")
                    else:
                        ocr_skipped += 1  # no digits -> honest-empty (no entry)
                except Exception as e:  # noqa: BLE001 — 429/ValueError -> skip, NOT Azure Read
                    warn(f"WARN: VLM read skipped for {d['det_id']} at {ts_key}ms ({e}); "
                         "honest-empty (no read_text entry).")
                    ocr_skipped += 1
    elif not dry_run and vlm is None:
        # Count un-pinned OCR targets as skipped (TPM-gated / no creds). Hero unaffected.
        for dets in detect_map.values():
            for d in dets:
                if d["det_id"] not in read_text_map:
                    ocr_skipped += 1

    # 4) events — derive scorer_det from the pin-resolution result (NOT a raw literal).
    events = build_events(manifest, detect_map, pin_resolution, read_text_map, warn=warn)

    # 5) assemble + canonicalize
    cache = assemble_cache(clip_block, detect_map, read_text_map, events)

    # 6) stills (skip in dry-run)
    still_files: list[str] = []
    if not dry_run:
        still_files = emit_stills(manifest, video, frames_dir, warn=warn, log=log)

    summary = {
        "clip": clip_block["id"],
        "detect_frames": len(detect_map),
        "total_dets": sum(len(v) for v in detect_map.values()),
        "read_text": len(read_text_map),
        "events": len(events),
        "ocr_skipped": ocr_skipped,
        "per_frame": per_frame_summary,
        "stills": still_files,
    }

    written = False
    if not dry_run:
        atomic_write_json(out, cache)
        written = True

    return {"cache": cache, "summary": summary, "written": written,
            "ocr_skipped": ocr_skipped, "still_files": still_files, "out": out}


def build_events(manifest, detect_map, pin_resolution, read_text_map, *, warn=print) -> list[dict]:
    """Build the events block. scorer_det is rewritten from the pin resolution that produced
    the read_text, so the join chain (events.scorer_det -> detect det_id -> read_text) is
    guaranteed before write. Events whose scorer cannot be grounded are still written but warn."""
    # map manifest scorer_det (semantic id) -> resolved id (they are the same id, but go
    # through the resolution so we PROVE the pin actually landed on a real det).
    resolved_ids = set(pin_resolution.values())
    all_det_ids = {d["det_id"] for dets in detect_map.values() for d in dets}
    out = []
    for ev in manifest.get("events", []):
        scorer = ev.get("scorer_det")
        # Prefer the pin-resolved id (proves the linkage); fall back to literal only if it is
        # already a real det id (so we never invent a phantom scorer).
        if scorer in resolved_ids:
            resolved = scorer
        elif scorer in all_det_ids:
            resolved = scorer
        else:
            warn(f"WARN: event {ev.get('ts_ms')}ms scorer_det '{scorer}' is not a minted "
                 "det_id (pin may not have landed); event written but verdict may flip.")
            resolved = scorer
        out.append({
            "ts_ms": ev["ts_ms"],
            "type": ev["type"],
            "scorer_det": resolved,
            "box": round_box(ev["box"]) if ev.get("box") else None,
        })
    return out


def assemble_cache(clip_block, detect_map, read_text_map, events) -> dict:
    """Canonicalize for determinism: numeric-sorted detect keys, det_id-sorted detections,
    rounded boxes."""
    detect_out: dict[str, list[dict]] = {}
    for ts_key in sorted(detect_map, key=lambda k: int(k)):
        dets = sorted(detect_map[ts_key], key=lambda d: d["det_id"])
        detect_out[ts_key] = [
            {
                "det_id": d["det_id"],
                "cls": d["cls"],
                "confidence": d["confidence"],
                "box": round_box(d["box"]),
            }
            for d in dets
        ]
    return {
        "clip": clip_block,
        "detect": detect_out,
        "read_text": dict(sorted(read_text_map.items())),
        "events": sorted(events, key=lambda e: e["ts_ms"]),
    }


def atomic_write_json(path: Path, data: dict) -> None:
    """Write canonical JSON atomically (temp + os.replace) so a concurrent Cache.load never
    reads a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# Stills — ffmpeg-direct JPEG to the exact filenames the frontend expects
# --------------------------------------------------------------------------- #

def emit_stills(manifest, video: Path, frames_dir: Path, *, warn=print, log=print) -> list[str]:
    """Write one JPEG per unique evidence frame into frames_dir, using the EXACT filenames
    the manifest declares (reproducing goal-4000.jpg / scorer-4625.jpg; frameSrc is NOT
    edited). ffmpeg-direct (-q:v 2, .jpg) avoids the Pillow dependency."""
    if not shutil.which("ffmpeg"):
        warn("WARN: ffmpeg not on PATH; skipping stills.")
        return []
    if not video.exists():
        warn(f"WARN: source video not found ({video}); skipping stills.")
        return []
    frames_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    seen: set[str] = set()
    for ef in manifest.get("evidence_frames", []):
        fname = ef["filename"]
        if fname in seen:
            continue
        seen.add(fname)
        dest = frames_dir / fname
        ts = ef["ts_ms"] / 1000.0
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", str(ts), "-i", str(video), "-frames:v", "1", "-q:v", "2", str(dest)]
        try:
            subprocess.run(cmd, check=True)
            written.append(fname)
            log(f"still: {ef['ts_ms']}ms -> {dest}")
        except (subprocess.CalledProcessError, OSError) as e:
            warn(f"WARN: still extraction failed for {fname} at {ef['ts_ms']}ms: {e}; skipping.")
    return written


# --------------------------------------------------------------------------- #
# Self-validation — structural cache check + hero-program replay
# --------------------------------------------------------------------------- #

def validate_cache_structure(cache: dict) -> list[str]:
    """Enforce docs/cache-schema.md in-code. Returns a list of error strings (empty = OK)."""
    errs: list[str] = []
    clip = cache.get("clip", {})
    for k in ("id", "width", "height", "duration_ms"):
        if k not in clip or clip[k] is None:
            errs.append(f"clip.{k} missing")
    detect = cache.get("detect", {})
    read_text = cache.get("read_text", {})
    events = cache.get("events", [])
    all_det_ids: set[str] = set()
    for ts_key, dets in detect.items():
        if not ts_key.lstrip("-").isdigit():
            errs.append(f"detect key {ts_key!r} is not an integer-ms string")
        seen_ids: set[str] = set()
        for d in dets:
            did = d.get("det_id")
            if not did:
                errs.append(f"detection at {ts_key} missing det_id")
                continue
            if did in seen_ids:
                errs.append(f"duplicate det_id {did!r} within frame {ts_key}")
            seen_ids.add(did)
            all_det_ids.add(did)
            b = d.get("box", {})
            for coord in ("x", "y", "w", "h"):
                v = b.get(coord)
                if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                    errs.append(f"det {did} box.{coord}={v} not in [0,1]")
                elif round(v, 4) != v:
                    errs.append(f"det {did} box.{coord}={v} has >4 decimals")
    for did in read_text:
        if did not in all_det_ids:
            errs.append(f"read_text key {did!r} is not a real det_id in detect")
    for rt in read_text.values():
        if rt.get("source") not in ("live", "cached", "pinned"):
            errs.append(f"read_text source {rt.get('source')!r} not in {{live,cached,pinned}}")
    for ev in events:
        scorer = ev.get("scorer_det")
        if scorer is not None and scorer not in all_det_ids:
            errs.append(f"event {ev.get('ts_ms')}ms scorer_det {scorer!r} not in detect map")
    return errs


def self_validate_replay(cache_path: Path, *, program_path: Path | None = None) -> tuple[bool, str]:
    """Replay the hero program over the freshly written cache and re-validate the run-doc
    against run_doc.schema.json. Reuses the run_program.py machinery (Cache/Interpreter +
    jsonschema). Returns (ok, message)."""
    program_path = program_path or DEFAULT_PROGRAM
    try:
        import jsonschema
    except ImportError:
        return False, "jsonschema not installed (pip install -r requirements.txt)"
    from interpreter import Cache, Interpreter
    from validate_program import strip_comments

    program = strip_comments(json.loads(program_path.read_text()))["program"]
    run_doc = Interpreter(Cache.load(cache_path)).run(program)

    rd_schema = json.loads(RUNDOC_SCHEMA.read_text())
    rd_errs = sorted(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc),
                     key=lambda e: list(e.path))
    if rd_errs:
        msgs = "; ".join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
                         for e in rd_errs[:5])
        return False, f"run-doc invalid: {msgs}"
    verdict = run_doc["findings"]["verdict"]
    return True, verdict


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _print_summary(summary: dict, *, out=print) -> None:
    out(f"\nclip: {summary['clip']}  "
        f"({summary['detect_frames']} detect frame(s), {summary['total_dets']} det(s), "
        f"{summary['read_text']} read_text, {summary['events']} event(s))")
    out("per-frame detect:")
    for ts, n in summary["per_frame"]:
        mark = "OK" if n else "··"
        out(f"  [{mark}] {ts:>6}ms -> {n} person(s)")
    if summary["ocr_skipped"]:
        out(f"ocr skipped (honest-empty): {summary['ocr_skipped']}")
    if summary["stills"]:
        out(f"stills: {', '.join(summary['stills'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Precompute a clip's replay cache + stills.")
    ap.add_argument("--clip", default=None, help="clip id (resolves clips/<id>/manifest.json)")
    ap.add_argument("--manifest", type=Path, default=None, help="explicit manifest path")
    ap.add_argument("--query", default=None, help="query id (informational; cache is per-clip)")
    ap.add_argument("--dry-run", action="store_true", help="sample + report; no Azure, no writes")
    ap.add_argument("--require-ocr", action="store_true", help="hard-fail (exit 1) if any OCR read is skipped")
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE_OUT, help="cache output path")
    ap.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR, help="stills output dir")
    args = ap.parse_args(argv)

    load_dotenv(API_DIR / ".env")

    # Resolve the manifest (exit 2 on absence — never synthesize defaults).
    if args.manifest:
        mpath = args.manifest
    elif args.clip:
        mpath = manifest_path_for_clip(args.clip)
    else:
        print("ERROR: pass --clip <id> or --manifest <path>.", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(mpath)
    except FileNotFoundError:
        print(f"ERROR: manifest not found: {mpath}\n"
              "  (a clip with no manifest is a setup problem — author "
              "clips/<id>/manifest.json first.)", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, KeyError) as e:
        print(f"ERROR: manifest invalid: {e}", file=sys.stderr)
        return 2

    if not shutil.which("ffmpeg") and not args.dry_run:
        print("ERROR: ffmpeg not found on PATH (needed to extract frames). "
              "Use --dry-run to plan without extraction.", file=sys.stderr)
        return 2

    try:
        res = run_precompute(
            manifest,
            dry_run=args.dry_run,
            require_ocr=args.require_ocr,
            out=args.out,
            frames_dir=args.frames_dir,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    _print_summary(res["summary"])

    if args.dry_run:
        print("\ndry-run: no Azure calls, no writes.")
        return 0

    # --require-ocr opts into hard-fail on any skipped read.
    if args.require_ocr and res["ocr_skipped"]:
        print(f"\nFAIL: --require-ocr set but {res['ocr_skipped']} OCR read(s) were skipped.",
              file=sys.stderr)
        return 1

    # Structural self-validation of the written cache.
    struct = validate_cache_structure(res["cache"])
    if struct:
        print(f"\nFAIL: cache structure invalid ({len(struct)} error(s)):", file=sys.stderr)
        for m in struct[:10]:
            print(f"  - {m}", file=sys.stderr)
        return 1

    # Replay self-validation: the cache must replay the hero program to a valid run-doc.
    ok, msg = self_validate_replay(res["out"])
    if not ok:
        print(f"\nFAIL: self-validation replay failed: {msg}", file=sys.stderr)
        return 1
    print(f"\nwrote {res['out']}")
    print(f"self-validated: replays to a valid run-doc — findings: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
