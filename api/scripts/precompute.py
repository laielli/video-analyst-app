#!/usr/bin/env python3
"""
Automated clip -> replay-cache precompute pipeline.

Turns a raw clip (starting with single-goal.mov) + a manifest of human knowledge into the
two artifacts the demo replays today:
  (a) frame stills under web/public/frames/  (the EvidencePanel renders these), and
  (b) a per-clip replay cache JSON in the exact shape api/interpreter/cache.py loads
      (detect map keyed by sampled-frame ms, read_text map keyed by det_id, events, clip).

No hand-annotation: Azure AI Vision provides the detect boxes (the real per-frame primitive),
manifest pins + gpt-4o VLM provide read_text, and the manifest carries the goal events and the
clip->.mov resolution. The pipeline mints stable det_ids, resolves pins by nearest-box, derives
events' scorer_det from that SAME pin resolution (the verdict linchpin), self-validates by
replaying the hero program over the generated cache, and writes idempotently (re-run = no-op).

    python scripts/precompute.py --clip single-goal --query hero-10-first-goal
    python scripts/precompute.py --manifest clips/single-goal/manifest.json --dry-run
    python scripts/precompute.py --clip single-goal --query hero-10-first-goal --require-ocr

Exit 0 = cache written + self-validated · 1 = produced but validation/grounding failed (or
--require-ocr and a read was skipped) · 2 = setup problem (no ffmpeg / no manifest / no creds
when required).

OPEN KNOBS (implementer's choices, justified in the PR):
  - frame sampling: detect on EVERY sampled ts in the window (cheap, ~13 frames @ 8fps; the
    verdict-relevant frames always carry a detect key — the dependency invariant holds).
  - det_id minting: p{N} by descending confidence per frame (highest = p0); manifest pins
    RENAME a matched det to a semantic id (p_messi) by nearest-box. Reproduces the hero cache.
  - JPEG production: ffmpeg-direct (-q:v, .jpg suffix to the destination), avoiding a Pillow dep.

Replay-mode note: this writes the cache; the interpreter replays it with zero live calls.
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
# put scripts/ on the path so we can reuse ocr_probe's extract_frame + load_dotenv, and api/
# so the interpreter package + vision adapters resolve.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(API_DIR))

from ocr_probe import extract_frame, load_dotenv  # noqa: E402

DEFAULT_OUT = API_DIR / "examples" / "hero_cache.json"
FRAMES_DIR = REPO_ROOT / "web" / "public" / "frames"
HERO_PROGRAM = API_DIR / "examples" / "hero_program.json"
RUNDOC_SCHEMA = API_DIR / "schema" / "run_doc.schema.json"

# jersey sub-region of a person box, mirroring op_crop math (primitives.py:77-79).
JERSEY = {"dx": 0.22, "dy": 0.10, "dw": 0.56, "dh": 0.22}
JERSEY_SCALE = 3.0  # upscale the crop so small digits clear the reader's minimum glyph size.


# --------------------------------------------------------------------------- errors

class SetupError(Exception):
    """A setup problem -> exit 2 (no ffmpeg / no manifest / missing required creds)."""


class GroundingError(Exception):
    """Produced output that failed validation/grounding -> exit 1."""


# --------------------------------------------------------------------------- manifest

def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SetupError(f"manifest not found: {path}")
    try:
        m = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SetupError(f"manifest is not valid JSON ({path}): {e}")
    for key in ("clip", "sampling"):
        if key not in m:
            raise SetupError(f"manifest {path} missing required '{key}' block")
    clip = m["clip"]
    if "source" not in clip:
        raise SetupError(
            f"manifest {path} clip block has no 'source' (.mov path) — the pipeline resolves the "
            "clip->.mov mapping from the manifest, not canned.py"
        )
    return m


def manifest_for_clip(clip_id: str) -> Path:
    """Resolve clips/<id>/manifest.json. Absent -> SetupError (exit 2), never synthesize."""
    return REPO_ROOT / "clips" / clip_id / "manifest.json"


def resolve_source(manifest: dict, manifest_path: Path) -> Path:
    """Resolve the .mov path from the manifest 'source' field (relative to the manifest dir)."""
    src = manifest["clip"]["source"]
    p = Path(src)
    if not p.is_absolute():
        p = (manifest_path.parent / p).resolve()
    return p


# --------------------------------------------------------------------------- clip block

def probe_dims(video: Path) -> tuple[int, int, int] | None:
    """(width, height, duration_ms) via ffprobe, or None if it can't probe."""
    if not shutil.which("ffprobe") or not video.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "json", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout
        data = json.loads(out)
        st = data["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        dur_ms = int(round(float(data["format"]["duration"]) * 1000))
        return w, h, dur_ms
    except (subprocess.CalledProcessError, KeyError, IndexError, ValueError):
        return None


def build_clip_block(manifest: dict, video: Path, registry_clip: dict | None) -> dict:
    """
    Reconcile clip dims: registry (canned.CLIPS, unedited) is the source of truth; the manifest
    fills any gap; ffprobe is a last-resort fallback. fps is the clip's NATIVE rate, not sampling.
    """
    mclip = manifest["clip"]
    base = {"id": mclip["id"]}
    # Prefer the registry, then manifest, then ffprobe.
    reg = registry_clip or {}
    probed = probe_dims(video)
    for field in ("width", "height", "duration_ms"):
        if field in reg:
            base[field] = reg[field]
        elif field in mclip:
            base[field] = mclip[field]
    if probed and ("width" not in base or "height" not in base or "duration_ms" not in base):
        pw, ph, pdur = probed
        base.setdefault("width", pw)
        base.setdefault("height", ph)
        base.setdefault("duration_ms", pdur)
    # fps: registry, then manifest; default 60 if truly unknown.
    base["fps"] = reg.get("fps", mclip.get("fps", 60))
    missing = [k for k in ("width", "height", "duration_ms") if k not in base]
    if missing:
        raise SetupError(
            f"could not determine clip {missing} (no registry/manifest value and ffprobe failed)"
        )
    return base


def registry_clip_for(clip_id: str) -> dict | None:
    try:
        import canned  # noqa: E402
    except Exception:  # noqa: BLE001
        return None
    return canned.CLIPS.get(clip_id)


# --------------------------------------------------------------------------- sampling

def sample_timestamps(sampling: dict) -> list[int]:
    """
    SAME stride the interpreter derives in op_sample_frames (primitives.py:28-29):
    stride = max(1, round(1000/fps)); ts = range(start, end+1, stride). Keys MUST align with
    what op_detect looks up so the verdict-relevant frames have a detect key.
    """
    start, end, fps = sampling["start_ms"], sampling["end_ms"], sampling["fps"]
    stride = max(1, round(1000 / fps))
    return list(range(start, end + 1, stride))


# --------------------------------------------------------------------------- det_id minting

def mint_det_ids(detections: list[dict]) -> list[dict]:
    """
    Stable per-frame ids: p{N} by DESCENDING confidence (highest = p0). Deterministic tie-break
    by box (x, then y) so equal confidences don't reorder run-to-run. Returns NEW dicts.
    """
    ordered = sorted(
        detections,
        key=lambda d: (-(d.get("confidence") or 0.0), d["box"]["x"], d["box"]["y"]),
    )
    return [{**d, "det_id": f"p{i}"} for i, d in enumerate(ordered)]


# --------------------------------------------------------------------------- box helpers

def round_box(b: dict, p: int = 4) -> dict:
    return {k: round(float(b[k]), p) for k in ("x", "y", "w", "h")}


def box_distance(a: dict, b: dict) -> float:
    """Squared distance between box top-left corners — for nearest-box pin matching."""
    return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2


def jersey_crop_box(person_box: dict) -> dict:
    b = person_box
    return {
        "x": b["x"] + JERSEY["dx"] * b["w"],
        "y": b["y"] + JERSEY["dy"] * b["h"],
        "w": JERSEY["dw"] * b["w"],
        "h": JERSEY["dh"] * b["h"],
    }


def crop_arg_pixels(norm_box: dict, width: int, height: int) -> str:
    """Normalized 0-1 box -> SOURCE-pixel 'x,y,w,h' crop arg for extract_frame."""
    x = int(round(norm_box["x"] * width))
    y = int(round(norm_box["y"] * height))
    w = max(1, int(round(norm_box["w"] * width)))
    h = max(1, int(round(norm_box["h"] * height)))
    return f"{x},{y},{w},{h}"


# --------------------------------------------------------------------------- detect

def detect_frame(video: Path, ts_ms: int, detect_classes: list[str], azure_vision,
                 *, warn) -> list[dict]:
    """
    Extract the frame, run Azure detect, filter to detect_classes (case-insensitive matching
    op_detect at primitives.py:48), normalize into cache detection dicts (det_ids minted later).
    Guards extract_frame's sys.exit(2) so one bad frame doesn't abort the run.
    """
    try:
        image_bytes = _safe_extract(video, ts_ms / 1000.0, crop=None, scale=1.0)
    except SetupError as e:
        warn(f"frame {ts_ms}ms: extraction failed ({e}); skipping")
        return []
    if image_bytes is None:
        warn(f"frame {ts_ms}ms: extraction returned nothing; skipping")
        return []
    try:
        result = azure_vision.analyze(image_bytes, detect=True, read=False)
    except Exception as e:  # noqa: BLE001 — surface SDK/auth/TPM errors as warnings, keep going
        warn(f"frame {ts_ms}ms: Azure detect failed ({type(e).__name__}: {e}); skipping")
        return []
    classes = [c.lower() for c in detect_classes]
    dets: list[dict] = []
    for o in result.objects:
        if classes and o.cls.lower() not in classes:
            continue
        dets.append({
            "cls": o.cls,
            "confidence": round(o.confidence, 4) if o.confidence is not None else None,
            "box": round_box({"x": o.box.x, "y": o.box.y, "w": o.box.w, "h": o.box.h}),
        })
    return dets


def _safe_extract(video: Path, ts_s: float, crop: str | None, scale: float):
    """
    Call ocr_probe.extract_frame but convert its sys.exit(2) into a SetupError we can catch,
    so one unreadable frame doesn't kill the whole pipeline.
    """
    try:
        return extract_frame(video, ts_s, crop, scale)
    except SystemExit as e:
        raise SetupError(f"extract_frame exited ({e.code})") from e


# --------------------------------------------------------------------------- pins

def apply_pins(detect: dict, manifest: dict, *, warn) -> dict[str, str]:
    """
    Resolve manifest pins by nearest-box AFTER detect + minting. RENAMES the matched det to the
    pin's semantic det_id (e.g. p_messi) IN PLACE in `detect`. Returns a map of
    pin_det_id -> resolved det_id (== pin_det_id on success) used to derive events' scorer_det.
    Pins with no matchable frame/detection are warned and dropped.
    """
    resolved: dict[str, str] = {}
    for pin in manifest.get("pins", []):
        pin_id = pin["det_id"]
        match = pin.get("match", {})
        ts = match.get("frame_ts_ms")
        target_box = match.get("nearest_box")
        frame_dets = detect.get(str(ts)) if ts is not None else None
        if not frame_dets or target_box is None:
            warn(f"pin {pin_id!r}: no detections at frame {ts}ms to match; pin dropped")
            continue
        nearest = min(frame_dets, key=lambda d: box_distance(d["box"], target_box))
        old_id = nearest["det_id"]
        nearest["det_id"] = pin_id
        resolved[pin_id] = pin_id
        if old_id != pin_id:
            # det_ids must stay unique within the frame; nothing else held pin_id, so we're safe.
            pass
    return resolved


# --------------------------------------------------------------------------- read_text / OCR

def read_jersey_numbers(detect: dict, manifest: dict, video: Path, clip: dict, azure_vlm_factory,
                        *, require_ocr, warn) -> dict[str, dict]:
    """
    Build the read_text map. OCR ladder (BINDING order): (1) manifest pin -> source 'pinned',
    verbatim; (2) gpt-4o VLM -> source 'live' when AZURE_OPENAI_* available + digits returned;
    (3) skip -> write NO entry (honest-empty, op_read_text degrades). Azure Read is NOT a rung.

    azure_vlm_factory is a zero-arg callable returning an AzureVLM (we pin construction to
    from_env per the test seam); call it lazily so we only touch creds when a VLM read is needed.
    """
    read_text: dict[str, dict] = {}

    # (1) pins win — write read_text under the (already renamed) pin det_id.
    pin_by_id = {p["det_id"]: p for p in manifest.get("pins", [])}
    pinned_ids = set()
    for pin_id, pin in pin_by_id.items():
        # only honor pins whose det actually exists in detect (apply_pins renamed it)
        if not _det_exists(detect, pin_id):
            continue
        entry = {"text": str(pin["text"]), "confidence": pin.get("confidence"), "source": "pinned"}
        if pin.get("note"):
            entry["note"] = pin["note"]
        read_text[pin_id] = entry
        pinned_ids.add(pin_id)

    # (2)+(3) VLM read for the remaining jersey-target dets (default: all detected persons not pinned).
    vlm = None
    vlm_unavailable = False
    targets = _ocr_targets(detect, pinned_ids)
    for ts_ms, det in targets:
        if vlm_unavailable:
            _skip_read(det["det_id"], require_ocr, warn, reason="VLM unavailable")
            continue
        if vlm is None:
            try:
                vlm = azure_vlm_factory()
            except Exception as e:  # noqa: BLE001 — missing creds / ValueError
                vlm_unavailable = True
                warn(f"gpt-4o VLM unavailable ({type(e).__name__}: {e}); skipping all VLM reads")
                _skip_read(det["det_id"], require_ocr, warn, reason="VLM unavailable")
                continue
        rt = _vlm_read_one(vlm, video, clip, ts_ms, det, warn=warn)
        if rt is not None and rt.get("text"):
            read_text[det["det_id"]] = rt
        else:
            _skip_read(det["det_id"], require_ocr, warn, reason="no legible digits")
    return read_text


def _ocr_targets(detect: dict, pinned_ids: set[str]) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for ts in sorted(int(k) for k in detect):
        for det in detect[str(ts)]:
            if det["det_id"] in pinned_ids:
                continue
            out.append((ts, det))
    return out


def _vlm_read_one(vlm, video: Path, clip: dict, ts_ms: int, det: dict, *, warn) -> dict | None:
    crop_box = jersey_crop_box(det["box"])
    crop_arg = crop_arg_pixels(crop_box, clip["width"], clip["height"])
    try:
        image_bytes = _safe_extract(video, ts_ms / 1000.0, crop=crop_arg, scale=JERSEY_SCALE)
    except SetupError as e:
        warn(f"det {det['det_id']} @ {ts_ms}ms: crop extraction failed ({e}); skipping read")
        return None
    if image_bytes is None:
        return None
    try:
        vr = vlm.read_jersey_number(image_bytes)
    except Exception as e:  # noqa: BLE001 — 429 / ValueError / SDK error -> skip (NOT Azure Read)
        warn(f"det {det['det_id']} @ {ts_ms}ms: VLM read failed ({type(e).__name__}: {e}); skipping")
        return None
    if vr.digits:
        return {"text": vr.digits, "confidence": None, "source": "live"}
    return None


def _skip_read(det_id: str, require_ocr: bool, warn, *, reason: str) -> None:
    warn(f"det {det_id}: read skipped ({reason}) — no read_text entry (honest-empty)")
    if require_ocr:
        raise GroundingError(f"--require-ocr: read for det {det_id} was skipped ({reason})")


def _det_exists(detect: dict, det_id: str) -> bool:
    return any(d["det_id"] == det_id for dets in detect.values() for d in dets)


# --------------------------------------------------------------------------- events

def build_events(manifest: dict, detect: dict, resolved_pins: dict[str, str], *, warn) -> list[dict]:
    """
    THE VERDICT LINCHPIN. Derive each event's scorer_det from the SAME pin resolution (not the
    manifest literal). Assert scorer_det is a minted det_id AND appears in detect at some cached
    ts — else op_answer's overlay re-scan drops the focus frame while the verdict says Yes.
    """
    events: list[dict] = []
    for ev in manifest.get("events", []):
        scorer_literal = ev.get("scorer_det")
        # resolve through the pin map when the literal names a pin id; else use it directly.
        scorer = resolved_pins.get(scorer_literal, scorer_literal)
        if not _det_exists(detect, scorer):
            raise GroundingError(
                f"event @ {ev.get('ts_ms')}ms: scorer_det {scorer!r} is not a minted det in the "
                "detect map — the grounded verdict would be a lie (op_answer overlay re-scan "
                "would drop the focus frame). Check the pin nearest_box matches a real detection."
            )
        out = {"ts_ms": ev["ts_ms"], "type": ev["type"], "scorer_det": scorer}
        if "box" in ev:
            out["box"] = round_box(ev["box"])
        events.append(out)
    return events


# --------------------------------------------------------------------------- canonical cache

def canonicalize(clip: dict, detect: dict, read_text: dict, events: list[dict],
                 *, regenerated_at: str | None) -> dict:
    """Deterministic output: sort detect keys numerically, detections by det_id, round boxes."""
    out_detect: dict[str, list[dict]] = {}
    for ts in sorted(int(k) for k in detect):
        dets = sorted(detect[str(ts)], key=lambda d: d["det_id"])
        out_detect[str(ts)] = [
            {"det_id": d["det_id"], "cls": d["cls"], "confidence": d["confidence"],
             "box": round_box(d["box"])}
            for d in dets
        ]
    cache: dict = {
        "clip": {
            "id": clip["id"], "width": clip["width"], "height": clip["height"],
            "duration_ms": clip["duration_ms"], "fps": clip["fps"],
        },
        "detect": out_detect,
        "read_text": {k: read_text[k] for k in sorted(read_text)},
        "events": sorted(events, key=lambda e: e["ts_ms"]),
    }
    if regenerated_at is not None:
        cache["regenerated_at"] = regenerated_at
    return cache


def serialize_cache(cache: dict) -> str:
    return json.dumps(cache, indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- structural validation

def validate_cache_structure(cache: dict) -> list[str]:
    """In-code structural check per docs/cache-schema.md (no separate JSON Schema file)."""
    errs: list[str] = []
    clip = cache.get("clip", {})
    for k in ("id", "width", "height", "duration_ms", "fps"):
        if k not in clip:
            errs.append(f"clip missing '{k}'")
    detect = cache.get("detect", {})
    all_det_ids: set[str] = set()
    for ts, dets in detect.items():
        try:
            int(ts)
        except (TypeError, ValueError):
            errs.append(f"detect key {ts!r} is not an int-string")
        seen: set[str] = set()
        for d in dets:
            for k in ("det_id", "cls", "confidence", "box"):
                if k not in d:
                    errs.append(f"detect[{ts}] detection missing '{k}'")
            did = d.get("det_id")
            if did in seen:
                errs.append(f"detect[{ts}] duplicate det_id {did!r}")
            seen.add(did)
            all_det_ids.add(did)
            b = d.get("box", {})
            for k in ("x", "y", "w", "h"):
                v = b.get(k)
                if not isinstance(v, (int, float)) or not (0 <= v <= 1):
                    errs.append(f"detect[{ts}] {did} box.{k}={v!r} not in [0,1]")
                elif round(v, 4) != v:
                    errs.append(f"detect[{ts}] {did} box.{k}={v!r} has >4 decimals")
    for det_id, rt in cache.get("read_text", {}).items():
        if det_id not in all_det_ids:
            errs.append(f"read_text key {det_id!r} is not a real det_id")
        if rt.get("source") not in ("live", "cached", "pinned"):
            errs.append(f"read_text[{det_id}] source {rt.get('source')!r} not in live/cached/pinned")
        if not rt.get("text"):
            errs.append(f"read_text[{det_id}] has empty text")
    for ev in cache.get("events", []):
        for k in ("ts_ms", "type", "scorer_det"):
            if k not in ev:
                errs.append(f"event missing '{k}'")
        if ev.get("scorer_det") not in all_det_ids:
            errs.append(f"event scorer_det {ev.get('scorer_det')!r} is not in the detect map")
    return errs


# --------------------------------------------------------------------------- self-validating replay

def self_validate_replay(out_path: Path) -> tuple[bool, str]:
    """
    Replay the pinned hero program over the freshly written cache and assert the run-doc validates
    against run_doc.schema.json (reusing run_program's gates). Returns (ok, message).
    """
    import jsonschema  # noqa: E402
    from interpreter import Cache, Interpreter  # noqa: E402

    program = json.loads(HERO_PROGRAM.read_text())["program"]
    run_doc = Interpreter(Cache.load(out_path)).run(program)
    schema = json.loads(RUNDOC_SCHEMA.read_text())
    errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(run_doc), key=lambda e: list(e.path))
    if errs:
        msg = "; ".join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errs[:5])
        return False, f"run-doc invalid: {msg}"
    verdict = run_doc["findings"]["verdict"]
    return True, verdict


# --------------------------------------------------------------------------- stills

def emit_stills(manifest: dict, video: Path, *, dry_run: bool, warn, log) -> list[str]:
    """
    Write one JPEG per unique evidence ts into web/public/frames/, named by the manifest. JPEG via
    ffmpeg-direct (-q:v) to the destination (extract_frame returns in-memory PNG and writes no file).
    """
    written: list[str] = []
    stills = manifest.get("stills", [])
    seen_ts: set[int] = set()
    for still in stills:
        ts = still["frame_ts_ms"]
        fname = still["filename"]
        if ts in seen_ts:
            continue
        seen_ts.add(ts)
        dest = FRAMES_DIR / fname
        if dry_run:
            log(f"  [dry-run] would write still {fname} from {ts}ms")
            written.append(fname)
            continue
        if extract_frame_to_jpeg(video, ts / 1000.0, dest, warn=warn):
            written.append(fname)
            log(f"  wrote still {dest.relative_to(REPO_ROOT)} ({ts}ms)")
    return written


def extract_frame_to_jpeg(video: Path, ts_s: float, dest: Path, *, warn, q: int = 2) -> bool:
    """ffmpeg-direct JPEG to dest. Returns True on success. Guards ffmpeg/video absence."""
    if not shutil.which("ffmpeg"):
        warn("ffmpeg not on PATH; cannot write stills")
        return False
    if not video.exists():
        warn(f"video not found ({video}); cannot write stills")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=dest.parent) as tmp:
        tmp_path = Path(tmp.name)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(ts_s), "-i", str(video),
           "-frames:v", "1", "-q:v", str(q), str(tmp_path)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        warn(f"ffmpeg failed writing {dest.name}: {e}")
        tmp_path.unlink(missing_ok=True)
        return False
    os.replace(tmp_path, dest)  # atomic
    return True


# --------------------------------------------------------------------------- atomic write

def atomic_write(path: Path, text: str) -> None:
    """Write via temp file + os.replace so a partial write never yields truncated JSON to readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False, dir=path.parent,
                                     encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------- pipeline core

def run_precompute(manifest: dict, manifest_path: Path, *, dry_run: bool, require_ocr: bool,
                   out: Path, azure_vision_factory=None, azure_vlm_factory=None,
                   regenerated_at: str | None = None, log=print) -> dict:
    """
    Core pipeline (importable for tests). Returns a result dict:
      {cache, out, written (bool), stills, verdict, ok, dry_run, warnings}
    Raises SetupError (exit 2) / GroundingError (exit 1) for fatal problems.

    azure_vision_factory / azure_vlm_factory default to AzureVision.from_env / AzureVLM.from_env
    (constructed via from_env per the test mock seam — never AzureVision(ep,key) directly).
    """
    warnings: list[str] = []

    def warn(msg: str) -> None:
        warnings.append(msg)
        log(f"  WARN: {msg}")

    clip_id = manifest["clip"]["id"]
    video = resolve_source(manifest, manifest_path)
    registry = registry_clip_for(clip_id)
    clip = build_clip_block(manifest, video, registry)

    sampling = manifest["sampling"]
    timestamps = sample_timestamps(sampling)
    detect_classes = manifest.get("detect_classes", ["person"])
    log(f"clip:   {clip['id']} ({clip['width']}x{clip['height']}, {clip['duration_ms']}ms)")
    log(f"sample: {sampling['start_ms']}-{sampling['end_ms']}ms @ {sampling['fps']}fps "
        f"-> {len(timestamps)} frames (stride {max(1, round(1000 / sampling['fps']))}ms)")

    if dry_run:
        # Sample + report only — NO Azure calls, NO writes.
        log("\n[dry-run] frames to detect on:")
        for ts in timestamps:
            log(f"  - {ts}ms")
        stills = emit_stills(manifest, video, dry_run=True, warn=warn, log=log)
        log(f"\n[dry-run] would write cache -> {out}")
        log(f"[dry-run] stills: {', '.join(stills) or '(none)'}")
        return {"cache": None, "out": out, "written": False, "stills": stills,
                "verdict": None, "ok": True, "dry_run": True, "warnings": warnings}

    # --- detect (Azure) over every sampled ts ---
    if azure_vision_factory is None:
        from vision.azure_vision import AzureVision  # noqa: E402
        azure_vision_factory = AzureVision.from_env
    try:
        azure_vision = azure_vision_factory()
    except Exception as e:  # noqa: BLE001 — missing creds: detect degrades to empty everywhere
        warn(f"Azure Vision unavailable ({type(e).__name__}: {e}); all detect frames will be empty")
        azure_vision = None

    detect: dict[str, list[dict]] = {}
    for ts in timestamps:
        raw = detect_frame(video, ts, detect_classes, azure_vision, warn=warn) if azure_vision else []
        minted = mint_det_ids(raw)
        if minted:
            detect[str(ts)] = minted
        log(f"  detect {ts}ms -> {len(minted)} {detect_classes}")

    # --- pins (rename matched dets) BEFORE read_text/events so det_ids are stable ---
    resolved_pins = apply_pins(detect, manifest, warn=warn)

    # --- read_text (OCR ladder: pin -> VLM -> skip) ---
    if azure_vlm_factory is None:
        from vision.vlm_read import AzureVLM  # noqa: E402
        azure_vlm_factory = AzureVLM.from_env
    read_text = read_jersey_numbers(detect, manifest, video, clip, azure_vlm_factory,
                                    require_ocr=require_ocr, warn=warn)

    # --- events (verdict linchpin: scorer_det from pin resolution, asserted in-detect) ---
    events = build_events(manifest, detect, resolved_pins, warn=warn)

    # --- canonical cache + structural validation ---
    cache = canonicalize(clip, detect, read_text, events, regenerated_at=regenerated_at)
    struct_errs = validate_cache_structure(cache)
    if struct_errs:
        raise GroundingError("cache structure invalid: " + "; ".join(struct_errs[:5]))

    # --- atomic write ---
    atomic_write(out, serialize_cache(cache))
    log(f"\nwrote cache -> {out}")

    # --- self-validating replay ---
    ok, msg = self_validate_replay(out)
    if not ok:
        raise GroundingError(msg)
    log(f"self-validate: run-doc valid — verdict: {msg!r}")

    # --- stills ---
    stills = emit_stills(manifest, video, dry_run=False, warn=warn, log=log)

    return {"cache": cache, "out": out, "written": True, "stills": stills,
            "verdict": msg, "ok": True, "dry_run": False, "warnings": warnings}


# --------------------------------------------------------------------------- CLI

def _resolve_manifest_path(args) -> Path:
    if args.manifest:
        return Path(args.manifest)
    if args.clip:
        return manifest_for_clip(args.clip)
    raise SetupError("provide --clip <id> or --manifest <path>")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Precompute a clip's replay cache + stills.")
    ap.add_argument("--clip", help="clip id (resolves clips/<id>/manifest.json)")
    ap.add_argument("--manifest", help="explicit manifest path (alternative to --clip)")
    ap.add_argument("--query", help="query id (for the report; e.g. hero-10-first-goal)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"cache output (default {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true", help="sample + report; no Azure calls, no writes")
    ap.add_argument("--require-ocr", action="store_true", help="hard-fail (exit 1) if any OCR read is skipped")
    args = ap.parse_args(argv)

    load_dotenv(API_DIR / ".env")

    try:
        manifest_path = _resolve_manifest_path(args)
        manifest = load_manifest(manifest_path)
    except SetupError as e:
        print(f"ERROR (setup): {e}", file=sys.stderr)
        return 2

    if args.query:
        print(f"query:  {args.query}")
    print(f"manifest: {manifest_path}\n")

    try:
        # NB: regenerated_at intentionally NOT set — it would break byte-identical idempotency.
        # (Plumbed through run_precompute as an opt-in for callers that don't need that guarantee.)
        result = run_precompute(
            manifest, manifest_path,
            dry_run=args.dry_run, require_ocr=args.require_ocr, out=args.out,
        )
    except SetupError as e:
        print(f"ERROR (setup): {e}", file=sys.stderr)
        return 2
    except GroundingError as e:
        print(f"ERROR (grounding): {e}", file=sys.stderr)
        return 1

    if result["dry_run"]:
        return 0
    print(f"\ndone — {result['out']} ({len(result['warnings'])} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
