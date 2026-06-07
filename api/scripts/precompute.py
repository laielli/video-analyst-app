#!/usr/bin/env python3
"""
Automated clip -> cache precompute pipeline.

Turns a raw clip (e.g. single-goal.mov) + a per-clip manifest (clips/<id>/manifest.json)
into the exact two artifacts the demo replays:

  (a) frame stills under web/public/frames/ (real JPEGs, the names EvidencePanel hardcodes), and
  (b) a per-clip replay cache JSON in the precise shape api/interpreter/cache.py loads:
      clip{} + detect{"<ts_ms>": [..]} + read_text{"<det_id>": {..}} + events[..].

The pipeline reuses the ffmpeg extraction from ocr_probe.extract_frame and the vision
adapters (AzureVision.analyze for detect, AzureVLM.read_jersey_number for jersey OCR). It
degrades gracefully when Azure creds are absent / TPM-gated, and is idempotent: re-running
over the same inputs writes a byte-identical cache JSON.

    python scripts/precompute.py --clip single-goal --query hero-10-first-goal
    python scripts/precompute.py --clip single-goal --query hero-10-first-goal --dry-run
    python scripts/precompute.py --manifest ../clips/single-goal/manifest.json --require-ocr

Exit codes:
  0 = cache written + self-validated (replays the pinned program to a grounded answer).
  1 = produced but validation/grounding failed (or --require-ocr and an OCR read was skipped).
  2 = setup problem (no ffmpeg / no manifest / no creds when required).

OPEN-KNOB choices (see docs/cache-schema.md + the PR body):
  - frame-sampling: detect on EVERY sampled ts in the window (cheap; default).
  - det_id minting: `p{N}` by descending confidence per frame (p0 = highest); manifest pins
    rename the nearest-box detection to a semantic id (p_messi).
  - JPEG production: ffmpeg-direct (.jpg + -q:v), avoiding a Pillow dependency.

Prereqs: ffmpeg on PATH. Azure creds in api/.env (see .env.example) only for the live detect
/ VLM read paths; the pin-only hero path needs neither (it stays honest-empty / pinned).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent
sys.path.insert(0, str(HERE))      # ocr_probe (extract_frame, load_dotenv)
sys.path.insert(0, str(API_DIR))   # interpreter, vision, canned

from ocr_probe import extract_frame, load_dotenv  # noqa: E402  (scripts-importing-scripts path hack)

DEFAULT_CACHE_OUT = API_DIR / "examples" / "hero_cache.json"
FRAMES_DIR = REPO_ROOT / "web" / "public" / "frames"
RUNDOC_SCHEMA = API_DIR / "schema" / "run_doc.schema.json"
HERO_PROGRAM = API_DIR / "examples" / "hero_program.json"

VALID_SOURCES = ("live", "cached", "pinned")


# --------------------------------------------------------------------------- #
# diagnostics                                                                  #
# --------------------------------------------------------------------------- #
def _warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


def _err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


class PrecomputeError(Exception):
    """Setup-level failure -> the CLI maps this to exit 2."""


# --------------------------------------------------------------------------- #
# manifest                                                                     #
# --------------------------------------------------------------------------- #
def _strip_comments(obj):
    """Drop dict keys starting with '_' (authoring comments), recursively."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def resolve_manifest_path(clip: str | None, manifest: str | Path | None) -> Path:
    """A clip id maps to clips/<id>/manifest.json; --manifest overrides. Missing -> error (exit 2)."""
    if manifest is not None:
        p = Path(manifest)
    elif clip is not None:
        p = REPO_ROOT / "clips" / clip / "manifest.json"
    else:
        raise PrecomputeError("specify --clip <id> or --manifest <path>")
    if not p.exists():
        raise PrecomputeError(
            f"no manifest at {p} — author clips/<id>/manifest.json first "
            "(the pipeline never synthesizes defaults)"
        )
    return p


def load_manifest(path: Path) -> dict:
    m = _strip_comments(json.loads(Path(path).read_text()))
    m["__path__"] = str(Path(path).resolve())
    for key in ("clip", "sampling"):
        if key not in m:
            raise PrecomputeError(f"manifest {path} missing required '{key}' block")
    if "source" not in m["clip"]:
        raise PrecomputeError(f"manifest {path} clip block missing 'source' (.mov path)")
    return m


def resolve_source(manifest: dict) -> Path:
    """The clip's .mov, resolved relative to the manifest file."""
    base = Path(manifest["__path__"]).parent
    return (base / manifest["clip"]["source"]).resolve()


def sampled_timestamps(sampling: dict) -> list[int]:
    """The SAME stride the interpreter samples on (op_sample_frames): stride=round(1000/fps)."""
    start, end, fps = sampling["start_ms"], sampling["end_ms"], sampling["fps"]
    stride = max(1, round(1000 / fps))
    return list(range(start, end + 1, stride))


# --------------------------------------------------------------------------- #
# geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def _round_box(box: dict, p: int = 4) -> dict:
    return {k: round(float(box[k]), p) for k in ("x", "y", "w", "h")}


def _box_center(box: dict) -> tuple[float, float]:
    return box["x"] + box["w"] / 2.0, box["y"] + box["h"] / 2.0


def _center_dist(a: dict, b: dict) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return math.hypot(ax - bx, ay - by)


def _jersey_subbox(person_box: dict) -> dict:
    """The jersey crop region — same math as op_crop region 'jersey' (primitives.py:77-79)."""
    b = person_box
    return {
        "x": b["x"] + 0.22 * b["w"],
        "y": b["y"] + 0.10 * b["h"],
        "w": 0.56 * b["w"],
        "h": 0.22 * b["h"],
    }


def _norm_box_to_px_crop(box: dict, width: int, height: int) -> str:
    """Normalized 0-1 box -> SOURCE-pixel 'x,y,w,h' crop arg for extract_frame."""
    x = int(round(box["x"] * width))
    y = int(round(box["y"] * height))
    w = max(1, int(round(box["w"] * width)))
    h = max(1, int(round(box["h"] * height)))
    return f"{x},{y},{w},{h}"


# --------------------------------------------------------------------------- #
# frame extraction (guarded — extract_frame sys.exit(2)s on error)            #
# --------------------------------------------------------------------------- #
def safe_extract_frame(video: Path, ts_ms: int, crop: str | None, scale: float) -> bytes | None:
    """Wrap ocr_probe.extract_frame so one bad frame doesn't abort the whole run.

    extract_frame sys.exit(2)s on ffmpeg/video errors and may raise CalledProcessError;
    we catch both and return None so the caller can degrade that frame to empty.
    """
    try:
        return extract_frame(video, ts_ms / 1000.0, crop=crop, scale=scale)
    except SystemExit:
        _warn(f"frame extraction failed at {ts_ms}ms (ffmpeg/video error) — degrading to empty")
        return None
    except subprocess.CalledProcessError as e:
        _warn(f"ffmpeg failed at {ts_ms}ms: {e} — degrading to empty")
        return None


# --------------------------------------------------------------------------- #
# Phase 2 — detect + det_id minting                                            #
# --------------------------------------------------------------------------- #
def _mint_det_ids(detections: list[dict]) -> list[dict]:
    """`p{N}` by descending confidence (p0 = highest). Stable tiebreak by box position."""
    ordered = sorted(
        detections,
        key=lambda d: (-(d.get("confidence") or 0.0), d["box"]["x"], d["box"]["y"]),
    )
    out = []
    for i, d in enumerate(ordered):
        out.append({**d, "det_id": f"p{i}"})
    return out


def detect_frame(vision, video: Path, ts_ms: int, classes: list[str]) -> list[dict]:
    """Run detect on one sampled frame; return minted, class-filtered detection dicts.

    `vision` is an AzureVision (or test double) with .analyze(bytes, detect, read). When it
    is None (no creds) the frame contributes nothing (honest-empty).
    """
    if vision is None:
        return []
    img = safe_extract_frame(video, ts_ms, crop=None, scale=1.0)
    if img is None:
        return []
    try:
        result = vision.analyze(img, detect=True, read=False)
    except Exception as e:  # noqa: BLE001 — surface SDK/auth errors but keep the run alive
        _warn(f"Azure detect failed at {ts_ms}ms: {e} — degrading to empty")
        return []
    want = {c.lower() for c in classes}
    raw = []
    for d in result.objects:
        if d.cls.lower() not in want:
            continue
        b = d.box.rounded()
        raw.append({
            "cls": d.cls.lower(),
            "confidence": round(d.confidence, 4) if d.confidence is not None else None,
            "box": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
        })
    return _mint_det_ids(raw)


# --------------------------------------------------------------------------- #
# Phase 2.5 — pin resolution (rename minted det -> semantic id)               #
# --------------------------------------------------------------------------- #
def resolve_pins(detect_map: dict, pins: list[dict]) -> dict:
    """Match each manifest pin to the nearest detection box at its frame and RENAME that
    det's det_id to the pin's semantic id. Returns {pin_id: resolved_det_id}.

    Mutates detect_map in place (det_ids stay unique per frame). One ordered step BEFORE
    read_text/events so the join key is stable.
    """
    resolution: dict[str, str] = {}
    for pin in pins:
        pin_id = pin["det_id"]
        match = pin["match"]
        ts = int(match["frame_ts_ms"])
        key = str(ts)
        dets = detect_map.get(key, [])
        if not dets:
            _warn(f"pin '{pin_id}': no detections at {ts}ms to match against — pin unresolved")
            continue
        target = _round_box(match["nearest_box"])
        nearest = min(dets, key=lambda d: _center_dist(d["box"], target))
        old_id = nearest["det_id"]
        # Avoid colliding with an existing semantic id already in this frame.
        for d in dets:
            if d is not nearest and d["det_id"] == pin_id:
                d["det_id"] = old_id  # swap, keep uniqueness
        nearest["det_id"] = pin_id
        resolution[pin_id] = pin_id
        if old_id != pin_id:
            print(f"  pin '{pin_id}': renamed {old_id} -> {pin_id} at {ts}ms "
                  f"(center-dist {_center_dist(nearest['box'], target):.4f})")
    return resolution


# --------------------------------------------------------------------------- #
# Phase 3 — read_text / OCR ladder                                            #
# --------------------------------------------------------------------------- #
def _find_det(detect_map: dict, det_id: str) -> tuple[int, dict] | None:
    for key, dets in detect_map.items():
        for d in dets:
            if d["det_id"] == det_id:
                return int(key), d
    return None


def build_read_text(
    detect_map: dict,
    manifest: dict,
    resolution: dict,
    *,
    video: Path,
    vlm,
    require_ocr: bool,
) -> tuple[dict, list[str]]:
    """Build the read_text map via the binding OCR ladder:
      (1) manifest pin  -> source 'pinned', verbatim text.
      (2) gpt-4o VLM    -> source 'live' if it returns digits (when AZURE_OPENAI_* present).
      (3) skip          -> NO read_text entry (honest-empty; op_read_text returns None).
    Azure Read is deliberately NOT a rung. Returns (read_text_map, skipped_det_ids).
    """
    read_text: dict[str, dict] = {}
    skipped: list[str] = []
    region = manifest.get("ocr_region", "jersey")
    clip = manifest["clip"]
    W, H = int(clip["width"]), int(clip["height"])

    # Rung 1: pins (always honored, even with no creds — keeps the hero #10 correct).
    pinned_ids = set()
    for pin in manifest.get("pins", []):
        resolved = resolution.get(pin["det_id"])
        if resolved is None:
            continue  # unresolved pin -> no read_text (warned in resolve_pins)
        entry = {"text": str(pin["text"]), "confidence": None, "source": "pinned"}
        if pin.get("note"):
            entry["note"] = pin["note"]
        read_text[resolved] = entry
        pinned_ids.add(resolved)

    # Rung 2/3: every OCR-target det without a pin -> VLM, else skip.
    targets = _ocr_target_ids(detect_map)
    for det_id in targets:
        if det_id in pinned_ids:
            continue
        found = _find_det(detect_map, det_id)
        if found is None:
            continue
        ts, det = found
        if vlm is None:
            # No VLM configured -> skip (honest-empty).
            skipped.append(det_id)
            continue
        sub = _jersey_subbox(det["box"]) if region == "jersey" else det["box"]
        crop = _norm_box_to_px_crop(sub, W, H)
        img = safe_extract_frame(video, ts, crop=crop, scale=3.0)
        if img is None:
            skipped.append(det_id)
            continue
        try:
            vread = vlm.read_jersey_number(img)
        except Exception as e:  # noqa: BLE001 — 429 / ValueError / SDK error -> skip (D-DR5)
            _warn(f"VLM read failed for {det_id} at {ts}ms: {e} — skipping (honest-empty)")
            skipped.append(det_id)
            continue
        if vread.digits:
            read_text[det_id] = {"text": vread.digits, "confidence": None, "source": "live"}
        else:
            skipped.append(det_id)

    return read_text, skipped


def _ocr_target_ids(detect_map: dict) -> list[str]:
    """All det_ids in the cache (every person crop is an OCR candidate). Deterministic order."""
    ids: list[str] = []
    for key in sorted(detect_map, key=lambda k: int(k)):
        for d in sorted(detect_map[key], key=lambda d: d["det_id"]):
            ids.append(d["det_id"])
    return ids


# --------------------------------------------------------------------------- #
# Phase 2 (cont.) — events block (the verdict linchpin)                       #
# --------------------------------------------------------------------------- #
def build_events(manifest: dict, detect_map: dict, resolution: dict) -> list[dict]:
    """Build events[], deriving each scorer_det from the SAME pin resolution (not a literal).

    Asserts each scorer_det is a minted det_id AND appears in the detect map, so op_answer's
    overlay re-scan finds the focus frame. Raises ValueError if an event can't be grounded.
    """
    minted = {d["det_id"] for dets in detect_map.values() for d in dets}
    events: list[dict] = []
    for ev in manifest.get("events", []):
        pin_id = ev.get("scorer_pin")
        if pin_id is None:
            raise ValueError(f"event @ {ev.get('ts_ms')}ms missing 'scorer_pin' (events derive "
                             "scorer_det from pin resolution, never a literal)")
        scorer_det = resolution.get(pin_id)
        if scorer_det is None:
            raise ValueError(f"event @ {ev.get('ts_ms')}ms: pin '{pin_id}' did not resolve to a "
                             "detection (cannot ground the scorer)")
        if scorer_det not in minted:
            raise ValueError(f"event @ {ev.get('ts_ms')}ms: scorer_det '{scorer_det}' is not a "
                             "minted det_id in the detect map")
        events.append({
            "ts_ms": int(ev["ts_ms"]),
            "type": ev.get("type", "goal"),
            "scorer_det": scorer_det,
            "box": _round_box(ev["box"]),
        })
    return events


# --------------------------------------------------------------------------- #
# Phase 4 — stills                                                             #
# --------------------------------------------------------------------------- #
def emit_stills(manifest: dict, video: Path, out_dir: Path, *, dry_run: bool) -> list[str]:
    """Write a real JPEG per unique evidence ts using ffmpeg-direct (.jpg + -q:v), into out_dir.

    Filenames come from the manifest (each evidence frame declares its output filename), so the
    unchanged EvidencePanel.frameSrc lookup keeps working. Returns the filenames written.
    """
    written: list[str] = []
    seen: set[int] = set()
    for ef in manifest.get("evidence_frames", []):
        ts = int(ef["ts_ms"])
        filename = ef["filename"]
        if ts in seen:
            continue
        seen.add(ts)
        if dry_run:
            written.append(filename)
            continue
        dest = out_dir / filename
        if _ffmpeg_still(video, ts, dest):
            written.append(filename)
        else:
            _warn(f"could not emit still {filename} at {ts}ms")
    return written


def _ffmpeg_still(video: Path, ts_ms: int, dest: Path) -> bool:
    """Extract one full frame as a JPEG straight to `dest` (ffmpeg-direct, no Pillow)."""
    if not shutil.which("ffmpeg"):
        _warn("ffmpeg not on PATH — cannot emit stills")
        return False
    if not video.exists():
        _warn(f"video not found ({video}) — cannot emit stills")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=dest.parent) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(ts_ms / 1000.0), "-i", str(video),
        "-frames:v", "1", "-q:v", "2", str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _warn(f"ffmpeg still failed at {ts_ms}ms: {e}")
        tmp_path.unlink(missing_ok=True)
        return False
    os.replace(tmp_path, dest)  # atomic into place
    return True


# --------------------------------------------------------------------------- #
# Phase 1/5 — cache assembly + canonical write                                #
# --------------------------------------------------------------------------- #
def probe_clip_dims(video: Path) -> dict | None:
    """ffprobe fallback for width/height/duration (registry/manifest is the source of truth)."""
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
        dur = float(data.get("format", {}).get("duration", 0)) * 1000.0
        return {
            "width": int(stream.get("width", 0)) or None,
            "height": int(stream.get("height", 0)) or None,
            "duration_ms": int(round(dur)) or None,
        }
    except (subprocess.CalledProcessError, ValueError, KeyError, json.JSONDecodeError):
        return None


def build_clip_block(manifest: dict, video: Path) -> dict:
    """clip block: manifest/registry is source of truth; ffprobe only fills missing values."""
    mc = manifest["clip"]
    clip = {
        "id": mc["id"],
        "width": mc.get("width"),
        "height": mc.get("height"),
        "duration_ms": mc.get("duration_ms"),
        "fps": mc.get("fps"),
    }
    if None in (clip["width"], clip["height"], clip["duration_ms"]):
        probed = probe_clip_dims(video)
        if probed:
            for k in ("width", "height", "duration_ms"):
                if clip[k] is None and probed.get(k):
                    clip[k] = probed[k]
    missing = [k for k in ("width", "height", "duration_ms") if clip.get(k) is None]
    if missing:
        raise PrecomputeError(f"clip dims {missing} unknown (manifest lacks them and ffprobe "
                              "could not supply them)")
    if clip["fps"] is None:
        clip["fps"] = 30
    return clip


def canonicalize(cache: dict) -> dict:
    """Deterministic shape: numeric-sorted detect keys, det_id-sorted detections, rounded boxes."""
    detect = {}
    for key in sorted(cache["detect"], key=lambda k: int(k)):
        dets = sorted(cache["detect"][key], key=lambda d: d["det_id"])
        detect[str(int(key))] = [
            {
                "det_id": d["det_id"],
                "cls": d["cls"],
                "confidence": d["confidence"],
                "box": _round_box(d["box"]),
            }
            for d in dets
        ]
    read_text = {}
    for det_id in sorted(cache["read_text"]):
        rt = cache["read_text"][det_id]
        entry = {"text": rt["text"], "confidence": rt.get("confidence"), "source": rt["source"]}
        if rt.get("note"):
            entry["note"] = rt["note"]
        read_text[det_id] = entry
    events = sorted(
        ({"ts_ms": int(e["ts_ms"]), "type": e["type"],
          "scorer_det": e["scorer_det"], "box": _round_box(e["box"])} for e in cache["events"]),
        key=lambda e: e["ts_ms"],
    )
    return {"clip": cache["clip"], "detect": detect, "read_text": read_text, "events": events}


def write_cache_atomic(cache: dict, out: Path) -> None:
    """Atomic write (temp + os.replace) — a partial write would 500 server.py's Cache.load."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cache, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=out.parent) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, out)


# --------------------------------------------------------------------------- #
# structural self-validation (per docs/cache-schema.md — no separate schema)  #
# --------------------------------------------------------------------------- #
def validate_cache_structure(cache: dict) -> list[str]:
    """In-code structural checks matching docs/cache-schema.md. Empty list = valid."""
    errs: list[str] = []
    clip = cache.get("clip", {})
    for k in ("id", "width", "height", "duration_ms"):
        if k not in clip:
            errs.append(f"clip missing '{k}'")

    detect = cache.get("detect", {})
    minted: set[str] = set()
    for key, dets in detect.items():
        if str(int(key)) != str(key):
            errs.append(f"detect key '{key}' is not a canonical int-string")
        seen_ids: set[str] = set()
        for d in dets:
            for k in ("det_id", "cls", "box"):
                if k not in d:
                    errs.append(f"detection at {key} missing '{k}'")
            did = d.get("det_id")
            if did in seen_ids:
                errs.append(f"detect[{key}] has duplicate det_id '{did}' (must be unique per frame)")
            seen_ids.add(did)
            minted.add(did)
            box = d.get("box", {})
            for c in ("x", "y", "w", "h"):
                v = box.get(c)
                if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                    errs.append(f"detect[{key}] det '{did}' box.{c}={v} not in 0..1")
                elif round(v, 4) != v:
                    errs.append(f"detect[{key}] det '{did}' box.{c}={v} has >4 decimals")

    for det_id, rt in cache.get("read_text", {}).items():
        if det_id not in minted:
            errs.append(f"read_text key '{det_id}' is not a real det_id in detect")
        if rt.get("source") not in VALID_SOURCES:
            errs.append(f"read_text['{det_id}'].source '{rt.get('source')}' not in {VALID_SOURCES}")
        if not rt.get("text"):
            errs.append(f"read_text['{det_id}'] has empty text (omit the entry instead)")

    for ev in cache.get("events", []):
        sd = ev.get("scorer_det")
        if sd not in minted:
            errs.append(f"event @ {ev.get('ts_ms')}ms scorer_det '{sd}' not in the detect map")
        for k in ("ts_ms", "type", "box"):
            if k not in ev:
                errs.append(f"event @ {ev.get('ts_ms')}ms missing '{k}'")
    return errs


# --------------------------------------------------------------------------- #
# replay self-validation (Interpreter + run_doc.schema.json)                  #
# --------------------------------------------------------------------------- #
def replay_and_validate(cache_path: Path) -> tuple[bool, str]:
    """Run the pinned hero program over the written cache; assert the run-doc validates and
    grounds. Returns (ok, message). Reuses the run_program.py contract (validate + interpret)."""
    try:
        import jsonschema
    except ImportError:
        return False, "jsonschema not installed (pip install -r requirements.txt)"
    from interpreter import Cache, Interpreter  # noqa: E402
    from validate_program import strip_comments  # noqa: E402

    program = strip_comments(json.loads(HERO_PROGRAM.read_text()))["program"]
    run_doc = Interpreter(Cache.load(cache_path)).run(program)

    rd_schema = json.loads(RUNDOC_SCHEMA.read_text())
    rd_errs = list(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc))
    if rd_errs:
        return False, f"run-doc invalid ({len(rd_errs)} error(s)): {rd_errs[0].message}"
    f = run_doc["findings"]
    grounded = not f.get("partial") and f.get("answer") not in (None, "", "Unknown")
    if not grounded:
        return False, f"run-doc valid but did not ground (verdict: {f.get('verdict')!r})"
    return True, f"replay OK — verdict {f.get('verdict')!r}"


# --------------------------------------------------------------------------- #
# Azure constructors (patchable test seam — always via from_env())            #
# --------------------------------------------------------------------------- #
def make_vision():
    """AzureVision.from_env() or None when creds are absent. Constructed via from_env so tests
    can monkeypatch that single seam."""
    if not (os.environ.get("AZURE_VISION_ENDPOINT") and os.environ.get("AZURE_VISION_KEY")):
        return None
    from vision.azure_vision import AzureVision  # noqa: E402
    try:
        return AzureVision.from_env()
    except Exception as e:  # noqa: BLE001
        _warn(f"AzureVision unavailable: {e}")
        return None


def make_vlm():
    """AzureVLM.from_env() or None when AZURE_OPENAI_* is absent. from_env seam for tests."""
    if not (os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY")):
        return None
    from vision.vlm_read import AzureVLM  # noqa: E402
    try:
        return AzureVLM.from_env()
    except Exception as e:  # noqa: BLE001
        _warn(f"AzureVLM unavailable (read_text will degrade to pin/skip): {e}")
        return None


# --------------------------------------------------------------------------- #
# orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def run_precompute(
    manifest: dict,
    *,
    dry_run: bool = False,
    require_ocr: bool = False,
    out: Path = DEFAULT_CACHE_OUT,
    frames_dir: Path = FRAMES_DIR,
    vision=None,
    vlm=None,
) -> dict:
    """Core pipeline. Returns a result dict:
      {exit_code, cache, cache_path, stills, skipped_reads, replay_msg}.

    `vision`/`vlm` default to from_env() constructors; tests inject doubles or rely on the
    monkeypatched from_env. dry_run = sample + report, no Azure calls, no writes.
    """
    video = resolve_source(manifest)
    sampling = manifest["sampling"]
    ts_list = sampled_timestamps(sampling)
    classes = manifest.get("detect_classes", ["person"])

    print(f"clip:     {manifest['clip']['id']}  source={video}")
    print(f"sampling: {sampling['start_ms']}-{sampling['end_ms']}ms @ {sampling['fps']}fps "
          f"-> {len(ts_list)} frames (stride {max(1, round(1000 / sampling['fps']))})")
    print(f"detect:   classes={classes}")

    if dry_run:
        print("\n[dry-run] would extract + detect these sampled frames (no Azure calls, no writes):")
        for ts in ts_list:
            print(f"  - {ts}ms")
        for ef in manifest.get("evidence_frames", []):
            print(f"  still -> {ef['filename']} @ {ef['ts_ms']}ms")
        return {"exit_code": 0, "cache": None, "cache_path": None,
                "stills": [], "skipped_reads": [], "replay_msg": "dry-run"}

    if vision is None:
        vision = make_vision()
    if vlm is None:
        vlm = make_vlm()
    if vision is None:
        _warn("no AzureVision creds — detect frames will be empty (pin-only hero path still "
              "grounds via the pinned read + events).")

    # Phase 2: detect every sampled ts.
    detect_map: dict[str, list[dict]] = {}
    for ts in ts_list:
        dets = detect_frame(vision, video, ts, classes)
        if dets:
            detect_map[str(ts)] = dets
        print(f"  detect {ts}ms -> {len(dets)} {classes}")

    # Phase 2.5: resolve pins (rename minted dets) BEFORE read_text/events.
    resolution = resolve_pins(detect_map, manifest.get("pins", []))

    # Phase 3: read_text via the OCR ladder.
    read_text, skipped = build_read_text(
        detect_map, manifest, resolution, video=video, vlm=vlm, require_ocr=require_ocr,
    )

    # Phase 2 (cont.): events derived from the same pin resolution.
    events = build_events(manifest, detect_map, resolution)

    # Phase 1: clip block + assemble + canonicalize.
    clip = build_clip_block(manifest, video)
    cache = canonicalize({"clip": clip, "detect": detect_map,
                          "read_text": read_text, "events": events})

    struct_errs = validate_cache_structure(cache)
    if struct_errs:
        _err(f"generated cache failed structural validation ({len(struct_errs)} error(s)):")
        for m in struct_errs[:10]:
            _err(f"  - {m}")
        return {"exit_code": 1, "cache": cache, "cache_path": None,
                "stills": [], "skipped_reads": skipped, "replay_msg": "structural-fail"}

    if require_ocr and skipped:
        _err(f"--require-ocr: {len(skipped)} OCR read(s) skipped: {skipped}")
        return {"exit_code": 1, "cache": cache, "cache_path": None,
                "stills": [], "skipped_reads": skipped, "replay_msg": "require-ocr-skip"}

    # Phase 4: stills (real JPEGs, manifest-declared names).
    stills = emit_stills(manifest, video, frames_dir, dry_run=False)

    # Phase 5/6.3: atomic overwrite-in-place.
    write_cache_atomic(cache, out)
    print(f"\nwrote cache -> {out}")
    if stills:
        print(f"wrote stills -> {frames_dir}: {', '.join(stills)}")

    # Phase 6.2: self-validating replay.
    ok, msg = replay_and_validate(out)
    print(msg)
    if not ok:
        _err("generated cache did not replay the hero program to a grounded answer")
        return {"exit_code": 1, "cache": cache, "cache_path": out,
                "stills": stills, "skipped_reads": skipped, "replay_msg": msg}

    return {"exit_code": 0, "cache": cache, "cache_path": out,
            "stills": stills, "skipped_reads": skipped, "replay_msg": msg}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Precompute a clip's replay cache + stills.")
    ap.add_argument("--clip", default=None, help="clip id -> clips/<id>/manifest.json")
    ap.add_argument("--manifest", default=None, help="explicit manifest path (overrides --clip)")
    ap.add_argument("--query", default=None, help="query id (informational; the manifest drives ingest)")
    ap.add_argument("--dry-run", action="store_true", help="sample + report; no Azure calls, no writes")
    ap.add_argument("--require-ocr", action="store_true", help="hard-fail (exit 1) if any OCR read is skipped")
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE_OUT, help="cache output path")
    args = ap.parse_args(argv)

    load_dotenv(API_DIR / ".env")

    try:
        manifest_path = resolve_manifest_path(args.clip, args.manifest)
        manifest = load_manifest(manifest_path)
    except PrecomputeError as e:
        _err(str(e))
        return 2

    print(f"manifest: {manifest_path}")
    if args.query:
        print(f"query:    {args.query}")

    if not args.dry_run and not shutil.which("ffmpeg"):
        _err("ffmpeg not found on PATH (needed to extract frames + stills).")
        return 2

    try:
        result = run_precompute(
            manifest, dry_run=args.dry_run, require_ocr=args.require_ocr, out=args.out,
        )
    except PrecomputeError as e:
        _err(str(e))
        return 2
    except ValueError as e:
        _err(str(e))
        return 1

    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
