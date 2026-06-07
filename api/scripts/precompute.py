#!/usr/bin/env python3
"""
Automated clip -> replay-cache precompute pipeline.

Turns a raw clip (+ a per-clip manifest carrying the human knowledge the vision models
can't supply) into the exact two artifacts the demo replays today:

  (a) frame stills under  web/public/frames/   (real JPEGs the EvidencePanel renders)
  (b) a per-clip replay cache JSON in the shape api/interpreter/cache.py loads
      (detect map keyed by sampled-frame ms, read_text map keyed by det_id, events, clip)

ViperGPT layering: this is a Layer-2/3 batch job. It runs the *real* Azure AI Vision
`detect` over the sampled window (Layer 3), mints stable det_ids, applies the OCR ladder
(manifest pin -> gpt-4o VLM -> skip; Azure Read is deliberately NOT a rung — it misreads
the stylized WC font), then writes a cache the interpreter (Layer 2) replays with zero
live calls. See docs/cache-schema.md for the cache contract.

    # full precompute for the hero clip + query (calls Azure detect on the real .mov):
    python scripts/precompute.py --clip single-goal --query hero-10-first-goal

    # plan + report only, no Azure calls, no writes:
    python scripts/precompute.py --clip single-goal --query hero-10-first-goal --dry-run

    # by explicit manifest path, and to a scratch cache:
    python scripts/precompute.py --manifest clips/single-goal/manifest.json --out /tmp/c.json

Exit 0 = cache written + self-validated (replays the pinned program to a grounded answer).
       1 = produced but validation/grounding failed (or --require-ocr and a read was skipped).
       2 = setup problem (no ffmpeg / no manifest / no creds when required / bad manifest).

Prereqs: ffmpeg/ffprobe on PATH; AZURE_VISION_* in api/.env for live detect (see .env.example).
With no creds the live path can't run here; the pytest suite mocks Azure + ffmpeg end to end.
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

# scripts-importing-scripts path hack (same pattern as run_program.py / codegen_probe.py):
# put scripts/ first so `ocr_probe` resolves, then api/ so `interpreter` / `vision` resolve.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(API_DIR))

from ocr_probe import extract_frame, load_dotenv  # noqa: E402  (reuse ffmpeg + .env loader)

DEFAULT_OUT = API_DIR / "examples" / "hero_cache.json"
FRAMES_DIR = REPO_ROOT / "web" / "public" / "frames"
RUNDOC_SCHEMA = API_DIR / "schema" / "run_doc.schema.json"

# jersey sub-region of a person box — MUST match op_crop (primitives.py:77-79) so the
# crop we OCR is the crop the interpreter would show.
JERSEY = {"dx": 0.22, "dy": 0.10, "dw": 0.56, "dh": 0.22}


# --------------------------------------------------------------------------- errors


class PrecomputeError(Exception):
    """Raised for setup problems (exit 2) and grounding/validation failures (exit 1)."""

    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- manifest


def resolve_manifest_path(*, clip: str | None, manifest: Path | None) -> Path:
    """`--manifest PATH` wins; else clips/<clip>/manifest.json. Absent -> exit 2 (never synthesize)."""
    if manifest is not None:
        path = manifest if manifest.is_absolute() else (REPO_ROOT / manifest)
        if not path.exists():
            raise PrecomputeError(f"manifest not found: {manifest}", 2)
        return path
    if clip is None:
        raise PrecomputeError("provide --clip <id> or --manifest <path>.", 2)
    path = REPO_ROOT / "clips" / clip / "manifest.json"
    if not path.exists():
        raise PrecomputeError(
            f"no manifest for clip '{clip}' at {path.relative_to(REPO_ROOT)} — "
            f"author it first (the pipeline never synthesizes defaults).",
            2,
        )
    return path


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PrecomputeError(f"manifest is not valid JSON ({path}): {e}", 2)
    for key in ("clip", "sampling"):
        if key not in data:
            raise PrecomputeError(f"manifest missing required '{key}' block ({path})", 2)
    if "id" not in data["clip"] or "source" not in data["clip"]:
        raise PrecomputeError("manifest clip must carry 'id' and 'source'", 2)
    return data


def resolve_source(manifest: dict, manifest_path: Path) -> Path:
    """Resolve the clip's .mov path. Manifest `source` is relative to the manifest's dir."""
    src = manifest["clip"]["source"]
    p = Path(src)
    if not p.is_absolute():
        p = (manifest_path.parent / p).resolve()
    return p


# --------------------------------------------------------------------------- clip block


def probe_dims(video: Path) -> dict | None:
    """ffprobe the clip for width/height/duration_ms/fps. Returns None on any failure."""
    if not shutil.which("ffprobe") or not video.exists():
        return None
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration", "-of", "json", str(video),
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        info = json.loads(out)
        stream = info["streams"][0]
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration_ms": int(round(float(info["format"]["duration"]) * 1000)),
            "fps": round(fps, 3),
        }
    except Exception:  # noqa: BLE001 — ffprobe is best-effort; manifest/registry fall back
        return None


def build_clip_block(manifest: dict, video: Path) -> dict:
    """Clip metadata for the cache. Registry (canned.CLIPS) is source of truth; ffprobe is a
    fallback; manifest values backstop both. Registry is NOT edited."""
    clip_id = manifest["clip"]["id"]
    block = {"id": clip_id, "width": None, "height": None, "duration_ms": None, "fps": None}

    # 1) manifest (lowest precedence, always present as a backstop)
    for k in ("width", "height", "duration_ms", "fps"):
        if manifest["clip"].get(k) is not None:
            block[k] = manifest["clip"][k]

    # 2) ffprobe (fills any gap; does not override the registry below)
    probed = probe_dims(video)
    if probed:
        for k, v in probed.items():
            if block.get(k) is None:
                block[k] = v

    # 3) registry — source of truth for the dims it carries
    try:
        from canned import CLIPS  # noqa: E402

        reg = CLIPS.get(clip_id)
        if reg:
            for k in ("width", "height", "duration_ms", "fps"):
                if reg.get(k) is not None:
                    block[k] = reg[k]
    except Exception:  # noqa: BLE001 — registry optional for standalone manifests
        pass

    for k in ("width", "height", "duration_ms"):
        if block.get(k) is None:
            raise PrecomputeError(
                f"could not resolve clip.{k} (no registry, ffprobe, or manifest value).", 2
            )
    return block


# --------------------------------------------------------------------------- sampling


def sample_ts(sampling: dict) -> list[int]:
    """Sampled-frame timestamps (ms). MUST match op_sample_frames (primitives.py:25-38):
    stride = max(1, round(1000/fps)); ts = range(start, end+1, stride)."""
    start, end, fps = sampling["start_ms"], sampling["end_ms"], sampling["fps"]
    stride = max(1, round(1000 / fps))
    return list(range(start, end + 1, stride))


# --------------------------------------------------------------------------- det_ids


def _round_box(box: dict) -> dict:
    return {k: round(float(box[k]), 4) for k in ("x", "y", "w", "h")}


def mint_det_ids(detections: list[dict]) -> list[dict]:
    """OPEN KNOB (det_id scheme): p{N} by DESCENDING confidence within the frame (highest = p0).
    Deterministic ties: confidence desc, then box (x,y,w,h) asc. Manifest pins later RENAME a
    matched det to a semantic id (p_messi). Azure returns no ids."""
    ordered = sorted(
        detections,
        key=lambda d: (
            -(d.get("confidence") if d.get("confidence") is not None else -1.0),
            d["box"]["x"], d["box"]["y"], d["box"]["w"], d["box"]["h"],
        ),
    )
    return [{**d, "det_id": f"p{i}"} for i, d in enumerate(ordered)]


def _box_center(box: dict) -> tuple[float, float]:
    return box["x"] + box["w"] / 2.0, box["y"] + box["h"] / 2.0


def _center_dist(a: dict, b: dict) -> float:
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return math.hypot(ax - bx, ay - by)


def nearest_det(frame_dets: list[dict], target_box: dict) -> dict | None:
    """The detection whose box center is closest to target_box's center, in one frame."""
    if not frame_dets:
        return None
    return min(frame_dets, key=lambda d: _center_dist(d["box"], target_box))


# --------------------------------------------------------------------------- detect map


def run_detect(
    video: Path,
    ts_list: list[int],
    classes: list[str],
    vision,
    *,
    extract=extract_frame,
) -> dict[int, list[dict]]:
    """Per sampled ts: extract the frame, run Azure detect, filter to classes (case-insensitive),
    mint det_ids. Returns ts -> [detection]. Guards extract_frame's sys.exit(2) so one bad frame
    doesn't abort the run."""
    want = {c.lower() for c in classes}
    detect: dict[int, list[dict]] = {}
    for ts in ts_list:
        try:
            frame_bytes = extract(video, ts / 1000.0, None, 1.0)
        except SystemExit:
            _warn(f"frame extraction failed at {ts}ms — skipping (empty detect for this frame).")
            detect[ts] = []
            continue
        result = vision.analyze(frame_bytes, detect=True, read=False)
        raw = [
            {
                "cls": d.cls,
                "confidence": round(d.confidence, 4) if d.confidence is not None else None,
                "box": _round_box({"x": d.box.x, "y": d.box.y, "w": d.box.w, "h": d.box.h}),
            }
            for d in result.objects
            if d.cls.lower() in want
        ]
        detect[ts] = mint_det_ids(raw)
    return detect


# --------------------------------------------------------------------------- pins


def apply_pins(detect: dict[int, list[dict]], pins: list[dict]) -> dict[str, dict]:
    """Resolve each manifest pin to a minted det by nearest box at the pin's match frame, then
    RENAME that det to the pin's semantic id ACROSS all frames it appears in (matched by nearest
    box per frame). Returns pin_det_id -> resolution info (matched frame, original minted id).

    Mutates `detect` in place. Errors (exit 2) if a pin can't be resolved — a silent miss would
    let the verdict drift off the wrong person."""
    resolutions: dict[str, dict] = {}
    for pin in pins:
        pin_id = pin["det_id"]
        match = pin["match"]
        frame_ts = int(match["frame_ts_ms"])
        target = match["nearest_box"]
        frame_dets = detect.get(frame_ts, [])
        anchor = nearest_det(frame_dets, target)
        if anchor is None:
            raise PrecomputeError(
                f"pin '{pin_id}' could not resolve: no detections at match frame {frame_ts}ms.", 2
            )
        original = anchor["det_id"]
        anchor_box = anchor["box"]
        resolutions[pin_id] = {"frame_ts_ms": frame_ts, "original_det_id": original}

        # Rename across all frames: in each frame, the det nearest the anchor box becomes pin_id,
        # but only if it's a tight match (so we don't grab an unrelated person in a frame where
        # the scorer isn't visible). Tightness: center within half the anchor box's diagonal.
        diag = math.hypot(anchor_box["w"], anchor_box["h"])
        for fts, dets in detect.items():
            cand = nearest_det(dets, anchor_box)
            if cand is None:
                continue
            if fts == frame_ts:
                cand["det_id"] = pin_id
                continue
            if _center_dist(cand["box"], anchor_box) <= diag / 2.0:
                cand["det_id"] = pin_id
        # Guarantee uniqueness within each frame (a rename could in theory collide).
        _dedupe_frame_ids(detect)
    return resolutions


def _dedupe_frame_ids(detect: dict[int, list[dict]]) -> None:
    """Ensure det_ids are unique within each frame after renames (defensive; pins target one
    nearest det per frame so collisions are not expected)."""
    for dets in detect.values():
        seen: set[str] = set()
        for d in dets:
            base = d["det_id"]
            new = base
            n = 1
            while new in seen:
                new = f"{base}_{n}"
                n += 1
            d["det_id"] = new
            seen.add(new)


# --------------------------------------------------------------------------- read_text (OCR)


def jersey_crop_pixels(person_box: dict, clip: dict) -> tuple[int, int, int, int]:
    """Normalized person box -> SOURCE-pixel jersey crop 'x,y,w,h' (the extract_frame --crop
    contract). Mirrors op_crop's jersey sub-region, then de-normalizes by clip dims."""
    W, H = clip["width"], clip["height"]
    nx = person_box["x"] + JERSEY["dx"] * person_box["w"]
    ny = person_box["y"] + JERSEY["dy"] * person_box["h"]
    nw = JERSEY["dw"] * person_box["w"]
    nh = JERSEY["dh"] * person_box["h"]
    return (
        max(0, int(round(nx * W))),
        max(0, int(round(ny * H))),
        max(1, int(round(nw * W))),
        max(1, int(round(nh * H))),
    )


def _find_det(detect: dict[int, list[dict]], det_id: str) -> tuple[int, dict] | None:
    """First (ts, detection) for a det_id, scanning frames in ascending ts."""
    for ts in sorted(detect):
        for d in detect[ts]:
            if d["det_id"] == det_id:
                return ts, d
    return None


def run_read_text(
    detect: dict[int, list[dict]],
    clip: dict,
    pins: list[dict],
    video: Path,
    vlm,
    *,
    require_ocr: bool,
    extract=extract_frame,
) -> dict[str, dict]:
    """OCR ladder (docs/cache-schema.md): (1) manifest pin -> source 'pinned'; (2) gpt-4o VLM
    -> source 'live' if digits; (3) skip (no entry). Azure Read is NOT a rung. Returns
    det_id -> read entry. `vlm` is an AzureVLM or None (None => no VLM rung available)."""
    read: dict[str, dict] = {}
    pinned_ids = set()

    # Rung 1: manifest pins (verbatim, trusted over any model).
    for pin in pins:
        det_id = pin["det_id"]
        entry = {"text": str(pin["text"]), "confidence": None, "source": "pinned"}
        if pin.get("note"):
            entry["note"] = pin["note"]
        read[det_id] = entry
        pinned_ids.add(det_id)

    # Rungs 2/3: VLM read for non-pinned OCR-target dets. Default OCR targets = every distinct
    # det that has a detection (we crop each to its jersey region). Pinned ids already resolved.
    all_ids = {d["det_id"] for dets in detect.values() for d in dets}
    for det_id in sorted(all_ids - pinned_ids):
        if vlm is None:
            if require_ocr:
                raise PrecomputeError(
                    f"--require-ocr set but no VLM available for det '{det_id}'.", 1
                )
            continue  # rung 3: skip, no entry
        located = _find_det(detect, det_id)
        if located is None:
            continue
        ts, det = located
        crop = ",".join(str(n) for n in jersey_crop_pixels(det["box"], clip))
        try:
            frame_bytes = extract(video, ts / 1000.0, crop, 3.0)
        except SystemExit:
            _warn(f"crop extraction failed for det '{det_id}' at {ts}ms — skipping its read.")
            if require_ocr:
                raise PrecomputeError(f"--require-ocr: crop extraction failed for '{det_id}'.", 1)
            continue
        try:
            vr = vlm.read_jersey_number(frame_bytes)
        except Exception as e:  # noqa: BLE001 — 429 / ValueError / SDK error => degrade (NOT Azure Read)
            _warn(f"VLM read failed for det '{det_id}' ({type(e).__name__}: {e}) — skipping (diagnostic-empty).")
            if require_ocr:
                raise PrecomputeError(f"--require-ocr: VLM read failed for '{det_id}': {e}", 1)
            continue
        if vr.digits:
            read[det_id] = {"text": vr.digits, "confidence": None, "source": "live"}
        elif require_ocr:
            raise PrecomputeError(f"--require-ocr: VLM read no digits for '{det_id}'.", 1)
        # else: no digits -> skip (rung 3)
    return read


# --------------------------------------------------------------------------- events


def build_events(manifest_events: list[dict], pin_resolutions: dict[str, dict]) -> list[dict]:
    """Build the events block. Each goal's scorer_det is DERIVED from the pin resolution (the
    pin's own semantic id), NOT an independent manifest literal — so scorer_det is guaranteed to
    be a real, OCR-carrying det_id. Manifest events carry `scorer_pin` (the pin id to resolve)."""
    events: list[dict] = []
    for ev in manifest_events:
        out = {"ts_ms": int(ev["ts_ms"]), "type": ev["type"]}
        pin_ref = ev.get("scorer_pin")
        if pin_ref is not None:
            if pin_ref not in pin_resolutions:
                raise PrecomputeError(
                    f"event @ {ev['ts_ms']}ms references scorer_pin '{pin_ref}' "
                    f"that no pin resolved.", 2
                )
            out["scorer_det"] = pin_ref  # the pinned semantic id (resolved above)
        elif ev.get("scorer_det") is not None:
            out["scorer_det"] = ev["scorer_det"]
        if ev.get("box") is not None:
            out["box"] = _round_box(ev["box"])
        events.append(out)
    return events


# --------------------------------------------------------------------------- assemble + validate


def canonical_cache(clip: dict, detect: dict[int, list[dict]], read: dict, events: list[dict]) -> dict:
    """Deterministic cache dict: detect keys sorted numerically (string keys), detections sorted
    by det_id, boxes already rounded to 4 places. Stable order => byte-identical re-runs."""
    detect_out: dict[str, list[dict]] = {}
    for ts in sorted(detect):
        dets = sorted(detect[ts], key=lambda d: d["det_id"])
        detect_out[str(ts)] = [
            {
                "det_id": d["det_id"],
                "cls": d["cls"],
                "confidence": d["confidence"],
                "box": _round_box(d["box"]),
            }
            for d in dets
        ]
    read_out = {det_id: read[det_id] for det_id in sorted(read)}
    return {"clip": clip, "detect": detect_out, "read_text": read_out, "events": events}


def assert_cache_invariants(cache: dict) -> None:
    """In-code structural check per docs/cache-schema.md (no separate JSON Schema). Raises
    PrecomputeError(exit 1) on any violation — a broken cache must fail loudly."""
    clip = cache.get("clip", {})
    for k in ("id", "width", "height", "duration_ms"):
        if k not in clip:
            raise PrecomputeError(f"cache.clip missing '{k}'", 1)

    all_det_ids: set[str] = set()
    for ts_key, dets in cache.get("detect", {}).items():
        if not (isinstance(ts_key, str) and ts_key.lstrip("-").isdigit()):
            raise PrecomputeError(f"detect key '{ts_key}' is not a stringified integer", 1)
        frame_ids: set[str] = set()
        for d in dets:
            for f in ("det_id", "cls", "box"):
                if f not in d:
                    raise PrecomputeError(f"detection at {ts_key} missing '{f}'", 1)
            if d["det_id"] in frame_ids:
                raise PrecomputeError(f"duplicate det_id '{d['det_id']}' in frame {ts_key}", 1)
            frame_ids.add(d["det_id"])
            b = d["box"]
            for c in ("x", "y", "w", "h"):
                if not (0.0 <= float(b[c]) <= 1.0):
                    raise PrecomputeError(f"box.{c}={b[c]} out of [0,1] for {d['det_id']}", 1)
            all_det_ids.add(d["det_id"])

    for det_id in cache.get("read_text", {}):
        if det_id not in all_det_ids:
            raise PrecomputeError(f"read_text key '{det_id}' is not a real det_id", 1)

    for ev in cache.get("events", []):
        for f in ("ts_ms", "type"):
            if f not in ev:
                raise PrecomputeError(f"event missing '{f}'", 1)
        scorer = ev.get("scorer_det")
        if scorer is not None and scorer not in all_det_ids:
            raise PrecomputeError(
                f"event scorer_det '{scorer}' is not in the detect map "
                f"(op_answer's overlay re-scan would drop the focus frame).", 1
            )


def atomic_write_json(path: Path, data: dict) -> None:
    """Write canonical JSON atomically (temp file + os.replace) so server.py's per-request
    Cache.load never reads a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def self_validate(out: Path, program_path: Path) -> dict:
    """Replay the pinned hero program over the freshly written cache and validate the run-doc
    against run_doc.schema.json (reuse run_program.py's gate). Returns the run-doc. Raises
    PrecomputeError(exit 1) if the cache can't replay the program to a valid, grounded doc."""
    import jsonschema  # noqa: E402

    from interpreter import Cache, Interpreter  # noqa: E402
    from validate_program import strip_comments  # noqa: E402

    program = strip_comments(json.loads(program_path.read_text()))["program"]
    run_doc = Interpreter(Cache.load(out)).run(program)

    rd_schema = json.loads(RUNDOC_SCHEMA.read_text())
    errs = sorted(
        jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc),
        key=lambda e: list(e.path),
    )
    if errs:
        msgs = "; ".join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errs[:5])
        raise PrecomputeError(f"generated cache failed run-doc validation: {msgs}", 1)
    return run_doc


# --------------------------------------------------------------------------- stills


def emit_stills(video: Path, evidence_frames: list[dict], *, extract=extract_frame) -> list[Path]:
    """Emit a real JPEG per unique evidence ts into web/public/frames/ using ffmpeg-direct
    (.jpg suffix, -q:v) — no Pillow dependency. Filenames come from the manifest, reproducing
    the two the unchanged EvidencePanel.frameSrc expects (goal-4000.jpg / scorer-4625.jpg).
    JPEG re-encode is not byte-reproducible; verify by dims. Idempotency excludes stills."""
    if not shutil.which("ffmpeg"):
        raise PrecomputeError("ffmpeg not found on PATH (needed to write stills).", 2)
    if not video.exists():
        raise PrecomputeError(f"video not found for stills: {video}", 2)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[int] = set()
    for ef in evidence_frames:
        ts = int(ef["ts_ms"])
        filename = ef["filename"]
        if ts in seen and (FRAMES_DIR / filename).exists():
            continue
        seen.add(ts)
        dest = FRAMES_DIR / filename
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(ts / 1000.0), "-i", str(video),
            "-frames:v", "1", "-q:v", "2", str(dest),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise PrecomputeError(f"ffmpeg failed writing {filename}: {e}", 2)
        written.append(dest)
    return written


# --------------------------------------------------------------------------- vision wiring


def make_vision():
    """Construct the Azure detect client via from_env (the test seam patches from_env)."""
    from vision.azure_vision import AzureVision  # noqa: E402

    return AzureVision.from_env()


def make_vlm():
    """Construct the gpt-4o VLM via from_env, or None if creds absent / construction fails.
    The test seam patches AzureVLM.from_env; a None return is the no-VLM rung."""
    if not (os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY")):
        return None
    try:
        from vision.vlm_read import AzureVLM  # noqa: E402

        return AzureVLM.from_env()
    except Exception as e:  # noqa: BLE001 — missing dep / bad creds => no VLM rung
        _warn(f"VLM unavailable ({type(e).__name__}: {e}) — OCR will degrade to pin-or-skip.")
        return None


# --------------------------------------------------------------------------- orchestration


def run_precompute(
    manifest: dict,
    manifest_path: Path,
    *,
    dry_run: bool = False,
    require_ocr: bool = False,
    out: Path = DEFAULT_OUT,
    program_path: Path | None = None,
    vision_factory=make_vision,
    vlm_factory=make_vlm,
    extract=extract_frame,
    emit_frames: bool = True,
) -> dict:
    """The pipeline. Returns a report dict. Raises PrecomputeError (carrying an exit code) on
    setup / grounding failures. With dry_run: sample + report only — zero Azure calls, no writes."""
    if program_path is None:
        program_path = API_DIR / "examples" / "hero_program.json"

    video = resolve_source(manifest, manifest_path)
    clip = build_clip_block(manifest, video)
    sampling = manifest["sampling"]
    ts_list = sample_ts(sampling)
    classes = manifest.get("detect_classes", ["person"])
    pins = manifest.get("pins", [])
    manifest_events = manifest.get("events", [])
    evidence_frames = manifest.get("evidence_frames", [])

    report = {
        "clip_id": clip["id"],
        "source": str(video),
        "sampled_ts": ts_list,
        "n_sampled": len(ts_list),
        "dry_run": dry_run,
        "out": str(out),
    }

    if dry_run:
        # No Azure, no writes — just report the plan (and confirm the .mov / ffmpeg are usable).
        report["ffmpeg"] = bool(shutil.which("ffmpeg"))
        report["video_present"] = video.exists()
        report["pins"] = [p["det_id"] for p in pins]
        report["evidence_frames"] = [ef["filename"] for ef in evidence_frames]
        report["wrote_cache"] = False
        report["wrote_stills"] = False
        return report

    # 1) detect over the sampled window (real Azure calls).
    vision = vision_factory()
    detect = run_detect(video, ts_list, classes, vision, extract=extract)
    report["n_detected_frames"] = sum(1 for d in detect.values() if d)
    report["n_detections"] = sum(len(d) for d in detect.values())

    # 2) resolve pins -> rename matched dets to semantic ids (det_ids now stable join keys).
    pin_resolutions = apply_pins(detect, pins)
    report["pin_resolutions"] = pin_resolutions

    # 3) read_text via the OCR ladder (pin -> VLM -> skip).
    vlm = vlm_factory()
    report["vlm_available"] = vlm is not None
    read = run_read_text(detect, clip, pins, video, vlm, require_ocr=require_ocr, extract=extract)
    report["n_reads"] = len(read)

    # 4) events — scorer_det derived from the pin resolution (the verdict linchpin).
    events = build_events(manifest_events, pin_resolutions)

    # 5) assemble canonical cache + invariant checks.
    cache = canonical_cache(clip, detect, read, events)
    assert_cache_invariants(cache)

    # 6) write atomically.
    atomic_write_json(out, cache)
    report["wrote_cache"] = True

    # 7) self-validate: replay the pinned program; assert the run-doc validates + grounds.
    run_doc = self_validate(out, program_path)
    report["verdict"] = run_doc["findings"]["verdict"]
    report["run_doc_valid"] = True

    # 8) stills (after the cache is sound). Excluded from idempotency byte-check.
    if emit_frames and evidence_frames:
        written = emit_stills(video, evidence_frames, extract=extract)
        report["wrote_stills"] = True
        report["stills"] = [str(p) for p in written]
    else:
        report["wrote_stills"] = False

    return report


# --------------------------------------------------------------------------- CLI


def _print_report(report: dict) -> None:
    print(f"clip:   {report['clip_id']}  source={report['source']}")
    print(f"sample: {report['n_sampled']} frames @ {report['sampled_ts'][0]}-{report['sampled_ts'][-1]}ms")
    if report.get("dry_run"):
        print(f"  [dry-run] ffmpeg={report.get('ffmpeg')} video_present={report.get('video_present')}")
        print(f"  [dry-run] pins={report.get('pins')} stills={report.get('evidence_frames')}")
        print("  [dry-run] no Azure calls, no writes.")
        return
    print(f"detect: {report.get('n_detections')} detections over {report.get('n_detected_frames')} frames")
    print(f"  pins resolved: {report.get('pin_resolutions')}")
    print(f"read:   {report.get('n_reads')} read_text entries (vlm_available={report.get('vlm_available')})")
    print(f"cache:  wrote {report.get('out')}  (valid={report.get('run_doc_valid')})")
    print(f"verdict: {report.get('verdict')}")
    if report.get("wrote_stills"):
        print(f"stills: {report.get('stills')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Precompute a clip's replay cache + stills.")
    ap.add_argument("--clip", default=None, help="clip id -> clips/<id>/manifest.json")
    ap.add_argument("--manifest", type=Path, default=None, help="explicit manifest path (overrides --clip)")
    ap.add_argument("--query", default=None, help="query id (selects the program to self-validate against)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="cache output path")
    ap.add_argument("--dry-run", action="store_true", help="sample + report only; no Azure calls, no writes")
    ap.add_argument("--require-ocr", action="store_true", help="hard-fail (exit 1) if any OCR read is skipped")
    args = ap.parse_args(argv)

    load_dotenv(API_DIR / ".env")

    try:
        manifest_path = resolve_manifest_path(clip=args.clip, manifest=args.manifest)
        manifest = load_manifest(manifest_path)

        # Resolve the program to self-validate against, from the query id (canned registry).
        program_path = API_DIR / "examples" / "hero_program.json"
        if args.query:
            try:
                from canned import by_id  # noqa: E402

                q = by_id(args.query)
                if q and q.get("program"):
                    program_path = Path(q["program"])
            except Exception:  # noqa: BLE001 — registry optional
                pass

        # Live path needs Azure Vision creds (unless dry-run). Fail clearly otherwise.
        if not args.dry_run and not (
            os.environ.get("AZURE_VISION_ENDPOINT") and os.environ.get("AZURE_VISION_KEY")
        ):
            print(
                "ERROR: AZURE_VISION_* not set — the live detect pass needs Azure AI Vision "
                "creds (api/.env, see .env.example). Use --dry-run to plan without Azure.",
                file=sys.stderr,
            )
            return 2

        report = run_precompute(
            manifest,
            manifest_path,
            dry_run=args.dry_run,
            require_ocr=args.require_ocr,
            out=args.out,
            program_path=program_path,
        )
        _print_report(report)
        return 0
    except PrecomputeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
