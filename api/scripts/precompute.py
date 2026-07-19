#!/usr/bin/env python3
"""
Automated clip -> cache precompute pipeline.

Turns a raw clip (starting with single-goal.mov) + a per-clip manifest into the artifacts
the demo replays today, with NO hand-annotation of vision output — including NO hardcoded
jersey digits. Every read_text entry is either a REAL gpt-4o VLM read or absent:

  (a) frame stills under web/public/frames/<clip_id>/  (one per sampled ts; EvidencePanel /
      the clip-scrubber render these)
  (b) a clip video under web/public/clips/<clip_id>.mp4 (manifest.video window, H.264)
  (c) web/lib/frames-manifest.json (stills + poster + clip-video metadata, read-modify-write)
  (d) a per-clip replay cache JSON in the exact shape api/interpreter/cache.py loads
      (`clip` block, `detect` map keyed by sampled-frame ms, `read_text` map keyed by
       det_id, `events` list) — see docs/cache-schema.md.

The human knowledge (which det is the scorer, the goal moment, the sampling window, and the
EXPECTED jersey digits for a pin) lives in clips/<id>/manifest.json. A pin RENAMES the nearest
det to a semantic id and DISCLOSES an expected read (`{"mode": "live_vlm", "expected_text":
"10", ...}`); the pipeline then requires a real gpt-4o VLM read of the full-person crop to
agree with that expectation before it will write the read — a mismatch is a hard GroundingError,
never a silent hardcoded override. Azure supplies the detections + the gpt-4o VLM read;
everything that grounds the verdict — det_ids, read_text linkage, events — is minted/verified
here, deterministically.

    python scripts/precompute.py --clip single-goal --query hero-10-first-goal
    python scripts/precompute.py --manifest ../clips/single-goal/manifest.json --dry-run
    python scripts/precompute.py --clip single-goal --require-ocr   # hard-fail if a read is skipped

Exit 0 = cache written + self-validated (replays the hero program to a grounded answer).
       1 = produced but validation / grounding failed (or --require-ocr and a read skipped).
       2 = setup problem (no ffmpeg / no manifest / no creds when required).

Reuses scripts/ocr_probe.py:extract_frame + load_dotenv (the scripts-import-scripts path
hack other probes use) and vision/azure_vision.py:AzureVision / vision/vlm_read.py:AzureVLM.

OPEN-KNOB choices (justified in the PR body):
  - frame sampling : detect on EVERY sampled ts in the window (12-13 frames @ 8fps, cheap).
  - det_id scheme  : p{N} by descending confidence per frame (p0 = highest); manifest pins
                     RENAME the nearest-box det to a semantic id (p_messi).
  - JPEG stills    : ffmpeg emits .jpg directly (-q:v) to the destination (no Pillow dep).
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
sys.path.insert(0, str(HERE))      # ocr_probe, validate_program, run_program (script imports)
sys.path.insert(0, str(API_DIR))   # vision, interpreter packages

# ocr_probe gives us the proven ffmpeg extraction + the no-dep .env loader.
from ocr_probe import extract_frame, load_dotenv  # noqa: E402

from limits import MAX_CAPTION_LEN  # noqa: E402 — per-caption free-text length cap

CLIPS_DIR = REPO_ROOT / "clips"
FRAMES_DIR = REPO_ROOT / "web" / "public" / "frames"
CLIPS_OUT_DIR = REPO_ROOT / "web" / "public" / "clips"
FRAMES_MANIFEST_PATH = REPO_ROOT / "web" / "lib" / "frames-manifest.json"
DEFAULT_OUT = API_DIR / "examples" / "hero_cache.json"


# --------------------------------------------------------------------------------------
# manifest loading + resolution
# --------------------------------------------------------------------------------------

def _strip_comments(obj):
    """Drop _-prefixed authoring keys recursively (same convention as validate_program)."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


class SetupError(Exception):
    """Raised for setup problems -> exit 2 (no manifest / no ffmpeg / missing creds when required)."""


class GroundingError(Exception):
    """Raised when the produced cache fails self-validation / grounding -> exit 1."""


def manifest_path_for_clip(clip_id: str) -> Path:
    return CLIPS_DIR / clip_id / "manifest.json"


def resolve_clip_query(clip_id: str | None, query_id: str | None) -> tuple[Path, Path]:
    """Resolve the (self-validation program_path, cache out) pair for a clip/query from the
    canned.QUERIES registry (the query's `program` + `cache` paths), so precomputing a NON-hero
    clip self-validates the RIGHT pinned program and writes the RIGHT cache. Keeps the hero
    back-compatible: with no clip/query (and no resolvable entry) it returns the hero defaults
    (DEFAULT_OUT + hero_program.json). A `--query` that names a real query wins; otherwise the
    first query whose `clip` matches `--clip` is used (so `--clip <id>` alone Just Works for a
    single-grounding-program clip). Returns (program_path, out)."""
    hero_program = API_DIR / "examples" / "hero_program.json"
    try:
        import canned  # noqa: E402 — guarded; precompute already imports canned in clip_block
    except Exception:  # noqa: BLE001 — registry optional -> hero defaults
        return hero_program, DEFAULT_OUT

    entry = None
    if query_id:
        entry = canned.by_id(query_id)
    if entry is None and clip_id:
        entry = next((q for q in canned.QUERIES if q.get("clip") == clip_id), None)
    if entry is None:
        return hero_program, DEFAULT_OUT
    return Path(entry["program"]), Path(entry["cache"])


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SetupError(
            f"no manifest at {path}\n"
            f"Author clips/<id>/manifest.json (see clips/single-goal/manifest.json). "
            f"This pipeline never synthesizes defaults for a missing manifest."
        )
    raw = json.loads(path.read_text())
    m = _strip_comments(raw)
    m["_dir"] = path.parent  # for resolving the relative `source` path
    return m


def resolve_source(manifest: dict) -> Path:
    """The clip .mov path, resolved relative to the manifest directory (carried IN the manifest
    so canned.py stays unchanged). Repo-root-relative paths also work."""
    src = manifest["clip"].get("source")
    if not src:
        raise SetupError("manifest clip.source is required (path to the .mov).")
    p = Path(src)
    if not p.is_absolute():
        p = (manifest["_dir"] / p).resolve()
    return p


# --------------------------------------------------------------------------------------
# clip dims (registry source of truth; ffprobe fallback)
# --------------------------------------------------------------------------------------

def clip_block(manifest: dict) -> dict:
    """Build the cache `clip` block. Source of truth = canned.CLIPS registry (NOT edited);
    fall back to manifest values, then ffprobe, if the registry has no entry."""
    m_clip = manifest["clip"]
    clip_id = m_clip["id"]
    block = {
        "id": clip_id,
        "width": m_clip.get("width"),
        "height": m_clip.get("height"),
        "duration_ms": m_clip.get("duration_ms"),
        "fps": m_clip.get("fps"),
    }
    # Registry wins for dims/duration when present (reconcile cache clip vs registry).
    try:
        from canned import CLIPS  # noqa: E402
        reg = CLIPS.get(clip_id)
        if reg:
            for k in ("width", "height", "duration_ms"):
                if reg.get(k) is not None:
                    block[k] = reg[k]
    except Exception:  # noqa: BLE001 — registry is optional here
        pass
    # ffprobe fallback only for anything still missing.
    if None in (block["width"], block["height"], block["duration_ms"]):
        probed = _ffprobe_dims(manifest)
        for k, v in probed.items():
            if block.get(k) is None and v is not None:
                block[k] = v
    return block


def _ffprobe_dims(manifest: dict) -> dict:
    out: dict = {"width": None, "height": None, "duration_ms": None}
    if not shutil.which("ffprobe"):
        return out
    try:
        video = resolve_source(manifest)
    except SetupError:
        return out
    if not video.exists():
        return out
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "json", str(video)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(res.stdout)
        st = (data.get("streams") or [{}])[0]
        out["width"] = st.get("width")
        out["height"] = st.get("height")
        dur = (data.get("format") or {}).get("duration")
        if dur is not None:
            out["duration_ms"] = int(round(float(dur) * 1000))
    except Exception:  # noqa: BLE001 — ffprobe is best-effort
        return {"width": None, "height": None, "duration_ms": None}
    return out


# --------------------------------------------------------------------------------------
# sampling — same stride as op_sample_frames (primitives.py:25-38)
# --------------------------------------------------------------------------------------

def sampled_timestamps(manifest: dict) -> list[int]:
    s = manifest["sampling"]
    start, end, fps = s["start_ms"], s["end_ms"], s["fps"]
    stride = max(1, round(1000 / fps))
    return list(range(start, end + 1, stride))


# --------------------------------------------------------------------------------------
# det_id minting (OPEN KNOB: p{N} by descending confidence per frame)
# --------------------------------------------------------------------------------------

def _round_box(box: dict) -> dict:
    return {k: round(float(box[k]), 4) for k in ("x", "y", "w", "h")}


def mint_det_ids(detections: list[dict]) -> list[dict]:
    """Assign p0,p1,... by DESCENDING confidence (highest = p0). Deterministic: ties broken
    by box (x,y,w,h) so a fixed input always yields the same ids. Input dicts: {cls,confidence,box}."""
    ordered = sorted(
        detections,
        key=lambda d: (
            -(d["confidence"] if d.get("confidence") is not None else -1.0),
            d["box"]["x"], d["box"]["y"], d["box"]["w"], d["box"]["h"],
        ),
    )
    out = []
    for i, d in enumerate(ordered):
        out.append({
            "det_id": f"p{i}",
            "cls": d["cls"],
            "confidence": round(d["confidence"], 4) if d.get("confidence") is not None else None,
            "box": _round_box(d["box"]),
        })
    return out


def _box_distance(a: dict, b: dict) -> float:
    """Squared L2 distance between two normalized boxes (center + size) — for nearest-box pin match."""
    acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return (acx - bcx) ** 2 + (acy - bcy) ** 2 + (a["w"] - b["w"]) ** 2 + (a["h"] - b["h"]) ** 2


def apply_pin_rename(detect: dict, pin: dict) -> str | None:
    """Rename the detection nearest the pin's box (at the pin's frame_ts_ms) to pin['det_id'].
    Returns the resolved det_id (the semantic id) or None if no detections at that frame.
    Mutates `detect` in place. This is step (i) of the ordered pin-resolution."""
    match = pin["match"]
    ts = match["frame_ts_ms"]
    target = match["nearest_box"]
    dets = detect.get(str(ts))
    if not dets:
        return None
    nearest = min(dets, key=lambda d: _box_distance(d["box"], target))
    nearest["det_id"] = pin["det_id"]
    return pin["det_id"]


# --------------------------------------------------------------------------------------
# OCR ladder (Phase 3) — pin > gpt-4o VLM > skip. Azure Read is deliberately NOT a rung.
# --------------------------------------------------------------------------------------

class OcrLadder:
    """Resolves a jersey read for a det. Constructs the VLM via from_env() (so tests can
    monkeypatch AzureVLM.from_env). Counts VLM calls for the zero-call test assertions."""

    def __init__(self, *, require_ocr: bool = False):
        self.require_ocr = require_ocr
        self._vlm = None
        self._vlm_attempted = False
        self.vlm_calls = 0
        self.warnings: list[str] = []

    def _get_vlm(self):
        if not self._vlm_attempted:
            self._vlm_attempted = True
            try:
                from vision.vlm_read import AzureVLM  # noqa: E402
                self._vlm = AzureVLM.from_env()
            except Exception as e:  # noqa: BLE001 — missing creds / ValueError -> degrade
                self.warnings.append(f"gpt-4o VLM unavailable ({e}); jersey reads will skip.")
                self._vlm = None
        return self._vlm

    def read(self, *, det_id: str, pin_read: dict | None, crop_bytes: bytes | None) -> dict | None:
        """Return a read_text entry dict, or None to SKIP (write no entry, honest-empty).

        `pin_read` (when present) is a DISCLOSED expectation, never hardcoded text:
        {"mode": "live_vlm", "expected_text": "10", "note": "..."?}. A pin is a human-verification
        GATE, not a degradable rung — it always requires a real gpt-4o read that agrees with the
        expectation; unavailable creds or a digit mismatch are both a hard GroundingError (never a
        silent skip, never a hardcoded override)."""
        # Rung 1: manifest pin — mandatory REAL VLM read, verified against the disclosed expectation.
        if pin_read is not None:
            mode = pin_read.get("mode")
            if mode != "live_vlm":
                raise GroundingError(
                    f"pin read for '{det_id}' has unsupported mode {mode!r}; only 'live_vlm' is "
                    f"supported (hardcoded pinned text is retired)."
                )
            expected = str(pin_read["expected_text"])
            vlm = self._get_vlm()
            if vlm is None:
                raise GroundingError(
                    f"pin '{det_id}' expects a live gpt-4o read of '{expected}' but no VLM is "
                    f"available (missing AZURE_OPENAI_* creds) — this is a hard-fail "
                    f"human-verification gate, not a degrade-to-skip rung."
                )
            if crop_bytes is None:
                raise GroundingError(
                    f"pin '{det_id}' expects a live gpt-4o read of '{expected}' but no crop was "
                    f"produced (frame/crop provider failed)."
                )
            self.vlm_calls += 1
            res = vlm.read_jersey_number(crop_bytes)
            if res.digits != expected:
                raise GroundingError(
                    f"pin '{det_id}' expected gpt-4o to read '{expected}' but it read "
                    f"{res.digits!r} (raw={res.raw!r}) — the honest human-verification gate "
                    f"failed; fix the pin's match box / expected_text, do not hardcode a text "
                    f"override."
                )
            return {
                "text": res.digits,
                "confidence": None,
                "source": "live",
                "note": "gpt-4o read of the full player crop at precompute; expectation "
                        "disclosed and verified",
            }

        # Rung 2: gpt-4o VLM (source=live, accept if digits).
        vlm = self._get_vlm()
        if vlm is not None and crop_bytes is not None:
            try:
                self.vlm_calls += 1
                res = vlm.read_jersey_number(crop_bytes)
                if res.digits:
                    return {"text": res.digits, "confidence": None, "source": "live"}
                # Legible call but no digits -> honest-empty (skip).
                if self.require_ocr:
                    raise GroundingError(f"--require-ocr: VLM returned no digits for {det_id}")
                self.warnings.append(f"VLM read no digits for {det_id}; skipping (honest-empty).")
                return None
            except GroundingError:
                raise
            except Exception as e:  # noqa: BLE001 — 429 / TPM / SDK error -> degrade (NOT Azure Read)
                self.warnings.append(f"VLM read failed for {det_id} ({e}); skipping.")
                if self.require_ocr:
                    raise GroundingError(f"--require-ocr and VLM read failed for {det_id}: {e}")
                return None

        # Rung 3: skip (no VLM available / no crop).
        if self.require_ocr:
            raise GroundingError(f"--require-ocr but no OCR rung succeeded for {det_id}")
        return None


# --------------------------------------------------------------------------------------
# detect (Phase 2) — extract frame -> AzureVision.analyze -> mint det_ids
# --------------------------------------------------------------------------------------

def run_detect(manifest: dict, timestamps: list[int], *, frame_provider, vision) -> tuple[dict, dict]:
    """Returns (detect_map, raw_objects_by_ts). frame_provider(ts) -> png bytes | None (guarded).
    vision is an AzureVision-like with .analyze(bytes, detect=True, read=False)."""
    classes = [c.lower() for c in manifest.get("detect_classes", ["person"])]
    detect: dict = {}
    raw_by_ts: dict = {}
    for ts in timestamps:
        img = frame_provider(ts)
        if img is None:
            continue  # guarded: one bad frame doesn't abort the run
        result = vision.analyze(img, detect=True, read=False)
        objs = []
        for d in result.objects:
            if d.cls.lower() not in classes:
                continue
            b = d.box.rounded() if hasattr(d.box, "rounded") else d.box
            box = {"x": b.x, "y": b.y, "w": b.w, "h": b.h} if hasattr(b, "x") else dict(b)
            objs.append({"cls": d.cls, "confidence": d.confidence, "box": box})
        raw_by_ts[ts] = objs
        minted = mint_det_ids(objs)
        if minted:
            detect[str(ts)] = minted
    return detect, raw_by_ts


def run_describe_scene(manifest: dict, timestamps: list[int], *, frame_provider, vision) -> dict:
    """Scene-caption precompute (Azure Image Analysis CAPTION), gated by a manifest opt-in
    `sampling.caption_frames` (default false — captions add cost + are not needed for the existing
    grounding shapes). Returns a `captions` map keyed by sampled-frame ts ms, mirroring run_detect's
    detect map. Each entry: {text (<= MAX_CAPTION_LEN), confidence, box in [0,1]}. Returns {} when
    the opt-in is off so the default manifest yields NO captions key."""
    if not manifest.get("sampling", {}).get("caption_frames", False):
        return {}
    captions: dict = {}
    for ts in timestamps:
        img = frame_provider(ts)
        if img is None:
            continue  # guarded: one bad frame doesn't abort the run
        result = vision.analyze(img, detect=False, read=False, caption=True)
        entries = []
        for c in getattr(result, "captions", []):
            b = c.box.rounded() if hasattr(c.box, "rounded") else c.box
            box = {"x": b.x, "y": b.y, "w": b.w, "h": b.h} if hasattr(b, "x") else dict(b)
            entries.append({
                "text": (c.text or "")[:MAX_CAPTION_LEN],
                "confidence": c.confidence,
                "box": _round_box(box),
            })
        if entries:
            captions[str(ts)] = entries
    return captions


# --------------------------------------------------------------------------------------
# the linchpin: pins + read_text + events in ONE ordered step (Phase 2.5 / 3 / 4)
# --------------------------------------------------------------------------------------

def crop_jersey_box(person_box: dict) -> dict:
    """op_crop jersey math (primitives.py:77-79), normalized. NOT used by the precompute VLM
    reads any more (see full_person_crop_box) — kept for op_crop parity elsewhere."""
    return {
        "x": round(person_box["x"] + 0.22 * person_box["w"], 4),
        "y": round(person_box["y"] + 0.10 * person_box["h"], 4),
        "w": round(0.56 * person_box["w"], 4),
        "h": round(0.22 * person_box["h"], 4),
    }


def full_person_crop_box(person_box: dict, margin: float = 0.05) -> dict:
    """FULL person box + ~5% margin per side, clamped to [0,1] — the VLM read crop. Replaces
    the brittle jersey-box heuristic (crop_jersey_box) for reads: proven live (vlm_probe) that
    a full-person crop reads jersey digits exactly and survives divers / horizontal players,
    where a tight jersey-box crop cuts the number off."""
    x0 = max(0.0, person_box["x"] - margin * person_box["w"])
    y0 = max(0.0, person_box["y"] - margin * person_box["h"])
    x1 = min(1.0, person_box["x"] + person_box["w"] * (1 + margin))
    y1 = min(1.0, person_box["y"] + person_box["h"] * (1 + margin))
    return {"x": round(x0, 4), "y": round(y0, 4), "w": round(x1 - x0, 4), "h": round(y1 - y0, 4)}


def resolve_pins_reads_events(
    manifest: dict, detect: dict, clip: dict, *, ladder: OcrLadder, jersey_crop_provider
) -> tuple[dict, list[dict]]:
    """ONE ordered step (plan Phase 2.5):
      (i)   apply each pin by nearest-box to RENAME the matched det to its semantic id;
      (ii)  resolve that det's read_text via the OCR ladder (pin wins) under the SAME id;
      (iii) build events deriving scorer_det FROM the same pin-resolution result.
    Returns (read_text_map, events). Asserts the verdict linchpin before returning."""
    pins = manifest.get("pins", [])
    read_text: dict = {}

    # (i) rename matched dets to semantic ids; remember the resolved id per pin.
    resolved_pin_ids: dict[str, str] = {}  # pin det_id -> actually-resolved det_id
    pin_read_for: dict[str, dict] = {}      # resolved det_id -> pin read spec
    for pin in pins:
        resolved = apply_pin_rename(detect, pin)
        if resolved is None:
            raise GroundingError(
                f"pin '{pin.get('det_id')}' matched no detection at "
                f"frame_ts_ms={pin['match']['frame_ts_ms']} (no det to rename)."
            )
        resolved_pin_ids[pin["det_id"]] = resolved
        if pin.get("read") is not None:
            pin_read_for[resolved] = pin["read"]

    # (ii) read_text for OCR-target dets, in deterministic order. A det is an OCR target if it
    #      has a pin (a MANDATORY live_vlm verification, see OcrLadder.read) OR the manifest opts
    #      into reading every jersey (sampling.read_all_jerseys, default false). Every read — pinned
    #      or not — is a REAL gpt-4o VLM call over the FULL-PERSON crop (+5% margin, see
    #      full_person_crop_box); no hardcoded text ever enters the cache.
    read_all = bool(manifest.get("sampling", {}).get("read_all_jerseys", False))
    for ts in sorted(detect, key=int):
        for d in detect[ts]:
            det_id = d["det_id"]
            if det_id in read_text:
                continue  # already read at an earlier (or same-id) frame
            pin_read = pin_read_for.get(det_id)
            if pin_read is None and not read_all:
                continue  # not an OCR target -> honest-empty (no entry, no VLM call)
            crop_box = full_person_crop_box(d["box"])
            crop_bytes = jersey_crop_provider(int(ts), crop_box)
            entry = ladder.read(det_id=det_id, pin_read=pin_read, crop_bytes=crop_bytes)
            if entry is not None:
                read_text[det_id] = entry

    # (iii) events deriving scorer_det from the pin resolution (NOT an independent literal).
    events: list[dict] = []
    for ev in manifest.get("events", []):
        scorer_pin = ev.get("scorer_pin")
        scorer_det = resolved_pin_ids.get(scorer_pin, scorer_pin)
        if scorer_det is None:
            raise GroundingError(f"event at {ev.get('ts_ms')}ms has no scorer_pin/scorer_det.")
        events.append({
            "ts_ms": ev["ts_ms"],
            "type": ev["type"],
            "scorer_det": scorer_det,
            "box": _round_box(ev["box"]),
        })

    # linchpin assertions (before writing).
    all_det_ids = {d["det_id"] for dets in detect.values() for d in dets}
    for ev in events:
        sd = ev["scorer_det"]
        if sd not in all_det_ids:
            raise GroundingError(
                f"event scorer_det '{sd}' is not a minted det_id in `detect` "
                f"(op_answer's overlay re-scan would drop the focus frame)."
            )
    return read_text, events


# --------------------------------------------------------------------------------------
# canonical JSON + atomic write (Phase 5)
# --------------------------------------------------------------------------------------

def canonical_cache(clip: dict, detect: dict, read_text: dict, events: list[dict],
                    captions: dict | None = None) -> dict:
    """Deterministic output: detect keys numeric-sorted, detections by det_id, boxes 4dp. `captions`
    is keyword-defaulted (captions=None) so existing 4-arg call sites + the committed caches stay
    byte-identical; a `captions` block is emitted ONLY when a non-empty map is supplied (describe_scene
    opt-in)."""
    sorted_detect = {}
    for k in sorted(detect, key=int):
        dets = sorted(detect[k], key=lambda d: d["det_id"])
        sorted_detect[k] = [
            {"det_id": d["det_id"], "cls": d["cls"], "confidence": d["confidence"], "box": _round_box(d["box"])}
            for d in dets
        ]
    out = {
        "clip": {
            "id": clip["id"], "width": clip["width"], "height": clip["height"],
            "duration_ms": clip["duration_ms"], "fps": clip["fps"],
        },
        "detect": sorted_detect,
        "read_text": {k: read_text[k] for k in sorted(read_text)},
        "events": sorted(events, key=lambda e: e["ts_ms"]),
    }
    if captions:
        sorted_captions = {}
        for k in sorted(captions, key=int):
            sorted_captions[k] = [
                {"text": c["text"], "confidence": c.get("confidence"), "box": _round_box(c["box"])}
                for c in captions[k]
            ]
        out["captions"] = sorted_captions
    return out


def cache_to_json(cache: dict) -> str:
    return json.dumps(cache, indent=2, ensure_ascii=False) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------------------
# cache shape validation (per docs/cache-schema.md — no separate schema file)
# --------------------------------------------------------------------------------------

def validate_cache_shape(cache: dict) -> list[str]:
    errs: list[str] = []
    if "clip" not in cache:
        errs.append("missing `clip`")
    else:
        for k in ("id", "width", "height", "duration_ms", "fps"):
            if cache["clip"].get(k) is None:
                errs.append(f"clip.{k} is missing/null")
    detect = cache.get("detect", {})
    read_text = cache.get("read_text", {})
    events = cache.get("events", [])
    all_det_ids: set[str] = set()
    for ts, dets in detect.items():
        if not str(ts).lstrip("-").isdigit():
            errs.append(f"detect key '{ts}' is not an integer ms")
        seen = set()
        for d in dets:
            for k in ("det_id", "cls", "box"):
                if k not in d:
                    errs.append(f"detect[{ts}] detection missing '{k}'")
            did = d.get("det_id")
            if did in seen:
                errs.append(f"detect[{ts}] duplicate det_id '{did}' in frame")
            seen.add(did)
            all_det_ids.add(did)
            box = d.get("box", {})
            for k in ("x", "y", "w", "h"):
                v = box.get(k)
                if not isinstance(v, (int, float)) or not (0 <= v <= 1):
                    errs.append(f"detect[{ts}] {did} box.{k}={v!r} not in [0,1]")
    for did, rt in read_text.items():
        if did not in all_det_ids:
            errs.append(f"read_text key '{did}' is not a real det_id in `detect`")
        if rt.get("source") not in ("live", "cached", "pinned"):
            errs.append(f"read_text[{did}].source={rt.get('source')!r} not in live/cached/pinned")
        if not rt.get("text"):
            errs.append(f"read_text[{did}] has no text")
    for ev in events:
        for k in ("ts_ms", "type", "scorer_det", "box"):
            if k not in ev:
                errs.append(f"event missing '{k}'")
        if ev.get("scorer_det") not in all_det_ids:
            errs.append(f"event scorer_det '{ev.get('scorer_det')}' not in `detect`")
    # captions slice (describe_scene opt-in): keyed by sampled ts ms; each entry needs `text`
    # (length-bound by MAX_CAPTION_LEN — captions are model free-text), a `box` in [0,1], and a
    # confidence field (may be null). Absent when the clip was precomputed without caption_frames.
    captions = cache.get("captions", {})
    for ts, caps in captions.items():
        if not str(ts).lstrip("-").isdigit():
            errs.append(f"captions key '{ts}' is not an integer ms")
        for c in caps:
            if "confidence" not in c:
                errs.append(f"captions[{ts}] entry missing 'confidence'")
            text = c.get("text")
            if not text:
                errs.append(f"captions[{ts}] entry has no text")
            elif len(text) > MAX_CAPTION_LEN:
                errs.append(f"captions[{ts}] text is {len(text)} chars; max is {MAX_CAPTION_LEN}")
            box = c.get("box", {})
            for k in ("x", "y", "w", "h"):
                v = box.get(k)
                if not isinstance(v, (int, float)) or not (0 <= v <= 1):
                    errs.append(f"captions[{ts}] box.{k}={v!r} not in [0,1]")
    return errs


# --------------------------------------------------------------------------------------
# self-validating replay (Phase 6.2) — replay the hero program over the produced cache
# --------------------------------------------------------------------------------------

def self_validate_replay(cache: dict, *, program_path: Path, expect_grounded: bool = True) -> dict:
    """Load the produced cache, run the clip's pinned program, and assert it conforms to
    run_doc.schema.json. GENERALIZED (multi-clip): the only structural requirement is an `answer`
    step (the evidence-grounded verdict — every clip's grounding program ends in one), NOT a
    temporal_order step, so a count->answer or read_text->answer program is a valid self-validation
    target too. When `expect_grounded` is True (the manifest declares a positive claim — goal
    events for a temporal clip, or any grounding query for a count/readout clip), assert the
    interpreter's own grounded discriminator: findings.grounded is True. The subject-specific
    expectations ('Yes' / '#10' / '7' / the count) are left to the dedicated replay tests — this
    gate proves the produced cache REPLAYS to a grounded answer, generically, without hardcoding
    the hero's verdict string. `expect_grounded=False` (a degraded/no-claim cache) only asserts the
    run-doc schema + answer-step presence (a valid honest-ungrounded 'No' is the correct answer)."""
    import jsonschema
    from interpreter import Cache, Interpreter  # noqa: E402
    from validate_program import strip_comments  # noqa: E402

    program = strip_comments(json.loads(program_path.read_text()))["program"]

    # Build a Cache directly from the in-memory dict (avoid a temp file).
    run_cache = Cache(cache)
    run_doc = Interpreter(run_cache).run(program)

    rd_schema = json.loads((API_DIR / "schema" / "run_doc.schema.json").read_text())
    rd_errs = sorted(jsonschema.Draft202012Validator(rd_schema).iter_errors(run_doc),
                     key=lambda e: list(e.path))
    if rd_errs:
        msgs = "; ".join(f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in rd_errs)
        raise GroundingError(f"run-doc does not conform to run_doc.schema.json: {msgs}")

    answer = next((t for t in run_doc["trace"] if t["op"] == "answer"), None)
    if answer is None:
        raise GroundingError("program lacks an answer step (the evidence-grounded verdict)")

    if not expect_grounded:
        return run_doc  # degraded / no-claim cache: a valid honest-ungrounded answer is correct.

    # Positive claim -> the produced cache must REPLAY to a grounded answer. Assert the
    # interpreter's own grounded bit rather than a hardcoded verdict string ('Yes' / '10'), so this
    # generalizes across count / readout / temporal clips. The verdict text is checked per-clip in
    # the dedicated replay tests.
    if run_doc["findings"].get("grounded") is not True:
        raise GroundingError(
            f"expected a grounded answer but findings.grounded="
            f"{run_doc['findings'].get('grounded')!r} (reason={run_doc['findings'].get('reason')!r})"
        )
    return run_doc


# --------------------------------------------------------------------------------------
# frame stills (Phase 4) — ffmpeg emits ONE .jpg per sampled ts to web/public/frames/<clip_id>/
# --------------------------------------------------------------------------------------

def emit_stills(manifest: dict, *, video: Path | None, frames_dir: Path = FRAMES_DIR,
                dry_run: bool = False) -> list[str]:
    """Write one JPEG per sampled ts (every frame the pipeline detected on, not a curated
    evidence subset) to frames_dir/<clip_id>/f<ts>.jpg — scale width 960 keep-aspect, -q:v 3.
    Returns the clip-relative paths written (e.g. 'bernabeu-counter/f10000.jpg'). Skips (returns
    []) if the .mov / ffmpeg is unavailable so the cache still gets produced."""
    written: list[str] = []
    if dry_run:
        return written
    if not shutil.which("ffmpeg") or video is None or not video.exists():
        return written
    clip_id = manifest["clip"]["id"]
    dest_dir = frames_dir / clip_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ts in sampled_timestamps(manifest):
        dest = dest_dir / f"f{ts}.jpg"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", str(ts / 1000.0), "-i", str(video),
               "-frames:v", "1", "-vf", "scale=960:-2", "-q:v", "3", str(dest)]
        try:
            subprocess.run(cmd, check=True)
            written.append(f"{clip_id}/f{ts}.jpg")
        except Exception:  # noqa: BLE001 — a bad frame shouldn't abort the run
            continue
    return written


# --------------------------------------------------------------------------------------
# clip video (Phase 4.5) — ffmpeg cuts manifest.video's window to web/public/clips/<id>.mp4
# --------------------------------------------------------------------------------------

def emit_clip_video(manifest: dict, *, video: Path | None, clips_dir: Path = CLIPS_OUT_DIR,
                    dry_run: bool = False) -> str | None:
    """Cut the manifest's `video` window (cut_start_ms/cut_end_ms) from the source .mov to
    clips_dir/<clip_id>.mp4: H.264 (yuv420p), scale=-2:720, -crf 23, no audio, faststart (so the
    <video> tag can start playback before the whole file downloads). Guarded exactly like
    emit_stills (ffmpeg presence + source existence); also skips when the manifest carries no
    `video` block (older/degraded manifests). Returns the filename written, or None when skipped."""
    vid = manifest.get("video")
    if dry_run or not vid:
        return None
    if not shutil.which("ffmpeg") or video is None or not video.exists():
        return None
    clip_id = manifest["clip"]["id"]
    clips_dir.mkdir(parents=True, exist_ok=True)
    dest = clips_dir / f"{clip_id}.mp4"
    start_s = vid["cut_start_ms"] / 1000.0
    dur_s = (vid["cut_end_ms"] - vid["cut_start_ms"]) / 1000.0
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_s), "-i", str(video), "-t", str(dur_s),
        "-vf", "scale=-2:720", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-an", "-movflags", "+faststart", str(dest),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:  # noqa: BLE001 — a bad cut shouldn't abort the run
        return None
    return f"{clip_id}.mp4"


# --------------------------------------------------------------------------------------
# frames-manifest (Phase 4.6) — web/lib/frames-manifest.json, read-modify-write, atomic
# --------------------------------------------------------------------------------------

def build_frames_manifest_entry(manifest: dict, *, stills_ts: list[int],
                                width: int, height: int) -> dict:
    """The EXACT per-clip shape the frontend reads: width, height, stills (sorted sampled ts
    ints), still_path (a '{ts}'-templated route), poster_ts (manifest field), and — only when
    the manifest carries a `video` block — video.{src,offset_ms,duration_ms}. Key order is
    fixed (this dict's insertion order) so a re-run is byte-identical."""
    clip_id = manifest["clip"]["id"]
    entry: dict = {
        "width": width,
        "height": height,
        "stills": sorted(int(ts) for ts in stills_ts),
        "still_path": f"/frames/{clip_id}/f{{ts}}.jpg",
        "poster_ts": manifest.get("poster_ts"),
    }
    vid = manifest.get("video")
    if vid:
        entry["video"] = {
            "src": f"/clips/{clip_id}.mp4",
            "offset_ms": vid["cut_start_ms"],
            "duration_ms": vid["cut_end_ms"] - vid["cut_start_ms"],
        }
    return entry


def emit_frames_manifest(manifest: dict, *, clip: dict, stills_ts: list[int],
                         path: Path = FRAMES_MANIFEST_PATH, dry_run: bool = False) -> dict:
    """Read-modify-write path: merges THIS clip's entry into the existing frames-manifest.json
    (or starts a fresh one), preserving every other clip's entry untouched, then writes back with
    deterministic (sorted-by-clip-id) key order, atomically. Returns the full merged dict (mostly
    for test/report introspection); a `dry_run` skips reading AND writing entirely."""
    if dry_run:
        return {}
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 — a corrupt stub shouldn't abort the run
            existing = {}
    clip_id = manifest["clip"]["id"]
    existing[clip_id] = build_frames_manifest_entry(
        manifest, stills_ts=stills_ts, width=clip["width"], height=clip["height"],
    )
    ordered = {k: existing[k] for k in sorted(existing)}
    atomic_write(path, json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
    return ordered


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def run_precompute(
    manifest: dict,
    *,
    dry_run: bool = False,
    require_ocr: bool = False,
    out: Path = DEFAULT_OUT,
    program_path: Path | None = None,
    vision=None,
    vlm_ladder: OcrLadder | None = None,
    frame_provider=None,
    jersey_crop_provider=None,
    emit_frames: bool = True,
) -> dict:
    """Pipeline core. Returns a report dict. Injectable seams (vision/ladder/providers) let the
    pytest suite mock Azure + ffmpeg with zero real calls; defaults wire the real adapters."""
    program_path = program_path or (API_DIR / "examples" / "hero_program.json")
    load_dotenv(API_DIR / ".env")

    clip = clip_block(manifest)
    timestamps = sampled_timestamps(manifest)

    # Resolve the .mov (may be absent in this worktree — guarded).
    try:
        video = resolve_source(manifest)
    except SetupError:
        video = None

    # default frame_provider: guarded extract_frame (PNG bytes), one bad frame won't abort.
    if frame_provider is None:
        def frame_provider(ts: int):  # noqa: ANN001
            if video is None or not video.exists():
                return None
            try:
                return extract_frame(video, ts / 1000.0, crop=None, scale=1.0)
            except SystemExit:
                return None
            except Exception:  # noqa: BLE001
                return None

    # default jersey_crop_provider: pixel crop (normalized box * dims) at scale 2 for the VLM —
    # the box passed in is the FULL-PERSON + margin box (full_person_crop_box), not the jersey box.
    if jersey_crop_provider is None:
        def jersey_crop_provider(ts: int, jbox: dict):  # noqa: ANN001
            if video is None or not video.exists():
                return None
            W, H = clip["width"], clip["height"]
            if not (W and H):
                return None
            x = int(round(jbox["x"] * W)); y = int(round(jbox["y"] * H))
            w = int(round(jbox["w"] * W)); h = int(round(jbox["h"] * H))
            if w <= 0 or h <= 0:
                return None
            try:
                return extract_frame(video, ts / 1000.0, crop=f"{x},{y},{w},{h}", scale=2.0)
            except SystemExit:
                return None
            except Exception:  # noqa: BLE001
                return None

    # default vision adapter via from_env (so tests patch AzureVision.from_env).
    if vision is None and not dry_run:
        try:
            from vision.azure_vision import AzureVision  # noqa: E402
            vision = AzureVision.from_env()
        except Exception as e:  # noqa: BLE001
            raise SetupError(f"Azure Vision unavailable ({e}); set AZURE_VISION_* in api/.env.")

    ladder = vlm_ladder or OcrLadder(require_ocr=require_ocr)

    # --- dry run: sample + report, NO Azure calls, NO writes ---
    if dry_run:
        return {
            "dry_run": True,
            "clip": clip,
            "timestamps": timestamps,
            "n_frames": len(timestamps),
            "out": str(out),
            "video": str(video) if video else None,
            "warnings": ["dry-run: no Azure calls, no writes"],
            "exit": 0,
        }

    # --- detect ---
    detect, _raw = run_detect(manifest, timestamps, frame_provider=frame_provider, vision=vision)
    if not detect:
        raise GroundingError("no detections produced for any sampled frame (cache would be empty).")

    # --- pins + read_text + events (one ordered step) ---
    read_text, events = resolve_pins_reads_events(
        manifest, detect, clip, ladder=ladder, jersey_crop_provider=jersey_crop_provider,
    )

    # --- describe_scene captions (manifest opt-in sampling.caption_frames; {} otherwise) ---
    captions = run_describe_scene(manifest, timestamps, frame_provider=frame_provider, vision=vision)

    # --- canonical cache + shape validation ---
    cache = canonical_cache(clip, detect, read_text, events, captions=captions)
    shape_errs = validate_cache_shape(cache)
    if shape_errs:
        raise GroundingError("cache shape invalid: " + "; ".join(shape_errs))

    # --- self-validating replay ---
    # A manifest that declares goal events is making a positive claim -> assert the grounded
    # Yes chain. A manifest with no events validly grounds to 'No' (honest-empty degraded clip).
    run_doc = self_validate_replay(
        cache, program_path=program_path,
        expect_grounded=bool(manifest.get("events")),
    )

    # --- write cache atomically ---
    text = cache_to_json(cache)
    atomic_write(out, text)

    # --- stills + clip video + frames-manifest (asset pipeline; all gated on emit_frames) ---
    stills = emit_stills(manifest, video=video, dry_run=not emit_frames)
    clip_video = emit_clip_video(manifest, video=video, dry_run=not emit_frames)
    frames_manifest = emit_frames_manifest(
        manifest, clip=clip, stills_ts=timestamps, dry_run=not emit_frames,
    )

    return {
        "dry_run": False,
        "clip": clip,
        "timestamps": timestamps,
        "detect_frames": sorted(detect, key=int),
        "n_detect": sum(len(v) for v in detect.values()),
        "read_text_ids": sorted(read_text),
        "events": events,
        "clip_video": clip_video,
        "frames_manifest_clip_ids": sorted(frames_manifest),
        "stills": stills,
        "vlm_calls": ladder.vlm_calls,
        "warnings": ladder.warnings,
        "out": str(out),
        "verdict": run_doc["findings"]["verdict"],
        "exit": 0,
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _print_report(report: dict) -> None:
    if report.get("dry_run"):
        print(f"# DRY RUN — {report['n_frames']} frames sampled (no Azure, no writes)")
        print(f"clip:   {report['clip']['id']} "
              f"({report['clip']['width']}x{report['clip']['height']})")
        print(f"video:  {report['video']}")
        print(f"would write -> {report['out']}")
        ts = report["timestamps"]
        print(f"sampled ts (ms): {ts[:6]}{' ...' if len(ts) > 6 else ''}  "
              f"({len(ts)} total, stride {ts[1]-ts[0] if len(ts) > 1 else '-'})")
        return
    print(f"# precompute OK -> {report['out']}")
    print(f"clip:    {report['clip']['id']} ({report['clip']['width']}x{report['clip']['height']})")
    print(f"detect:  {report['n_detect']} dets across {len(report['detect_frames'])} frame(s) "
          f"{report['detect_frames'][:6]}")
    print(f"reads:   {report['read_text_ids']}")
    print(f"events:  {[(e['ts_ms'], e['scorer_det']) for e in report['events']]}")
    print(f"stills:  {len(report['stills'])} written {report['stills'][:3] or '(none — .mov/ffmpeg absent)'}")
    print(f"video:   {report['clip_video'] or '(none — .mov/ffmpeg absent or no manifest.video)'}")
    print(f"frames-manifest clips: {report['frames_manifest_clip_ids'] or '(dry-run/no-emit)'}")
    print(f"VLM calls: {report['vlm_calls']}")
    for w in report.get("warnings", []):
        print(f"  warn: {w}")
    print(f"verdict: {report['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clip -> replay cache precompute pipeline.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--clip", help="clip id; loads clips/<id>/manifest.json")
    g.add_argument("--manifest", type=Path, help="explicit path to a manifest.json")
    ap.add_argument("--query", default="hero-10-first-goal",
                    help="canned query id this cache backs; resolves the pinned program self-validated "
                         "+ the per-clip cache out (default hero-10-first-goal)")
    ap.add_argument("--dry-run", action="store_true", help="sample + report; no Azure calls, no writes")
    ap.add_argument("--require-ocr", action="store_true", help="hard-fail (exit 1) if any OCR read is skipped")
    ap.add_argument("--out", type=Path, default=None,
                    help="cache output path (default: resolved per-clip from --clip/--query, hero fallback)")
    args = ap.parse_args(argv)

    try:
        if args.manifest:
            mpath = args.manifest
        elif args.clip:
            mpath = manifest_path_for_clip(args.clip)
        else:
            print("ERROR: pass --clip <id> or --manifest <path>.", file=sys.stderr)
            return 2
        manifest = load_manifest(mpath)
    except SetupError as e:
        print(f"ERROR (setup): {e}", file=sys.stderr)
        return 2

    # Resolve the right pinned program (self-validation target) + cache out for this clip/query
    # from the canned registry. An explicit --out always wins; otherwise use the resolved path.
    program_path, resolved_out = resolve_clip_query(args.clip, args.query)
    out = args.out if args.out is not None else resolved_out

    try:
        report = run_precompute(
            manifest, dry_run=args.dry_run, require_ocr=args.require_ocr, out=out,
            program_path=program_path,
        )
    except SetupError as e:
        print(f"ERROR (setup): {e}", file=sys.stderr)
        return 2
    except GroundingError as e:
        print(f"ERROR (grounding/validation): {e}", file=sys.stderr)
        return 1

    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
