"""
Shared fixtures + test doubles for the precompute pipeline suite.

TWO mock seams (the only network/IO the pipeline touches):
  (a) Azure — `FakeVision` returns canned `AnalysisResult`s; `FakeVLM` returns canned
      `VlmRead`s. We monkeypatch `precompute.make_vision` / `precompute.make_vlm` (the
      from_env construction seam) so no real creds/SDK are needed and call-counting works.
  (b) ffmpeg — we monkeypatch `precompute.extract_frame` and `precompute._ffmpeg_still`
      so the suite never shells out to ffmpeg or touches the gitignored single-goal.mov.

Everything is driven off a temp manifest mirroring clips/single-goal/manifest.json, so the
suite is hermetic and the committed examples/hero_cache.json is never overwritten.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
SCRIPTS = API_DIR / "scripts"
for p in (str(API_DIR), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import precompute  # noqa: E402
from vision.azure_vision import AnalysisResult, Detection, Box  # noqa: E402
from vision.vlm_read import VlmRead  # noqa: E402


# --------------------------------------------------------------------------- #
# canned detect output — mirrors the real hero-cache boxes at 4625ms          #
# --------------------------------------------------------------------------- #
# (x,y,w,h normalized, confidence) — p_messi is the conf-0.753 left box.
HERO_DETECTIONS_4625 = [
    ("person", 0.753, (0.0294, 0.1891, 0.2212, 0.6276)),  # -> p0 -> pinned p_messi
    ("person", 0.629, (0.7959, 0.2649, 0.1277, 0.2534)),  # -> p1
    ("person", 0.598, (0.2452, 0.5739, 0.2618, 0.2553)),  # -> p2
    ("person", 0.567, (0.8793, 0.0,    0.1122, 0.2505)),  # -> p3
    ("person", 0.562, (0.0,    0.0355, 0.0705, 0.7726)),  # -> p4
]


def _analysis(detections) -> AnalysisResult:
    objs = [
        Detection(cls=cls, confidence=conf, box=Box(*box))
        for (cls, conf, box) in detections
    ]
    return AnalysisResult(width=1872, height=1042, objects=objs, lines=[])


class FakeVision:
    """Test double for AzureVision. analyze() returns canned detections by sampled ts.

    `by_ts` maps ts_ms -> list of (cls, conf, (x,y,w,h)); but analyze() only gets image
    BYTES, so we encode the ts in the bytes the fake extract_frame returns (see fake_extract).
    """

    def __init__(self, by_ts: dict[int, list]):
        self.by_ts = by_ts
        self.calls: list[int] = []

    def analyze(self, image_bytes: bytes, detect: bool = True, read: bool = True):
        ts = _ts_from_bytes(image_bytes)
        self.calls.append(ts)
        return _analysis(self.by_ts.get(ts, []))


class FakeVLM:
    """Test double for AzureVLM.read_jersey_number. Configurable per-call behavior."""

    def __init__(self, digits: str = "", raises: Exception | None = None):
        self.digits = digits
        self.raises = raises
        self.calls = 0

    def read_jersey_number(self, image_bytes: bytes, prompt: str | None = None) -> VlmRead:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return VlmRead(raw=self.digits, digits=self.digits)


# --------------------------------------------------------------------------- #
# ffmpeg seam — fake extract_frame encodes the ts in the returned bytes        #
# --------------------------------------------------------------------------- #
def _ts_from_bytes(b: bytes) -> int:
    try:
        return int(b.decode().split(":")[1].split("|")[0])
    except (ValueError, IndexError, UnicodeDecodeError):
        return -1


def make_fake_extract(call_log: list | None = None):
    """A drop-in for ocr_probe.extract_frame that never shells out. Encodes ts into bytes so
    FakeVision can key its canned detections on the sampled ts."""
    def _fake(video, ts_seconds, crop=None, scale=1.0):
        ts_ms = int(round(ts_seconds * 1000))
        if call_log is not None:
            call_log.append({"ts_ms": ts_ms, "crop": crop, "scale": scale})
        return f"FRAME:{ts_ms}|crop={crop}|scale={scale}".encode()
    return _fake


# --------------------------------------------------------------------------- #
# manifest fixture (mirrors clips/single-goal/manifest.json)                   #
# --------------------------------------------------------------------------- #
HERO_MANIFEST = {
    "clip": {
        "id": "single-goal", "source": "single-goal.mov",
        "width": 1872, "height": 1042, "duration_ms": 5067, "fps": 60,
    },
    "sampling": {"start_ms": 3500, "end_ms": 5000, "fps": 8},
    "detect_classes": ["person"],
    "ocr_region": "jersey",
    "pins": [
        {
            "match": {"frame_ts_ms": 4625,
                      "nearest_box": {"x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276}},
            "det_id": "p_messi", "text": "10", "note": "hand-verified hero #10",
        }
    ],
    "events": [
        {"ts_ms": 4000, "type": "goal", "scorer_pin": "p_messi",
         "box": {"x": 0.86, "y": 0.34, "w": 0.1, "h": 0.34}}
    ],
    "evidence_frames": [
        {"ts_ms": 4000, "filename": "goal-4000.jpg"},
        {"ts_ms": 4625, "filename": "scorer-4625.jpg"},
    ],
}


@pytest.fixture
def tmp_manifest(tmp_path):
    """Write the hero manifest into tmp_path and a dummy source .mov next to it; load it."""
    clip_dir = tmp_path / "clips" / "single-goal"
    clip_dir.mkdir(parents=True)
    mov = tmp_path / "single-goal.mov"
    mov.write_bytes(b"\x00")  # presence-only; extract_frame is mocked
    manifest = json.loads(json.dumps(HERO_MANIFEST))
    manifest["clip"]["source"] = "../../single-goal.mov"
    mpath = clip_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    return precompute.load_manifest(mpath)


@pytest.fixture
def hero_detect_map():
    """Detections only at 4625ms — the verdict-bearing sampled frame (others degrade empty)."""
    return {4625: HERO_DETECTIONS_4625}


@pytest.fixture
def patch_ffmpeg(monkeypatch):
    """Patch the ffmpeg seams: extract_frame -> fake bytes; _ffmpeg_still -> writes a stub jpg."""
    log: list = []
    monkeypatch.setattr(precompute, "extract_frame", make_fake_extract(log))

    def _fake_still(video, ts_ms, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Minimal valid-ish JPEG header + footer so it is a real (tiny) file on disk.
        dest.write_bytes(b"\xff\xd8\xff\xe0" + f"STILL:{ts_ms}".encode() + b"\xff\xd9")
        return True

    monkeypatch.setattr(precompute, "_ffmpeg_still", _fake_still)
    # ffmpeg presence check in main(): pretend it exists.
    monkeypatch.setattr(precompute.shutil, "which", lambda name: "/usr/bin/" + name)
    return log


@pytest.fixture
def patch_azure(monkeypatch, hero_detect_map):
    """Default Azure seam: FakeVision with the hero detections, no VLM (pin-only hero path)."""
    vision = FakeVision(hero_detect_map)
    monkeypatch.setattr(precompute, "make_vision", lambda: vision)
    monkeypatch.setattr(precompute, "make_vlm", lambda: None)
    return vision
