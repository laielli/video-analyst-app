"""
First pytest setup for api/. Two mock seams keep the suite hermetic — no Azure creds, no
ffmpeg, no gitignored .mov:

  (a) Azure: fixtures build canned AnalysisResult / VlmRead and hand the pipeline factory
      callables (monkeypatching AzureVision.from_env / AzureVLM.from_env semantics by passing
      `azure_vision_factory` / `azure_vlm_factory`). Spies count calls for zero-call asserts.
  (b) ffmpeg: extract_frame is patched per-test to return canned bytes — the pipeline never
      shells out to ffmpeg or touches the .mov.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))

from vision.azure_vision import AnalysisResult, Detection, Box  # noqa: E402
from vision.vlm_read import VlmRead  # noqa: E402

FIXTURES = HERE / "fixtures"


# --------------------------------------------------------------------------- fake Azure objects

def make_analysis_result(detections: list[dict], *, width: int = 1872, height: int = 1042) -> AnalysisResult:
    """Build a real AnalysisResult from [{cls,confidence,box}] dicts (box normalized 0-1)."""
    objs = [
        Detection(cls=d["cls"], confidence=d.get("confidence"),
                  box=Box(**d["box"]))
        for d in detections
    ]
    return AnalysisResult(width=width, height=height, objects=objs, lines=[])


# The five real hero-frame person boxes (mirrors examples/hero_cache.json @ 4625ms, pre-mint).
HERO_FRAME_4625 = [
    {"cls": "person", "confidence": 0.753, "box": {"x": 0.0294, "y": 0.1891, "w": 0.2212, "h": 0.6276}},
    {"cls": "person", "confidence": 0.598, "box": {"x": 0.2452, "y": 0.5739, "w": 0.2618, "h": 0.2553}},
    {"cls": "person", "confidence": 0.629, "box": {"x": 0.7959, "y": 0.2649, "w": 0.1277, "h": 0.2534}},
    {"cls": "person", "confidence": 0.567, "box": {"x": 0.8793, "y": 0.0, "w": 0.1122, "h": 0.2505}},
    {"cls": "person", "confidence": 0.562, "box": {"x": 0.0, "y": 0.0355, "w": 0.0705, "h": 0.7726}},
]


class FakeVision:
    """Canned AzureVision: returns HERO_FRAME_4625 only at the 4625ms frame, else empty.

    `analyze(image_bytes, detect, read)` ignores image_bytes (ffmpeg is mocked). We tag the
    extracted bytes with the ts so we can vary detections per frame; the extract_frame mock
    encodes the ts in the returned bytes.
    """

    def __init__(self, per_ts: dict[int, list[dict]] | None = None):
        self.per_ts = per_ts if per_ts is not None else {4625: HERO_FRAME_4625}
        self.calls = 0

    def analyze(self, image_bytes: bytes, detect: bool = True, read: bool = True) -> AnalysisResult:
        self.calls += 1
        ts = _ts_from_bytes(image_bytes)
        return make_analysis_result(self.per_ts.get(ts, []))


class FakeVLM:
    """Canned AzureVLM. `behavior` is 'digits' | 'none' | callable(image_bytes)->VlmRead.
    Or set `raises` to an Exception to simulate 429/missing-creds at read time."""

    def __init__(self, behavior="none", *, raises: Exception | None = None):
        self.behavior = behavior
        self.raises = raises
        self.calls = 0

    def read_jersey_number(self, image_bytes: bytes, prompt: str | None = None) -> VlmRead:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        if callable(self.behavior):
            return self.behavior(image_bytes)
        if self.behavior == "digits":
            return VlmRead(raw="10", digits="10")
        return VlmRead(raw="NONE", digits="")


def _ts_from_bytes(image_bytes: bytes) -> int:
    """Decode the ts the fake extract_frame encoded into the bytes (b'frame@<ts>...')."""
    try:
        s = image_bytes.decode("utf-8", "ignore")
        if s.startswith("frame@"):
            return int(s.split("@", 1)[1].split(":", 1)[0])
    except (ValueError, IndexError):
        pass
    return -1


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def hero_manifest() -> dict:
    import json
    return json.loads((REPO_ROOT / "clips" / "single-goal" / "manifest.json").read_text())


@pytest.fixture
def hero_manifest_path() -> Path:
    return REPO_ROOT / "clips" / "single-goal" / "manifest.json"


@pytest.fixture
def fake_extract(monkeypatch):
    """
    Patch precompute.extract_frame to return ts-tagged canned bytes (no ffmpeg, no .mov).
    Records calls so tests can assert zero/specific extractions. The ts is taken from the
    SECOND positional arg (ts in seconds) and encoded so FakeVision can vary by frame.
    """
    import precompute

    calls: list[tuple] = []

    def _fake(video, ts_s, crop=None, scale=1.0):
        ts_ms = int(round(ts_s * 1000))
        calls.append((str(video), ts_ms, crop, scale))
        return f"frame@{ts_ms}:crop={crop}:scale={scale}".encode()

    monkeypatch.setattr(precompute, "extract_frame", _fake)
    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


@pytest.fixture
def no_stills(monkeypatch):
    """Stub out still emission so tests don't depend on ffmpeg/the .mov for JPEGs."""
    import precompute
    written: list = []

    def _stub(manifest, video, *, dry_run, warn, log):
        out = [s["filename"] for s in manifest.get("stills", [])]
        written.extend(out)
        return out

    monkeypatch.setattr(precompute, "emit_stills", _stub)
    _stub.written = written  # type: ignore[attr-defined]
    return _stub
