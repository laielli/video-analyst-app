"""
Shared fixtures for the precompute pipeline tests. The two mock seams the plan mandates:

  (a) Azure   — monkeypatch AzureVision.from_env / AzureVLM.from_env to return canned
                fakes. The pipeline constructs both via from_env (never AzureVision(ep,key)
                directly), so patching from_env fully isolates the network.
  (b) ffmpeg  — patch precompute.extract_frame to return canned bytes, so tests never
                depend on ffmpeg or the gitignored single-goal.mov.

Tests import the pipeline module by name (pytest.ini pythonpath = . scripts).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# vision dataclasses for building canned detect/read results
from vision.azure_vision import AnalysisResult, Detection, Box  # noqa: E402
from vision.vlm_read import VlmRead  # noqa: E402

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent
REAL_MANIFEST = REPO_ROOT / "clips" / "single-goal" / "manifest.json"


# --------------------------------------------------------------------------- #
# The 5 hero detections at 4625ms (mirrors the committed hero_cache, but with NO
# det_ids — Azure returns none; the pipeline mints them). p_messi is the conf-0.75
# left box, which is the highest-confidence det -> becomes p0 -> pin renames to p_messi.
# --------------------------------------------------------------------------- #
HERO_DETECTIONS = [
    Detection("person", 0.753, Box(0.0294, 0.1891, 0.2212, 0.6276)),  # the #10 (highest conf)
    Detection("person", 0.629, Box(0.7959, 0.2649, 0.1277, 0.2534)),
    Detection("person", 0.598, Box(0.2452, 0.5739, 0.2618, 0.2553)),
    Detection("person", 0.567, Box(0.8793, 0.0, 0.1122, 0.2505)),
    Detection("person", 0.562, Box(0.0, 0.0355, 0.0705, 0.7726)),
]


class FakeAzureVision:
    """Canned AzureVision: returns HERO_DETECTIONS only at the 4625ms frame, empty elsewhere.
    Records analyze() calls so tests can assert call counts / that read is never requested."""

    # 4625ms frame bytes carry this sentinel so the fake can key detections to that frame.
    SENTINEL_4625 = b"FRAME@4625"

    def __init__(self):
        self.calls = []

    def analyze(self, image_bytes, detect=True, read=True):
        self.calls.append({"detect": detect, "read": read, "bytes": image_bytes})
        objs = list(HERO_DETECTIONS) if image_bytes == self.SENTINEL_4625 else []
        return AnalysisResult(width=1872, height=1042, objects=objs, lines=[])


class CountingVLM:
    """Canned AzureVLM that records read_jersey_number calls. Default: returns no digits
    (so the ladder degrades to skip). Configure .responses or .raise_with to vary."""

    def __init__(self, digits="", raise_with=None):
        self.calls = []
        self._digits = digits
        self._raise = raise_with

    def read_jersey_number(self, image_bytes, prompt=None):
        self.calls.append(image_bytes)
        if self._raise is not None:
            raise self._raise
        return VlmRead(raw=self._digits, digits=self._digits)


@pytest.fixture
def hero_manifest():
    """The real committed manifest (comment-stripped) for single-goal."""
    import precompute
    return precompute.load_manifest(REAL_MANIFEST)


@pytest.fixture
def fake_extract(monkeypatch):
    """Patch precompute.extract_frame so no ffmpeg / no .mov is needed. Full-frame extracts
    at 4625ms return the SENTINEL bytes (so FakeAzureVision yields the hero dets); all other
    ts / any crop return distinct non-sentinel bytes. Records calls."""
    import precompute
    calls = []

    def _fake(video, ts, crop=None, scale=1.0):
        calls.append({"ts": ts, "crop": crop, "scale": scale})
        ts_ms = round(ts * 1000)
        if crop is None and ts_ms == 4625:
            return FakeAzureVision.SENTINEL_4625
        return f"FRAME@{ts_ms}/crop={crop}/scale={scale}".encode()

    monkeypatch.setattr(precompute, "extract_frame", _fake)
    _fake.calls = calls
    return _fake


@pytest.fixture
def patch_vision(monkeypatch):
    """Install a FakeAzureVision via AzureVision.from_env (the seam the pipeline uses).
    Also force the env-creds guard to pass. Returns the fake for call assertions."""
    import vision.azure_vision as av
    fake = FakeAzureVision()
    monkeypatch.setenv("AZURE_VISION_ENDPOINT", "https://fake.example")
    monkeypatch.setenv("AZURE_VISION_KEY", "fake-key")
    monkeypatch.setattr(av.AzureVision, "from_env", classmethod(lambda cls: fake))
    return fake


@pytest.fixture
def no_vlm(monkeypatch):
    """No Azure OpenAI creds -> the VLM rung is unavailable (the hero default)."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)


@pytest.fixture
def out_paths(tmp_path):
    """Isolated cache + frames-dir under tmp so tests never touch the committed artifacts."""
    return {"cache": tmp_path / "cache.json", "frames": tmp_path / "frames"}


def write_temp_manifest(tmp_path, manifest: dict) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p
