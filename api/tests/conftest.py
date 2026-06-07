"""
Fixtures + mock seams for the precompute pipeline tests.

TWO mock seams (no real Azure, no ffmpeg, no .mov):
  (a) Azure — fake AzureVision / AzureVLM with from_env() classmethods, returning canned
      AnalysisResult / VlmRead. Call-counting so zero-call assertions are real.
  (b) ffmpeg — frame_provider / jersey_crop_provider are injected as callables returning
      canned bytes; tests never touch ffmpeg or the gitignored .mov.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Reuse the real dataclasses so the fakes are structurally identical to production.
from vision.azure_vision import AnalysisResult, Detection, Box  # noqa: E402
from vision.vlm_read import VlmRead  # noqa: E402

API_DIR = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---- canned Azure detect output (mirrors the real boxes in hero_cache.json @ 4.625s) ----
# Intentionally NOT in det_id order or confidence order, so minting is exercised.
HERO_OBJECTS = [
    Detection(cls="person", confidence=0.598, box=Box(0.2452, 0.5739, 0.2618, 0.2553)),  # p1
    Detection(cls="person", confidence=0.753, box=Box(0.0294, 0.1891, 0.2212, 0.6276)),  # scorer -> p0 -> p_messi
    Detection(cls="person", confidence=0.629, box=Box(0.7959, 0.2649, 0.1277, 0.2534)),
    Detection(cls="person", confidence=0.567, box=Box(0.8793, 0.0, 0.1122, 0.2505)),
    Detection(cls="ball",   confidence=0.40,  box=Box(0.5, 0.5, 0.05, 0.05)),            # filtered out
    Detection(cls="person", confidence=0.562, box=Box(0.0, 0.0355, 0.0705, 0.7726)),
]


class FakeAzureVision:
    """Stands in for AzureVision. analyze() returns the same objects for every frame, so the
    sampled-frame stride is what determines which ts carry detections. Counts calls."""
    instances: list["FakeAzureVision"] = []

    def __init__(self, objects=None):
        self.objects = list(HERO_OBJECTS if objects is None else objects)
        self.analyze_calls = 0
        FakeAzureVision.instances.append(self)

    @classmethod
    def from_env(cls):
        return cls()

    def analyze(self, image_bytes, detect=True, read=True):
        self.analyze_calls += 1
        return AnalysisResult(
            width=1872, height=1042,
            objects=list(self.objects) if detect else [],
            lines=[],
        )


class CallSpyVLM:
    """Fake AzureVLM. read_jersey_number returns a configured VlmRead (or raises). Counts calls.
    A class-level call counter lets zero-call tests assert across all instances."""
    total_calls = 0

    def __init__(self, digits="", raise_exc=None):
        self.digits = digits
        self.raise_exc = raise_exc
        self.calls = 0

    @classmethod
    def from_env(cls):
        # default: a benign VLM that reads nothing legible (returns no digits).
        return cls(digits="")

    def read_jersey_number(self, image_bytes, prompt=None):
        CallSpyVLM.total_calls += 1
        self.calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        return VlmRead(raw=self.digits or "NONE", digits=self.digits)


@pytest.fixture(autouse=True)
def _reset_spies():
    FakeAzureVision.instances = []
    CallSpyVLM.total_calls = 0
    yield


@pytest.fixture
def manifest():
    """The real hero manifest (comment-stripped), with `source` neutralized so no .mov is needed."""
    import precompute
    raw = json.loads((API_DIR.parent / "clips" / "single-goal" / "manifest.json").read_text())
    m = precompute._strip_comments(raw)
    m["_dir"] = API_DIR.parent / "clips" / "single-goal"
    return m


@pytest.fixture
def fake_frame_provider():
    """frame_provider(ts) -> canned PNG-ish bytes for every ts (ffmpeg never invoked)."""
    def provider(ts):
        return b"\x89PNG\r\n\x1a\n" + ts.to_bytes(4, "big")
    return provider


@pytest.fixture
def fake_crop_provider():
    """jersey_crop_provider(ts, box) -> canned bytes."""
    def provider(ts, box):
        return b"\x89PNG-crop" + ts.to_bytes(4, "big")
    return provider


@pytest.fixture
def hero_program_path():
    return API_DIR / "examples" / "hero_program.json"
