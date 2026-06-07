"""
Shared pytest fixtures + fakes for the precompute pipeline.

TWO mock seams (both required by the plan):
  (a) Azure — `fake_vision` / `fake_vlm` return canned AnalysisResult / VlmRead, with call
      counting so we can assert "VLM never called" / "Azure Read never called". The pipeline
      constructs via `*_factory` injection (or AzureVision.from_env / AzureVLM.from_env at the
      CLI seam), so tests inject the fakes directly — never the live SDK.
  (b) ffmpeg — `fake_extract` returns canned bytes keyed by (ts, crop), counting calls. Nothing
      here touches ffmpeg or the gitignored .mov.

The detection layout mirrors the committed hero_cache.json: at 4625ms, five `person` boxes; the
highest-confidence one (0.753) sits at the far left (x≈0.029) — that is the box the manifest pin
targets, so it becomes `p_messi`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "scripts"))

from vision.azure_vision import AnalysisResult, Box, Detection  # noqa: E402
from vision.vlm_read import VlmRead  # noqa: E402

CLIP_W, CLIP_H = 1872, 1042

# Five person boxes at the scorer frame (4625ms), mirroring hero_cache.json. The leftmost,
# highest-confidence box is the scorer (#10) the pin targets.
HERO_BOXES_4625 = [
    {"conf": 0.753, "box": (0.0294, 0.1891, 0.2212, 0.6276)},  # scorer -> p_messi
    {"conf": 0.629, "box": (0.7959, 0.2649, 0.1277, 0.2534)},
    {"conf": 0.598, "box": (0.2452, 0.5739, 0.2618, 0.2553)},
    {"conf": 0.567, "box": (0.8793, 0.0, 0.1122, 0.2505)},
    {"conf": 0.562, "box": (0.0, 0.0355, 0.0705, 0.7726)},
]


def _result(boxes, cls="person") -> AnalysisResult:
    objs = [
        Detection(cls=cls, confidence=b["conf"], box=Box(*b["box"]))
        for b in boxes
    ]
    return AnalysisResult(width=CLIP_W, height=CLIP_H, objects=objs, lines=[])


class FakeVision:
    """Stand-in for AzureVision. Returns the hero person boxes on EVERY sampled frame (so the
    scorer is detectable across the window). Counts analyze() calls and records read=True calls
    (Azure Read must never be requested by the pipeline)."""

    def __init__(self, boxes=None):
        self._boxes = boxes if boxes is not None else HERO_BOXES_4625
        self.analyze_calls = 0
        self.read_requested = 0

    def analyze(self, image_bytes, detect=True, read=True):
        self.analyze_calls += 1
        if read:
            self.read_requested += 1
        return _result(self._boxes if detect else [])


class FakeVLM:
    """Stand-in for AzureVLM. Configurable: returns digits, returns NONE, or raises (TPM 429)."""

    def __init__(self, *, digits="", raise_exc=None):
        self._digits = digits
        self._raise = raise_exc
        self.calls = 0

    def read_jersey_number(self, image_bytes, prompt=None):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return VlmRead(raw=self._digits or "NONE", digits=self._digits)


class FakeExtract:
    """Stand-in for ocr_probe.extract_frame. Returns deterministic bytes per (ts, crop); counts
    calls. Never touches ffmpeg or the .mov."""

    def __init__(self):
        self.calls = []

    def __call__(self, video, ts, crop, scale=1.0):
        self.calls.append({"ts": ts, "crop": crop, "scale": scale})
        return f"FRAME[{ts}|{crop}|{scale}]".encode()


@pytest.fixture
def fake_vision():
    return FakeVision()


@pytest.fixture
def fake_vlm_digits():
    return FakeVLM(digits="7")


@pytest.fixture
def fake_vlm_none():
    return FakeVLM(digits="")


@pytest.fixture
def fake_vlm_429():
    return FakeVLM(raise_exc=RuntimeError("429 Too Many Requests (TPM quota)"))


@pytest.fixture
def fake_extract():
    return FakeExtract()


@pytest.fixture
def manifest_path() -> Path:
    return REPO_ROOT / "clips" / "single-goal" / "manifest.json"


@pytest.fixture
def manifest(manifest_path) -> dict:
    return json.loads(manifest_path.read_text())


@pytest.fixture
def hero_program_path() -> Path:
    return API_DIR / "examples" / "hero_program.json"
