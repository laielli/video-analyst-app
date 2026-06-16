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
from vision.azure_vision import AnalysisResult, Caption, Detection, Box  # noqa: E402
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


# Canned CAPTION output (describe_scene). One whole-image caption with a full-frame box —
# mirrors the real Azure CAPTION shape (no region).
HERO_CAPTION = Caption(
    text="A football player runs with the ball on a stadium pitch.",
    confidence=0.83,
    box=Box(0.0, 0.0, 1.0, 1.0),
)


class FakeAzureVision:
    """Stands in for AzureVision. analyze() returns the same objects for every frame, so the
    sampled-frame stride is what determines which ts carry detections. Counts calls. With
    caption=True it returns one canned Caption (the describe_scene CAPTION feature)."""
    instances: list["FakeAzureVision"] = []

    def __init__(self, objects=None, caption=None):
        self.objects = list(HERO_OBJECTS if objects is None else objects)
        self.caption = HERO_CAPTION if caption is None else caption
        self.analyze_calls = 0
        FakeAzureVision.instances.append(self)

    @classmethod
    def from_env(cls):
        return cls()

    def analyze(self, image_bytes, detect=True, read=False, caption=False):
        # Azure Read is deliberately NOT a rung in the OCR ladder (Resolved #2): its known
        # 22->77 jersey-font misread would poison the cache. Enforce that suite-wide — any
        # read=True from the pipeline is a regression. (Guard ported from PR #6's
        # azure_read_spy, hardened from a single test into a fixture-wide invariant.)
        assert not read, "Azure Read must NOT be a rung in the OCR ladder (Resolved #2)"
        self.analyze_calls += 1
        return AnalysisResult(
            width=1872, height=1042,
            objects=list(self.objects) if detect else [],
            lines=[],
            captions=[self.caption] if (caption and self.caption is not None) else [],
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


@pytest.fixture(autouse=True)
def _reset_codegen_budget(monkeypatch):
    """`server._codegen_calls` is a module-global set at import that NOTHING else resets — a test
    that mutates the budget counter would poison every later free-text test via ordering. Reset it
    to 0 per test via monkeypatch (auto-restored). Imported lazily so non-server tests don't pay
    the server import cost / a hard dependency."""
    import server
    monkeypatch.setattr(server, "_codegen_calls", 0)
    yield


@pytest.fixture(autouse=True)
def _isolated_run_store(tmp_path, monkeypatch):
    """Redirect the process-wide run cache to a per-test tmp_path-rooted FileRunStore so the
    free-text cache / permalink tests never read or write the real api/.run_cache/ (which would
    make codegen-call-count + poisoning assertions order-dependent across runs). Monkeypatched so
    it auto-restores."""
    import server
    import run_cache
    store = run_cache.FileRunStore(tmp_path / "run_cache", max_entries=1000)
    monkeypatch.setattr(server, "_run_store", store)
    yield store


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


# ---- codegen mock seam (free-text / clip-aware-prompt tests) ----------------------------
# There is no codegen mock in production conftest; the free-text path calls codegen.enabled()
# and codegen.AzureCodegen.from_env().generate(question, clip=...). FakeCodegen stands in for
# AzureCodegen with a from_env() classmethod, returning a configured program (or raising), and
# RECORDS the clip it was called with so prompt-injection tests can inspect it.

class FakeCodegen:
    """Stands in for codegen.AzureCodegen. generate() returns a configured program (list of DSL
    steps) or raises a configured exception. Records (question, clip) per call and counts calls
    across instances so zero-call assertions (e.g. Interpreter.run never reached) are real."""
    last_clip = None
    total_calls = 0

    def __init__(self, program=None, raise_exc=None):
        self._program = program if program is not None else []
        self._raise_exc = raise_exc
        self.calls = []

    @classmethod
    def from_env(cls):
        return cls()

    def generate(self, question, clip=None):
        FakeCodegen.total_calls += 1
        self.calls.append((question, clip))
        FakeCodegen.last_clip = clip
        if self._raise_exc is not None:
            raise self._raise_exc
        return list(self._program)


@pytest.fixture
def mock_codegen(monkeypatch):
    """Returns an installer: call install(program=..., raise_exc=...) to monkeypatch the free-text
    path's codegen seam (codegen.enabled()->True and AzureCodegen.from_env().generate->fake). Use
    install(enabled=False) for the codegen-disabled path. Resets the FakeCodegen counters."""
    import codegen

    FakeCodegen.total_calls = 0
    FakeCodegen.last_clip = None

    def install(program=None, raise_exc=None, enabled=True):
        monkeypatch.setattr(codegen, "enabled", lambda: enabled)
        fake = FakeCodegen(program=program, raise_exc=raise_exc)
        monkeypatch.setattr(codegen.AzureCodegen, "from_env", classmethod(lambda cls: fake))
        return fake

    return install


@pytest.fixture
def hero_program(hero_program_path):
    """The committed hero program (comment-stripped) as a plain step list, for codegen mocking."""
    from validate_program import strip_comments
    return strip_comments(json.loads(hero_program_path.read_text()))["program"]
