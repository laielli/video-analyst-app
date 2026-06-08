"""
Vision-adapter tests for vision/azure_vision.py (AzureVision) and vision/vlm_read.py (AzureVLM),
exercised via FAKE SDK clients injected into the adapter — no real Azure calls.

Both adapters lazily import their SDK inside __init__ and store self._client. We bypass __init__
(which would need real creds) by constructing via __new__ and injecting a duck-typed fake client
whose attributes mirror the SDK result shapes the adapter reads.

CAVEAT (per plan): analyze() still executes `from ...models import VisualFeatures` at call time,
and both constructors import the SDK BEFORE cred validation. So azure-ai-vision-imageanalysis /
openai must be importable (they're in requirements.txt). Tests that need the SDK import are
skipped if it is absent, so the suite still runs in a minimal local env.
"""
from __future__ import annotations

import importlib.util

import pytest

from vision.azure_vision import (
    AzureVision, AnalysisResult, Detection, TextLine, Box, _norm_xywh, _box_from_polygon,
)
from vision.vlm_read import AzureVLM, VlmRead


_HAS_AZURE_VISION = importlib.util.find_spec("azure.ai.vision.imageanalysis") is not None
_HAS_OPENAI = importlib.util.find_spec("openai") is not None

requires_azure_vision = pytest.mark.skipif(
    not _HAS_AZURE_VISION, reason="azure-ai-vision-imageanalysis not installed (needed for the SDK import)"
)
requires_openai = pytest.mark.skipif(
    not _HAS_OPENAI, reason="openai SDK not installed (needed for the AzureOpenAI import)"
)


# ---- fakes that duck-type the SDK result shapes -------------------------------------------

class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Tag:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence


class _BBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.width = w
        self.height = h


class _Object:
    def __init__(self, bb, tags):
        self.bounding_box = bb
        self.tags = tags


class _Word:
    def __init__(self, text, confidence):
        self.text = text
        self.confidence = confidence


class _Line:
    def __init__(self, text, polygon, words):
        self.text = text
        self.bounding_polygon = polygon
        self.words = words


class _Block:
    def __init__(self, lines):
        self.lines = lines


class _Objects:
    def __init__(self, lst):
        self.list = lst


class _Read:
    def __init__(self, blocks):
        self.blocks = blocks


class _Metadata:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class FakeImageAnalysisClient:
    """Returns a duck-typed result mirroring the SDK shape analyze() reads."""
    def __init__(self, width=200, height=100, objects=None, read=None):
        self._result = type("R", (), {})()
        self._result.metadata = _Metadata(width, height)
        self._result.objects = _Objects(objects) if objects is not None else None
        self._result.read = _Read(read) if read is not None else None
        self.calls = []

    def analyze(self, image_data=None, visual_features=None):
        self.calls.append({"image_data": image_data, "visual_features": visual_features})
        return self._result


def _vision_with(client):
    obj = AzureVision.__new__(AzureVision)
    obj._client = client
    return obj


# ---- azure_vision object parsing ----------------------------------------------------------

@requires_azure_vision
def test_azure_vision_object_parsing():
    # bounding box in PIXELS; W=200,H=100 -> normalized box
    client = FakeImageAnalysisClient(
        width=200, height=100,
        objects=[_Object(_BBox(20, 10, 100, 50), [_Tag("person", 0.987654)])],
    )
    av = _vision_with(client)
    res = av.analyze(b"img", detect=True, read=False)
    assert isinstance(res, AnalysisResult)
    assert res.width == 200 and res.height == 100
    assert len(res.objects) == 1
    d = res.objects[0]
    assert d.cls == "person"
    assert d.confidence == round(0.987654, 4)  # rounded to 4
    # _norm_xywh: x/W, y/H, w/W, h/H
    assert d.box.x == pytest.approx(20 / 200)
    assert d.box.y == pytest.approx(10 / 100)
    assert d.box.w == pytest.approx(100 / 200)
    assert d.box.h == pytest.approx(50 / 100)


@requires_azure_vision
def test_azure_vision_object_without_tags_defaults():
    client = FakeImageAnalysisClient(objects=[_Object(_BBox(0, 0, 10, 10), [])])
    av = _vision_with(client)
    res = av.analyze(b"img", detect=True, read=False)
    d = res.objects[0]
    assert d.cls == "object"
    assert d.confidence is None


@requires_azure_vision
def test_azure_vision_read_ocr_parsing():
    poly = [_Point(10, 20), _Point(110, 20), _Point(110, 70), _Point(10, 70)]
    line = _Line("10", poly, [_Word("1", 0.9), _Word("0", 0.7)])
    client = FakeImageAnalysisClient(width=200, height=100, read=[_Block([line])])
    av = _vision_with(client)
    res = av.analyze(b"img", detect=False, read=True)
    assert len(res.lines) == 1
    t = res.lines[0]
    assert t.text == "10"
    # mean word confidence (0.9 + 0.7) / 2 = 0.8, rounded to 4
    assert t.confidence == 0.8
    # polygon -> box via min/max then normalized
    expected = _box_from_polygon(poly, 200, 100)
    assert t.box.x == pytest.approx(expected.x)
    assert t.box.w == pytest.approx(expected.w)
    assert t.words == [{"text": "1", "confidence": 0.9}, {"text": "0", "confidence": 0.7}]


@requires_azure_vision
def test_azure_vision_read_line_without_words_has_none_confidence():
    poly = [_Point(0, 0), _Point(10, 0), _Point(10, 10), _Point(0, 10)]
    line = _Line("--", poly, [])
    client = FakeImageAnalysisClient(read=[_Block([line])])
    av = _vision_with(client)
    res = av.analyze(b"img", detect=False, read=True)
    assert res.lines[0].confidence is None


@requires_azure_vision
def test_azure_vision_feature_flags_passed_to_client():
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    client = FakeImageAnalysisClient(objects=[], read=[])
    av = _vision_with(client)
    av.analyze(b"img", detect=True, read=True)
    feats = client.calls[0]["visual_features"]
    assert VisualFeatures.OBJECTS in feats
    assert VisualFeatures.READ in feats


@requires_azure_vision
def test_azure_vision_no_features_raises():
    # analyze() imports VisualFeatures BEFORE this raise (so SDK must be importable).
    av = _vision_with(FakeImageAnalysisClient())
    with pytest.raises(ValueError, match="at least one"):
        av.analyze(b"img", detect=False, read=False)


# ---- to_dict shape (pure, no client) ------------------------------------------------------

def test_analysis_result_to_dict_shape_and_rounding():
    res = AnalysisResult(
        width=200, height=100,
        objects=[Detection(cls="person", confidence=0.5, box=Box(0.123456, 0.2, 0.3, 0.4))],
        lines=[TextLine(text="10", confidence=0.8, box=Box(0.111111, 0.2, 0.3, 0.4), words=[])],
    )
    d = res.to_dict()
    assert d["width"] == 200 and d["height"] == 100
    o = d["objects"][0]
    assert o["class"] == "person"  # `cls` -> `class`
    assert o["confidence"] == 0.5
    # box rounded to 4
    assert o["box"]["x"] == round(0.123456, 4)
    assert set(o["box"]) == {"x", "y", "w", "h"}
    line = d["lines"][0]
    assert line["text"] == "10"
    assert line["box"]["x"] == round(0.111111, 4)
    assert "words" in line


# ---- vlm_read digit extraction ------------------------------------------------------------

class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeOpenAIClient:
    """Duck-types client.chat.completions.create(...) -> a completion with one choice."""
    def __init__(self, content):
        self._content = content
        self.create_calls = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.create_calls.append(kwargs)
                return _Completion(outer._content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _vlm_with(client, deployment="my-gpt4o"):
    obj = AzureVLM.__new__(AzureVLM)
    obj._client = client
    obj._deployment = deployment  # read_jersey_number reads self._deployment
    return obj


def test_vlm_read_extracts_digits_from_hash_number():
    client = FakeOpenAIClient("#10")
    vlm = _vlm_with(client)
    out = vlm.read_jersey_number(b"\x89PNG-img")
    assert isinstance(out, VlmRead)
    assert out.raw == "#10"
    assert out.digits == "10"
    # request shape: deterministic, capped, base64 PNG data URL
    kw = client.create_calls[0]
    assert kw["model"] == "my-gpt4o"
    assert kw["temperature"] == 0
    assert kw["max_tokens"] == 16
    content = kw["messages"][0]["content"]
    img = next(c for c in content if c["type"] == "image_url")
    assert img["image_url"]["url"].startswith("data:image/png;base64,")


def test_vlm_read_none_yields_empty_digits():
    vlm = _vlm_with(FakeOpenAIClient("NONE"))
    out = vlm.read_jersey_number(b"img")
    assert out.raw == "NONE"
    assert out.digits == ""


def test_vlm_read_strips_and_handles_empty_content():
    vlm = _vlm_with(FakeOpenAIClient("  22  "))
    out = vlm.read_jersey_number(b"img")
    assert out.raw == "22"
    assert out.digits == "22"


def test_vlm_read_custom_prompt_is_passed():
    client = FakeOpenAIClient("7")
    vlm = _vlm_with(client)
    vlm.read_jersey_number(b"img", prompt="read it")
    content = client.create_calls[0]["messages"][0]["content"]
    text = next(c for c in content if c["type"] == "text")
    assert text["text"] == "read it"


# ---- cred guards (SDK imported BEFORE cred validation in both constructors) ---------------

@requires_azure_vision
def test_azure_vision_blank_creds_raise():
    with pytest.raises(ValueError):
        AzureVision("", "")


@requires_openai
def test_azure_vlm_blank_creds_raise():
    with pytest.raises(ValueError):
        AzureVLM("", "", "")
