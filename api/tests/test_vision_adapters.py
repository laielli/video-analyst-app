"""
Vision-adapter tests for vision/azure_vision.py (AzureVision) and vision/vlm_read.py (AzureVLM).

Both adapters lazily import their SDK inside __init__ and store self._client. We bypass __init__
via __new__ and inject a fake duck-typed client, then call analyze()/read_jersey_number()
directly. The fakes mirror the SDK result shapes the adapters read.

CAVEAT: analyze() still imports `VisualFeatures` at call time, and both constructors import their
SDK BEFORE cred validation, so these tests require the SDKs to be importable (they're in
requirements.txt and CI installs them). We do NOT refactor production code to avoid that.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision.azure_vision import AzureVision, AnalysisResult, Detection, TextLine, Box, _norm_xywh, _box_from_polygon
from vision.vlm_read import AzureVLM, VlmRead


# ----------------------------------------------------------------------------------------
# Fake Azure Image Analysis client
# ----------------------------------------------------------------------------------------

def _pt(x, y):
    return SimpleNamespace(x=x, y=y)


class FakeImageAnalysisClient:
    """Duck-types ImageAnalysisClient.analyze(...) -> result with .metadata/.objects/.read."""

    def __init__(self, width=200, height=100, objects=None, read_blocks=None):
        self._width = width
        self._height = height
        self._objects = objects
        self._read_blocks = read_blocks

    def analyze(self, image_data=None, visual_features=None):
        metadata = SimpleNamespace(width=self._width, height=self._height)
        objects = None
        if self._objects is not None:
            objects = SimpleNamespace(list=self._objects)
        read = None
        if self._read_blocks is not None:
            read = SimpleNamespace(blocks=self._read_blocks)
        return SimpleNamespace(metadata=metadata, objects=objects, read=read)


def _fake_object(name, conf, x, y, w, h):
    tag = SimpleNamespace(name=name, confidence=conf)
    bb = SimpleNamespace(x=x, y=y, width=w, height=h)
    return SimpleNamespace(tags=[tag], bounding_box=bb)


def _fake_line(text, words, polygon_pts):
    word_objs = [SimpleNamespace(text=t, confidence=c) for t, c in words]
    poly = [_pt(px, py) for px, py in polygon_pts]
    return SimpleNamespace(text=text, words=word_objs, bounding_polygon=poly)


def make_vision(client):
    obj = AzureVision.__new__(AzureVision)
    obj._client = client
    return obj


# ----------------------------------------------------------------------------------------
# azure_vision object parsing
# ----------------------------------------------------------------------------------------

def test_azure_vision_object_parsing():
    # W=200, H=100. One object at pixel (20,10,40,30) -> normalized (0.1, 0.1, 0.2, 0.3).
    client = FakeImageAnalysisClient(
        width=200, height=100,
        objects=[_fake_object("person", 0.987654, 20, 10, 40, 30)],
    )
    res = make_vision(client).analyze(b"img", detect=True, read=False)
    assert isinstance(res, AnalysisResult)
    assert res.width == 200 and res.height == 100
    d = res.objects[0]
    assert d.cls == "person"
    assert d.confidence == 0.9877  # rounded to 4
    assert d.box.x == pytest.approx(0.1)
    assert d.box.y == pytest.approx(0.1)
    assert d.box.w == pytest.approx(0.2)
    assert d.box.h == pytest.approx(0.3)
    assert res.lines == []


def test_azure_vision_object_without_tags_defaults_to_object():
    obj = SimpleNamespace(tags=[], bounding_box=SimpleNamespace(x=0, y=0, width=10, height=10))
    client = FakeImageAnalysisClient(width=100, height=100, objects=[obj])
    res = make_vision(client).analyze(b"img", detect=True, read=False)
    assert res.objects[0].cls == "object"
    assert res.objects[0].confidence is None


# ----------------------------------------------------------------------------------------
# azure_vision read / OCR parsing
# ----------------------------------------------------------------------------------------

def test_azure_vision_read_parsing_mean_conf_and_polygon_box():
    # Polygon spans pixels x:[10,50], y:[20,40] on a 100x100 frame -> box (0.1,0.2,0.4,0.2).
    line = _fake_line(
        "10",
        words=[("1", 0.9), ("0", 0.7)],  # mean = 0.8
        polygon_pts=[(10, 20), (50, 20), (50, 40), (10, 40)],
    )
    block = SimpleNamespace(lines=[line])
    client = FakeImageAnalysisClient(width=100, height=100, read_blocks=[block])
    res = make_vision(client).analyze(b"img", detect=False, read=True)
    assert res.objects == []
    tl = res.lines[0]
    assert isinstance(tl, TextLine)
    assert tl.text == "10"
    assert tl.confidence == 0.8  # mean of word confidences
    assert tl.box.x == pytest.approx(0.1)
    assert tl.box.y == pytest.approx(0.2)
    assert tl.box.w == pytest.approx(0.4)
    assert tl.box.h == pytest.approx(0.2)
    assert tl.words == [{"text": "1", "confidence": 0.9}, {"text": "0", "confidence": 0.7}]


def test_azure_vision_read_line_without_words_has_none_confidence():
    line = SimpleNamespace(text="x", words=None,
                           bounding_polygon=[_pt(0, 0), _pt(10, 0), _pt(10, 10), _pt(0, 10)])
    block = SimpleNamespace(lines=[line])
    client = FakeImageAnalysisClient(width=100, height=100, read_blocks=[block])
    res = make_vision(client).analyze(b"img", detect=False, read=True)
    assert res.lines[0].confidence is None


# ----------------------------------------------------------------------------------------
# feature flags
# ----------------------------------------------------------------------------------------

def test_azure_vision_requires_at_least_one_feature():
    v = make_vision(FakeImageAnalysisClient())
    with pytest.raises(ValueError, match="at least one"):
        v.analyze(b"img", detect=False, read=False)


# ----------------------------------------------------------------------------------------
# to_dict shape (pure, no client)
# ----------------------------------------------------------------------------------------

def test_analysis_result_to_dict_shape_and_rounding():
    res = AnalysisResult(
        width=10, height=20,
        objects=[Detection(cls="person", confidence=0.5, box=Box(0.123456, 0.2, 0.3, 0.4))],
        lines=[TextLine(text="10", confidence=0.8, box=Box(0.111111, 0.2, 0.3, 0.4),
                        words=[{"text": "10", "confidence": 0.8}])],
    )
    d = res.to_dict()
    assert d["width"] == 10 and d["height"] == 20
    o = d["objects"][0]
    assert o["class"] == "person"
    assert o["confidence"] == 0.5
    assert o["box"]["x"] == 0.1235  # rounded to 4
    line = d["lines"][0]
    assert line["text"] == "10"
    assert line["box"]["x"] == 0.1111
    assert line["words"] == [{"text": "10", "confidence": 0.8}]


# ----------------------------------------------------------------------------------------
# pure box helpers
# ----------------------------------------------------------------------------------------

def test_norm_xywh_divides_by_dimensions():
    b = _norm_xywh(20, 10, 40, 30, 200, 100)
    assert (b.x, b.y, b.w, b.h) == (0.1, 0.1, 0.2, 0.3)


def test_box_from_polygon_min_max_extent():
    pts = [_pt(10, 20), _pt(50, 25), _pt(40, 40), _pt(15, 38)]
    b = _box_from_polygon(pts, 100, 100)
    assert b.x == pytest.approx(0.1)   # min x = 10
    assert b.y == pytest.approx(0.2)   # min y = 20
    assert b.w == pytest.approx(0.4)   # 50 - 10
    assert b.h == pytest.approx(0.2)   # 40 - 20


# ----------------------------------------------------------------------------------------
# from_env cred guards (SDK import happens BEFORE cred validation)
# ----------------------------------------------------------------------------------------

def test_azure_vision_constructor_requires_creds():
    with pytest.raises(ValueError):
        AzureVision("", "")


# ----------------------------------------------------------------------------------------
# AzureVLM digit extraction
# ----------------------------------------------------------------------------------------

class FakeOpenAIClient:
    """Duck-types AzureOpenAI: .chat.completions.create(...) -> choices[0].message.content."""

    def __init__(self, content):
        self._content = content
        self.last_call = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_call = kwargs
        msg = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_vlm(client, deployment="gpt-4o"):
    obj = AzureVLM.__new__(AzureVLM)
    obj._client = client
    obj._deployment = deployment
    return obj


def test_vlm_extracts_digits_from_hash_number():
    client = FakeOpenAIClient("#10")
    vlm = make_vlm(client, deployment="my-deploy")
    res = vlm.read_jersey_number(b"\x89PNG")
    assert isinstance(res, VlmRead)
    assert res.raw == "#10"
    assert res.digits == "10"
    # request shape: temperature 0, max_tokens 16, model = deployment, base64 data URL.
    call = client.last_call
    assert call["temperature"] == 0
    assert call["max_tokens"] == 16
    assert call["model"] == "my-deploy"
    image = call["messages"][0]["content"][1]["image_url"]["url"]
    assert image.startswith("data:image/png;base64,")


def test_vlm_none_response_yields_empty_digits():
    res = make_vlm(FakeOpenAIClient("NONE")).read_jersey_number(b"img")
    assert res.raw == "NONE"
    assert res.digits == ""


def test_vlm_strips_and_handles_empty_content():
    res = make_vlm(FakeOpenAIClient("  22  ")).read_jersey_number(b"img")
    assert res.raw == "22"
    assert res.digits == "22"
    # None content coerces to empty.
    res2 = make_vlm(FakeOpenAIClient(None)).read_jersey_number(b"img")
    assert res2.raw == ""
    assert res2.digits == ""


def test_vlm_constructor_requires_creds():
    with pytest.raises(ValueError):
        AzureVLM("", "", "")
