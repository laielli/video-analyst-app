"""
Vision-adapter tests for vision/azure_vision.py (AzureVision) and vision/vlm_read.py (AzureVLM),
exercised via a FAKE SDK client injected into the adapter (no real Azure calls).

Both adapters lazily import their SDK inside __init__ and store self._client. We bypass __init__
with __new__ and assign self._client (and, for the VLM, self._deployment) directly, then call the
analyze()/read_jersey_number() methods. The fakes are duck-typed to the SDK result shape the
adapter reads. CAVEAT: analyze() still imports VisualFeatures at call time and the constructors
import their SDK before cred validation — the SDKs are in requirements.txt so they import in CI.
"""
from __future__ import annotations

import base64

import pytest

from vision.azure_vision import AzureVision, AnalysisResult, Detection, Box
from vision.vlm_read import AzureVLM, VlmRead


# ---------------------------------------------------------------------------------------
# fake Azure AI Vision SDK result shapes
# ---------------------------------------------------------------------------------------

class _Tag:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence


class _BBox:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = x, y, w, h


class _Obj:
    def __init__(self, bbox, tags):
        self.bounding_box = bbox
        self.tags = tags


class _Objects:
    def __init__(self, objs):
        self.list = objs


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Word:
    def __init__(self, text, confidence):
        self.text, self.confidence = text, confidence


class _Line:
    def __init__(self, text, polygon, words):
        self.text = text
        self.bounding_polygon = polygon
        self.words = words


class _Block:
    def __init__(self, lines):
        self.lines = lines


class _Read:
    def __init__(self, blocks):
        self.blocks = blocks


class _Metadata:
    def __init__(self, w, h):
        self.width, self.height = w, h


class _Result:
    def __init__(self, metadata, objects=None, read=None):
        self.metadata = metadata
        self.objects = objects
        self.read = read


class FakeImageAnalysisClient:
    """Records the last analyze() kwargs and returns a configured _Result."""
    def __init__(self, result):
        self._result = result
        self.last_kwargs = None

    def analyze(self, **kwargs):
        self.last_kwargs = kwargs
        return self._result


def _make_vision(result):
    obj = AzureVision.__new__(AzureVision)
    obj._client = FakeImageAnalysisClient(result)
    return obj


# ---------------------------------------------------------------------------------------
# azure_vision object parsing
# ---------------------------------------------------------------------------------------

def test_azure_vision_object_parsing_normalizes_box_and_rounds_confidence():
    W, H = 200, 100
    result = _Result(
        _Metadata(W, H),
        objects=_Objects([_Obj(_BBox(40, 20, 100, 50), [_Tag("person", 0.123456)])]),
    )
    res = _make_vision(result).analyze(b"img", detect=True, read=False)
    assert res.width == W and res.height == H
    d = res.objects[0]
    assert d.cls == "person"
    # confidence rounded to 4 places.
    assert d.confidence == 0.1235
    # box normalized by W/H.
    assert d.box.x == 40 / W
    assert d.box.y == 20 / H
    assert d.box.w == 100 / W
    assert d.box.h == 50 / H


def test_azure_vision_object_without_tags_defaults_class():
    result = _Result(_Metadata(100, 100), objects=_Objects([_Obj(_BBox(0, 0, 10, 10), [])]))
    res = _make_vision(result).analyze(b"img", detect=True, read=False)
    assert res.objects[0].cls == "object"
    assert res.objects[0].confidence is None


# ---------------------------------------------------------------------------------------
# azure_vision read / OCR parsing
# ---------------------------------------------------------------------------------------

def test_azure_vision_read_parsing_mean_word_conf_and_polygon_box():
    W, H = 100, 100
    poly = [_Pt(10, 20), _Pt(30, 20), _Pt(30, 40), _Pt(10, 40)]
    line = _Line("10", poly, [_Word("1", 0.8), _Word("0", 0.6)])
    result = _Result(_Metadata(W, H), read=_Read([_Block([line])]))
    res = _make_vision(result).analyze(b"img", detect=False, read=True)
    tl = res.lines[0]
    assert tl.text == "10"
    # mean of word confidences (0.8 + 0.6) / 2 = 0.7.
    assert tl.confidence == 0.7
    # polygon -> bounding box normalized: x0=10,y0=20, w=20, h=20.
    assert tl.box.x == 10 / W
    assert tl.box.y == 20 / H
    assert tl.box.w == 20 / W
    assert tl.box.h == 20 / H
    assert tl.words == [{"text": "1", "confidence": 0.8}, {"text": "0", "confidence": 0.6}]


def test_azure_vision_read_line_without_words_has_none_confidence():
    poly = [_Pt(0, 0), _Pt(5, 0), _Pt(5, 5), _Pt(0, 5)]
    line = _Line("x", poly, [])
    result = _Result(_Metadata(10, 10), read=_Read([_Block([line])]))
    res = _make_vision(result).analyze(b"img", detect=False, read=True)
    assert res.lines[0].confidence is None


# ---------------------------------------------------------------------------------------
# feature flags
# ---------------------------------------------------------------------------------------

def test_azure_vision_analyze_requires_at_least_one_feature():
    # analyze(detect=False, read=False) raises ValueError (imports VisualFeatures first).
    vis = _make_vision(_Result(_Metadata(10, 10)))
    with pytest.raises(ValueError, match="at least one"):
        vis.analyze(b"img", detect=False, read=False)


def test_azure_vision_passes_visual_features_to_client():
    from azure.ai.vision.imageanalysis.models import VisualFeatures

    # both flags -> exactly [OBJECTS, READ] (analyze() appends detect-then-read).
    vis = _make_vision(_Result(_Metadata(10, 10), objects=_Objects([]), read=_Read([])))
    vis.analyze(b"img", detect=True, read=True)
    assert vis._client.last_kwargs["visual_features"] == [VisualFeatures.OBJECTS, VisualFeatures.READ]
    # detect-only -> only OBJECTS; read-only -> only READ. Pins WHICH feature each flag maps to,
    # so swapping the read branch for a second OBJECTS (or dropping a branch) would fail here.
    vis_d = _make_vision(_Result(_Metadata(10, 10), objects=_Objects([])))
    vis_d.analyze(b"img", detect=True, read=False)
    assert vis_d._client.last_kwargs["visual_features"] == [VisualFeatures.OBJECTS]
    vis_r = _make_vision(_Result(_Metadata(10, 10), read=_Read([])))
    vis_r.analyze(b"img", detect=False, read=True)
    assert vis_r._client.last_kwargs["visual_features"] == [VisualFeatures.READ]


# ---------------------------------------------------------------------------------------
# AnalysisResult.to_dict shape (pure, no client)
# ---------------------------------------------------------------------------------------

def test_analysis_result_to_dict_shape_and_rounding():
    res = AnalysisResult(
        width=100, height=50,
        objects=[Detection(cls="person", confidence=0.5, box=Box(0.123456, 0.2, 0.3, 0.4))],
        lines=[],
    )
    d = res.to_dict()
    assert d["width"] == 100 and d["height"] == 50
    o = d["objects"][0]
    assert o["class"] == "person"
    assert o["confidence"] == 0.5
    # box rounded to 4 places.
    assert o["box"] == {"x": 0.1235, "y": 0.2, "w": 0.3, "h": 0.4}
    assert d["lines"] == []


# ---------------------------------------------------------------------------------------
# vlm_read digit extraction
# ---------------------------------------------------------------------------------------

class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _ChatResp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _ChatResp(self._content)


class _Chat:
    def __init__(self, content):
        self.completions = _Completions(content)


class FakeOpenAIClient:
    def __init__(self, content):
        self.chat = _Chat(content)


def _make_vlm(content):
    obj = AzureVLM.__new__(AzureVLM)
    obj._client = FakeOpenAIClient(content)
    obj._deployment = "x"  # read_jersey_number reads self._deployment
    return obj


def test_vlm_extracts_digits_and_sends_expected_request():
    vlm = _make_vlm("#10")
    out = vlm.read_jersey_number(b"\x89PNGfake")
    assert isinstance(out, VlmRead)
    assert out.raw == "#10"
    assert out.digits == "10"
    kwargs = vlm._client.chat.completions.last_kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 16
    assert kwargs["model"] == "x"
    # the image is sent as a base64 data URL carrying the ACTUAL input bytes,
    # not just a constant prefix.
    content = kwargs["messages"][0]["content"]
    img = next(c for c in content if c["type"] == "image_url")
    url = img["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNGfake"


def test_vlm_none_response_yields_empty_digits():
    vlm = _make_vlm("NONE")
    out = vlm.read_jersey_number(b"img")
    assert out.raw == "NONE"
    assert out.digits == ""


def test_vlm_handles_null_content():
    vlm = _make_vlm(None)
    out = vlm.read_jersey_number(b"img")
    assert out.raw == ""
    assert out.digits == ""


# ---------------------------------------------------------------------------------------
# from_env / constructor cred guards (SDK imports before cred validation -> needs SDK importable)
# ---------------------------------------------------------------------------------------

def test_azure_vision_constructor_requires_creds():
    with pytest.raises(ValueError):
        AzureVision("", "")


def test_azure_vlm_constructor_requires_creds():
    with pytest.raises(ValueError):
        AzureVLM("", "", "")
