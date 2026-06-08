"""
Vision-adapter tests for vision/azure_vision.py (AzureVision) and vision/vlm_read.py (AzureVLM),
exercised via fake SDK clients injected into the adapter. Both adapters lazily import their SDK in
__init__ and store self._client; we bypass __init__ with __new__ and set _client (and _deployment
for the VLM) directly, then call the public methods. The fake clients return duck-typed results
mirroring the SDK shapes analyze()/chat.completions.create read. No real Azure call is made.

NOTE: analyze() still imports VisualFeatures from the SDK at call time, and both constructors
import their SDK before cred validation — so the suite needs azure-ai-vision-imageanalysis + openai
importable (both are in requirements.txt / installed in CI).
"""
from __future__ import annotations

import base64

import pytest

from vision.azure_vision import AzureVision, AnalysisResult, Detection, Box, _norm_xywh, _box_from_polygon
from vision.vlm_read import AzureVLM, VlmRead


# ---- duck-typed fake Azure AI Vision SDK result -----------------------------------------

class _FakeMeta:
    def __init__(self, w, h):
        self.width = w
        self.height = h


class _FakeTag:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence


class _FakeBB:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.width, self.height = x, y, w, h


class _FakeObject:
    def __init__(self, bb, tags):
        self.bounding_box = bb
        self.tags = tags


class _FakeObjects:
    def __init__(self, objs):
        self.list = objs


class _FakePoint:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeWord:
    def __init__(self, text, confidence):
        self.text = text
        self.confidence = confidence


class _FakeLine:
    def __init__(self, text, polygon, words):
        self.text = text
        self.bounding_polygon = polygon
        self.words = words


class _FakeBlock:
    def __init__(self, lines):
        self.lines = lines


class _FakeRead:
    def __init__(self, blocks):
        self.blocks = blocks


class _FakeAnalysisResult:
    def __init__(self, width, height, objects=None, read=None):
        self.metadata = _FakeMeta(width, height)
        self.objects = objects
        self.read = read


class FakeImageAnalysisClient:
    """Records the analyze() call and returns a configured fake result."""
    def __init__(self, result):
        self._result = result
        self.calls = []

    def analyze(self, image_data=None, visual_features=None):
        self.calls.append({"image_data": image_data, "visual_features": visual_features})
        return self._result


def _vision_with(result):
    obj = AzureVision.__new__(AzureVision)
    obj._client = FakeImageAnalysisClient(result)
    return obj


# ---- AzureVision: object parsing --------------------------------------------------------

def test_azure_vision_object_parsing_normalizes_and_rounds():
    W, H = 1000, 500
    res = _FakeAnalysisResult(W, H, objects=_FakeObjects([
        _FakeObject(_FakeBB(100, 50, 200, 100), [_FakeTag("person", 0.987654321)]),
    ]))
    out = _vision_with(res).analyze(b"img", detect=True, read=False)
    assert isinstance(out, AnalysisResult)
    assert out.width == W and out.height == H
    d = out.objects[0]
    assert d.cls == "person"
    assert d.confidence == round(0.987654321, 4)  # rounded to 4
    # _norm_xywh divides by W/H.
    assert d.box.x == 100 / W and d.box.y == 50 / H
    assert d.box.w == 200 / W and d.box.h == 100 / H


def test_azure_vision_object_defaults_when_no_tags():
    res = _FakeAnalysisResult(100, 100, objects=_FakeObjects([
        _FakeObject(_FakeBB(0, 0, 10, 10), []),  # no tags
    ]))
    out = _vision_with(res).analyze(b"img", detect=True, read=False)
    d = out.objects[0]
    assert d.cls == "object"  # fallback class
    assert d.confidence is None


# ---- AzureVision: read / OCR parsing ----------------------------------------------------

def test_azure_vision_read_parsing_mean_conf_and_polygon_box():
    W, H = 200, 100
    poly = [_FakePoint(20, 10), _FakePoint(60, 10), _FakePoint(60, 30), _FakePoint(20, 30)]
    line = _FakeLine("10", poly, [_FakeWord("1", 0.8), _FakeWord("0", 0.6)])
    res = _FakeAnalysisResult(W, H, read=_FakeRead([_FakeBlock([line])]))
    out = _vision_with(res).analyze(b"img", detect=False, read=True)
    tl = out.lines[0]
    assert tl.text == "10"
    # mean of word confidences, rounded to 4.
    assert tl.confidence == round((0.8 + 0.6) / 2, 4)
    # polygon -> box: min/max over xs/ys, normalized.
    assert tl.box.x == 20 / W and tl.box.y == 10 / H
    assert tl.box.w == (60 - 20) / W and tl.box.h == (30 - 10) / H
    assert tl.words == [{"text": "1", "confidence": 0.8}, {"text": "0", "confidence": 0.6}]


def test_azure_vision_read_line_with_no_words_has_none_conf():
    res = _FakeAnalysisResult(100, 100, read=_FakeRead([_FakeBlock([
        _FakeLine("x", [_FakePoint(0, 0), _FakePoint(10, 10)], []),
    ])]))
    out = _vision_with(res).analyze(b"img", detect=False, read=True)
    assert out.lines[0].confidence is None


def test_azure_vision_passes_correct_visual_features():
    res = _FakeAnalysisResult(100, 100, objects=_FakeObjects([]), read=_FakeRead([]))
    v = _vision_with(res)
    v.analyze(b"img", detect=True, read=True)
    feats = v._client.calls[0]["visual_features"]
    # one feature per enabled flag (OBJECTS + READ).
    assert len(feats) == 2


# ---- AzureVision: feature flags ---------------------------------------------------------

def test_azure_vision_requires_at_least_one_feature():
    res = _FakeAnalysisResult(100, 100)
    with pytest.raises(ValueError, match="at least one"):
        _vision_with(res).analyze(b"img", detect=False, read=False)


# ---- AzureVision: to_dict shape (pure) --------------------------------------------------

def test_analysis_result_to_dict_shape_and_rounding():
    ar = AnalysisResult(
        width=10, height=20,
        objects=[Detection(cls="person", confidence=0.5, box=Box(0.123456, 0.2, 0.3, 0.4))],
        lines=[],
    )
    d = ar.to_dict()
    assert d["width"] == 10 and d["height"] == 20
    o = d["objects"][0]
    assert o["class"] == "person"  # 'cls' -> 'class'
    assert o["confidence"] == 0.5
    # box rounded to 4 places.
    assert o["box"]["x"] == round(0.123456, 4)
    assert set(o["box"]) == {"x", "y", "w", "h"}


# ---- helper functions -------------------------------------------------------------------

def test_norm_xywh_and_box_from_polygon():
    b = _norm_xywh(50, 25, 100, 50, 200, 100)
    assert (b.x, b.y, b.w, b.h) == (0.25, 0.25, 0.5, 0.5)
    poly = [_FakePoint(10, 20), _FakePoint(40, 20), _FakePoint(40, 60), _FakePoint(10, 60)]
    bb = _box_from_polygon(poly, 100, 100)
    assert (bb.x, bb.y, bb.w, bb.h) == (0.1, 0.2, 0.3, 0.4)


# ---- AzureVision: cred guard ------------------------------------------------------------

def test_azure_vision_requires_credentials():
    with pytest.raises(ValueError, match="endpoint and key are required"):
        AzureVision("", "")


# ---- duck-typed fake Azure OpenAI (VLM) client ------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeChatResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeChatResp(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class FakeOpenAIClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def _vlm_with(content):
    obj = AzureVLM.__new__(AzureVLM)
    obj._client = FakeOpenAIClient(content)
    obj._deployment = "gpt-4o-test"
    return obj


# ---- AzureVLM: digit extraction ---------------------------------------------------------

def test_vlm_read_extracts_digits_from_hash_prefixed():
    vlm = _vlm_with("#10")
    out = vlm.read_jersey_number(b"\x89PNG")
    assert isinstance(out, VlmRead)
    assert out.raw == "#10"
    assert out.digits == "10"


def test_vlm_read_none_yields_empty_digits():
    vlm = _vlm_with("NONE")
    out = vlm.read_jersey_number(b"\x89PNG")
    assert out.raw == "NONE"
    assert out.digits == ""


def test_vlm_read_request_params_and_data_url():
    img = b"\x89PNG-bytes"
    vlm = _vlm_with("10")
    vlm.read_jersey_number(img)
    call = vlm._client.chat.completions.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 16
    assert call["model"] == "gpt-4o-test"  # reads self._deployment
    # the image is embedded as a base64 data URL.
    content = call["messages"][0]["content"]
    image_part = next(p for p in content if p["type"] == "image_url")
    expected_b64 = base64.b64encode(img).decode()
    assert image_part["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


# ---- AzureVLM: cred guard ---------------------------------------------------------------

def test_azure_vlm_requires_credentials():
    with pytest.raises(ValueError, match="endpoint, key, and deployment are required"):
        AzureVLM("", "", "")
