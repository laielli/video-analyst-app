"""
Vision-adapter tests for vision/azure_vision.py (AzureVision) and vision/vlm_read.py
(AzureVLM), exercised via a FAKE SDK client injected into the adapter.

Both adapters lazily import their SDK inside __init__ and store self._client. We bypass the
constructor (which calls the real SDK) via __new__ and set _client (and, for the VLM,
_deployment) directly, then call analyze()/read_jersey_number() against a duck-typed fake
result whose attributes mirror the SDK shape the adapter reads. analyze() still imports
VisualFeatures at call time, so azure-ai-vision-imageanalysis must be importable (it's in
requirements.txt). No production code is refactored.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vision.azure_vision import AzureVision, AnalysisResult, Detection, Box, TextLine
from vision.vlm_read import AzureVLM, VlmRead


# ---------------------------------------------------------------- fakes mirroring the SDK shape

def _pt(x, y):
    return SimpleNamespace(x=x, y=y)


def make_fake_result(*, width=1000, height=500, objects=None, lines=None):
    """Duck-type the ImageAnalysisClient.analyze() result the adapter reads:
    .metadata.width/height, .objects.list[*].bounding_box/.tags, .read.blocks[*].lines[*]."""
    obj_ns = None
    if objects is not None:
        obj_ns = SimpleNamespace(list=objects)
    read_ns = None
    if lines is not None:
        read_ns = SimpleNamespace(blocks=[SimpleNamespace(lines=lines)])
    return SimpleNamespace(
        metadata=SimpleNamespace(width=width, height=height),
        objects=obj_ns,
        read=read_ns,
    )


class FakeImageAnalysisClient:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def analyze(self, image_data, visual_features):
        self.calls.append({"image_data": image_data, "visual_features": visual_features})
        return self._result


def make_vision(result) -> AzureVision:
    obj = AzureVision.__new__(AzureVision)
    obj._client = FakeImageAnalysisClient(result)
    return obj


# ---------------------------------------------------------------- azure_vision: objects

def test_azure_vision_object_parsing_normalizes_and_rounds():
    bb = SimpleNamespace(x=100, y=50, width=200, height=100)
    tag = SimpleNamespace(name="person", confidence=0.598123)
    o = SimpleNamespace(bounding_box=bb, tags=[tag])
    result = make_fake_result(width=1000, height=500, objects=[o], lines=None)
    av = make_vision(result)

    out = av.analyze(b"img", detect=True, read=False)
    assert isinstance(out, AnalysisResult)
    assert out.width == 1000 and out.height == 500
    assert len(out.objects) == 1
    d = out.objects[0]
    assert d.cls == "person"
    assert d.confidence == round(0.598123, 4)  # rounded to 4
    # _norm_xywh divides by W/H: 100/1000, 50/500, 200/1000, 100/500.
    assert d.box.x == pytest.approx(0.1)
    assert d.box.y == pytest.approx(0.1)
    assert d.box.w == pytest.approx(0.2)
    assert d.box.h == pytest.approx(0.2)


def test_azure_vision_object_without_tags_defaults():
    bb = SimpleNamespace(x=0, y=0, width=10, height=10)
    o = SimpleNamespace(bounding_box=bb, tags=[])
    result = make_fake_result(objects=[o], lines=None)
    out = make_vision(result).analyze(b"img", detect=True, read=False)
    assert out.objects[0].cls == "object"
    assert out.objects[0].confidence is None


# ---------------------------------------------------------------- azure_vision: read / OCR

def test_azure_vision_read_parsing_mean_conf_and_polygon_box():
    words = [SimpleNamespace(text="10", confidence=0.8),
             SimpleNamespace(text="A", confidence=0.6)]
    poly = [_pt(100, 50), _pt(300, 50), _pt(300, 150), _pt(100, 150)]
    line = SimpleNamespace(text="10 A", words=words, bounding_polygon=poly)
    result = make_fake_result(width=1000, height=500, objects=None, lines=[line])
    out = make_vision(result).analyze(b"img", detect=False, read=True)

    assert len(out.lines) == 1
    tl = out.lines[0]
    assert isinstance(tl, TextLine)
    assert tl.text == "10 A"
    # mean word confidence = (0.8 + 0.6) / 2 = 0.7
    assert tl.confidence == pytest.approx(0.7)
    assert tl.words == [{"text": "10", "confidence": 0.8}, {"text": "A", "confidence": 0.6}]
    # polygon bbox: x0=100,y0=50,x1=300,y1=150 -> norm (0.1, 0.1, 0.2, 0.2)
    assert tl.box.x == pytest.approx(0.1)
    assert tl.box.y == pytest.approx(0.1)
    assert tl.box.w == pytest.approx(0.2)
    assert tl.box.h == pytest.approx(0.2)


def test_azure_vision_read_line_with_no_words_has_none_conf():
    poly = [_pt(0, 0), _pt(10, 0), _pt(10, 10), _pt(0, 10)]
    line = SimpleNamespace(text="", words=[], bounding_polygon=poly)
    result = make_fake_result(objects=None, lines=[line])
    out = make_vision(result).analyze(b"img", detect=False, read=True)
    assert out.lines[0].confidence is None
    assert out.lines[0].words == []


# ---------------------------------------------------------------- feature flags

def test_azure_vision_no_features_raises():
    result = make_fake_result(objects=[], lines=[])
    av = make_vision(result)
    # analyze() imports VisualFeatures BEFORE this raise -> SDK must be importable.
    with pytest.raises(ValueError, match="at least one of detect/read"):
        av.analyze(b"img", detect=False, read=False)


def test_azure_vision_passes_only_requested_features():
    bb = SimpleNamespace(x=0, y=0, width=10, height=10)
    o = SimpleNamespace(bounding_box=bb, tags=[SimpleNamespace(name="person", confidence=0.5)])
    result = make_fake_result(objects=[o], lines=[])
    av = make_vision(result)
    av.analyze(b"img", detect=True, read=False)
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    sent = av._client.calls[0]["visual_features"]
    assert VisualFeatures.OBJECTS in sent
    assert VisualFeatures.READ not in sent


# ---------------------------------------------------------------- to_dict shape (pure)

def test_to_dict_shape_and_rounding():
    ar = AnalysisResult(
        width=100, height=200,
        objects=[Detection(cls="person", confidence=0.5,
                            box=Box(0.123456, 0.2, 0.3, 0.4))],
        lines=[TextLine(text="10", confidence=0.9,
                        box=Box(0.111111, 0.2, 0.3, 0.4),
                        words=[{"text": "10", "confidence": 0.9}])],
    )
    d = ar.to_dict()
    assert d["width"] == 100 and d["height"] == 200
    obj = d["objects"][0]
    assert obj["class"] == "person"        # cls -> "class"
    assert obj["confidence"] == 0.5
    assert obj["box"]["x"] == round(0.123456, 4)  # box rounded to 4
    ln = d["lines"][0]
    assert ln["text"] == "10"
    assert ln["box"]["x"] == round(0.111111, 4)
    assert ln["words"] == [{"text": "10", "confidence": 0.9}]


# ---------------------------------------------------------------- vlm_read

class FakeChatCompletions:
    def __init__(self, content):
        self._content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        msg = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


class FakeOpenAIClient:
    def __init__(self, content):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(content))


def make_vlm(content) -> AzureVLM:
    obj = AzureVLM.__new__(AzureVLM)
    obj._client = FakeOpenAIClient(content)
    obj._deployment = "gpt-4o-test"
    return obj


def test_vlm_extracts_digits_and_request_shape():
    vlm = make_vlm("#10")
    out = vlm.read_jersey_number(b"\x89PNG-bytes")
    assert isinstance(out, VlmRead)
    assert out.raw == "#10"
    assert out.digits == "10"
    kwargs = vlm._client.chat.completions.last_kwargs
    assert kwargs["model"] == "gpt-4o-test"  # reads self._deployment
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 16
    # the image_url carries a base64 data URL.
    content = kwargs["messages"][0]["content"]
    img = next(c for c in content if c["type"] == "image_url")
    assert img["image_url"]["url"].startswith("data:image/png;base64,")


def test_vlm_none_means_no_digits():
    vlm = make_vlm("NONE")
    out = vlm.read_jersey_number(b"img")
    assert out.raw == "NONE"
    assert out.digits == ""


def test_vlm_strips_and_filters_non_digits():
    vlm = make_vlm("  number is 7!  ")
    out = vlm.read_jersey_number(b"img")
    assert out.raw == "number is 7!"  # stripped
    assert out.digits == "7"


def test_vlm_custom_prompt_used():
    vlm = make_vlm("10")
    vlm.read_jersey_number(b"img", prompt="read it")
    content = vlm._client.chat.completions.last_kwargs["messages"][0]["content"]
    text = next(c for c in content if c["type"] == "text")
    assert text["text"] == "read it"


# ---------------------------------------------------------------- cred guards (constructors)

def test_azure_vision_from_env_empty_creds_raise():
    # The SDK import happens BEFORE cred validation, so SDK must be importable; we assert ValueError.
    with pytest.raises(ValueError, match="endpoint and key are required"):
        AzureVision("", "")


def test_azure_vlm_empty_creds_raise():
    with pytest.raises(ValueError, match="endpoint, key, and deployment are required"):
        AzureVLM("", "", "")
