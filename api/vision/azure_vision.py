"""
Azure AI Vision Image Analysis 4.0 adapter — the two real per-frame primitives:
`detect` (OBJECTS) and `read_text` (READ / OCR). Stateless: one image in, structured
results out. Boxes are returned in NORMALIZED 0-1 coordinates (relative to the analyzed
frame), which is exactly the overlay contract the UI consumes (overlay-normalized-stills
decision) — so the same payload feeds the interpreter, the run-doc cache, and the canvas.

Why this service and not Video Indexer: keeping detection + OCR as stateless image
primitives preserves the glass-box thesis (the generated program does the orchestration,
not a managed black box). See the design doc, Premise 2.

SDK: azure-ai-vision-imageanalysis (stable 1.0.x).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class Box:
    """Axis-aligned box in normalized 0-1 coords (x,y = top-left; w,h = size)."""
    x: float
    y: float
    w: float
    h: float

    def rounded(self, p: int = 4) -> "Box":
        return Box(round(self.x, p), round(self.y, p), round(self.w, p), round(self.h, p))


@dataclass
class Detection:
    cls: str
    confidence: float | None
    box: Box


@dataclass
class TextLine:
    text: str
    confidence: float | None  # mean of word confidences
    box: Box
    words: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    width: int
    height: int
    objects: list[Detection]
    lines: list[TextLine]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "objects": [
                {"class": d.cls, "confidence": d.confidence, "box": asdict(d.box.rounded())}
                for d in self.objects
            ],
            "lines": [
                {"text": l.text, "confidence": l.confidence, "box": asdict(l.box.rounded()), "words": l.words}
                for l in self.lines
            ],
        }


def _norm_xywh(x: float, y: float, w: float, h: float, W: int, H: int) -> Box:
    return Box(x / W, y / H, w / W, h / H)


def _box_from_polygon(points, W: int, H: int) -> Box:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return _norm_xywh(x0, y0, x1 - x0, y1 - y0, W, H)


class AzureVision:
    """Thin wrapper over ImageAnalysisClient. Construct with explicit creds or from_env()."""

    def __init__(self, endpoint: str, key: str):
        # Imported lazily so `validate_program.py` and `--help` work without the SDK installed.
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.core.credentials import AzureKeyCredential

        if not endpoint or not key:
            raise ValueError("Azure Vision endpoint and key are required.")
        self._client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    @classmethod
    def from_env(cls) -> "AzureVision":
        return cls(
            endpoint=os.environ.get("AZURE_VISION_ENDPOINT", ""),
            key=os.environ.get("AZURE_VISION_KEY", ""),
        )

    def analyze(self, image_bytes: bytes, detect: bool = True, read: bool = True) -> AnalysisResult:
        from azure.ai.vision.imageanalysis.models import VisualFeatures

        features = []
        if detect:
            features.append(VisualFeatures.OBJECTS)
        if read:
            features.append(VisualFeatures.READ)
        if not features:
            raise ValueError("Request at least one of detect/read.")

        result = self._client.analyze(image_data=image_bytes, visual_features=features)
        W = result.metadata.width
        H = result.metadata.height

        objects: list[Detection] = []
        if getattr(result, "objects", None) is not None:
            for o in result.objects.list:
                bb = o.bounding_box
                tag = o.tags[0] if o.tags else None
                objects.append(
                    Detection(
                        cls=tag.name if tag else "object",
                        confidence=round(tag.confidence, 4) if tag else None,
                        box=_norm_xywh(bb.x, bb.y, bb.width, bb.height, W, H),
                    )
                )

        lines: list[TextLine] = []
        if getattr(result, "read", None) is not None:
            for block in result.read.blocks:
                for line in block.lines:
                    words = [
                        {"text": w.text, "confidence": round(w.confidence, 4)}
                        for w in (line.words or [])
                    ]
                    conf = round(sum(w["confidence"] for w in words) / len(words), 4) if words else None
                    lines.append(
                        TextLine(
                            text=line.text,
                            confidence=conf,
                            box=_box_from_polygon(line.bounding_polygon, W, H),
                            words=words,
                        )
                    )

        return AnalysisResult(width=W, height=H, objects=objects, lines=lines)
