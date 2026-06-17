"""
Per-clip replay cache. In replay mode (design D1) the interpreter reads detect/read_text
outputs from here instead of calling Azure, so the hero run is deterministic and free.
Same shape a real precompute pass would populate (real Azure outputs + hand-verified pins).
"""
from __future__ import annotations

import json
from pathlib import Path


class Cache:
    def __init__(self, data: dict):
        self.clip: dict = data["clip"]
        self._detect: dict = data.get("detect", {})      # "ts_ms" -> [detection,...]
        self._read: dict = data.get("read_text", {})      # det_id -> {text,confidence,source,note}
        self._events: list = data.get("events", [])       # [{ts_ms,type,scorer_det,box}]
        self._captions: dict = data.get("captions", {})   # "ts_ms" -> [{text,confidence,box},...]

    @classmethod
    def load(cls, path: str | Path) -> "Cache":
        return cls(json.loads(Path(path).read_text()))

    def analyzed_frames(self) -> list[int]:
        """Sampled-frame timestamps (ms) that have cached detections."""
        return sorted(int(k) for k in self._detect)

    def detections_at(self, ts_ms: int) -> list[dict]:
        return self._detect.get(str(ts_ms), [])

    def captions_at(self, ts_ms: int) -> list[dict]:
        """Scene captions cached at a sampled-frame ts (ms). Mirrors detections_at: the
        describe_scene op reads from here in replay mode instead of calling Azure CAPTION."""
        return self._captions.get(str(ts_ms), [])

    def read_text_for(self, det_id: str) -> dict | None:
        return self._read.get(det_id)

    def goal_events(self) -> list[dict]:
        return sorted((e for e in self._events if e.get("type") == "goal"), key=lambda e: e["ts_ms"])
