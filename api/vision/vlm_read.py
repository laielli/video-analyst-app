"""
gpt-4o vision reader (Azure OpenAI) — the accurate `read_text` backend for stylized
jersey numbers, where Azure AI Vision Read fires but misreads the WC font (e.g. 22 -> 77).
A VLM reads the digits in context and is robust to the stylized glyphs and mild warping.

Still a discrete, single-purpose primitive call (one crop in, one number out), so the
glass-box thesis holds: the generated program orchestrates; this just backs one step.

SDK: openai>=1.40 (AzureOpenAI client).
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass

DEFAULT_PROMPT = (
    "You are reading the squad number printed on a football (soccer) player's shirt. "
    "Reply with ONLY the digits of the number, no words, no punctuation (e.g. 10). "
    "If a name is also visible you may ignore it. If no number is legible, reply NONE."
)


@dataclass
class VlmRead:
    raw: str           # exactly what the model returned
    digits: str        # digits extracted from raw, or "" if none


class AzureVLM:
    """Thin wrapper over Azure OpenAI chat-completions vision. from_env() or explicit creds."""

    def __init__(self, endpoint: str, key: str, deployment: str, api_version: str = "2024-10-21"):
        from openai import AzureOpenAI  # lazy: keep import optional until actually used

        if not (endpoint and key and deployment):
            raise ValueError("Azure OpenAI endpoint, key, and deployment are required.")
        self._client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)
        self._deployment = deployment

    @classmethod
    def from_env(cls) -> "AzureVLM":
        return cls(
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            key=os.environ.get("AZURE_OPENAI_KEY", ""),
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    def read_jersey_number(self, image_bytes: bytes, prompt: str | None = None) -> VlmRead:
        b64 = base64.b64encode(image_bytes).decode()
        resp = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        return VlmRead(raw=raw, digits=digits)
