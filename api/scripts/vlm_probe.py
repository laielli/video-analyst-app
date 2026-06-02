#!/usr/bin/env python3
"""
gpt-4o vision jersey-read probe (Azure OpenAI) — the accurate live-read path when
Azure AI Vision Read misreads the stylized WC jersey font (it called Lozano's 22 a 77).

Same interface as ocr_probe.py:

    python vlm_probe.py --ts 4.6 --crop 230,250,300,240 --scale 3 --expect 10   # Messi
    python vlm_probe.py --ts 3.9 --crop 200,225,360,260 --scale 3 --expect 22   # Lozano
    python vlm_probe.py --image /tmp/pre22.png --expect 22

Exit 0 = read the expected number · 1 = read something else / nothing · 2 = setup problem.

Prereqs: an Azure OpenAI resource with a gpt-4o deployment. Put AZURE_OPENAI_* in api/.env
(see .env.example). Unlike OCR, NO preprocessing is needed — gpt-4o reads the raw crop.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
sys.path.insert(0, str(HERE))      # for ocr_probe (shared frame extraction)
sys.path.insert(0, str(API_DIR))   # for vision.vlm_read

from ocr_probe import extract_frame, load_dotenv, DEFAULT_VIDEO  # noqa: E402

OPENAI_HINT = """\
Azure OpenAI is not configured. Provision a gpt-4o deployment (you already have the
resource group from the Vision step):

  az cognitiveservices account create -n va-openai -g video-analyst-rg -l eastus \\
    --kind OpenAI --sku S0 --yes
  az cognitiveservices account deployment create -g video-analyst-rg -n va-openai \\
    --deployment-name gpt-4o --model-name gpt-4o --model-version 2024-11-20 \\
    --model-format OpenAI --sku-name GlobalStandard --sku-capacity 10
  az cognitiveservices account show     -n va-openai -g video-analyst-rg --query properties.endpoint -o tsv
  az cognitiveservices account keys list -n va-openai -g video-analyst-rg --query key1 -o tsv

Then add to api/.env:
  AZURE_OPENAI_ENDPOINT=<endpoint>
  AZURE_OPENAI_KEY=<key1>
  AZURE_OPENAI_DEPLOYMENT=gpt-4o
  AZURE_OPENAI_API_VERSION=2024-10-21

If the deployment create fails on capacity/version, try --model-version 2024-08-06 or
a different region; list options with: az cognitiveservices account list-models -n va-openai -g video-analyst-rg\
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="gpt-4o vision jersey-number probe")
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--ts", type=float, default=4.6)
    ap.add_argument("--crop", default=None, help="pixel crop 'x,y,w,h'")
    ap.add_argument("--scale", type=float, default=1.0, help="upscale factor (gpt-4o rarely needs it)")
    ap.add_argument("--image", type=Path, default=None, help="use this image instead of extracting")
    ap.add_argument("--expect", default="10", help="jersey number we expect (default 10)")
    args = ap.parse_args()

    load_dotenv(API_DIR / ".env")
    if not (os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_KEY")):
        print(OPENAI_HINT, file=sys.stderr)
        return 2

    if args.image:
        if not args.image.exists():
            print(f"ERROR: image not found: {args.image}", file=sys.stderr)
            return 2
        image_bytes = args.image.read_bytes()
        source = str(args.image)
    else:
        image_bytes = extract_frame(args.video, args.ts, args.crop, args.scale)
        source = f"{args.video.name} @ {args.ts}s" + (f" crop={args.crop}" if args.crop else "") + (f" x{args.scale}" if args.scale != 1.0 else "")

    try:
        from vision.vlm_read import AzureVLM
        out = AzureVLM.from_env().read_jersey_number(image_bytes)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR calling Azure OpenAI: {e}", file=sys.stderr)
        return 2

    print(f"# source: {source}")
    print(f"gpt-4o read -> raw={out.raw!r}  digits={out.digits!r}\n")

    want = str(args.expect).strip()
    if out.digits == want:
        print(f"T1 PASS — gpt-4o read {want!r} off the jersey (exact). Live read works.")
        return 0
    if want in out.digits:
        print(f"T1 PARTIAL — expected {want!r}, got {out.digits!r} (contains it). Check the crop.")
        return 1
    print(f"T1 FAIL — expected {want!r}, gpt-4o read {out.digits or out.raw!r}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
