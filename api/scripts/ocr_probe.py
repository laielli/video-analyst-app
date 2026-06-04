#!/usr/bin/env python3
"""
T1 / D16 OCR harness — the kill-criterion check: does Azure AI Vision actually read
the scorer's jersey number off OUR footage?

It extracts a frame from the hero clip (single-goal.mov) at a timestamp, optionally
crops to the jersey region, runs Azure detect + read, prints what it found, and reports
whether the expected number (default "10", Messi) appears in the OCR output.

    # full T1 check on the scorer frame (Messi #10 legible ~4.6s):
    python ocr_probe.py --ts 4.6 --expect 10

    # tighter test: crop to Messi's torso first (px x,y,w,h on the 1872x1042 frame):
    python ocr_probe.py --ts 4.6 --crop 90,80,560,520 --expect 10

    # run against an already-extracted image instead of the video:
    python ocr_probe.py --image ~/.gstack/.../clip/scorer-10-crop.png --expect 10

Exit 0 = expected number read (T1 PASS) · 1 = not found (T1 FAIL) · 2 = setup problem.

Prereqs: a provisioned Azure AI Vision resource. Put creds in api/.env (see .env.example)
or export AZURE_VISION_ENDPOINT / AZURE_VISION_KEY. ffmpeg must be on PATH for extraction.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
API_DIR = HERE.parent
REPO_ROOT = API_DIR.parent
DEFAULT_VIDEO = REPO_ROOT / "single-goal.mov"

PROVISION_HINT = """\
Azure AI Vision is not configured. The clip looks great visually; this is the last
step to close T1 for real. Minimal provision (you have an Azure account):

  az login
  az group create -n video-analyst-rg -l eastus
  az cognitiveservices account create \\
    -n va-vision -g video-analyst-rg -l eastus \\
    --kind ComputerVision --sku S1 --yes
  az cognitiveservices account show    -n va-vision -g video-analyst-rg \\
    --query properties.endpoint -o tsv          # -> AZURE_VISION_ENDPOINT
  az cognitiveservices account keys list -n va-vision -g video-analyst-rg \\
    --query key1 -o tsv                          # -> AZURE_VISION_KEY

Then copy api/.env.example to api/.env, paste both values, and re-run this script.
(ComputerVision S1 is the Image Analysis 4.0 resource; scales by usage, cents for the demo.)\
"""


def load_dotenv(env_path: Path) -> None:
    """Minimal .env loader (no dependency). Does not override already-set vars."""
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def extract_frame(video: Path, ts: float, crop: str | None, scale: float = 1.0) -> bytes:
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH (needed to extract the frame).", file=sys.stderr)
        sys.exit(2)
    if not video.exists():
        print(f"ERROR: video not found: {video}", file=sys.stderr)
        sys.exit(2)
    vf = []
    if crop:
        try:
            x, y, w, h = (int(n) for n in crop.split(","))
        except ValueError:
            print("ERROR: --crop must be 'x,y,w,h' in pixels.", file=sys.stderr)
            sys.exit(2)
        vf.append(f"crop={w}:{h}:{x}:{y}")
    if scale and scale != 1.0:
        # Upscale (lanczos) so small jersey numbers clear Azure Read's minimum glyph size.
        vf.append(f"scale=iw*{scale}:ih*{scale}:flags=lanczos")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        out = Path(tmp.name)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(ts), "-i", str(video)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-frames:v", "1", str(out)]
    subprocess.run(cmd, check=True)
    data = out.read_bytes()
    out.unlink(missing_ok=True)
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="T1 OCR harness: does Azure read the jersey number?")
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="hero clip (default: repo single-goal.mov)")
    ap.add_argument("--ts", type=float, default=4.6, help="timestamp in seconds (default 4.6 = #10 legible)")
    ap.add_argument("--crop", default=None, help="optional pixel crop 'x,y,w,h' before OCR")
    ap.add_argument("--scale", type=float, default=1.0, help="upscale factor after crop (helps Read on small numbers, e.g. 3)")
    ap.add_argument("--image", type=Path, default=None, help="use this image instead of extracting from --video")
    ap.add_argument("--expect", default="10", help="jersey number we expect to read (default 10)")
    ap.add_argument("--out", type=Path, default=None, help="write normalized results JSON here")
    args = ap.parse_args()

    load_dotenv(API_DIR / ".env")
    if not os.environ.get("AZURE_VISION_ENDPOINT") or not os.environ.get("AZURE_VISION_KEY"):
        print(PROVISION_HINT, file=sys.stderr)
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
        from vision.azure_vision import AzureVision  # noqa: E402  (after dotenv load)
    except ImportError:
        sys.path.insert(0, str(API_DIR))
        from vision.azure_vision import AzureVision  # type: ignore

    try:
        result = AzureVision.from_env().analyze(image_bytes, detect=True, read=True)
    except Exception as e:  # noqa: BLE001 — surface any SDK/auth error plainly
        print(f"ERROR calling Azure AI Vision: {e}", file=sys.stderr)
        return 2

    print(f"# source: {source}  ({result.width}x{result.height})\n")

    print(f"detect -> {len(result.objects)} object(s):")
    for d in result.objects:
        b = d.box.rounded()
        print(f"  - {d.cls:<10} conf={d.confidence}  box=({b.x},{b.y},{b.w},{b.h})")

    print(f"\nread_text -> {len(result.lines)} line(s):")
    for l in result.lines:
        b = l.box.rounded()
        print(f"  - {l.text!r:<16} conf={l.confidence}  box=({b.x},{b.y},{b.w},{b.h})")

    if args.out:
        args.out.write_text(json.dumps({"source": source, **result.to_dict()}, indent=2))
        print(f"\nwrote {args.out}")

    want = str(args.expect).strip().lower()
    hits = [l for l in result.lines if want in l.text.strip().lower()]
    print()
    if hits:
        best = max(hits, key=lambda l: (l.confidence or 0))
        print(f"T1 PASS — read {want!r} in {best.text!r} (conf={best.confidence}).")
        print("Azure reads the scorer's number off this footage. Kill-criterion retired.")
        return 0
    print(f"T1 FAIL — {want!r} not found in any OCR line.")
    print("Try --crop to tighten on the jersey, a different --ts, or hand-correct the cache for the hero query (design doc: 'The Assignment').")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
